from __future__ import annotations
import copy
import shutil
from pathlib import Path
import yaml

from mlip_pipeline.config import build_gen_config, project_paths, gen_paths, load_yaml
from mlip_pipeline.models import GenerationState, FitResult, LabelResult, STEPS
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.utils.logging import step_header, info, warn, success, error

_LABEL_STEPS = {"label", "label_local", "label_hpc", "convert"}


def _replicate_schedule(config: dict) -> list[list[int]]:
    """Return the full replicate schedule, falling back to [replicate] if not set."""
    schedule = config.get("explore", {}).get("replicate_schedule")
    if schedule:
        return [list(t) for t in schedule]
    return [list(config.get("explore", {}).get("replicate", [1, 1, 1]))]


def run_single_generation(
    base_config: dict,
    generation: int,
    *,
    force: bool = False,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    prev_state: GenerationState | None = None,
) -> GenerationState:
    config   = build_gen_config(base_config, generation)
    resolved = project_paths(config)
    paths    = gen_paths(config, resolved)

    gen_tag = f"gen_{str(generation).zfill(2)}"

    config["fit"]["output_subdir"]     = f"{gen_tag}/fit"
    config["explore"]["output_subdir"] = f"{gen_tag}/explore"
    config["select"]["output_subdir"]  = f"{gen_tag}/select"
    config["select"]["input_subdir"]   = f"{gen_tag}/explore"
    config["label"]["output_subdir"]   = f"{gen_tag}/label"
    config["label"]["input_subdir"]    = f"{gen_tag}/select"
    config.setdefault("convert", {})["gen_subdir"] = f"{gen_tag}/convert"

    gen_dir = paths["gen_dir"]
    ensure_dir(gen_dir)

    snapshot = gen_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.dump(config, default_flow_style=False))

    state_file = gen_dir / "state.json"
    if state_file.exists():
        state = GenerationState.load(gen_dir)
        if state.status == "completed":
            success(f"Generation {generation:02d} already completed — skipping.")
            return state
        # Reset 'failed' status so steps can resume
        if state.status == "failed":
            state.status = "running"
            state.error = None
            state.save()
    else:
        state = GenerationState.init(str(generation).zfill(2), gen_dir)

    # Apply the replicate tier that was active when we last ran (for resume)
    schedule = _replicate_schedule(config)
    tier = min(state.replicate_tier, len(schedule) - 1)
    config["explore"]["replicate"] = schedule[tier]

    step_header(f"Generation {generation:02d}", str(generation).zfill(2))

    active = [
        s for s in STEPS
        if (only_steps is None or s in only_steps)
        and s not in (skip_steps or [])
    ]

    try:
        for step in active:
            # ———————————————————————————————————————————————
            # Skip logic: step already in completed_steps OR label already
            # written to disk (so label step never re-runs on hpc retry).
            # ———————————————————————————————————————————————
            if state.is_step_done(step):
                info(f"  step '{step}' already done — skipping.")
                continue

            # If label task dirs already exist on disk, skip label and go
            # straight to label_hpc / label_local without re-writing them.
            label_dir = resolved["runs_root"] / config["label"]["output_subdir"]
            if step == "label" and state.label_prepared:
                info(f"  step 'label' already prepared (task dirs on disk) — skipping.")
                state.mark_step_done("label")
                continue

            state.mark_step_start(step)
            step_header(step.upper())

            # ── fit ────────────────────────────────────────────────────
            if step == "fit":
                from mlip_pipeline.fit.trainer import train_potential, resolve_train_cfg
                from mlip_pipeline.checks import check_training_cfg_count

                train_cfg = resolve_train_cfg(config["fit"], config, resolved)

                if prev_state is not None:
                    prev_selected = (
                        resolved["runs_root"]
                        / f"gen_{str(generation - 1).zfill(2)}/select"
                        / config["select"].get("selected_filename", "selected.cfg")
                    )
                    check_training_cfg_count(
                        current_train_cfg=train_cfg,
                        prev_train_cfg=(
                            Path(prev_state.merged_cfg)
                            if prev_state.merged_cfg else None
                        ),
                        prev_selected_cfg=prev_selected,
                        generation=generation,
                        strict=True,
                    )

                result = train_potential(config, resolved)
                state.model_path = str(result.model_path)

            # ── explore ────────────────────────────────────────────────
            elif step == "explore":
                from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
                from mlip_pipeline.explore.runner import run_exploration_runs
                fit_dir = paths.get("fit_dir", paths["runs_root"] / "fit")
                fit_result = (
                    FitResult(run_dir=fit_dir, model_path=Path(state.model_path))
                    if state.model_path
                    else FitResult.load_manifest(fit_dir)
                )
                explore_dir = create_exploration_runs(config, resolved, fit_result)
                run_exploration_runs(config, resolved, explore_dir)

            # ── select ────────────────────────────────────────────────
            elif step == "select":
                fit_dir = paths.get("fit_dir", paths["runs_root"] / "fit")
                fit_result = (
                    FitResult(run_dir=fit_dir, model_path=Path(state.model_path))
                    if state.model_path
                    else FitResult.load_manifest(fit_dir)
                )
                from mlip_pipeline.select.runner import run_selection
                sel_result = run_selection(config, resolved, fit_result)

                # No candidates — try growing the cell via replicate_schedule
                while sel_result.selected_count == 0:
                    next_tier = state.replicate_tier + 1
                    if next_tier >= len(schedule):
                        # All tiers exhausted — genuinely converged
                        warn(
                            f"Generation {generation:02d}: no candidates at any "
                            f"replicate tier — marking as converged."
                        )
                        for s in _LABEL_STEPS:
                            if s in active and not state.is_step_done(s):
                                state.mark_step_done(s)
                        break

                    next_rep = schedule[next_tier]
                    current_rep = schedule[state.replicate_tier]
                    info(
                        f"Generation {generation:02d}: 0 candidates at replicate "
                        f"{current_rep} — growing cell to {next_rep} and re-running explore."
                    )

                    # Advance tier in state BEFORE wiping explore, so a crash
                    # here resumes at the new (bigger) tier, not the old one.
                    state.replicate_tier = next_tier
                    state.save()
                    config["explore"]["replicate"] = next_rep

                    # Wipe the previous explore output so LAMMPS inputs are
                    # regenerated with the new cell size.
                    explore_out = resolved["runs_root"] / config["explore"]["output_subdir"]
                    if explore_out.exists():
                        shutil.rmtree(explore_out)

                    # Re-run explore + select with bigger cell
                    from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
                    from mlip_pipeline.explore.runner import run_exploration_runs
                    explore_dir = create_exploration_runs(config, resolved, fit_result)
                    run_exploration_runs(config, resolved, explore_dir)
                    sel_result = run_selection(config, resolved, fit_result)

            # ── label ─────────────────────────────────────────────────
            elif step == "label":
                from mlip_pipeline.label.runner import run_labeling
                run_labeling(config, paths)
                state.label_prepared = True
                state.save()

            # ── label_local ──────────────────────────────────────────
            elif step == "label_local":
                from mlip_pipeline.label.local_runner import run_vasp_local
                label_dir = (
                    paths.get("label_dir")
                    or resolved["runs_root"] / config["label"]["output_subdir"]
                )
                label_result = LabelResult.load_manifest(label_dir)
                run_vasp_local(label_result, config)

            # ── label_hpc ──────────────────────────────────────────
            elif step == "label_hpc":
                from mlip_pipeline.label.runner import run_labeling
                from mlip_pipeline.io.dardel import submit_label_jobs
                label_dir = (
                    paths.get("label_dir")
                    or resolved["runs_root"] / config["label"]["output_subdir"]
                )
                # Only re-prepare label task dirs if they haven't been written yet
                if not state.label_prepared:
                    run_labeling(config, paths)
                    state.label_prepared = True
                    state.save()
                label_result = LabelResult.load_manifest(label_dir)
                submit_label_jobs(label_result, config)
                outcars = list(label_result.label_root.glob("task.*/OUTCAR"))
                if not outcars:
                    raise RuntimeError(
                        f"label_hpc: no OUTCARs found in {label_result.label_root}. "
                        "VASP jobs likely failed — check job.sh env_block and vasp_cmd."
                    )

            # ── convert ──────────────────────────────────────────────
            elif step == "convert":
                label_dir = (
                    paths.get("label_dir")
                    or resolved["runs_root"] / config["label"]["output_subdir"]
                )
                label_result = LabelResult.load_manifest(label_dir)
                from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
                merged_cfg_path = convert_outcars_to_cfg(label_result, config, resolved)
                state.merged_cfg = str(merged_cfg_path)

            state.mark_step_done(step)

    except Exception as exc:
        state.mark_failed(exc)
        error(f"Generation {generation:02d} failed at '{state.current_step}': {exc}")
        raise

    state.mark_done()
    success(f"Generation {generation:02d} complete.")
    return state


def run_loop(
    base_config_path: str | Path,
    start_gen: int,
    end_gen: int,
    *,
    force: bool = False,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    stop_on_failure: bool = True,
) -> list[GenerationState]:
    base_config = load_yaml(base_config_path)
    states: list[GenerationState] = []
    failed: list[int] = []

    for gen in range(start_gen, end_gen + 1):
        info(f"\n{'='*60}\nStarting generation {gen:02d} / {end_gen:02d}\n{'='*60}")

        prev_state: GenerationState | None = states[-1] if states else None
        if prev_state is None and gen > start_gen:
            prev_gen_dir = resolved_paths_for(base_config, gen - 1)
            if prev_gen_dir and prev_gen_dir.exists():
                try:
                    prev_state = GenerationState.load(prev_gen_dir)
                except Exception:
                    warn(f"  Could not load state for gen_{gen-1:02d} — skipping integrity check.")

        if states and states[-1].merged_cfg:
            base_config = copy.deepcopy(base_config)
            base_config.setdefault("training", {})["origin_cfg"] = states[-1].merged_cfg
            info(f"Using gen_{gen-1:02d} merged_cfg as training set.")

        try:
            state = run_single_generation(
                base_config, gen,
                skip_steps=skip_steps,
                only_steps=only_steps,
                prev_state=prev_state,
            )
            states.append(state)
        except Exception as exc:
            error(f"Generation {gen:02d} failed: {exc}")
            failed.append(gen)
            if stop_on_failure:
                error("Stopping. Re-run with same args to resume.")
                break
            warn("stop_on_failure=False — continuing.")

    if failed:
        warn(f"Loop complete with {len(failed)} failure(s): {failed}")
    else:
        success(f"All generations {start_gen}–{end_gen} complete.")

    return states


def resolved_paths_for(base_config: dict, generation: int) -> Path | None:
    """Return the gen_dir Path for a given generation, or None on error."""
    try:
        config   = build_gen_config(base_config, generation)
        resolved = project_paths(config)
        paths    = gen_paths(config, resolved)
        return paths["gen_dir"]
    except Exception:
        return None
