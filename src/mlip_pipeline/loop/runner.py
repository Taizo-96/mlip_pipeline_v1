from __future__ import annotations
import copy
import select as _select
import shutil
import sys
from pathlib import Path
import yaml

from mlip_pipeline.config import build_gen_config, project_paths, gen_paths, load_yaml
from mlip_pipeline.models import GenerationState, FitResult, LabelResult, STEPS
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.utils.logging import step_header, info, warn, success, error

_LABEL_STEPS = {"label", "label_local", "label_hpc", "convert"}
_RELABEL_TIMEOUT = 120  # seconds before auto-confirming re-label


def _ask_relabel(label_dir: Path, timeout: int = _RELABEL_TIMEOUT) -> bool:
    prompt = (
        f"\n[label_hpc retry] The previous VASP submission failed.\n"
        f"  Label dir : {label_dir}\n"
        f"  Wipe it and re-prepare label task dirs before re-submitting?\n"
        f"  [Y/n] (auto-yes in {timeout}s): "
    )
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = _select.select([sys.stdin], [], [], timeout)
    if ready:
        answer = sys.stdin.readline().strip().lower()
        return answer not in ("n", "no")
    sys.stdout.write("\n(timeout — defaulting to yes)\n")
    sys.stdout.flush()
    return True


def _replicate_schedule(config: dict) -> list[list[int]]:
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
    config.setdefault("convert", {})["gen_subdir"] = gen_tag

    gen_dir = paths["gen_dir"]
    ensure_dir(gen_dir)

    snapshot = gen_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.dump(config, default_flow_style=False))

    state_file = gen_dir / "state.json"
    if state_file.exists():
        state = GenerationState.load(gen_dir)

        if force:
            steps_to_force = set(only_steps) if only_steps else set(STEPS)
            removed = [s for s in state.completed_steps if s in steps_to_force]
            if removed:
                state.completed_steps = [s for s in state.completed_steps
                                          if s not in steps_to_force]
                if state.status == "completed":
                    state.status = "running"
                    state.completed_at = None
                state.error = None
                state.save()
                warn(f"Generation {generation:02d}: --force reset step(s): {removed}")

        if state.status == "completed":
            success(f"Generation {generation:02d} already completed — skipping.")
            return state
        if state.status == "failed":
            state.status = "running"
            state.error = None
            state.save()
    else:
        state = GenerationState.init(str(generation).zfill(2), gen_dir)

    # ── Replicate-tier inheritance ─────────────────────────────────────────
    # For a brand-new state (tier == 0) inherit from the previous generation's
    # *converged* tier so the cell never regresses.  We use
    # prev_state.converged_replicate_tier (set at mark_done) rather than
    # replicate_tier so we get the tier that actually *worked*, not one that
    # was merely reached mid-run.
    schedule = _replicate_schedule(config)
    if state.replicate_tier == 0 and prev_state is not None:
        inherited_tier = min(
            prev_state.converged_replicate_tier, len(schedule) - 1
        )
        if inherited_tier > 0:
            info(
                f"Generation {generation:02d}: inheriting replicate tier "
                f"{inherited_tier} ({schedule[inherited_tier]}) "
                f"from gen_{prev_state.generation}."
            )
            state.replicate_tier = inherited_tier
            state.save()

    tier = min(state.replicate_tier, len(schedule) - 1)
    config["explore"]["replicate"] = schedule[tier]

    step_header(f"Generation {generation:02d}", str(generation).zfill(2))

    active = [
        s for s in STEPS
        if (only_steps is None or s in only_steps)
        and s not in (skip_steps or [])
    ]

    label_dir = resolved["runs_root"] / config["label"]["output_subdir"]

    try:
        for step in active:
            if state.is_step_done(step):
                info(f"  step '{step}' already done — skipping.")
                continue

            if step == "label" and state.label_prepared:
                info(f"  step 'label' already prepared (task dirs on disk) — skipping.")
                state.mark_step_done("label")
                continue

            if step == "label" and label_dir.exists() and any(label_dir.iterdir()):
                if _ask_relabel(label_dir):
                    warn(f"  Wiping {label_dir} before re-labelling.")
                    shutil.rmtree(label_dir)
                else:
                    info("  Keeping existing label dir — jumping straight to submission.")
                    state.label_prepared = True
                    state.mark_step_done("label")
                    state.save()
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

                while sel_result.selected_count == 0:
                    next_tier = state.replicate_tier + 1
                    if next_tier >= len(schedule):
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

                    state.replicate_tier = next_tier
                    state.save()
                    config["explore"]["replicate"] = next_rep

                    explore_out = resolved["runs_root"] / config["explore"]["output_subdir"]
                    if explore_out.exists():
                        shutil.rmtree(explore_out)

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
                label_result = LabelResult.load_manifest(label_dir)
                run_vasp_local(label_result, config)

            # ── label_hpc ──────────────────────────────────────────
            elif step == "label_hpc":
                from mlip_pipeline.label.runner import run_labeling
                from mlip_pipeline.io.dardel import submit_label_jobs
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
                label_result = LabelResult.load_manifest(label_dir)
                from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
                merged_cfg_path = convert_outcars_to_cfg(label_result, config, resolved)
                state.merged_cfg = str(merged_cfg_path)

            state.mark_step_done(step)

    except Exception as exc:
        if state.current_step == "label_hpc":
            if "label" in state.completed_steps:
                state.completed_steps.remove("label")
            state.label_prepared = False
            warn(
                "label_hpc failed — 'label' step has been reset so task dirs "
                "will be re-prepared on next run."
            )
        state.mark_failed(exc)
        error(f"Generation {generation:02d} failed at '{state.current_step}': {exc}")
        raise

    state.mark_done()
    report_path = state.save_convergence_report(schedule)
    info(f"Generation {generation:02d}: convergence report → {report_path}")
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

        # ── Resolve prev_state from disk when not yet in the in-memory list ──
        # This covers the case where --start-gen N > 1, so states[] is empty
        # at the first iteration but gen N-1 already ran in a previous session.
        prev_state: GenerationState | None = states[-1] if states else None
        if prev_state is None and gen > 1:
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
                force=force,
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
    try:
        config   = build_gen_config(base_config, generation)
        resolved = project_paths(config)
        paths    = gen_paths(config, resolved)
        return paths["gen_dir"]
    except Exception:
        return None
