from __future__ import annotations

"""
loop/runner.py — unified generation loop runner.

Combines:
  - replicate-tier scheduling, convergence reports, force/skip/only flags,
    stop_on_failure, _ask_relabel, label_prepared guard  (from old runner.py)
  - SyncRetryExhausted soft-failure, evaluate step, _config_for_gen helper
    (from old loop.py)

Public API (used by cli.py):
    run_loop(base_config_path, start_gen, end_gen, *, force, skip_steps,
             only_steps, stop_on_failure) -> list[GenerationState]
    run_single_generation(base_config, generation, *, force, skip_steps,
                          only_steps, prev_state) -> GenerationState
"""

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


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config_for_gen(base_config: dict, generation: int) -> dict:
    """
    Return a generation-specific config dict.

    Delegates to build_gen_config for any YAML-level overrides, then sets
    per-generation output_subdir for the steps that write into runs/gen_NN/.
    The 'convert' step is intentionally excluded — it must always accumulate
    into a single directory (datasets/converted_cfg/) across all generations.
    """
    cfg = build_gen_config(base_config, generation)
    tag = f"gen_{str(generation).zfill(2)}"
    for section in ("fit", "explore", "select", "label"):
        if section in cfg:
            cfg[section]["output_subdir"] = f"{tag}/{section}"
    cfg["select"]["input_subdir"] = f"{tag}/explore"
    cfg["label"]["input_subdir"]  = f"{tag}/select"
    cfg.setdefault("convert", {})["gen_subdir"] = tag
    return cfg


def _replicate_schedule(config: dict) -> list[list[int]]:
    schedule = config.get("explore", {}).get("replicate_schedule")
    if schedule:
        return [list(t) for t in schedule]
    return [list(config.get("explore", {}).get("replicate", [1, 1, 1]))]


def _prev_convert_dir(resolved: dict, config: dict, generation: int) -> Path | None:
    """Return the directory that holds gen N-1's convert_manifest.json."""
    if generation <= 1:
        return None
    convert_cfg  = config.get("convert", {})
    accum_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    prev_gen_tag = f"gen_{str(generation - 1).zfill(2)}"
    candidate    = resolved["datasets_root"] / accum_subdir / prev_gen_tag
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Single-generation runner
# ---------------------------------------------------------------------------

def run_single_generation(
    base_config: dict,
    generation: int,
    *,
    force: bool = False,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    prev_state: GenerationState | None = None,
) -> GenerationState:
    config   = _config_for_gen(base_config, generation)
    resolved = project_paths(config)
    paths    = gen_paths(config, resolved)

    gen_tag = f"gen_{str(generation).zfill(2)}"
    gen_dir = paths["gen_dir"]
    ensure_dir(gen_dir)

    # Persist a config snapshot for reproducibility
    snapshot = gen_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.dump(config, default_flow_style=False))

    # ── Load or initialise generation state ─────────────────────────────
    state_file = gen_dir / "state.json"
    if state_file.exists():
        state = GenerationState.load(gen_dir)

        if force:
            steps_to_force = set(only_steps) if only_steps else set(STEPS)
            removed = [s for s in state.completed_steps if s in steps_to_force]
            if removed:
                state.completed_steps = [
                    s for s in state.completed_steps if s not in steps_to_force
                ]
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

    # ── Replicate-tier inheritance from previous generation ──────────────
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

    # ── Step loop ────────────────────────────────────────────────────────
    try:
        for step in active:
            if state.is_step_done(step):
                info(f"  step '{step}' already done — skipping.")
                continue

            # label_prepared guard: skip re-preparing task dirs on resume
            if step == "label" and state.label_prepared:
                info("  step 'label' already prepared (task dirs on disk) — skipping.")
                state.mark_step_done("label")
                continue

            # ask before wiping an existing label dir
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

            # ── fit ───────────────────────────────────────────────────────
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
                        prev_convert_dir=_prev_convert_dir(resolved, config, generation),
                    )

                result = train_potential(config, resolved)
                state.model_path = str(result.model_path)

            # ── explore ───────────────────────────────────────────────────
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

            # ── select ────────────────────────────────────────────────────
            elif step == "select":
                from mlip_pipeline.select.runner import run_selection

                fit_dir = paths.get("fit_dir", paths["runs_root"] / "fit")
                fit_result = (
                    FitResult(run_dir=fit_dir, model_path=Path(state.model_path))
                    if state.model_path
                    else FitResult.load_manifest(fit_dir)
                )
                sel_result = run_selection(config, resolved, fit_result)

                # Auto-escalate replicate tier if no candidates found
                while sel_result.selected_count == 0:
                    next_tier = state.replicate_tier + 1
                    if next_tier >= len(schedule):
                        warn(
                            f"Generation {generation:02d}: no candidates at any "
                            "replicate tier — marking as converged."
                        )
                        for s in _LABEL_STEPS:
                            if s in active and not state.is_step_done(s):
                                state.mark_step_done(s)
                        break

                    next_rep    = schedule[next_tier]
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
                    sel_result  = run_selection(config, resolved, fit_result)

            # ── label (prepare VASP input dirs) ───────────────────────────
            elif step == "label":
                from mlip_pipeline.label.runner import run_labeling
                run_labeling(config, paths)
                state.label_prepared = True
                state.save()

            # ── label_local ───────────────────────────────────────────────
            elif step == "label_local":
                from mlip_pipeline.label.local_runner import run_vasp_local
                label_result = LabelResult.load_manifest(label_dir)
                run_vasp_local(label_result, config)

            # ── label_hpc ─────────────────────────────────────────────────
            elif step == "label_hpc":
                from mlip_pipeline.label.runner import run_labeling
                from mlip_pipeline.io.dardel import submit_label_jobs, SyncRetryExhausted

                if not state.label_prepared:
                    run_labeling(config, paths)
                    state.label_prepared = True
                    state.save()

                label_result = LabelResult.load_manifest(label_dir)
                try:
                    submit_label_jobs(label_result, config)
                except SyncRetryExhausted as exc:
                    # dardel.py already retried for 24 h — soft-fail this
                    # generation so the loop can continue to the next one.
                    if "label" in state.completed_steps:
                        state.completed_steps.remove("label")
                    state.label_prepared = False
                    warn(
                        "label_hpc failed — 'label' step has been reset so task "
                        "dirs will be re-prepared on next run."
                    )
                    state.mark_failed(exc)
                    error(
                        f"Generation {generation:02d} failed at 'label_hpc': "
                        f"rsync sync-back failed (exit 255) — SSH connection "
                        f"lost or remote path missing."
                    )
                    raise  # caught by run_loop; respects stop_on_failure

                outcars = list(label_result.label_root.glob("task.*/OUTCAR"))
                if not outcars:
                    raise RuntimeError(
                        f"label_hpc: no OUTCARs found in {label_result.label_root}. "
                        "VASP jobs likely failed — check job.sh env_block and vasp_cmd."
                    )

            # ── convert ───────────────────────────────────────────────────
            elif step == "convert":
                from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
                label_result   = LabelResult.load_manifest(label_dir)
                convert_result = convert_outcars_to_cfg(label_result, config, resolved)
                state.merged_cfg = str(convert_result.merged_cfg)

            # ── evaluate ──────────────────────────────────────────────────
            elif step == "evaluate":
                from mlip_pipeline.evaluate.runner import run_evaluation
                fit_dir = paths.get("fit_dir", paths["runs_root"] / "fit")
                fit_result = (
                    FitResult(run_dir=fit_dir, model_path=Path(state.model_path))
                    if state.model_path
                    else FitResult.load_manifest(fit_dir)
                )
                run_evaluation(config, resolved, fit_result)

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


# ---------------------------------------------------------------------------
# Multi-generation loop
# ---------------------------------------------------------------------------

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

        # Carry the previous generation's state for tier inheritance
        prev_state: GenerationState | None = states[-1] if states else None
        if prev_state is None and gen > 1:
            prev_gen_dir = resolved_paths_for(base_config, gen - 1)
            if prev_gen_dir and prev_gen_dir.exists():
                try:
                    prev_state = GenerationState.load(prev_gen_dir)
                except Exception:
                    warn(f"  Could not load state for gen_{gen - 1:02d} — skipping integrity check.")

        # Thread the accumulated training set forward
        if states and states[-1].merged_cfg:
            base_config = copy.deepcopy(base_config)
            base_config.setdefault("training", {})["origin_cfg"] = states[-1].merged_cfg
            info(f"Using gen_{gen - 1:02d} merged_cfg as training set.")

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
            warn("stop_on_failure=False — continuing to next generation.")

    if failed:
        warn(f"Loop complete with {len(failed)} failure(s): {failed}")
    else:
        success(f"All generations {start_gen}–{end_gen} complete.")

    return states


def resolved_paths_for(base_config: dict, generation: int) -> Path | None:
    """Return the gen_dir Path for a generation, or None on error."""
    try:
        config   = _config_for_gen(base_config, generation)
        resolved = project_paths(config)
        paths    = gen_paths(config, resolved)
        return paths["gen_dir"]
    except Exception:
        return None
