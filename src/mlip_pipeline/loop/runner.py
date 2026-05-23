from __future__ import annotations

"""
loop/runner.py — unified generation loop runner.

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
import threading
from pathlib import Path

import yaml

from mlip_pipeline.config import build_gen_config, project_paths, gen_paths, load_yaml
from mlip_pipeline.models import GenerationState, FitResult, LabelResult, LoopResult, STEPS
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.utils.logging import step_header, info, warn, success, error

_LABEL_STEPS = {"label", "label_local", "label_hpc", "convert"}
_RELABEL_TIMEOUT = 120  # seconds before auto-confirming re-label


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _ask_relabel(label_dir: Path, timeout: int = _RELABEL_TIMEOUT) -> bool:
    prompt = (
        f"\n[label_hpc] The label directory already exists.\n"
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


def _resolve_fit_result(config: dict, paths: dict, state: GenerationState) -> FitResult:
    """
    Build a FitResult for the evaluate step using the best available source:
      1. state.model_path  (set when fit ran inside the loop runner)
      2. fit_manifest.json (written by train_potential)
      3. config fallback   (construct from fit.output_subdir + trained_potential_name)

    Raises RuntimeError if none of the above yields an existing model file.
    """
    fit_dir = paths.get("fit_dir") or (paths["runs_root"] / config["fit"]["output_subdir"])

    # Source 1: state.model_path
    if state.model_path:
        mp = Path(state.model_path)
        if mp.exists():
            return FitResult(run_dir=fit_dir, model_path=mp)

    # Source 2: fit_manifest.json
    manifest = fit_dir / "fit_manifest.json"
    if manifest.exists():
        return FitResult.load_manifest(fit_dir)

    # Source 3: config fallback — construct path from config keys
    potential_name = config.get("fit", {}).get("trained_potential_name", "Pb.mtp")
    mp = fit_dir / potential_name
    if mp.exists():
        return FitResult(run_dir=fit_dir, model_path=mp)

    raise RuntimeError(
        f"Cannot locate trained model for generation {state.generation}.\n"
        f"  Checked: state.model_path={state.model_path!r}\n"
        f"           fit_manifest: {manifest}\n"
        f"           config fallback: {mp}\n"
        "Ensure the 'fit' step completed successfully for this generation."
    )


# ---------------------------------------------------------------------------
# Helper: load all GenerationState objects that exist on disk under runs_root
# ---------------------------------------------------------------------------

def _load_all_states(runs_root: Path) -> list[GenerationState]:
    """
    Walk runs_root and return a sorted list of ALL GenerationState objects
    found on disk (gen_NN/state.json), regardless of which invocation created
    them.

    This is the single source of truth used by both loop_manifest.json and
    loop summary plots, ensuring both always reflect the full run history
    including generations from earlier invocations or migration scripts.
    """
    result: list[tuple[int, GenerationState]] = []
    for state_path in sorted(runs_root.glob("gen_*/state.json")):
        try:
            state = GenerationState.load(state_path.parent)
            gen_num = int(state.generation)
            result.append((gen_num, state))
        except Exception as exc:  # noqa: BLE001
            warn(f"  loop_manifest: could not load {state_path}: {exc}")
    return [s for _, s in sorted(result)]


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

    gen_dir = paths["gen_dir"]
    ensure_dir(gen_dir)

    # Persist a config snapshot for reproducibility
    snapshot = gen_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.dump(config, default_flow_style=False))

    # ── Load or initialise generation state ────────────────────────────────────
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
                if state.status in ("completed", "converged"):
                    state.status = "running"
                    state.completed_at = None
                state.error = None
                state.save()
                warn(f"Generation {generation:02d}: --force reset step(s): {removed}")

        if state.status == "completed":
            success(f"Generation {generation:02d} already completed — skipping.")
            return state

        if state.status == "converged":
            success(
                f"Generation {generation:02d} already converged (0 candidates at all "
                "replicate tiers) — skipping."
            )
            return state

    else:
        state = GenerationState.init(str(generation), gen_dir)
        if prev_state is not None and prev_state.converged_replicate_tier > 0:
            state.replicate_tier = prev_state.converged_replicate_tier
            state.save()
            info(
                f"Generation {generation:02d}: inheriting replicate_tier="
                f"{state.replicate_tier} from gen_{int(prev_state.generation):02d}."
            )

    schedule = _replicate_schedule(config)
    config["explore"]["replicate"] = schedule[state.replicate_tier]

    skip_set  = set(skip_steps or [])
    active = [
        s for s in STEPS
        if not state.is_step_done(s)
        and s not in skip_set
        and (only_steps is None or s in only_steps)
    ]
    if not active:
        info(f"Generation {generation:02d}: all requested steps already done.")
        state.mark_done()
        report_path = state.save_convergence_report(schedule)
        info(f"Generation {generation:02d}: convergence report → {report_path}")
        success(f"Generation {generation:02d} complete.")
        return state

    label_dir = paths.get("label_dir", paths["runs_root"] / config["label"]["output_subdir"])

    SyncRetryExhausted: type = Exception
    _generation_converged = False

    try:
        for step in active:
            step_header(f"Generation {generation:02d} — {step}")
            state.mark_step_start(step)

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

            elif step == "explore":
                from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
                from mlip_pipeline.explore.runner import run_exploration_runs

                fit_result = _resolve_fit_result(config, paths, state)
                explore_dir = create_exploration_runs(config, resolved, fit_result)
                run_exploration_runs(config, resolved, explore_dir)

            elif step == "select":
                from mlip_pipeline.select.runner import run_selection

                fit_result = _resolve_fit_result(config, paths, state)
                sel_result = run_selection(config, resolved, fit_result)

                while sel_result.selected_count == 0:
                    next_tier = state.replicate_tier + 1
                    if next_tier >= len(schedule):
                        warn(
                            f"Generation {generation:02d}: 0 candidates at all "
                            f"{len(schedule)} replicate tier(s) — "
                            "potential is converged. Stopping loop."
                        )
                        _generation_converged = True
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

                if _generation_converged:
                    break

            elif step == "label":
                from mlip_pipeline.label.runner import run_labeling
                run_labeling(config, paths)
                state.label_prepared = True
                state.save()

            elif step == "label_local":
                from mlip_pipeline.label.local_runner import run_vasp_local
                label_result = LabelResult.load_manifest(label_dir)
                run_vasp_local(label_result, config)

            elif step == "label_hpc":
                from mlip_pipeline.label.runner import run_labeling
                from mlip_pipeline.io.dardel import (
                    submit_label_jobs,
                    sync_outputs_from_dardel,
                    SyncRetryExhausted,
                )

                if not state.label_prepared:
                    run_labeling(config, paths)
                    state.label_prepared = True
                    state.save()

                label_result = LabelResult.load_manifest(label_dir)
                outcars = list(label_result.label_root.glob("task.*/OUTCAR"))

                if state.jobs_submitted and outcars:
                    info(
                        f"  label_hpc: jobs already submitted and "
                        f"{len(outcars)} OUTCAR(s) found locally — "
                        "skipping sync-inputs / submit / watch, "
                        "retrying sync-back only."
                    )
                    dardel_cfg       = config["label"]["dardel"]
                    user             = dardel_cfg["user"]
                    host             = dardel_cfg.get("host", "dardel.pdc.kth.se")
                    remote_root      = dardel_cfg["remote_root"]
                    runs_root_p      = Path(config.get("runs_root", "runs"))
                    label_subdir     = config["label"]["output_subdir"]
                    remote_label_dir = f"{remote_root}/{runs_root_p}/{label_subdir}"
                    sync_outputs_from_dardel(
                        label_result.label_root, user, host, remote_label_dir,
                        dardel_cfg=dardel_cfg,
                    )

                elif state.jobs_submitted and not outcars:
                    warn(
                        f"  label_hpc: jobs_submitted=True but no OUTCARs found "
                        f"in {label_result.label_root} — VASP jobs likely failed. "
                        "Resetting and re-preparing label dirs."
                    )
                    state.jobs_submitted  = False
                    state.label_prepared  = False
                    state.save()
                    if label_dir.exists():
                        if _ask_relabel(label_dir):
                            warn(f"  Wiping {label_dir}.")
                            shutil.rmtree(label_dir)
                        else:
                            info("  Keeping existing label dir.")
                    run_labeling(config, paths)
                    state.label_prepared = True
                    state.save()
                    label_result = LabelResult.load_manifest(label_dir)
                    submit_label_jobs(label_result, config)
                    state.jobs_submitted = True
                    state.save()

                else:
                    submit_label_jobs(label_result, config)
                    state.jobs_submitted = True
                    state.save()

                if state.jobs_submitted and not state.evaluation_done:
                    try:
                        fit_result_bg = _resolve_fit_result(config, paths, state)
                    except RuntimeError as _e:
                        fit_result_bg = None
                        warn(f"  label_hpc: skipping background eval — {_e}")

                    if fit_result_bg is not None:
                        _config_snap   = copy.deepcopy(config)
                        _resolved_snap = copy.deepcopy(resolved)
                        _state_ref     = state

                        def _bg_eval(
                            cfg=_config_snap,
                            res=_resolved_snap,
                            fr=fit_result_bg,
                            st=_state_ref,
                        ):
                            info(f"  [bg-eval gen {st.generation}] Starting background evaluation...")
                            try:
                                from mlip_pipeline.evaluate.runner import run_evaluation
                                eval_result = run_evaluation(cfg, res, fr)
                                st.evaluation_done     = True
                                st.evaluation_failed   = False
                                st.evaluation_manifest = str(eval_result.eval_dir / "eval_manifest.json")
                                st.save()
                                info(
                                    f"  [bg-eval gen {st.generation}] Done — "
                                    f"{len(eval_result.plot_paths)} plot(s)."
                                )
                            except Exception as exc:  # noqa: BLE001
                                warn(f"  [bg-eval gen {st.generation}] FAILED: {exc}")
                                st.evaluation_done   = False
                                st.evaluation_failed = True
                                st.save()

                        bg_thread = threading.Thread(
                            target=_bg_eval,
                            daemon=True,
                            name=f"bg-eval-gen{generation:02d}",
                        )
                        bg_thread.start()
                        info(
                            f"  label_hpc: background evaluation thread started "
                            f"(thread={bg_thread.name})."
                        )

                outcars = list(label_result.label_root.glob("task.*/OUTCAR"))
                if not outcars:
                    raise RuntimeError(
                        f"label_hpc: no OUTCARs found in {label_result.label_root}. "
                        "VASP jobs likely failed — check job.sh env_block and vasp_cmd."
                    )

            elif step == "convert":
                from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
                label_result   = LabelResult.load_manifest(label_dir)
                convert_result = convert_outcars_to_cfg(label_result, config, resolved)
                state.merged_cfg = str(convert_result.merged_cfg)

            elif step == "evaluate":
                from mlip_pipeline.evaluate.runner import run_evaluation
                fit_result = _resolve_fit_result(config, paths, state)
                eval_result = run_evaluation(config, resolved, fit_result)
                state.evaluation_done     = True
                state.evaluation_failed   = False
                state.evaluation_manifest = str(eval_result.eval_dir / "eval_manifest.json")
                state.save()

            state.mark_step_done(step)

    except Exception as exc:
        if state.current_step == "label_hpc":
            if isinstance(exc, SyncRetryExhausted):
                warn(
                    "label_hpc: sync-back retry deadline reached — "
                    "label_prepared and jobs_submitted preserved for smart resume."
                )
            else:
                warn(
                    "label_hpc failed — state preserved for smart resume on next run."
                )
            state.mark_failed(exc)
        else:
            state.mark_failed(exc)
        raise

    if _generation_converged:
        state.converged_replicate_tier = state.replicate_tier
        state.status = "converged"
        state.current_step = None
        state.completed_at = _now()
        state.save()
        report_path = state.save_convergence_report(schedule)
        info(f"Generation {generation:02d}: convergence report → {report_path}")
        success(f"Generation {generation:02d} converged — loop will stop.")
        return state

    state.mark_done()
    report_path = state.save_convergence_report(schedule)
    info(f"Generation {generation:02d}: convergence report → {report_path}")
    success(f"Generation {generation:02d} complete.")
    return state


# ---------------------------------------------------------------------------
# Module-level _now() helper
# ---------------------------------------------------------------------------
from datetime import datetime as _datetime  # noqa: E402
def _now() -> str:
    return _datetime.utcnow().isoformat(timespec="seconds") + "Z"


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
    loop_started_at = _now()

    _cfg0     = _config_for_gen(base_config, start_gen)
    _resolved = project_paths(_cfg0)
    runs_root = _resolved["runs_root"]

    stop_reason = "completed"

    try:
        for gen in range(start_gen, end_gen + 1):
            info(f"\n{'='*60}\nStarting generation {gen:02d} / {end_gen:02d}\n{'='*60}")

            prev_state: GenerationState | None = states[-1] if states else None
            if prev_state is None and gen > 1:
                prev_gen_dir = resolved_paths_for(base_config, gen - 1)
                if prev_gen_dir and prev_gen_dir.exists():
                    try:
                        prev_state = GenerationState.load(prev_gen_dir)
                    except Exception:
                        warn(f"  Could not load state for gen_{gen - 1:02d} — skipping integrity check.")

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
                if stop_on_failure:
                    stop_reason = "failed"
                    error("Stopping. Re-run with same args to resume.")
                    break
                warn("stop_on_failure=False — continuing to next generation.")
                continue

            if state.status == "converged":
                stop_reason = "converged"
                success(
                    f"Potential converged at generation {gen:02d} — "
                    "no new structures found at any replicate tier. Loop complete."
                )
                break

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        warn("run_loop interrupted by user.")

    # ── Single source of truth: scan ALL gen_NN dirs on disk ──────────────────────
    # Both loop_manifest.json and loop summary plots consume this same list,
    # so they always agree and neither can drift from the other.
    all_disk_states = _load_all_states(runs_root)

    # ── Write loop_manifest.json ───────────────────────────────────────────
    if all_disk_states:
        try:
            loop_result = LoopResult.from_states(
                runs_root=runs_root,
                states=all_disk_states,
                start_gen=min(int(s.generation) for s in all_disk_states),
                requested_end_gen=end_gen,
                stop_reason=stop_reason,
                started_at=loop_started_at,
            )
            manifest_path = loop_result.save()
            info(f"Loop manifest → {manifest_path}")
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not write loop_manifest.json (non-fatal): {exc}")

    failed_count = sum(1 for s in all_disk_states if s.status == "failed")
    if failed_count:
        warn(f"Loop complete with {failed_count} failure(s).")
    else:
        success(f"Loop finished (last gen processed: {gen:02d}).")

    # ── Loop summary plots ───────────────────────────────────────────────
    # Use all_disk_states (same source as loop_manifest) so that plots always
    # cover the same generation range as the manifest — including gen_00 and
    # any generation from a prior invocation.  Include all generations that
    # have at least some data (completed, converged, or failed mid-way) so
    # that partial data for the last generation is still shown.
    plottable = [
        s for s in all_disk_states
        if s.status in ("completed", "converged", "failed")
    ]
    if len(plottable) >= 2:
        try:
            from mlip_pipeline.evaluate.loop_plots import collect_loop_records, plot_loop_summary

            plot_gens = [int(s.generation) for s in plottable]
            records   = collect_loop_records(runs_root, plot_gens)
            if records:
                output_dir = runs_root / "loop_summary"
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_paths = plot_loop_summary(records, output_dir)
                info(f"Loop summary plots written to {output_dir}:")
                for p in plot_paths:
                    info(f"  {p}")
        except Exception as exc:  # noqa: BLE001
            warn(f"Loop summary plotting failed (non-fatal): {exc}")

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
