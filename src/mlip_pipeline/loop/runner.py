from __future__ import annotations
import copy
from pathlib import Path
import yaml

from mlip_pipeline.config import build_gen_config, project_paths, gen_paths, load_yaml
from mlip_pipeline.models import GenerationState, FitResult, LabelResult, STEPS
from mlip_pipeline.utils.fs import ensure_dir, write_json
from mlip_pipeline.utils.logging import step_header, info, warn, success, error


def run_single_generation(
    base_config: dict,
    generation: int,
    *,
    force: bool = False,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
) -> GenerationState:
    config  = build_gen_config(base_config, generation)
    resolved = project_paths(config)
    paths = gen_paths(config, resolved)

    gen_tag = f"gen_{str(generation).zfill(2)}"

    # Inject per-gen subdirs for all steps
    config["fit"]["output_subdir"]     = f"{gen_tag}/fit"
    config["explore"]["output_subdir"] = f"{gen_tag}/explore"
    config["select"]["output_subdir"]  = f"{gen_tag}/select"
    config["select"]["input_subdir"]   = f"{gen_tag}/explore"
    config["label"]["output_subdir"]   = f"{gen_tag}/label"
    config["label"]["input_subdir"]    = f"{gen_tag}/select"

    # convert: inject a per-gen storage subdir but do NOT touch output_subdir
    # (output_subdir = accumulation root = where train.cfg lives, must be stable)
    config.setdefault("convert", {})["gen_subdir"] = f"{gen_tag}/convert"

    gen_dir = paths["gen_dir"]
    ensure_dir(gen_dir)

    # Snapshot config for reproducibility (written once, never overwritten)
    snapshot = gen_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.dump(config, default_flow_style=False))

    state_file = gen_dir / "state.json"
    if state_file.exists():
        state = GenerationState.load(gen_dir)
        if state.status == "completed":
            success(f"Generation {generation:02d} already completed — skipping.")
            return state
    else:
        state = GenerationState.init(str(generation).zfill(2), gen_dir)

    step_header(f"Generation {generation:02d}", str(generation).zfill(2))

    active = [
        s for s in STEPS
        if (only_steps is None or s in only_steps)
        and s not in (skip_steps or [])
    ]

    try:
        for step in active:
            if state.is_step_done(step):
                info(f"  step '{step}' already done — skipping.")
                continue

            state.mark_step_start(step)
            step_header(step.upper())

            # ── fit ────────────────────────────────────────────────────
            if step == "fit":
                from mlip_pipeline.fit.trainer import train_potential
                result = train_potential(config, paths)
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
                explore_dir = create_exploration_runs(config, paths, fit_result)
                run_exploration_runs(config, paths, explore_dir)

            # ── select ──────────────────────────────────────────────────
            elif step == "select":
                fit_dir = paths.get("fit_dir", paths["runs_root"] / "fit")
                fit_result = (
                    FitResult(run_dir=fit_dir, model_path=Path(state.model_path))
                    if state.model_path
                    else FitResult.load_manifest(fit_dir)
                )
                from mlip_pipeline.select.runner import run_selection
                run_selection(config, paths, fit_result)

            # ── label (prepare VASP input dirs) ─────────────────────────
            elif step == "label":
                from mlip_pipeline.label.runner import run_labeling
                run_labeling(config, paths)

            # ── label_local (run VASP locally via mpirun) ───────────────
            elif step == "label_local":
                from mlip_pipeline.label.local_runner import run_vasp_local
                label_dir = paths.get("label_dir",
                    paths["runs_root"] / config["label"]["output_subdir"])
                label_result = LabelResult.load_manifest(label_dir)
                run_vasp_local(label_result, config)

            # ── label_hpc (submit to Dardel + wait for OUTCARs) ─────────
            elif step == "label_hpc":
                from mlip_pipeline.label.runner import run_labeling
                from mlip_pipeline.io.dardel import submit_label_jobs
                label_result = run_labeling(config, paths)
                submit_label_jobs(label_result, config)
                label_root = label_result.label_root
                outcars = list(label_root.glob("task.*/OUTCAR"))
                if not outcars:
                    raise RuntimeError(
                        f"label_hpc: no OUTCARs found in {label_root}. "
                        "VASP jobs likely failed — check job.sh env_block and vasp_cmd."
                    )

            # ── convert ───────────────────────────────────────────────
            elif step == "convert":
                label_dir = paths.get("label_dir",
                    paths["runs_root"] / config["label"]["output_subdir"])
                label_result = LabelResult.load_manifest(label_dir)
                from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
                # returns Path to the merged train.cfg in the accumulation root
                merged_cfg_path = convert_outcars_to_cfg(label_result, config, paths)
                state.merged_cfg = str(merged_cfg_path)

            # ── mark done ──────────────────────────────────────────────
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
    """
    Run generations start_gen … end_gen (inclusive) sequentially.

    - Fully idempotent: completed gens and steps are skipped on re-run.
    - Passes merged_cfg from gen N → gen N+1 automatically.
    - Interrupt anytime; re-run with the same command to resume.
    """
    base_config = load_yaml(base_config_path)
    states: list[GenerationState] = []
    failed: list[int] = []

    for gen in range(start_gen, end_gen + 1):
        info(f"\n{'='*60}\nStarting generation {gen:02d} / {end_gen:02d}\n{'='*60}")

        # Chain: propagate previous gen's merged_cfg as next gen's train set
        if states and states[-1].merged_cfg:
            base_config = copy.deepcopy(base_config)
            base_config.setdefault("training", {})["origin_cfg"] = states[-1].merged_cfg
            info(f"Using gen_{gen-1:02d} merged_cfg as training set.")

        try:
            state = run_single_generation(
                base_config, gen,
                skip_steps=skip_steps,
                only_steps=only_steps,
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
