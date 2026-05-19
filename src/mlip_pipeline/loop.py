from __future__ import annotations

"""
loop.py — Generation loop runner.

Called by `mlip-pipeline run-loop` via cli.py.
Manages per-generation state, skips completed steps, and runs
each step in order. Supports --local mode which replaces the
Dardel HPC step with a local VASP run.
"""

import traceback
from pathlib import Path

from mlip_pipeline.models import GenerationState, LabelResult


# Steps executed in --local mode (no Dardel)
LOCAL_STEPS = ["fit", "explore", "select", "label", "label_local", "convert", "evaluate"]

# Steps executed in HPC mode (Dardel)
HPC_STEPS   = ["fit", "explore", "select", "label", "label_hpc",   "convert", "evaluate"]


def _gen_tag(gen: int) -> str:
    return f"gen_{gen:02d}"


def run_loop(
    config: dict,
    resolved_paths: dict,
    start_gen: int,
    end_gen: int,
    local: bool = False,
) -> None:
    """
    Run generations [start_gen, end_gen] inclusive.
    Failed generations are logged; the loop continues with the next.
    """
    from mlip_pipeline.fit.trainer import train_potential
    from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
    from mlip_pipeline.explore.runner import run_exploration_runs
    from mlip_pipeline.select.runner import run_selection
    from mlip_pipeline.label.runner import run_labeling
    from mlip_pipeline.label.local_runner import run_vasp_local
    from mlip_pipeline.io.dardel import submit_label_jobs
    from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
    from mlip_pipeline.evaluate.runner import run_evaluation
    from mlip_pipeline.models import FitResult

    runs_root = resolved_paths["runs_root"]
    steps = LOCAL_STEPS if local else HPC_STEPS
    failures = []

    print(f"\n{'='*60}")
    print(f"Starting generation {start_gen:02d} / {end_gen:02d}")
    print(f"{'='*60}")

    for gen in range(start_gen, end_gen + 1):
        tag     = _gen_tag(gen)
        gen_dir = runs_root / tag
        gen_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"\u25b6  Generation {gen:02d} \u00b7 {tag}")
        print(f"{'─'*60}")

        # Load or initialise state
        state_file = gen_dir / "state.json"
        if state_file.exists():
            state = GenerationState.load(gen_dir)
        else:
            state = GenerationState.init(generation=tag, gen_dir=gen_dir)

        # --- helpers that inject gen-specific config overrides -------------
        def _config_for_gen() -> dict:
            """Return a shallow config copy with generation-specific overrides."""
            import copy
            cfg = copy.deepcopy(config)
            cfg["generation"] = tag
            # Inject gen-specific subdirs so each step writes to gen_NN/
            for section in ("fit", "explore", "select", "label", "convert"):
                if section in cfg:
                    cfg[section].setdefault("output_subdir", f"{tag}/{section}")
                    # Override so files land under gen_NN/
                    cfg[section]["output_subdir"] = f"{tag}/{section}"
            return cfg

        gen_cfg = _config_for_gen()

        def _build_fit_result() -> FitResult:
            fit_dir = runs_root / tag / "fit"
            return FitResult(
                run_dir=fit_dir,
                model_path=fit_dir / config["fit"]["trained_potential_name"],
                log_path=fit_dir / "train.log",
            )

        def _build_label_result() -> LabelResult:
            label_root = runs_root / tag / "label"
            return LabelResult.load_manifest(label_root)

        try:
            for step in steps:
                if state.is_step_done(step):
                    _ts = __import__('datetime').datetime.now().strftime('%H:%M:%S')
                    print(f"{_ts}    step '{step}' already done \u2014 skipping.")
                    continue

                _ts = __import__('datetime').datetime.now().strftime('%H:%M:%S')
                print(f"\n{'─'*60}")
                print(f"\u25b6  {step.upper()}")
                print(f"{'─'*60}")

                state.mark_step_start(step)

                # ── fit ────────────────────────────────────────────────
                if step == "fit":
                    result = train_potential(gen_cfg, resolved_paths)
                    state.model_path = str(result.model_path)

                # ── explore ────────────────────────────────────────────
                elif step == "explore":
                    fit_result = _build_fit_result()
                    explore_root = create_exploration_runs(gen_cfg, resolved_paths, fit_result)
                    run_exploration_runs(gen_cfg, resolved_paths, explore_root)

                # ── select ─────────────────────────────────────────────
                elif step == "select":
                    fit_result = _build_fit_result()
                    run_selection(gen_cfg, resolved_paths, fit_result)

                # ── label (create VASP input dirs) ─────────────────────
                elif step == "label":
                    run_labeling(gen_cfg, resolved_paths)

                # ── label_local (run VASP locally) ──────────────────────
                elif step == "label_local":
                    label_result = _build_label_result()
                    run_vasp_local(label_result, gen_cfg)

                # ── label_hpc (submit to Dardel + sync back) ───────────
                elif step == "label_hpc":
                    label_result = _build_label_result()
                    submit_label_jobs(label_result, gen_cfg)

                # ── convert ────────────────────────────────────────────
                elif step == "convert":
                    label_root = runs_root / tag / "label"
                    label_result = LabelResult.load_manifest(label_root)
                    result = convert_outcars_to_cfg(label_result, gen_cfg, resolved_paths)
                    state.merged_cfg = str(result.merged_cfg)

                # ── evaluate ───────────────────────────────────────────
                elif step == "evaluate":
                    fit_result = _build_fit_result()
                    run_evaluation(gen_cfg, resolved_paths, fit_result)

                state.mark_step_done(step)

            state.mark_done()
            print(f"\n\u2713 Generation {gen:02d} complete.")

        except Exception as exc:  # noqa: BLE001
            state.mark_failed(exc)
            print(f"\nERROR Generation {gen:02d} failed at '{state.current_step}': {exc}")
            traceback.print_exc()
            failures.append(gen)
            print("Stopping. Re-run with same args to resume.")
            break

    if failures:
        print(f"\nWARNING Loop complete with {len(failures)} failure(s): {failures}")
    else:
        print(f"\n\u2713 All generations {start_gen:02d}\u2013{end_gen:02d} completed successfully.")
