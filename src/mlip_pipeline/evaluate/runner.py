from __future__ import annotations

import math
from pathlib import Path

from mlip_pipeline.models import FitResult, EvaluationResult
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.evaluate.log_parser import parse_train_log, write_metrics_csv
from mlip_pipeline.evaluate.parity import run_calculate_efs, parse_cfg_efs, build_parity_data
from mlip_pipeline.evaluate.gamma import (
    parse_grades_with_provenance,
    grades_to_floats,
    write_grades_json,
    load_grades_json,
    GAMMA_GRADES_FILENAME,
)
from mlip_pipeline.evaluate import plots


def _resolve_train_cfg(config: dict, resolved_paths: dict) -> Path:
    """Resolve training cfg with the same fallback chain as trainer.py."""
    fit_cfg = config["fit"]
    if "train_cfg" in fit_cfg:
        p = Path(fit_cfg["train_cfg"])
        return (resolved_paths["project_root"] / p).resolve() if not p.is_absolute() else p

    convert_cfg = config.get("convert", {})
    if convert_cfg.get("output_subdir"):
        return (
            resolved_paths["datasets_root"]
            / convert_cfg["output_subdir"]
            / convert_cfg.get("merge_name", "train.cfg")
        ).resolve()

    training_block = config["training"]
    return (
        resolved_paths["datasets_root"]
        / training_block.get("output_subdir", "pb_cfg")
        / training_block.get("merge_name", "train.cfg")
    ).resolve()


def _collect_grade_cfgs(config: dict, resolved_paths: dict) -> list[Path]:
    """Return the list of cfg files that contain extrapolation grades."""
    explore_cfg  = config.get("explore", {})
    explore_root = resolved_paths["runs_root"] / explore_cfg.get("output_subdir", "")
    presel_name  = (
        explore_cfg.get("active_learning", {})
        .get("save_extrapolative_to", "preselected.cfg")
    )

    grade_cfgs: list[Path] = []
    if explore_root.exists():
        grade_cfgs = sorted(explore_root.rglob(presel_name))

    # fallback: selected.cfg from select step
    if not grade_cfgs:
        select_cfg  = config.get("select", {})
        select_root = resolved_paths["runs_root"] / select_cfg.get("output_subdir", "")
        sel_path    = select_root / select_cfg.get("selected_filename", "selected.cfg")
        if sel_path.exists():
            grade_cfgs = [sel_path]

    return grade_cfgs


# ---------------------------------------------------------------------------
# COMPUTE PHASE  (expensive — calls mlp calculate_efs)
# ---------------------------------------------------------------------------

def run_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result: FitResult,
) -> EvaluationResult:
    """
    Full evaluation: run ``mlp calculate_efs``, parse results, produce plots,
    and write eval_manifest.json.

    The intermediate ``predicted_train.cfg``, ``metrics.csv``, and
    ``gamma_grades.json`` are persisted to ``<fit_dir>/eval/`` so that
    :func:`replot_evaluation` can regenerate plots without re-running the
    expensive MTP inference.
    """
    fit_cfg     = config["fit"]
    mlp_cmd     = fit_cfg.get("mlp_command", "mlp")
    mpi_command = fit_cfg.get("mpi_command")
    mpi_np      = fit_cfg.get("mpi_np")
    eval_dir    = ensure_dir(fit_result.run_dir / "eval")
    metrics: dict = {}

    log_path: Path | None = fit_result.log_path or (fit_result.run_dir / "train.log")

    # ── 1. Loss summary from train.log ──────────────────────────────────────
    if log_path is not None and log_path.exists():
        metrics = parse_train_log(log_path)
        if metrics:
            write_metrics_csv(metrics, eval_dir / "metrics.csv")
        else:
            print("  [loss]    WARNING: no RMSE summary found in train.log")
    else:
        print(f"  [loss]    WARNING: train.log not found at {log_path}")

    # ── 2. Parity data via calculate_efs  (THE EXPENSIVE STEP) ──────────────
    train_cfg     = _resolve_train_cfg(config, resolved_paths)
    predicted_cfg = eval_dir / "predicted_train.cfg"

    ref_records:  list = []
    pred_records: list = []

    if train_cfg.exists() and fit_result.model_path.exists():
        if mpi_command:
            _np_str = f" -n {mpi_np}" if mpi_np is not None else ""
            print(f"  [parity]  using MPI: {mpi_command}{_np_str} {mlp_cmd} calculate_efs ...")
        try:
            run_calculate_efs(
                mlp_cmd,
                fit_result.model_path,
                train_cfg,
                predicted_cfg,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
            )
            ref_records  = parse_cfg_efs(train_cfg)
            pred_records = parse_cfg_efs(predicted_cfg)
        except RuntimeError as exc:
            print(f"  [parity]  WARNING: calculate_efs failed: {exc}")
    else:
        missing = []
        if not train_cfg.exists():
            missing.append(f"train_cfg ({train_cfg})")
        if not fit_result.model_path.exists():
            missing.append(f"model ({fit_result.model_path})")
        print(f"  [parity]  WARNING: missing {', '.join(missing)}, skipping parity")

    # ── 3. Gamma data — collect with provenance and persist ─────────────────
    grade_cfgs    = _collect_grade_cfgs(config, resolved_paths)
    grade_records = parse_grades_with_provenance(grade_cfgs)
    grades_path: Path | None = None

    if grade_records:
        grades_path = write_grades_json(grade_records, eval_dir / GAMMA_GRADES_FILENAME)
        print(f"  [gamma]   {len(grade_records):,} grades persisted → {grades_path.name}")
    else:
        print("  [gamma]   WARNING: no grade values found")

    # ── 4. Plots + manifest ──────────────────────────────────────────────────
    result = _make_plots_and_manifest(
        eval_dir=eval_dir,
        metrics=metrics,
        ref_records=ref_records,
        pred_records=pred_records,
        grade_records=grade_records,
        gr