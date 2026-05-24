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


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def _rmse(ref: list[float], pred: list[float]) -> float:
    """Root-mean-square error between two equal-length float lists."""
    if not ref or not pred:
        return float("nan")
    n = min(len(ref), len(pred))
    return math.sqrt(sum((r - p) ** 2 for r, p in zip(ref[:n], pred[:n])) / n)


def _parity_rmse(parity: dict) -> tuple[float, float, float]:
    """
    Compute (rmse_energy, rmse_forces, rmse_stress) from a parity dict
    produced by build_parity_data().

    These are computed directly from DFT vs MTP residuals on the full
    training set -- the authoritative RMSE values for loop-summary plots.
    """
    rmse_e = _rmse(parity["energies_ref"], parity["energies_pred"])
    rmse_f = _rmse(parity["forces_ref"], parity["forces_pred"])
    rmse_s = (
        _rmse(parity["stress_ref"], parity["stress_pred"])
        if parity.get("stress_ref")
        else float("nan")
    )
    return rmse_e, rmse_f, rmse_s


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

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

    if not grade_cfgs:
        select_cfg  = config.get("select", {})
        select_root = resolved_paths["runs_root"] / select_cfg.get("output_subdir", "")
        sel_path    = select_root / select_cfg.get("selected_filename", "selected.cfg")
        if sel_path.exists():
            grade_cfgs = [sel_path]

    return grade_cfgs


# ---------------------------------------------------------------------------
# COMPUTE PHASE  (expensive -- calls mlp calculate_efs)
# ---------------------------------------------------------------------------

def run_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result: FitResult,
) -> EvaluationResult:
    """
    Full evaluation: run mlp calculate_efs, parse results, produce plots,
    and write eval_manifest.json.

    Persisted to <fit_dir>/eval/:
      predicted_train.cfg   -- MTP predictions (expensive, needed for replot)
      metrics.csv           -- RMSE scalars (parity-derived, canonical)
      gamma_grades.json     -- ALL raw gamma values with source provenance
      eval_manifest.json    -- typed summary + path to gamma_grades.json
    """
    fit_cfg     = config["fit"]
    mlp_cmd     = fit_cfg.get("mlp_command", "mlp")
    mpi_command = fit_cfg.get("mpi_command")
    mpi_np      = fit_cfg.get("mpi_np")
    eval_dir    = ensure_dir(fit_result.run_dir / "eval")

    log_path: Path | None = fit_result.log_path or (fit_result.run_dir / "train.log")

    # -- 1. Loss summary from train.log (kept as diagnostic _log values) --
    log_metrics: dict = {}
    if log_path is not None and log_path.exists():
        log_metrics = parse_train_log(log_path)
        if not log_metrics:
            print("  [loss]    WARNING: no RMSE summary found in train.log")
    else:
        print(f"  [loss]    WARNING: train.log not found at {log_path}")

    # -- 2. Parity data via calculate_efs  (THE EXPENSIVE STEP) -----------
    train_cfg     = _resolve_train_cfg(config, resolved_paths)
    predicted_cfg = eval_dir / "predicted_train.cfg"

    model_path = fit_result.resolve_model_path()

    ref_records:  list = []
    pred_records: list = []

    if train_cfg.exists() and model_path.exists():
        if mpi_command:
            _np_str = f" -n {mpi_np}" if mpi_np is not None else ""
            print(f"  [parity]  using MPI: {mpi_command}{_np_str} {mlp_cmd} calculate_efs ...")
        try:
            run_calculate_efs(
                mlp_cmd,
                model_path,
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
        if not model_path.exists():
            missing.append(f"model ({model_path})")
        print(f"  [parity]  WARNING: missing {', '.join(missing)}, skipping parity")

    # -- 3. Gamma data -- collect with provenance and persist -------------
    grade_cfgs    = _collect_grade_cfgs(config, resolved_paths)
    grade_records = parse_grades_with_provenance(grade_cfgs)
    grades_path: Path | None = None

    if grade_records:
        grades_path = write_grades_json(grade_records, eval_dir / GAMMA_GRADES_FILENAME)
        print(f"  [gamma]   {len(grade_records):,} grades persisted -> {grades_path.name}")
    else:
        print("  [gamma]   WARNING: no grade values found")

    # -- 4. Plots + manifest ----------------------------------------------
    return _make_plots_and_manifest(
        eval_dir=eval_dir,
        log_metrics=log_metrics,
        ref_records=ref_records,
        pred_records=pred_records,
        grade_records=grade_records,
        grades_path=grades_path,
        config=config,
    )


# ---------------------------------------------------------------------------
# PLOT PHASE  (cheap -- reads cached files)
# ---------------------------------------------------------------------------

def replot_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result: FitResult,
) -> EvaluationResult:
    """
    Regenerate all evaluation plots from cached intermediate files without
    re-running mlp calculate_efs.

    Requires predicted_train.cfg to exist (written by run_evaluation).
    gamma_grades.json is loaded if present; otherwise falls back to live
    cfg file parsing from explore/select dirs.
    """
    eval_dir      = fit_result.run_dir / "eval"
    predicted_cfg = eval_dir / "predicted_train.cfg"
    train_cfg     = _resolve_train_cfg(config, resolved_paths)

    if not predicted_cfg.exists():
        raise FileNotFoundError(
            f"predicted_train.cfg not found at {predicted_cfg}.\n"
            "Run 'evaluate' first to generate it."
        )

    # -- 1. Read log metrics for diagnostic reference only ----------------
    log_metrics: dict = {}
    metrics_csv = eval_dir / "metrics.csv"
    if metrics_csv.exists():
        import csv
        with metrics_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    log_metrics[row["metric"]] = float(row["value"])
                except (KeyError, ValueError):
                    pass
    else:
        log_path = fit_result.log_path or (fit_result.run_dir / "train.log")
        if log_path and log_path.exists():
            log_metrics = parse_train_log(log_path)

    # -- 2. Parse cached parity data --------------------------------------
    ref_records  = parse_cfg_efs(train_cfg)
    pred_records = parse_cfg_efs(predicted_cfg)
    print(f"  [replot]  {len(ref_records)} ref + {len(pred_records)} pred records from cache")

    # -- 3. Gamma: prefer cached json, fall back to live cfg parsing ------
    cached_grades_path = eval_dir / GAMMA_GRADES_FILENAME
    grade_records      = load_grades_json(cached_grades_path)
    grades_path: Path | None = cached_grades_path if grade_records else None

    if not grade_records:
        grade_cfgs    = _collect_grade_cfgs(config, resolved_paths)
        grade_records = parse_grades_with_provenance(grade_cfgs)
        if grade_records:
            grades_path = write_grades_json(grade_records, cached_grades_path)
            print(f"  [replot]  {len(grade_records):,} grades (re-)persisted -> {grades_path.name}")
    else:
        print(f"  [replot]  {len(grade_records):,} grades loaded from cache")

    # -- 4. Plots + manifest ----------------------------------------------
    return _make_plots_and_manifest(
        eval_dir=eval_dir,
        log_metrics=log_metrics,
        ref_records=ref_records,
        pred_records=pred_records,
        grade_records=grade_records,
        grades_path=grades_path,
        config=config,
    )


# ---------------------------------------------------------------------------
# Shared plotting + manifest helper
# ---------------------------------------------------------------------------

def _make_plots_and_manifest(
    eval_dir: Path,
    log_metrics: dict,
    ref_records: list,
    pred_records: list,
    grade_records: list[dict],
    grades_path: Path | None,
    config: dict,
) -> EvaluationResult:
    """
    Given already-loaded data, produce all PNG plots, write eval_manifest.json,
    and return an EvaluationResult.

    RMSE values written to metrics.csv and stored in EvaluationResult are
    computed from parity residuals (DFT vs MTP on the full training set).
    Log-parsed values are stored alongside with a _log suffix for diagnostics.
    """
    plot_paths: list[Path] = []

    # -- Parity plots + parity-derived RMSE (canonical) -------------------
    rmse_e = float("nan")
    rmse_f = float("nan")
    rmse_s = float("nan")

    if ref_records and pred_records:
        parity       = build_parity_data(ref_records, pred_records)
        parity_paths = plots.plot_parity(parity, eval_dir)
        plot_paths.extend(parity_paths)
        print(f"  [parity]  {len(ref_records)} configs -> {[p.name for p in parity_paths]}")

        rmse_e, rmse_f, rmse_s = _parity_rmse(parity)
        print(
            f"  [rmse]    parity-derived: "
            f"E={rmse_e:.6g}  F={rmse_f:.6g}  S={rmse_s:.6g}"
        )
    else:
        print("  [parity]  WARNING: no parity data available -- skipping parity plots")
        # Fall back to log values if parity is unavailable
        rmse_e = log_metrics.get("rmse_e", float("nan"))
        rmse_f = log_metrics.get("rmse_f", float("nan"))
        rmse_s = log_metrics.get("rmse_s", float("nan"))

    # -- Build canonical metrics dict (parity values are primary) ---------
    metrics: dict = {
        "rmse_e": rmse_e,
        "rmse_f": rmse_f,
        "rmse_s": rmse_s,
    }
    # Append log values as diagnostic reference
    for k, v in log_metrics.items():
        metrics.setdefault(f"{k}_log", v)

    # -- Per-quantity loss bar charts -------------------------------------
    if metrics:
        written = plots.plot_summary_metrics(metrics, eval_dir)
        plot_paths.extend(written)
        for k, v in metrics.items():
            print(f"  [loss]    {k} = {v:.6g}")
    else:
        print("  [loss]    WARNING: no metrics available for loss plots")

    # Write canonical metrics to CSV
    write_metrics_csv(metrics, eval_dir / "metrics.csv")

    # -- Gamma histogram --------------------------------------------------
    mean_gamma       = float("nan")
    max_gamma        = float("nan")
    frac_above_save  = float("nan")
    frac_above_break = float("nan")

    if grade_records:
        all_grades = grades_to_floats(grade_records)

        explore_cfg  = config.get("explore", {})
        al_cfg       = explore_cfg.get("active_learning", {})
        thresh_save  = al_cfg.get("threshold_save")
        thresh_break = al_cfg.get("threshold_break")
        thresholds   = {
            k: v
            for k, v in {"save": thresh_save, "break": thresh_break}.items()
            if v is not None
        }
        p = plots.plot_gamma_histogram(all_grades, thresholds, eval_dir / "gamma_hist.png")
        plot_paths.append(p)

        mean_gamma = float(sum(all_grades)) / len(all_grades)
        max_gamma  = float(max(all_grades))
        if thresh_save is not None:
            frac_above_save = sum(1 for g in all_grades if g > thresh_save) / len(all_grades)
        if thresh_break is not None:
            frac_above_break = sum(1 for g in all_grades if g > thresh_break) / len(all_grades)

        print(
            f"  [gamma]   {len(all_grades):,} grades -> {p.name}  "
            f"(mean={mean_gamma:.3f}, max={max_gamma:.3f})"
        )
    else:
        print("  [gamma]   WARNING: no grade values -- skipping histogram")

    print(f"\nEvaluation complete -- {len(plot_paths)} plot(s) in {eval_dir}/")

    result = EvaluationResult(
        rmse_energy=rmse_e,
        rmse_forces=rmse_f,
        rmse_stress=rmse_s,
        mean_gamma=mean_gamma,
        max_gamma=max_gamma,
        frac_above_save=frac_above_save,
        frac_above_break=frac_above_break,
        gamma_grades_path=grades_path,
        eval_dir=eval_dir,
        plot_paths={p.stem: p for p in plot_paths},
    )
    result.save_manifest()
    return result
