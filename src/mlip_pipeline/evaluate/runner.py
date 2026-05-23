from __future__ import annotations

import math
from pathlib import Path

from mlip_pipeline.models import FitResult, EvaluationResult
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.evaluate.log_parser import parse_train_log, write_metrics_csv
from mlip_pipeline.evaluate.parity import run_calculate_efs, parse_cfg_efs, build_parity_data
from mlip_pipeline.evaluate.gamma import parse_grades_from_cfg
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

    The intermediate ``predicted_train.cfg`` and ``metrics.csv`` are persisted
    to ``<fit_dir>/eval/`` so that :func:`replot_evaluation` can regenerate
    plots without re-running the expensive MTP inference.
    """
    fit_cfg     = config["fit"]
    mlp_cmd     = fit_cfg.get("mlp_command", "mlp")
    mpi_command = fit_cfg.get("mpi_command")
    mpi_np      = fit_cfg.get("mpi_np")
    eval_dir    = ensure_dir(fit_result.run_dir / "eval")
    plot_paths: list[Path] = []
    metrics: dict = {}

    log_path: Path | None = fit_result.log_path or (fit_result.run_dir / "train.log")

    # ── 1. Loss summary from train.log ───────────────────────────────────────
    if log_path is not None and log_path.exists():
        metrics = parse_train_log(log_path)
        if metrics:
            write_metrics_csv(metrics, eval_dir / "metrics.csv")
        else:
            print("  [loss]    WARNING: no RMSE summary found in train.log")
    else:
        print(f"  [loss]    WARNING: train.log not found at {log_path}")

    # ── 2. Parity data via calculate_efs  (THE EXPENSIVE STEP) ───────────────
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

    # ── 3. Gamma data ─────────────────────────────────────────────────────────
    grade_cfgs = _collect_grade_cfgs(config, resolved_paths)

    # ── 4. Plots (re-uses shared helper) ─────────────────────────────────────
    result = _make_plots_and_manifest(
        eval_dir=eval_dir,
        metrics=metrics,
        ref_records=ref_records,
        pred_records=pred_records,
        grade_cfgs=grade_cfgs,
        config=config,
    )
    return result


# ---------------------------------------------------------------------------
# PLOT PHASE  (cheap — reads cached predicted_train.cfg / metrics.csv)
# ---------------------------------------------------------------------------

def replot_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result: FitResult,
) -> EvaluationResult:
    """
    Regenerate all evaluation plots from cached intermediate files without
    re-running ``mlp calculate_efs``.

    Requires that :func:`run_evaluation` has been called at least once so that
    ``<eval_dir>/predicted_train.cfg`` and ``<eval_dir>/metrics.csv`` exist.
    Raises :class:`FileNotFoundError` if ``predicted_train.cfg`` is missing.
    """
    eval_dir      = fit_result.run_dir / "eval"
    predicted_cfg = eval_dir / "predicted_train.cfg"
    metrics_csv   = eval_dir / "metrics.csv"
    train_cfg     = _resolve_train_cfg(config, resolved_paths)

    if not predicted_cfg.exists():
        raise FileNotFoundError(
            f"predicted_train.cfg not found at {predicted_cfg}.\n"
            "Run 'evaluate' (or 'regenerate-eval') first to generate it."
        )

    # ── 1. Read cached metrics ─────────────────────────────────────────────
    metrics: dict = {}
    if metrics_csv.exists():
        import csv
        with metrics_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    metrics[row["metric"]] = float(row["value"])
                except (KeyError, ValueError):
                    pass
    else:
        # Fallback: re-parse train.log (still cheap)
        log_path = fit_result.log_path or (fit_result.run_dir / "train.log")
        if log_path and log_path.exists():
            metrics = parse_train_log(log_path)

    # ── 2. Parse cached parity data ────────────────────────────────────────
    ref_records  = parse_cfg_efs(train_cfg)
    pred_records = parse_cfg_efs(predicted_cfg)
    print(f"  [replot]  read {len(ref_records)} ref + {len(pred_records)} pred records from cache")

    # ── 3. Gamma data (always re-read from explore/select dirs) ────────────
    grade_cfgs = _collect_grade_cfgs(config, resolved_paths)

    # ── 4. Plots ───────────────────────────────────────────────────────────
    result = _make_plots_and_manifest(
        eval_dir=eval_dir,
        metrics=metrics,
        ref_records=ref_records,
        pred_records=pred_records,
        grade_cfgs=grade_cfgs,
        config=config,
    )
    return result


# ---------------------------------------------------------------------------
# Shared plotting + manifest helper
# ---------------------------------------------------------------------------

def _make_plots_and_manifest(
    eval_dir: Path,
    metrics: dict,
    ref_records: list,
    pred_records: list,
    grade_cfgs: list[Path],
    config: dict,
) -> EvaluationResult:
    """
    Given already-loaded data, produce all PNG plots, print progress lines,
    write eval_manifest.json, and return an :class:`EvaluationResult`.

    Called by both :func:`run_evaluation` and :func:`replot_evaluation`.
    """
    plot_paths: list[Path] = []

    # ── Loss summary bar chart ────────────────────────────────────────────
    if metrics:
        p = plots.plot_summary_metrics(metrics, eval_dir / "loss_summary.png")
        plot_paths.append(p)
        for k, v in metrics.items():
            print(f"  [loss]    {k} = {v:.6g}")
    else:
        print("  [loss]    WARNING: no metrics available for loss_summary plot")

    # ── Parity plots ──────────────────────────────────────────────────────
    if ref_records and pred_records:
        parity       = build_parity_data(ref_records, pred_records)
        parity_paths = plots.plot_parity(parity, eval_dir)
        plot_paths.extend(parity_paths)
        print(f"  [parity]  {len(ref_records)} configs → {[p.name for p in parity_paths]}")
    else:
        print("  [parity]  WARNING: no parity data available — skipping parity plots")

    # ── Gamma histogram ────────────────────────────────────────────────────
    mean_gamma      = float("nan")
    max_gamma       = float("nan")
    frac_above_save = float("nan")
    frac_above_break = float("nan")

    if grade_cfgs:
        all_grades: list[float] = []
        for gc in grade_cfgs:
            all_grades.extend(parse_grades_from_cfg(gc))

        if all_grades:
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
                f"  [gamma]   {len(all_grades):,} grades from "
                f"{len(grade_cfgs)} cfg(s) → {p.name}  "
                f"(mean={mean_gamma:.3f}, max={max_gamma:.3f})"
            )
        else:
            print("  [gamma]   WARNING: no grade values found in cfg files")
    else:
        print("  [gamma]   WARNING: no preselected/selected cfg found")

    print(f"\nEvaluation complete — {len(plot_paths)} plot(s) in {eval_dir}/")

    result = EvaluationResult(
        rmse_energy=metrics.get("rmse_e", float("nan")),
        rmse_forces=metrics.get("rmse_f", float("nan")),
        rmse_stress=metrics.get("rmse_s", float("nan")),
        mean_gamma=mean_gamma,
        max_gamma=max_gamma,
        frac_above_save=frac_above_save,
        frac_above_break=frac_above_break,
        eval_dir=eval_dir,
        plot_paths={p.stem: p for p in plot_paths},
    )
    result.save_manifest()
    return result
