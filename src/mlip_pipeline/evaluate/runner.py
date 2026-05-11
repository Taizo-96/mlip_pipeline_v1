from __future__ import annotations

from pathlib import Path

from mlip_pipeline.models import FitResult, EvaluateResult
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


def run_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result: FitResult,
) -> EvaluateResult:
    fit_cfg   = config["fit"]
    mlp_cmd   = fit_cfg.get("mlp_command", "mlp")
    eval_dir  = ensure_dir(fit_result.run_dir / "eval")
    plot_paths: list[Path] = []
    metrics: dict = {}

    # ── 1. Loss summary from train.log ────────────────────────────────────────
    if fit_result.log_path.exists():
        metrics = parse_train_log(fit_result.log_path)
        if metrics:
            metrics_csv = write_metrics_csv(metrics, eval_dir / "metrics.csv")
            p = plots.plot_summary_metrics(metrics, eval_dir / "loss_summary.png")
            plot_paths.append(p)
            for k, v in metrics.items():
                print(f"  [loss]    {k} = {v:.6g}")
        else:
            print("  [loss]    WARNING: no RMSE summary found in train.log")
    else:
        print(f"  [loss]    WARNING: train.log not found at {fit_result.log_path}")

    # ── 2. Parity plots via calculate_efs ────────────────────────────────────
    train_cfg     = _resolve_train_cfg(config, resolved_paths)
    predicted_cfg = eval_dir / "predicted_train.cfg"

    if train_cfg.exists() and fit_result.model_path.exists():
        try:
            run_calculate_efs(mlp_cmd, fit_result.model_path, train_cfg, predicted_cfg)
            ref_records  = parse_cfg_efs(train_cfg)
            pred_records = parse_cfg_efs(predicted_cfg)
            parity       = build_parity_data(ref_records, pred_records)
            parity_paths = plots.plot_parity(parity, eval_dir)
            plot_paths.extend(parity_paths)
            print(f"  [parity]  {len(ref_records)} configs → {[p.name for p in parity_paths]}")
        except RuntimeError as exc:
            print(f"  [parity]  WARNING: {exc}")
    else:
        missing = []
        if not train_cfg.exists():
            missing.append(f"train_cfg ({train_cfg})")
        if not fit_result.model_path.exists():
            missing.append(f"model ({fit_result.model_path})")
        print(f"  [parity]  WARNING: missing {', '.join(missing)}, skipping parity")

    # ── 3. Gamma histogram ────────────────────────────────────────────────────
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

    if grade_cfgs:
        all_grades: list[float] = []
        for gc in grade_cfgs:
            all_grades.extend(parse_grades_from_cfg(gc))

        if all_grades:
            al_cfg = explore_cfg.get("active_learning", {})
            thresholds = {
                k: v
                for k, v in {
                    "save":  al_cfg.get("threshold_save"),
                    "break": al_cfg.get("threshold_break"),
                }.items()
                if v is not None
            }
            p = plots.plot_gamma_histogram(
                all_grades, thresholds, eval_dir / "gamma_hist.png"
            )
            plot_paths.append(p)
            print(
                f"  [gamma]   {len(all_grades):,} grades from "
                f"{len(grade_cfgs)} cfg(s) → {p.name}"
            )
        else:
            print("  [gamma]   WARNING: no grade values found in cfg files")
    else:
        print("  [gamma]   WARNING: no preselected/selected cfg found")

    print(f"\nEvaluation complete — {len(plot_paths)} plot(s) in {eval_dir}/")
    return EvaluateResult(
        eval_dir=eval_dir,
        metrics_csv=eval_dir / "metrics.csv",
        plots=plot_paths,
    )
