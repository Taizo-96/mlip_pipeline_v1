"""mlip_pipeline/evaluate/runner.py
Parses MLIP-3 output and drives all evaluation plots.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from mlip_pipeline.evaluate import plots


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    rmse_energy: float
    rmse_forces: float
    rmse_stress: float
    eval_dir: Path
    plot_paths: dict[str, Path] = field(default_factory=dict)


# ── MLIP-3 log parser ─────────────────────────────────────────────────────────

def _parse_mlip3_log(log_path: Path) -> dict:
    """
    Parse the MLIP-3 ``calc-grade`` / ``calc-errors`` output.

    The parser is section-aware and intentionally lenient:
    lines that don't match the expected column count are skipped.
    Sections are detected by header keywords written by MLIP-3.
    """
    energy_ref, energy_pred = [], []
    force_ref,  force_pred  = [], []
    stress_ref, stress_pred = [], []
    gamma_vals              = []

    in_energy_section  = False
    in_force_section   = False
    in_stress_section  = False
    in_grade_section   = False

    with open(log_path) as fh:
        lines = fh.readlines()

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        # section header detection
        if "energy" in low and ("ref" in low or "predicted" in low):
            in_energy_section = True
            in_force_section = in_stress_section = in_grade_section = False
            continue
        if "force" in low and ("ref" in low or "predicted" in low):
            in_force_section = True
            in_energy_section = in_stress_section = in_grade_section = False
            continue
        if "stress" in low and ("ref" in low or "predicted" in low):
            in_stress_section = True
            in_energy_section = in_force_section = in_grade_section = False
            continue
        if "grade" in low or "gamma" in low:
            in_grade_section = True
            in_energy_section = in_force_section = in_stress_section = False
            continue
        if stripped.startswith("#") or not stripped:
            continue

        nums = []
        for tok in stripped.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass

        if not nums:
            continue

        if in_energy_section and len(nums) >= 2:
            energy_ref.append(nums[0])
            energy_pred.append(nums[1])
        elif in_force_section and len(nums) >= 6:
            force_ref.append(nums[0:3])
            force_pred.append(nums[3:6])
        elif in_stress_section and len(nums) >= 12:
            stress_ref.append(nums[0:6])
            stress_pred.append(nums[6:12])
        elif in_grade_section and len(nums) >= 1:
            gamma_vals.append(nums[-1])

    return {
        "energy_ref":  np.array(energy_ref),
        "energy_pred": np.array(energy_pred),
        "force_ref":   np.array(force_ref)  if force_ref  else np.empty((0, 3)),
        "force_pred":  np.array(force_pred) if force_pred else np.empty((0, 3)),
        "stress_ref":  np.array(stress_ref)  if stress_ref  else np.empty((0, 6)),
        "stress_pred": np.array(stress_pred) if stress_pred else np.empty((0, 6)),
        "gamma":       np.array(gamma_vals),
    }


def _rmse(ref, pred) -> float:
    r = np.asarray(ref, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    n = min(len(r), len(p))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((r[:n] - p[:n]) ** 2)))


# ── public entry point ────────────────────────────────────────────────────────

def run_evaluation(
    config: dict,
    resolved_paths: dict,
    fit_result,
    *,
    config_labels=None,
    gamma_threshold: float | None = None,
) -> EvaluationResult:
    """
    Parse MLIP-3 evaluation log, compute RMSEs, and produce all plots.

    Parameters
    ----------
    config_labels :
        Optional list/array of string labels (one per structure) used for
        the per-config-type RMSE bar chart.  If None that plot is skipped.
    gamma_threshold :
        The gamma threshold used in active learning; drawn as a vertical line
        on the gamma histogram.  Falls back to config["select"]["gamma_break"]
        if present.
    """
    eval_cfg = config.get("evaluate", {})
    eval_dir = resolved_paths["runs_root"] / eval_cfg.get("output_subdir", "evaluate")
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = fit_result.log_path

    # ── parse ─────────────────────────────────────────────────────────────────
    parity = _parse_mlip3_log(log_path)

    rmse_e = _rmse(parity["energy_ref"],  parity["energy_pred"])
    rmse_f = _rmse(parity["force_ref"],   parity["force_pred"])
    rmse_s = _rmse(parity["stress_ref"],  parity["stress_pred"])

    print(f"  [loss]    rmse_e = {rmse_e:.6g}")
    print(f"  [loss]    rmse_f = {rmse_f:.6g}")
    print(f"  [loss]    rmse_s = {rmse_s:.6g}")

    all_paths: dict[str, Path] = {}

    # 1. parity plots (E, F per x/y/z, S per Voigt component)
    parity_paths = plots.plot_parity(parity, eval_dir)
    all_paths.update(parity_paths)

    # 2. error histograms
    hist_paths = plots.plot_error_histograms(parity, eval_dir)
    all_paths.update(hist_paths)

    # 3. gamma histogram
    if len(parity["gamma"]) > 0:
        thr = gamma_threshold or config.get("select", {}).get("gamma_break")
        p   = plots.plot_gamma_histogram(parity["gamma"], eval_dir, threshold=thr)
        all_paths["gamma"] = p
    else:
        print("  [info] no gamma values in log; skipping gamma histogram.")

    # 4. per-config-type RMSE (optional)
    if config_labels is not None:
        p = plots.plot_per_config_rmse(parity, config_labels, eval_dir)
        all_paths["per_config_rmse"] = p

    return EvaluationResult(
        rmse_energy=rmse_e,
        rmse_forces=rmse_f,
        rmse_stress=rmse_s,
        eval_dir=eval_dir,
        plot_paths=all_paths,
    )
