from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "font.family": "sans-serif",
}


def plot_summary_metrics(metrics: dict, dest: Path) -> Path:
    """
    Bar chart of final RMSE values from the train.log summary.
    """
    labels, values = [], []
    label_map = {
        "rmse_e": "Energy\n(eV/atom)",
        "rmse_f": "Forces\n(eV/Å)",
        "rmse_s": "Stress\n(pressure)",
    }
    colors = ["#01696f", "#964219", "#7a39bb"]

    for i, (key, label) in enumerate(label_map.items()):
        if key in metrics:
            labels.append(label)
            values.append(metrics[key])

    if not values:
        return dest

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(4, len(values) * 2), 4))
        bars = ax.bar(labels, values, color=colors[: len(values)], width=0.5)
        ax.bar_label(bars, fmt="%.4g", padding=4, fontsize=9)
        ax.set_ylabel("RMS absolute difference")
        ax.set_title("Final training RMSE (from train.log summary)")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _parity_panel(ax, ref, pred, label: str, units: str, color: str) -> float:
    ref_arr  = np.array(ref)
    pred_arr = np.array(pred)
    lo = min(ref_arr.min(), pred_arr.min())
    hi = max(ref_arr.max(), pred_arr.max())
    margin = (hi - lo) * 0.05
    ax.scatter(ref_arr, pred_arr, s=6, alpha=0.35, color=color,
               linewidths=0, rasterized=True)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=0.8, label="ideal")
    rmse = math.sqrt(float(np.mean((pred_arr - ref_arr) ** 2)))
    ax.set_xlabel(f"DFT {label}  ({units})")
    ax.set_ylabel(f"MTP {label}  ({units})")
    ax.set_title(f"{label}  RMSE = {rmse:.4g} {units}")
    ax.legend(fontsize=8)
    return rmse


def plot_parity(parity: dict, dest_dir: Path) -> list[Path]:
    paths = []
    with plt.rc_context(_STYLE):

        # Energy
        fig, ax = plt.subplots(figsize=(5, 5))
        _parity_panel(ax, parity["energies_ref"], parity["energies_pred"],
                      "energy/atom", "eV", "#01696f")
        fig.tight_layout()
        p = dest_dir / "parity_energy.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # Forces
        fig, ax = plt.subplots(figsize=(5, 5))
        _parity_panel(ax, parity["forces_ref"], parity["forces_pred"],
                      "forces", "eV/Å", "#964219")
        fig.tight_layout()
        p = dest_dir / "parity_forces.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # Stress (optional)
        if parity.get("stress_ref") and len(parity["stress_ref"]) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            _parity_panel(ax, parity["stress_ref"], parity["stress_pred"],
                          "stress", "GPa", "#7a39bb")
            fig.tight_layout()
            p = dest_dir / "parity_stress.png"
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)

    return paths


def plot_gamma_histogram(
    grades: list[float],
    thresholds: dict,
    dest: Path,
) -> Path:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(grades, bins=60, color="#01696f", edgecolor="white",
                linewidth=0.3, rasterized=True)
        if "save" in thresholds:
            ax.axvline(
                thresholds["save"], color="#964219", linestyle="--",
                linewidth=1.4,
                label=f"save threshold  ({thresholds['save']})",
            )
        if "break" in thresholds:
            ax.axvline(
                thresholds["break"], color="#a12c7b", linestyle="--",
                linewidth=1.4,
                label=f"break threshold  ({thresholds['break']})",
            )
        ax.set_xlabel("Extrapolation grade  γ")
        ax.set_ylabel("Count")
        ax.set_title(f"Grade distribution  (n = {len(grades):,})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest
