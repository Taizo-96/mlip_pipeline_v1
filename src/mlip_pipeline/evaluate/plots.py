"""mlip_pipeline/evaluate/plots.py
Parity plots, error histograms, stress component breakdown,
force-error distributions, gamma histogram, and per-config-type RMSE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── colour palette (colour-blind friendly) ───────────────────────────────────
_C = dict(energy="#2176AE", force="#F7941D", stress_diag="#43AA8B",
          stress_offdiag="#C05299", gamma="#E63946", residual="#555555")

VOIGT_LABELS = ["xx", "yy", "zz", "xy", "xz", "yz"]
FORCE_LABELS = ["x", "y", "z"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_1d(arr: Any) -> np.ndarray:
    """Flatten to 1-D float64, dropping NaNs."""
    a = np.asarray(arr, dtype=float).ravel()
    return a[~np.isnan(a)]


def _align(ref: Any, pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return two equal-length 1-D arrays; warn and trim if sizes differ."""
    r, p = _to_1d(ref), _to_1d(pred)
    if len(r) != len(p):
        n = min(len(r), len(p))
        print(f"  [warn] size mismatch ref={len(r)} pred={len(p)}, trimming to {n}")
        r, p = r[:n], p[:n]
    return r, p


def _parity_panel(ax: plt.Axes, ref: Any, pred: Any,
                  xlabel: str, ylabel: str, color: str, title: str) -> None:
    r, p = _align(ref, pred)
    if len(r) == 0:
        ax.set_visible(False)
        return

    lo = min(r.min(), p.min())
    hi = max(r.max(), p.max())
    pad = (hi - lo) * 0.05 or 0.1
    lo -= pad; hi += pad

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(r, p, s=6, alpha=0.35, color=color, linewidths=0, zorder=2)

    rmse = float(np.sqrt(np.mean((r - p) ** 2)))
    ax.text(0.05, 0.93, f"RMSE = {rmse:.4g}", transform=ax.transAxes,
            fontsize=7, va="top")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="box")


def _hist_panel(ax: plt.Axes, errors: np.ndarray,
                xlabel: str, color: str, title: str) -> None:
    e = _to_1d(errors)
    if len(e) == 0:
        ax.set_visible(False)
        return
    ax.hist(e, bins=60, color=color, alpha=0.75, edgecolor="none")
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae  = float(np.mean(np.abs(e)))
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.text(0.97, 0.93,
            f"RMSE={rmse:.3g}\nMAE={mae:.3g}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right")
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)


# ── public API ────────────────────────────────────────────────────────────────

def plot_parity(parity: dict, eval_dir: Path) -> dict[str, Path]:
    """
    Main parity figure: energy, forces (per x/y/z), stress (per Voigt component).
    All arrays are normalised to (N, 3) and (N, 6) before plotting so that
    the old 'x and y must be the same size' crash cannot occur.
    Returns dict of {name: Path}.
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # ── energy ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    _parity_panel(ax,
                  parity["energy_ref"], parity["energy_pred"],
                  "DFT energy (eV/atom)", "MTP energy (eV/atom)",
                  _C["energy"], "Energy parity")
    fig.tight_layout()
    p = eval_dir / "parity_energy.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["energy"] = p

    # ── forces ────────────────────────────────────────────────────────────────
    force_ref  = np.asarray(parity["force_ref"],  dtype=float)
    force_pred = np.asarray(parity["force_pred"], dtype=float)
    if force_ref.ndim == 1:
        n = (len(force_ref) // 3) * 3
        force_ref  = force_ref[:n].reshape(-1, 3)
        force_pred = force_pred[:n].reshape(-1, 3)
    if force_ref.shape[0] > 0:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for i, (ax, lbl) in enumerate(zip(axes, FORCE_LABELS)):
            _parity_panel(ax,
                          force_ref[:, i], force_pred[:, i],
                          f"DFT F{lbl} (eV/\u00c5)", f"MTP F{lbl} (eV/\u00c5)",
                          _C["force"], f"Force {lbl} parity")
        fig.suptitle("Force parity (per component)", fontsize=10, y=1.01)
        fig.tight_layout()
        p = eval_dir / "parity_forces.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths["forces"] = p

    # ── stress ────────────────────────────────────────────────────────────────
    stress_ref  = np.asarray(parity["stress_ref"],  dtype=float)
    stress_pred = np.asarray(parity["stress_pred"], dtype=float)
    # normalise to (N_structures, 6) Voigt — this is what fixes the crash
    if stress_ref.ndim == 1:
        n6 = (len(stress_ref) // 6) * 6
        stress_ref  = stress_ref[:n6].reshape(-1, 6)
        stress_pred = stress_pred[:n6].reshape(-1, 6)
    if stress_ref.shape[0] > 0:
        colors = [_C["stress_diag"]] * 3 + [_C["stress_offdiag"]] * 3
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for i, (ax, lbl, col) in enumerate(zip(axes.ravel(), VOIGT_LABELS, colors)):
            _parity_panel(ax,
                          stress_ref[:, i], stress_pred[:, i],
                          f"DFT \u03c3{lbl} (GPa)", f"MTP \u03c3{lbl} (GPa)",
                          col, f"Stress {lbl}")
        fig.suptitle("Stress parity — diagonal (teal) vs off-diagonal (purple)",
                     fontsize=10)
        fig.tight_layout()
        p = eval_dir / "parity_stress.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths["stress"] = p

    return paths


def plot_error_histograms(parity: dict, eval_dir: Path) -> dict[str, Path]:
    """Residual histograms for energy, force components, and stress components."""
    eval_dir = Path(eval_dir)
    paths: dict[str, Path] = {}

    # energy
    r, p = _align(parity["energy_ref"], parity["energy_pred"])
    if len(r) > 0:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        _hist_panel(ax, p - r, "\u0394E = MTP \u2212 DFT (eV/atom)",
                    _C["energy"], "Energy error distribution")
        fig.tight_layout()
        pth = eval_dir / "hist_energy.png"
        fig.savefig(pth, dpi=150); plt.close(fig)
        paths["hist_energy"] = pth

    # forces
    force_ref  = np.asarray(parity["force_ref"],  dtype=float)
    force_pred = np.asarray(parity["force_pred"], dtype=float)
    if force_ref.ndim == 1:
        n = (len(force_ref) // 3) * 3
        force_ref  = force_ref[:n].reshape(-1, 3)
        force_pred = force_pred[:n].reshape(-1, 3)
    if force_ref.shape[0] > 0:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for i, (ax, lbl) in enumerate(zip(axes, FORCE_LABELS)):
            _hist_panel(ax, force_pred[:, i] - force_ref[:, i],
                        f"\u0394F{lbl} (eV/\u00c5)", _C["force"], f"Force {lbl} error")
        fig.suptitle("Force error distributions", fontsize=10, y=1.01)
        fig.tight_layout()
        pth = eval_dir / "hist_forces.png"
        fig.savefig(pth, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths["hist_forces"] = pth

    # stress
    stress_ref  = np.asarray(parity["stress_ref"],  dtype=float)
    stress_pred = np.asarray(parity["stress_pred"], dtype=float)
    if stress_ref.ndim == 1:
        n6 = (len(stress_ref) // 6) * 6
        stress_ref  = stress_ref[:n6].reshape(-1, 6)
        stress_pred = stress_pred[:n6].reshape(-1, 6)
    if stress_ref.shape[0] > 0:
        colors = [_C["stress_diag"]] * 3 + [_C["stress_offdiag"]] * 3
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for i, (ax, lbl, col) in enumerate(zip(axes.ravel(), VOIGT_LABELS, colors)):
            _hist_panel(ax, stress_pred[:, i] - stress_ref[:, i],
                        f"\u0394\u03c3{lbl} (GPa)", col, f"Stress {lbl} error")
        fig.suptitle("Stress error distributions", fontsize=10)
        fig.tight_layout()
        pth = eval_dir / "hist_stress.png"
        fig.savefig(pth, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths["hist_stress"] = pth

    return paths


def plot_gamma_histogram(gamma_values: Any, eval_dir: Path,
                         threshold: float | None = None) -> Path:
    """
    Distribution of MTP extrapolation grades (\u03b3) over the test set.

    Parameters
    ----------
    gamma_values : 1-D array of per-structure \u03b3 values
    threshold    : active-learning selection threshold drawn as a vertical line
    """
    eval_dir = Path(eval_dir)
    g = _to_1d(gamma_values)
    if len(g) == 0:
        print("  [warn] no gamma values provided; skipping gamma histogram.")
        return eval_dir / "hist_gamma.png"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(g, bins=60, color=_C["gamma"], alpha=0.75, edgecolor="none")
    if threshold is not None:
        ax.axvline(threshold, color="k", lw=1.2, ls="--",
                   label=f"threshold = {threshold}")
        frac = float(np.mean(g > threshold)) * 100
        ax.text(0.97, 0.93, f"{frac:.1f}% above threshold",
                transform=ax.transAxes, fontsize=7, ha="right", va="top")
        ax.legend(fontsize=7)
    ax.set_xlabel("Extrapolation grade \u03b3", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("MTP extrapolation grade distribution (test set)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    p = eval_dir / "hist_gamma.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_per_config_rmse(parity: dict, config_labels: Any,
                         eval_dir: Path) -> Path:
    """
    Per-configuration-type RMSE bar chart for energy (and optionally forces).

    Parameters
    ----------
    config_labels :
        Array-like of string labels, one per structure, matching the order
        of parity["energy_ref"].
    """
    eval_dir = Path(eval_dir)
    labels = np.asarray(config_labels)
    e_ref  = np.asarray(parity["energy_ref"],  dtype=float).ravel()
    e_pred = np.asarray(parity["energy_pred"], dtype=float).ravel()

    use_force = ("force_ref_per_struct" in parity and
                 "force_pred_per_struct" in parity)

    types  = sorted(set(labels.tolist()))
    e_rmse = []
    f_rmse = []

    for t in types:
        mask = labels == t
        er = e_ref[mask]; ep = e_pred[mask]
        e_rmse.append(float(np.sqrt(np.mean((ep - er) ** 2))) if len(er) else float("nan"))
        if use_force:
            fr_t = _to_1d(np.asarray(parity["force_ref_per_struct"])[mask])
            fp_t = _to_1d(np.asarray(parity["force_pred_per_struct"])[mask])
            r2, p2 = _align(fr_t, fp_t)
            f_rmse.append(float(np.sqrt(np.mean((p2 - r2) ** 2))) if len(r2) else float("nan"))

    x = np.arange(len(types))
    ncols = 2 if use_force else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].bar(x, e_rmse, color=_C["energy"], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(types, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("RMSE energy (eV/atom)", fontsize=8)
    axes[0].set_title("Per-config energy RMSE", fontsize=9)
    axes[0].tick_params(labelsize=7)

    if use_force:
        axes[1].bar(x, f_rmse, color=_C["force"], alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(types, rotation=30, ha="right", fontsize=7)
        axes[1].set_ylabel("RMSE force (eV/\u00c5)", fontsize=8)
        axes[1].set_title("Per-config force RMSE", fontsize=9)
        axes[1].tick_params(labelsize=7)

    fig.tight_layout()
    p = eval_dir / "rmse_per_config_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p
