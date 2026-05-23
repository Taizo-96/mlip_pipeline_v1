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

_TEAL   = "#01696f"
_BROWN  = "#964219"
_PURPLE = "#7a39bb"

# Distinct colours for multi-gen comparison panels (cycles if > 8 gens)
_GEN_COLORS = [
    "#01696f", "#964219", "#7a39bb", "#a13544",
    "#006494", "#437a22", "#da7101", "#d19900",
]


def plot_summary_metrics(metrics: dict, dest_dir: Path) -> list[Path]:
    """
    Produce one bar-chart PNG per RMSE quantity (energy, forces, stress).

    Each quantity lives on its own scale so the differences are readable.
    Returns a list of Paths for every file actually written.

    ``dest_dir`` is the directory that will contain the files
    (``loss_energy.png``, ``loss_forces.png``, ``loss_stress.png``).
    ``dest_dir`` is created if it does not exist.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("rmse_e", "Energy RMSE (eV/atom)", "Energy\n(eV/atom)", _TEAL,   "loss_energy.png"),
        ("rmse_f", "Forces RMSE (eV/Å)",   "Forces\n(eV/Å)",   _BROWN,  "loss_forces.png"),
        ("rmse_s", "Stress RMSE (GPa)",     "Stress\n(GPa)",    _PURPLE, "loss_stress.png"),
    ]

    written: list[Path] = []
    for key, title, x_label, color, fname in specs:
        if key not in metrics:
            continue
        value = metrics[key]
        dest  = dest_dir / fname

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(3.5, 4))
            bar = ax.bar([x_label], [value], color=color, width=0.4)
            ax.bar_label(bar, fmt="%.4g", padding=4, fontsize=9)
            ax.set_ylabel("RMS absolute difference")
            ax.set_title(title)
            ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(dest)
            plt.close(fig)

        written.append(dest)
    return written


def _parity_panel(ax, ref, pred, label: str, units: str, color: str,
                  gen_tag: str | None = None) -> float:
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
    title = f"{label}  RMSE = {rmse:.4g} {units}"
    if gen_tag:
        title = f"{gen_tag}\n{title}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    return rmse


def plot_parity(parity: dict, dest_dir: Path) -> list[Path]:
    paths = []
    with plt.rc_context(_STYLE):

        # Energy
        fig, ax = plt.subplots(figsize=(5, 5))
        _parity_panel(ax, parity["energies_ref"], parity["energies_pred"],
                      "energy/atom", "eV", _TEAL)
        fig.tight_layout()
        p = dest_dir / "parity_energy.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # Forces
        fig, ax = plt.subplots(figsize=(5, 5))
        _parity_panel(ax, parity["forces_ref"], parity["forces_pred"],
                      "forces", "eV/Å", _BROWN)
        fig.tight_layout()
        p = dest_dir / "parity_forces.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

        # Stress (optional)
        if parity.get("stress_ref") and len(parity["stress_ref"]) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            _parity_panel(ax, parity["stress_ref"], parity["stress_pred"],
                          "stress", "GPa", _PURPLE)
            fig.tight_layout()
            p = dest_dir / "parity_stress.png"
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)

    return paths


def plot_parity_comparison(
    gen_parities: list[tuple[int, dict]],
    quantities: list[str],
    dest_dir: Path,
) -> list[Path]:
    """
    Produce a side-by-side comparison parity plot for each requested quantity.

    Parameters
    ----------
    gen_parities
        List of (gen_number, parity_dict) pairs in display order.
        parity_dict must be the output of build_parity_data().
    quantities
        Subset of ['energy', 'forces', 'stress'] to plot.
    dest_dir
        Directory where output PNGs are written.

    Returns
    -------
    List of written PNG paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    n = len(gen_parities)
    if n == 0:
        return []

    # Map quantity name → (ref_key, pred_key, label, units)
    _QUANTITY_SPEC = {
        "energy": ("energies_ref", "energies_pred", "energy/atom", "eV"),
        "forces": ("forces_ref",   "forces_pred",   "forces",      "eV/Å"),
        "stress": ("stress_ref",   "stress_pred",   "stress",      "GPa"),
    }

    written: list[Path] = []

    with plt.rc_context(_STYLE):
        for qty in quantities:
            if qty not in _QUANTITY_SPEC:
                continue
            ref_key, pred_key, label, units = _QUANTITY_SPEC[qty]

            # Filter to gens that actually have data for this quantity
            valid = [
                (gn, p) for gn, p in gen_parities
                if p.get(ref_key) and len(p[ref_key]) > 0
            ]
            if not valid:
                continue

            ncols = len(valid)
            fig, axes = plt.subplots(
                1, ncols,
                figsize=(4.5 * ncols, 4.8),
                squeeze=False,
            )

            for col, (gen_num, parity) in enumerate(valid):
                ax    = axes[0][col]
                color = _GEN_COLORS[col % len(_GEN_COLORS)]
                gen_tag = f"gen_{gen_num:02d}"
                _parity_panel(
                    ax,
                    parity[ref_key],
                    parity[pred_key],
                    label,
                    units,
                    color,
                    gen_tag=gen_tag,
                )

            gen_range = "_".join(f"{gn:02d}" for gn, _ in valid)
            fname = f"parity_compare_{qty}_gens{gen_range}.png"
            fig.suptitle(
                f"Parity comparison — {label}  ({', '.join(f'gen_{gn:02d}' for gn, _ in valid)})",
                fontsize=12,
                y=1.01,
            )
            fig.tight_layout()
            dest = dest_dir / fname
            fig.savefig(dest, bbox_inches="tight")
            plt.close(fig)
            written.append(dest)

    return written


def plot_gamma_histogram(
    grades: list[float],
    thresholds: dict,
    dest: Path,
) -> Path:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(grades, bins=60, color=_TEAL, edgecolor="white",
                linewidth=0.3, rasterized=True)
        if "save" in thresholds:
            ax.axvline(
                thresholds["save"], color=_BROWN, linestyle="--",
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
