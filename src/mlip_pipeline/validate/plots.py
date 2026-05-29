"""Plotting helpers for the validate step."""
from __future__ import annotations

import math
from pathlib import Path

from mlip_pipeline.validate.models import (
    EosResult, ElasticResult,
    ThermalExpansionResult, RdfResult, MeltingResult, VacancyResult,
)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for validation plots") from exc


def plot_eos(result: EosResult, out_dir: Path) -> Path | None:
    if not result.volumes or not result.energies:
        return None
    plt = _require_matplotlib()
    try:
        import numpy as np
        from mlip_pipeline.validate.eos import _birch_murnaghan
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(result.volumes, result.energies, s=30, color="#2c7bb6", zorder=3, label="MTP")
    if result.fit_ok:
        v_fit = np.linspace(min(result.volumes), max(result.volumes), 200)
        B0_ev = result.B0 / 160.2176634
        e_fit = [_birch_murnaghan(v, result.V0, result.E0, B0_ev, result.B0p) for v in v_fit]
        ax.plot(v_fit, e_fit, color="#d7191c", lw=1.5,
                label=f"BM fit  B\u2080={result.B0:.1f} GPa")
        ax.axvline(result.V0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Volume per atom (\u00c5\u00b3)")
    ax.set_ylabel("Energy per atom (eV)")
    ax.set_title(f"EOS \u2014 {result.structure_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / f"eos_{result.structure_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_elastic_bar(results: list[ElasticResult], out_dir: Path) -> Path | None:
    valid = [r for r in results if r.compute_ok and r.C]
    if not valid:
        return None
    plt = _require_matplotlib()
    import numpy as np
    keys = sorted({k for r in valid for k in r.C})
    x = np.arange(len(keys))
    width = 0.8 / len(valid)
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 4))
    for i, r in enumerate(valid):
        vals = [r.C.get(k, float("nan")) for k in keys]
        ax.bar(x + i * width, vals, width, label=r.structure_id)
    ax.set_xticks(x + width * (len(valid) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Elastic constant (GPa)")
    ax.set_title("Elastic constants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "elastic_constants.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_melting(
    results: list[MeltingResult], out_dir: Path
) -> Path | None:
    """Horizontal bracket diagram showing T_melt estimate per structure."""
    valid = [r for r in results if r.compute_ok]
    if not valid:
        return None
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(valid) * 1.2)))
    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]

    for i, r in enumerate(valid):
        c = colors[i % len(colors)]
        y = i
        lo = r.T_bracket_lo if math.isfinite(r.T_bracket_lo) else r.T_melt
        hi = r.T_bracket_hi if math.isfinite(r.T_bracket_hi) else r.T_melt

        # Bracket bar
        if math.isfinite(lo) and math.isfinite(hi) and lo != hi:
            ax.barh(y, hi - lo, left=lo, height=0.4, color=c, alpha=0.35,
                    label=f"{r.structure_id} bracket")
        # T_melt point
        if math.isfinite(r.T_melt):
            ax.plot(r.T_melt, y, "D", color=c, ms=8, zorder=5,
                    label=f"{r.structure_id}  T_m={r.T_melt:.0f} K")
            ax.axvline(r.T_melt, color=c, ls="--", lw=0.8, alpha=0.5)

    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([r.structure_id for r in valid])
    ax.set_xlabel("Temperature (K)")
    ax.set_title("Melting temperature (two-phase coexistence)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "melting_temperature.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_thermal_expansion(
    results: list[ThermalExpansionResult], out_dir: Path
) -> Path | None:
    valid = [r for r in results if r.compute_ok and r.temperatures]
    if not valid:
        return None
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]
    for i, r in enumerate(valid):
        c = colors[i % len(colors)]
        ax.scatter(r.temperatures, r.volumes, s=30, color=c, zorder=3)
        if math.isfinite(r.alpha) and math.isfinite(r.V_ref):
            Ts = [min(r.temperatures), max(r.temperatures)]
            Vs = [r.V_ref + 3 * r.alpha * r.V_ref * (T - r.T_ref) for T in Ts]
            ax.plot(Ts, Vs, color=c, lw=1.5,
                    label=f"{r.structure_id}  \u03b1={r.alpha*1e6:.1f}\u00d710\u207b\u2076 K\u207b\u00b9")
        else:
            ax.plot([], [], color=c, label=r.structure_id)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Volume per atom (\u00c5\u00b3)")
    ax.set_title("Thermal expansion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "thermal_expansion.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_vacancy(
    results: list[VacancyResult], out_dir: Path
) -> Path | None:
    """Bar chart of vacancy formation energies across structures."""
    valid = [r for r in results if r.compute_ok and math.isfinite(r.E_vac)]
    if not valid:
        return None
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(max(4, len(valid) * 1.5), 4))
    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]
    for i, r in enumerate(valid):
        ax.bar(i, r.E_vac, color=colors[i % len(colors)], width=0.6,
               label=f"{r.structure_id}  {r.E_vac:.3f} eV")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([r.structure_id for r in valid])
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.set_title("Vacancy formation energy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "vacancy_formation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rdf(results: list[RdfResult], out_dir: Path) -> Path | None:
    valid = [r for r in results if r.compute_ok and r.r]
    if not valid:
        return None
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]
    for i, r in enumerate(valid):
        c = colors[i % len(colors)]
        ax.plot(r.r, r.g_r, color=c, lw=1.2,
                label=f"{r.structure_id} ({r.temperature:.0f} K)")
        if math.isfinite(r.first_peak_r):
            ax.axvline(r.first_peak_r, color=c, ls="--", lw=0.7, alpha=0.6)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("r (\u00c5)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial distribution function")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "rdf.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
