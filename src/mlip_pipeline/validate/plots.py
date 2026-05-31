"""Plotting helpers for the validate step.

Only comparison plots (MTP vs classical reference) are produced.
Standalone MTP-only plots have been removed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_pipeline.models import (
    EosResult, ElasticResult,
    ThermalExpansionResult, RdfResult, MeltingResult, VacancyResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from mlip_pipeline.validate.classical_reference import ClassicalReferenceResult


# ── Colour palette ─────────────────────────────────────────────────────
_MTP_COLOR = "#2c7bb6"
_REF_COLOR = "#d7191c"


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for validation plots") from exc


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise ImportError("numpy is required for validation plots") from exc


def _thexp_fit_line(
    r: ThermalExpansionResult,
) -> tuple[list[float], list[float]] | None:
    """Return (Ts, Vs) for a linear fit line spanning the MD temperature range.

    Reproduces the same polyfit that thermal_expansion.py uses, so V_ref is
    derived on-the-fly rather than stored on the dataclass.
    Returns None if the data or fit is unusable.
    """
    if not r.temperatures or not r.volumes or not math.isfinite(r.alpha):
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    Ts_arr = np.array(r.temperatures)
    Vs_arr = np.array(r.volumes)
    coeffs = np.polyfit(Ts_arr, Vs_arr, 1)
    T_span = [float(Ts_arr.min()), float(Ts_arr.max())]
    Vs_fit = [float(np.polyval(coeffs, T)) for T in T_span]
    return T_span, Vs_fit


# ── MTP vs classical reference comparison plots ─────────────────────────────

def _cmp_eos(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Overlay EOS curves (data + BM fit) for MTP vs reference, per structure."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
        from mlip_pipeline.validate.eos import _birch_murnaghan
    except ImportError:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label
    ref_map = {r.structure_id: r for r in ref_result.eos_results}

    for mtp_r in mtp_result.eos_results:
        sid   = mtp_r.structure_id
        ref_r = ref_map.get(sid)
        if ref_r is None or not mtp_r.fit_ok or not ref_r.fit_ok:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        # MTP scatter + BM fit
        if mtp_r.volumes:
            ax.scatter(mtp_r.volumes, mtp_r.energies, s=25,
                       color=_MTP_COLOR, zorder=3, label="MTP data")
            v_fit = np.linspace(min(mtp_r.volumes), max(mtp_r.volumes), 200)
            e_fit = [_birch_murnaghan(v, mtp_r.V0, mtp_r.E0,
                                      mtp_r.B0 / 160.2176634, mtp_r.B0p)
                     for v in v_fit]
            ax.plot(v_fit, e_fit, color=_MTP_COLOR, lw=1.8,
                    label=f"MTP fit  B\u2080={mtp_r.B0:.1f} GPa")

        # Reference BM fit
        v_lo = min(ref_r.volumes) if ref_r.volumes else (min(mtp_r.volumes) if mtp_r.volumes else None)
        v_hi = max(ref_r.volumes) if ref_r.volumes else (max(mtp_r.volumes) if mtp_r.volumes else None)
        if v_lo is None:
            plt.close(fig)
            continue
        if ref_r.volumes:
            ax.scatter(ref_r.volumes, ref_r.energies, s=25,
                       color=_REF_COLOR, zorder=3, marker="^",
                       label=f"{ref_lbl} data")
        v_ref = np.linspace(v_lo, v_hi, 200)
        e_ref = [_birch_murnaghan(v, ref_r.V0, ref_r.E0,
                                  ref_r.B0 / 160.2176634, ref_r.B0p)
                 for v in v_ref]
        ax.plot(v_ref, e_ref, color=_REF_COLOR, lw=1.8, ls="--",
                label=f"{ref_lbl} fit  B\u2080={ref_r.B0:.1f} GPa")

        ax.set_xlabel("Volume per atom (\u00c5\u00b3)")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title(f"EOS comparison \u2014 {sid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cmp_eos_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"eos_{sid}"] = path

    return paths


def _cmp_elastic(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Grouped bar chart of Cij and bulk/shear moduli, per structure."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label
    ref_map = {r.structure_id: r for r in ref_result.elastic_results}

    for mtp_r in mtp_result.elastic_results:
        sid   = mtp_r.structure_id
        ref_r = ref_map.get(sid)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue

        cij_keys = sorted(
            set((mtp_r.C or {}).keys()) | set((ref_r.C or {}).keys())
        )
        prop_keys = cij_keys + [
            k for k in ("B_voigt", "G_voigt")
            if getattr(mtp_r, k, None) is not None
            or getattr(ref_r, k, None) is not None
        ]
        if not prop_keys:
            continue

        def _get(r, k):
            if k in ("B_voigt", "G_voigt"):
                return getattr(r, k, float("nan")) or float("nan")
            return (r.C or {}).get(k, float("nan"))

        x     = np.arange(len(prop_keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(prop_keys) * 1.4), 4))
        ax.bar(x - width / 2, [_get(mtp_r, k) for k in prop_keys],
               width, color=_MTP_COLOR, label="MTP", alpha=0.85)
        ax.bar(x + width / 2, [_get(ref_r, k) for k in prop_keys],
               width, color=_REF_COLOR, label=ref_lbl, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(prop_keys)
        ax.set_ylabel("Elastic constant (GPa)")
        ax.set_title(f"Elastic constants comparison \u2014 {sid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cmp_elastic_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"elastic_{sid}"] = path

    return paths


def _cmp_melting(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Horizontal grouped bar chart of T_melt per structure."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    ref_map = {r.structure_id: r for r in ref_result.melting_results}
    pairs   = [
        (mtp_r, ref_map[mtp_r.structure_id])
        for mtp_r in mtp_result.melting_results
        if mtp_r.structure_id in ref_map
        and mtp_r.compute_ok
        and ref_map[mtp_r.structure_id].compute_ok
    ]
    if not pairs:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label
    sids    = [m.structure_id for m, _ in pairs]
    mtp_T   = [m.T_melt for m, _ in pairs]
    ref_T   = [r.T_melt for _, r in pairs]

    y     = np.arange(len(sids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(sids) * 1.2)))
    ax.barh(y - width / 2, mtp_T, width, color=_MTP_COLOR, label="MTP",  alpha=0.85)
    ax.barh(y + width / 2, ref_T, width, color=_REF_COLOR, label=ref_lbl, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sids)
    ax.set_xlabel("T\u2098\u2091\u2097\u209c (K)")
    ax.set_title("Melting temperature comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "cmp_melting.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["melting"] = path
    return paths


def _cmp_thermal_expansion(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Overlay V(T) scatter + linear fit for MTP vs reference, per structure."""
    paths: dict[str, Path] = {}
    ref_map = {r.structure_id: r for r in ref_result.thermal_expansion_results}
    pairs   = [
        (mtp_r, ref_map[mtp_r.structure_id])
        for mtp_r in mtp_result.thermal_expansion_results
        if mtp_r.structure_id in ref_map
        and mtp_r.compute_ok
        and ref_map[mtp_r.structure_id].compute_ok
        and mtp_r.temperatures
        and ref_map[mtp_r.structure_id].temperatures
    ]
    if not pairs:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label

    for mtp_r, ref_r in pairs:
        sid = mtp_r.structure_id
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(mtp_r.temperatures, mtp_r.volumes,
                   s=25, color=_MTP_COLOR, zorder=3)
        mtp_fit = _thexp_fit_line(mtp_r)
        if mtp_fit is not None:
            ax.plot(*mtp_fit, color=_MTP_COLOR, lw=1.8,
                    label=f"MTP  \u03b1={mtp_r.alpha*1e6:.1f}\u00d710\u207b\u2076 K\u207b\u00b9")
        else:
            ax.plot([], [], color=_MTP_COLOR, label="MTP")

        ax.scatter(ref_r.temperatures, ref_r.volumes,
                   s=25, color=_REF_COLOR, zorder=3, marker="^")
        ref_fit = _thexp_fit_line(ref_r)
        if ref_fit is not None:
            ax.plot(*ref_fit, color=_REF_COLOR, lw=1.8, ls="--",
                    label=f"{ref_lbl}  \u03b1={ref_r.alpha*1e6:.1f}\u00d710\u207b\u2076 K\u207b\u00b9")
        else:
            ax.plot([], [], color=_REF_COLOR, ls="--", label=ref_lbl)

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Volume per atom (\u00c5\u00b3)")
        ax.set_title(f"Thermal expansion comparison \u2014 {sid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cmp_thexp_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"thexp_{sid}"] = path

    return paths


def _cmp_vacancy(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Grouped bar chart of E_vac per structure."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    ref_map = {r.structure_id: r for r in ref_result.vacancy_results}
    pairs   = [
        (mtp_r, ref_map[mtp_r.structure_id])
        for mtp_r in mtp_result.vacancy_results
        if mtp_r.structure_id in ref_map
        and mtp_r.compute_ok
        and ref_map[mtp_r.structure_id].compute_ok
        and math.isfinite(mtp_r.E_vac)
        and math.isfinite(ref_map[mtp_r.structure_id].E_vac)
    ]
    if not pairs:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label
    sids    = [m.structure_id for m, _ in pairs]
    mtp_E   = [m.E_vac for m, _ in pairs]
    ref_E   = [r.E_vac for _, r in pairs]

    x     = np.arange(len(sids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(4, len(sids) * 1.8), 4))
    ax.bar(x - width / 2, mtp_E, width, color=_MTP_COLOR, label="MTP",  alpha=0.85)
    ax.bar(x + width / 2, ref_E, width, color=_REF_COLOR, label=ref_lbl, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.set_title("Vacancy formation energy comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "cmp_vacancy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["vacancy"] = path
    return paths


def _cmp_rdf(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Overlay g(r) curves for MTP vs reference, per structure."""
    paths: dict[str, Path] = {}
    ref_map = {r.structure_id: r for r in ref_result.rdf_results}
    pairs   = [
        (mtp_r, ref_map[mtp_r.structure_id])
        for mtp_r in mtp_result.rdf_results
        if mtp_r.structure_id in ref_map
        and mtp_r.compute_ok
        and ref_map[mtp_r.structure_id].compute_ok
        and mtp_r.r
        and ref_map[mtp_r.structure_id].r
    ]
    if not pairs:
        return paths

    plt     = _require_matplotlib()
    ref_lbl = ref_result.label

    for mtp_r, ref_r in pairs:
        sid = mtp_r.structure_id
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mtp_r.r, mtp_r.g_r, color=_MTP_COLOR, lw=1.4, label="MTP")
        ax.plot(ref_r.r, ref_r.g_r, color=_REF_COLOR, lw=1.4, ls="--",
                label=ref_lbl)
        if math.isfinite(mtp_r.first_peak_r):
            ax.axvline(mtp_r.first_peak_r, color=_MTP_COLOR,
                       ls=":", lw=0.9, alpha=0.7)
        if math.isfinite(ref_r.first_peak_r):
            ax.axvline(ref_r.first_peak_r, color=_REF_COLOR,
                       ls=":", lw=0.9, alpha=0.7)
        ax.axhline(1.0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("r (\u00c5)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"RDF comparison \u2014 {sid} ({mtp_r.temperature:.0f} K)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"cmp_rdf_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"rdf_{sid}"] = path

    return paths


def plot_comparison(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Generate all MTP vs classical reference comparison plots.

    Returns a merged dict mapping plot key -> Path for every file produced.
    """
    paths: dict[str, Path] = {}
    for fn in (_cmp_eos, _cmp_elastic, _cmp_melting,
               _cmp_thermal_expansion, _cmp_vacancy, _cmp_rdf):
        paths.update(fn(mtp_result, ref_result, out_dir))
    return paths
