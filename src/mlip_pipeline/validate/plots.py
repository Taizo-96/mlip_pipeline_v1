"""Plotting helpers for the validate step.

Only comparison plots (MTP vs classical reference) are produced.
Standalone MTP-only plots have been removed.

Public API
----------
plot_comparison(mtp_result, ref_result, out_dir)
    Pairwise MTP vs one reference — called once per reference.
plot_combined(mtp_result, ref_results, out_dir)
    All references on the same axes — called once after the loop.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mlip_pipeline.models import (
    EosResult, ElasticResult,
    ThermalExpansionResult, RdfResult, MeltingResult, VacancyResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from mlip_pipeline.validate.classical_reference import ClassicalReferenceResult


# ── Colour palettes ────────────────────────────────────────────────────
_MTP_COLOR = "#2c7bb6"
_REF_COLOR = "#d7191c"  # used for single-ref pairwise plots

# Palette for combined plots (MTP is always _MTP_COLOR; refs rotate through this)
_REF_PALETTE = [
    "#d7191c",  # red
    "#1a9641",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
    "#e41a1c",  # crimson (fallback cycle)
]


def _ref_color(idx: int) -> str:
    return _REF_PALETTE[idx % len(_REF_PALETTE)]


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


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise plots  (MTP vs one reference)
# ──────────────────────────────────────────────────────────────────────────────

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

        if mtp_r.volumes:
            ax.scatter(mtp_r.volumes, mtp_r.energies, s=25,
                       color=_MTP_COLOR, zorder=3, label="MTP data")
            v_fit = np.linspace(min(mtp_r.volumes), max(mtp_r.volumes), 200)
            e_fit = [_birch_murnaghan(v, mtp_r.V0, mtp_r.E0,
                                      mtp_r.B0 / 160.2176634, mtp_r.B0p)
                     for v in v_fit]
            ax.plot(v_fit, e_fit, color=_MTP_COLOR, lw=1.8,
                    label=f"MTP fit  B\u2080={mtp_r.B0:.1f} GPa")

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


# ──────────────────────────────────────────────────────────────────────────────
# Combined plots  (MTP + all references on one figure)
# ──────────────────────────────────────────────────────────────────────────────

def _combined_eos(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """All EOS curves on one plot per structure."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
        from mlip_pipeline.validate.eos import _birch_murnaghan
    except ImportError:
        return paths

    plt = _require_matplotlib()

    # Collect all structure ids that have a valid MTP EOS
    for mtp_r in mtp_result.eos_results:
        if not mtp_r.fit_ok:
            continue
        sid = mtp_r.structure_id

        # Check at least one ref is usable
        usable_refs = [
            (i, rr) for i, rr in enumerate(ref_results)
            for r in rr.eos_results
            if r.structure_id == sid and r.fit_ok
        ]
        if not usable_refs:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # MTP
        if mtp_r.volumes:
            v_range = np.linspace(min(mtp_r.volumes), max(mtp_r.volumes), 200)
            e_fit   = [_birch_murnaghan(v, mtp_r.V0, mtp_r.E0,
                                        mtp_r.B0 / 160.2176634, mtp_r.B0p)
                       for v in v_range]
            ax.scatter(mtp_r.volumes, mtp_r.energies, s=20,
                       color=_MTP_COLOR, zorder=3)
            ax.plot(v_range, e_fit, color=_MTP_COLOR, lw=2.0,
                    label=f"MTP  B\u2080={mtp_r.B0:.1f} GPa")

        # References
        for ref_idx, ref_rr in enumerate(ref_results):
            ref_r = next((r for r in ref_rr.eos_results
                          if r.structure_id == sid and r.fit_ok), None)
            if ref_r is None:
                continue
            col = _ref_color(ref_idx)
            ls  = ["--", "-.", ":", (0, (3, 1, 1, 1))][ref_idx % 4]
            if ref_r.volumes:
                ax.scatter(ref_r.volumes, ref_r.energies, s=20,
                           color=col, zorder=3, marker="^")
            v_lo = min(ref_r.volumes) if ref_r.volumes else min(mtp_r.volumes)
            v_hi = max(ref_r.volumes) if ref_r.volumes else max(mtp_r.volumes)
            v_ref = np.linspace(v_lo, v_hi, 200)
            e_ref = [_birch_murnaghan(v, ref_r.V0, ref_r.E0,
                                      ref_r.B0 / 160.2176634, ref_r.B0p)
                     for v in v_ref]
            ax.plot(v_ref, e_ref, color=col, lw=1.8, ls=ls,
                    label=f"{ref_rr.label}  B\u2080={ref_r.B0:.1f} GPa")

        ax.set_xlabel("Volume per atom (\u00c5\u00b3)")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title(f"EOS \u2014 {sid} \u2014 all potentials")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"combined_eos_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"eos_{sid}"] = path

    return paths


def _combined_elastic(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """Grouped bar chart with one group per property, one bar per potential."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    plt = _require_matplotlib()

    for mtp_r in mtp_result.elastic_results:
        if not mtp_r.compute_ok:
            continue
        sid = mtp_r.structure_id

        all_ref_r = [
            rr for rr in ref_results
            for r in rr.elastic_results
            if r.structure_id == sid and r.compute_ok
        ]
        if not all_ref_r:
            continue

        # Union of property keys across all potentials
        cij_keys = sorted(
            set((mtp_r.C or {}).keys())
            | {k for rr in ref_results
               for r in rr.elastic_results
               if r.structure_id == sid
               for k in (r.C or {}).keys()}
        )
        prop_keys = cij_keys + ["B_voigt", "G_voigt"]

        def _get(r, k):
            if k in ("B_voigt", "G_voigt"):
                return getattr(r, k, float("nan")) or float("nan")
            return (r.C or {}).get(k, float("nan"))

        n_pots  = 1 + len(all_ref_r)  # MTP + refs
        total_w = 0.8
        width   = total_w / n_pots
        x       = np.arange(len(prop_keys))
        offsets = np.linspace(-total_w / 2 + width / 2,
                               total_w / 2 - width / 2, n_pots)

        fig, ax = plt.subplots(figsize=(max(6, len(prop_keys) * 1.6), 4.5))
        ax.bar(x + offsets[0], [_get(mtp_r, k) for k in prop_keys],
               width, color=_MTP_COLOR, label="MTP", alpha=0.85)

        ref_idx_map = {}  # rr -> canonical index for colour
        counter = 0
        for rr in ref_results:
            ref_r = next((r for r in rr.elastic_results
                          if r.structure_id == sid and r.compute_ok), None)
            if ref_r is None:
                continue
            col = _ref_color(counter)
            ax.bar(x + offsets[1 + counter],
                   [_get(ref_r, k) for k in prop_keys],
                   width, color=col, label=rr.label, alpha=0.85)
            counter += 1

        ax.set_xticks(x)
        ax.set_xticklabels(prop_keys)
        ax.set_ylabel("Elastic constant (GPa)")
        ax.set_title(f"Elastic constants \u2014 {sid} \u2014 all potentials")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"combined_elastic_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"elastic_{sid}"] = path

    return paths


def _combined_melting(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """Horizontal grouped bar chart of T_melt, all potentials."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    plt  = _require_matplotlib()
    sids = [r.structure_id for r in mtp_result.melting_results if r.compute_ok]
    if not sids:
        return paths

    labels = ["MTP"] + [rr.label for rr in ref_results]
    n_pots = len(labels)
    total_h = 0.8
    height  = total_h / n_pots
    y       = np.arange(len(sids))
    offsets = np.linspace(-total_h / 2 + height / 2,
                           total_h / 2 - height / 2, n_pots)

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(sids) * 1.4)))

    mtp_T = [
        next((r.T_melt for r in mtp_result.melting_results
              if r.structure_id == sid and r.compute_ok), None)
        for sid in sids
    ]
    ax.barh(y + offsets[0], [t or 0 for t in mtp_T],
            height, color=_MTP_COLOR, label="MTP", alpha=0.85)

    for i, rr in enumerate(ref_results):
        col   = _ref_color(i)
        ref_T = [
            next((r.T_melt for r in rr.melting_results
                  if r.structure_id == sid and r.compute_ok), 0)
            for sid in sids
        ]
        ax.barh(y + offsets[1 + i], ref_T, height,
                color=col, label=rr.label, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(sids)
    ax.set_xlabel("T\u2098\u2091\u2097\u209c (K)")
    ax.set_title("Melting temperature \u2014 all potentials")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "combined_melting.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["melting"] = path
    return paths


def _combined_thermal_expansion(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """V(T) scatter + linear fit for all potentials on one plot."""
    paths: dict[str, Path] = {}
    plt = _require_matplotlib()

    sids = [r.structure_id for r in mtp_result.thermal_expansion_results
            if r.compute_ok and r.temperatures]
    if not sids:
        return paths

    for sid in sids:
        mtp_r = next((r for r in mtp_result.thermal_expansion_results
                      if r.structure_id == sid), None)
        if mtp_r is None:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.scatter(mtp_r.temperatures, mtp_r.volumes,
                   s=20, color=_MTP_COLOR, zorder=3)
        mtp_fit = _thexp_fit_line(mtp_r)
        lbl = f"MTP  \u03b1={mtp_r.alpha*1e6:.1f}\u00d710\u207b\u2076 K\u207b\u00b9"
        if mtp_fit:
            ax.plot(*mtp_fit, color=_MTP_COLOR, lw=2.0, label=lbl)
        else:
            ax.plot([], [], color=_MTP_COLOR, label="MTP")

        for i, rr in enumerate(ref_results):
            ref_r = next((r for r in rr.thermal_expansion_results
                          if r.structure_id == sid and r.compute_ok
                          and r.temperatures), None)
            if ref_r is None:
                continue
            col = _ref_color(i)
            ls  = ["--", "-.", ":", (0, (3, 1, 1, 1))][i % 4]
            ax.scatter(ref_r.temperatures, ref_r.volumes,
                       s=20, color=col, zorder=3, marker="^")
            ref_fit = _thexp_fit_line(ref_r)
            lbl_r = f"{rr.label}  \u03b1={ref_r.alpha*1e6:.1f}\u00d710\u207b\u2076 K\u207b\u00b9"
            if ref_fit:
                ax.plot(*ref_fit, color=col, lw=1.8, ls=ls, label=lbl_r)
            else:
                ax.plot([], [], color=col, ls=ls, label=rr.label)

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Volume per atom (\u00c5\u00b3)")
        ax.set_title(f"Thermal expansion \u2014 {sid} \u2014 all potentials")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"combined_thexp_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"thexp_{sid}"] = path

    return paths


def _combined_vacancy(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """Grouped bar chart of E_vac for all potentials."""
    paths: dict[str, Path] = {}
    try:
        import numpy as np
    except ImportError:
        return paths

    plt  = _require_matplotlib()
    sids = [r.structure_id for r in mtp_result.vacancy_results
            if r.compute_ok and math.isfinite(r.E_vac)]
    if not sids:
        return paths

    n_pots  = 1 + len(ref_results)
    total_w = 0.8
    width   = total_w / n_pots
    x       = np.arange(len(sids))
    offsets = np.linspace(-total_w / 2 + width / 2,
                           total_w / 2 - width / 2, n_pots)

    fig, ax = plt.subplots(figsize=(max(4, len(sids) * 2.0), 4.5))

    mtp_E = [next((r.E_vac for r in mtp_result.vacancy_results
                   if r.structure_id == sid and r.compute_ok), float("nan"))
             for sid in sids]
    ax.bar(x + offsets[0], mtp_E, width, color=_MTP_COLOR,
           label="MTP", alpha=0.85)

    for i, rr in enumerate(ref_results):
        col   = _ref_color(i)
        ref_E = [next((r.E_vac for r in rr.vacancy_results
                       if r.structure_id == sid and r.compute_ok), float("nan"))
                 for sid in sids]
        ax.bar(x + offsets[1 + i], ref_E, width, color=col,
               label=rr.label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sids)
    ax.set_ylabel("Vacancy formation energy (eV)")
    ax.set_title("Vacancy formation energy \u2014 all potentials")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "combined_vacancy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["vacancy"] = path
    return paths


def _combined_rdf(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """Overlay all g(r) curves on one plot per structure."""
    paths: dict[str, Path] = {}
    plt = _require_matplotlib()

    for mtp_r in mtp_result.rdf_results:
        if not mtp_r.compute_ok or not mtp_r.r:
            continue
        sid = mtp_r.structure_id

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(mtp_r.r, mtp_r.g_r, color=_MTP_COLOR, lw=2.0, label="MTP")
        if math.isfinite(mtp_r.first_peak_r):
            ax.axvline(mtp_r.first_peak_r, color=_MTP_COLOR,
                       ls=":", lw=0.9, alpha=0.6)

        for i, rr in enumerate(ref_results):
            ref_r = next((r for r in rr.rdf_results
                          if r.structure_id == sid and r.compute_ok and r.r),
                         None)
            if ref_r is None:
                continue
            col = _ref_color(i)
            ls  = ["--", "-.", ":", (0, (3, 1, 1, 1))][i % 4]
            ax.plot(ref_r.r, ref_r.g_r, color=col, lw=1.6, ls=ls,
                    label=rr.label)
            if math.isfinite(ref_r.first_peak_r):
                ax.axvline(ref_r.first_peak_r, color=col,
                           ls=":", lw=0.9, alpha=0.6)

        ax.axhline(1.0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("r (\u00c5)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"RDF \u2014 {sid} ({mtp_r.temperature:.0f} K) \u2014 all potentials")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"combined_rdf_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[f"rdf_{sid}"] = path

    return paths


def plot_combined(
    mtp_result: ValidationResult,
    ref_results: Sequence["ClassicalReferenceResult"],
    out_dir: Path,
) -> dict[str, Path]:
    """Generate combined plots with MTP + all reference potentials on one figure.

    Only called when two or more references are present.  Each plot file is
    named ``combined_<type>_<sid>.png`` and saved to ``out_dir``.

    Returns a dict mapping plot key -> Path for every file produced.
    """
    paths: dict[str, Path] = {}
    for fn in (_combined_eos, _combined_elastic, _combined_melting,
               _combined_thermal_expansion, _combined_vacancy, _combined_rdf):
        paths.update(fn(mtp_result, ref_results, out_dir))
    return paths
