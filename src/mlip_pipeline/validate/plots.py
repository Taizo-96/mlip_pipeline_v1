"""Plotting helpers for the validate step."""
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


# ── Colour palette (consistent across all plots) ─────────────────────────────
_MTP_COLOR = "#2c7bb6"
_REF_COLOR = "#d7191c"
_COLORS    = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for validation plots") from exc


# ── Individual MTP plots ──────────────────────────────────────────────────────

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
    ax.scatter(result.volumes, result.energies, s=30, color=_MTP_COLOR, zorder=3, label="MTP")
    if result.fit_ok:
        v_fit = np.linspace(min(result.volumes), max(result.volumes), 200)
        B0_ev = result.B0 / 160.2176634
        e_fit = [_birch_murnaghan(v, result.V0, result.E0, B0_ev, result.B0p) for v in v_fit]
        ax.plot(v_fit, e_fit, color=_REF_COLOR, lw=1.5,
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
        ax.bar(x + i * width, vals, width, label=r.structure_id, color=_COLORS[i % len(_COLORS)])
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

    for i, r in enumerate(valid):
        c = _COLORS[i % len(_COLORS)]
        y = i
        lo = r.T_bracket_lo if math.isfinite(r.T_bracket_lo) else r.T_melt
        hi = r.T_bracket_hi if math.isfinite(r.T_bracket_hi) else r.T_melt

        if math.isfinite(lo) and math.isfinite(hi) and lo != hi:
            ax.barh(y, hi - lo, left=lo, height=0.4, color=c, alpha=0.35,
                    label=f"{r.structure_id} bracket")
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
    for i, r in enumerate(valid):
        c = _COLORS[i % len(_COLORS)]
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
    for i, r in enumerate(valid):
        ax.bar(i, r.E_vac, color=_COLORS[i % len(_COLORS)], width=0.6,
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
    for i, r in enumerate(valid):
        c = _COLORS[i % len(_COLORS)]
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


# ── MTP vs classical reference comparison plots ───────────────────────────────

def plot_comparison(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> dict[str, Path]:
    """Generate side-by-side MTP vs classical reference comparison plots.

    Returns a dict mapping plot key → Path for every plot produced.
    """
    paths: dict[str, Path] = {}
    plt = _require_matplotlib()
    ref_lbl = ref_result.label

    # ── EOS overlay: MTP data+fit vs reference fit per structure ─────────────
    try:
        import numpy as np
        from mlip_pipeline.validate.eos import _birch_murnaghan
        _have_np = True
    except ImportError:
        _have_np = False

    ref_eos_map = {r.structure_id: r for r in ref_result.eos_results}
    for mtp_r in mtp_result.eos_results:
        sid   = mtp_r.structure_id
        ref_r = ref_eos_map.get(sid)
        if ref_r is None or not mtp_r.fit_ok or not ref_r.fit_ok or not _have_np:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        # MTP scatter + fit
        if mtp_r.volumes:
            ax.scatter(mtp_r.volumes, mtp_r.energies, s=25, color=_MTP_COLOR,
                       zorder=3, label="MTP data")
            v_fit = np.linspace(min(mtp_r.volumes), max(mtp_r.volumes), 200)
            B0_ev = mtp_r.B0 / 160.2176634
            e_fit = [_birch_murnaghan(v, mtp_r.V0, mtp_r.E0, B0_ev, mtp_r.B0p) for v in v_fit]
            ax.plot(v_fit, e_fit, color=_MTP_COLOR, lw=1.8,
                    label=f"MTP fit  B\u2080={mtp_r.B0:.1f} GPa")
        # Reference fit (using its own V range if available, else MTP range)
        if ref_r.volumes:
            v_ref = np.linspace(min(ref_r.volumes), max(ref_r.volumes), 200)
        elif mtp_r.volumes:
            v_ref = np.linspace(min(mtp_r.volumes), max(mtp_r.volumes), 200)
        else:
            plt.close(fig)
            continue
        B0_ev_ref = ref_r.B0 / 160.2176634
        e_ref = [_birch_murnaghan(v, ref_r.V0, ref_r.E0, B0_ev_ref, ref_r.B0p) for v in v_ref]
        ax.plot(v_ref, e_ref, color=_REF_COLOR, lw=1.8, ls="--",
                label=f"{ref_lbl} fit  B\u2080={ref_r.B0:.1f} GPa")

        ax.set_xlabel("Volume per atom (\u00c5\u00b3)")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title(f"EOS comparison \u2014 {sid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        key  = f"eos_{sid}"
        path = out_dir / f"cmp_eos_{sid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[key] = path

    # ── Elastic bar chart: grouped MTP vs reference ───────────────────────────
    ref_el_map = {r.structure_id: r for r in ref_result.elastic_results}
    pairs = [
        (mtp_r, ref_el_map[mtp_r.structure_id])
        for mtp_r in mtp_result.elastic_results
        if mtp_r.structure_id in ref_el_map
           and mtp_r.compute_ok and ref_el_map[mtp_r.structure_id].compute_ok
    ]
    if pairs:
        try:
            import numpy as np
        except ImportError:
            np = None
        for mtp_r, ref_r in pairs:
            sid  = mtp_r.structure_id
            keys = sorted(set((mtp_r.C or {}).keys()) | set((ref_r.C or {}).keys()))
            if not keys or np is None:
                continue
            x     = np.arange(len(keys))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.4), 4))
            mtp_vals = [(mtp_r.C or {}).get(k, float("nan")) for k in keys]
            ref_vals = [(ref_r.C or {}).get(k, float("nan")) for k in keys]
            ax.bar(x - width / 2, mtp_vals, width, color=_MTP_COLOR,
                   label="MTP", alpha=0.85)
            ax.bar(x + width / 2, ref_vals, width, color=_REF_COLOR,
                   label=ref_lbl, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(keys)
            ax.set_ylabel("Elastic constant (GPa)")
            ax.set_title(f"Elastic constants comparison \u2014 {sid}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            key  = f"elastic_{sid}"
            path = out_dir / f"cmp_elastic_{sid}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths[key] = path

    # ── Melting temperature comparison (horizontal bar) ───────────────────────
    ref_melt_map = {r.structure_id: r for r in ref_result.melting_results}
    melt_pairs = [
        (mtp_r, ref_melt_map[mtp_r.structure_id])
        for mtp_r in mtp_result.melting_results
        if mtp_r.structure_id in ref_melt_map
           and mtp_r.compute_ok and ref_melt_map[mtp_r.structure_id].compute_ok
    ]
    if melt_pairs:
        sids  = [m.structure_id for m, _ in melt_pairs]
        mtp_T = [m.T_melt for m, _ in melt_pairs]
        ref_T = [r.T_melt for _, r in melt_pairs]
        try:
            import numpy as np
            y     = np.arange(len(sids))
            width = 0.35
            fig, ax = plt.subplots(figsize=(6, max(2.5, len(sids) * 1.2)))
            ax.barh(y - width / 2, mtp_T, width, color=_MTP_COLOR, label="MTP", alpha=0.85)
            ax.barh(y + width / 2, ref_T, width, color=_REF_COLOR, label=ref_lbl, alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(sids)
            ax.set_xlabel("Melting temperature (K)")
            ax.set_title("Melting temperature comparison")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / "cmp_melting.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["melting"] = path
        except ImportError:
            pass

    # ── Thermal expansion comparison ──────────────────────────────────────────
    ref_thexp_map = {r.structure_id: r for r in ref_result.thermal_expansion_results}
    thexp_pairs = [
        (mtp_r, ref_thexp_map[mtp_r.structure_id])
        for mtp_r in mtp_result.thermal_expansion_results
        if mtp_r.structure_id in ref_thexp_map
           and mtp_r.compute_ok and ref_thexp_map[mtp_r.structure_id].compute_ok
           and mtp_r.temperatures
    ]
    if thexp_pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (mtp_r, ref_r) in enumerate(thexp_pairs):
            c_mtp = _COLORS[i * 2 % len(_COLORS)]
            c_ref = _COLORS[(i * 2 + 1) % len(_COLORS)]
            ax.scatter(mtp_r.temperatures, mtp_r.volumes, s=20,
                       color=c_mtp, zorder=3, alpha=0.7)
            if math.isfinite(mtp_r.alpha) and math.isfinite(mtp_r.V_ref):
                Ts = [min(mtp_r.temperatures), max(mtp_r.temperatures)]
                Vs = [mtp_r.V_ref + 3 * mtp_r.alpha * mtp_r.V_ref * (T - mtp_r.T_ref)
                      for T in Ts]
                ax.plot(Ts, Vs, color=c_mtp, lw=1.8,
                        label=f"MTP {mtp_r.structure_id}  \u03b1={mtp_r.alpha*1e6:.1f}e-6")
            if ref_r.temperatures:
                ax.scatter(ref_r.temperatures, ref_r.volumes, s=20,
                           color=c_ref, marker="^", zorder=3, alpha=0.7)
            if math.isfinite(ref_r.alpha) and math.isfinite(ref_r.V_ref) and ref_r.temperatures:
                Ts_r = [min(ref_r.temperatures), max(ref_r.temperatures)]
                Vs_r = [ref_r.V_ref + 3 * ref_r.alpha * ref_r.V_ref * (T - ref_r.T_ref)
                        for T in Ts_r]
                ax.plot(Ts_r, Vs_r, color=c_ref, lw=1.8, ls="--",
                        label=f"{ref_lbl} {ref_r.structure_id}  \u03b1={ref_r.alpha*1e6:.1f}e-6")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Volume per atom (\u00c5\u00b3)")
        ax.set_title("Thermal expansion comparison")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "cmp_thermal_expansion.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["thermal_expansion"] = path

    # ── Vacancy formation energy comparison ───────────────────────────────────
    ref_vac_map = {r.structure_id: r for r in ref_result.vacancy_results}
    vac_pairs = [
        (mtp_r, ref_vac_map[mtp_r.structure_id])
        for mtp_r in mtp_result.vacancy_results
        if mtp_r.structure_id in ref_vac_map
           and mtp_r.compute_ok and ref_vac_map[mtp_r.structure_id].compute_ok
           and math.isfinite(mtp_r.E_vac)
    ]
    if vac_pairs:
        try:
            import numpy as np
            sids      = [m.structure_id for m, _ in vac_pairs]
            mtp_evac  = [m.E_vac for m, _ in vac_pairs]
            ref_evac  = [r.E_vac for _, r in vac_pairs]
            y         = np.arange(len(sids))
            width     = 0.35
            fig, ax   = plt.subplots(figsize=(max(4, len(sids) * 1.8), 4))
            ax.bar(y - width / 2, mtp_evac, width, color=_MTP_COLOR, label="MTP", alpha=0.85)
            ax.bar(y + width / 2, ref_evac, width, color=_REF_COLOR, label=ref_lbl, alpha=0.85)
            ax.set_xticks(y)
            ax.set_xticklabels(sids)
            ax.set_ylabel("Vacancy formation energy (eV)")
            ax.set_title("Vacancy formation energy comparison")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / "cmp_vacancy.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["vacancy"] = path
        except ImportError:
            pass

    return paths
