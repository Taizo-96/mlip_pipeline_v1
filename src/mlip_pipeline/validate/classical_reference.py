"""Classical potential reference runner.

Runs the same physics validation steps (EOS, elastic, melting, vacancy,
thermal expansion) using a classical interatomic potential (e.g. EAM, MEAM)
and computes property deviations against MTP results.

This module is intentionally potential-agnostic: the caller supplies the
``pair_style`` and ``pair_coeff`` strings, so it works for any LAMMPS-
supported potential (EAM, MEAM, MEAM/SW, etc.).

Typical usage (Lee 2003 2NN-MEAM for Pb)
-----------------------------------------
    from mlip_pipeline.validate.classical_reference import run_classical_reference

    ref_result = run_classical_reference(
        config=config,
        pair_style="meam",
        pair_coeff="* * /path/to/library.meam Pb /path/to/Pb.meam Pb",
        out_dir=val_dir / "classical_ref",
        label="Lee2003-MEAM",
        mtp_result=validation_result,   # ValidationResult from MTP run
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.models import (
    EosResult, ElasticResult, MeltingResult,
    ThermalExpansionResult, VacancyResult, ValidationResult,
)
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate.melting import run_melting
from mlip_pipeline.validate.thermal_expansion import run_thermal_expansion
from mlip_pipeline.validate.vacancy import run_vacancy
from mlip_pipeline.validate import plots


# ── CLI helpers ──────────────────────────────────────────────────────────────

_W = 60


def _banner(title: str) -> None:
    pad = (_W - len(title) - 2) // 2
    print(f"\n{'─' * pad} {title} {'─' * (_W - pad - len(title) - 2)}")


def _step(tag: str, msg: str) -> None:
    print(f"  [{tag:<8}]  {msg}")


def _ok(tag: str, msg: str) -> None:
    print(f"  [{tag:<8}]  ✓  {msg}")


def _warn(tag: str, msg: str) -> None:
    print(f"  [{tag:<8}]  ⚠  {msg}")


# ------------------------------------------------------------------ #
# Result dataclass                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ClassicalReferenceResult:
    """Outcome of a classical potential reference validation run."""
    label: str
    pair_style: str
    pair_coeff: str
    eos_results: list[EosResult] = field(default_factory=list)
    elastic_results: list[ElasticResult] = field(default_factory=list)
    melting_results: list[MeltingResult] = field(default_factory=list)
    thermal_expansion_results: list[ThermalExpansionResult] = field(default_factory=list)
    vacancy_results: list[VacancyResult] = field(default_factory=list)
    deviations: dict = field(default_factory=dict)
    plot_paths: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Deviation helpers                                                     #
# ------------------------------------------------------------------ #

def _pct(mtp_val: Optional[float], ref_val: Optional[float]) -> Optional[float]:
    if mtp_val is None or ref_val is None:
        return None
    if ref_val == 0.0:
        return None
    return (mtp_val - ref_val) / abs(ref_val) * 100.0


def _collect_deviations(
    mtp_result: ValidationResult,
    ref_eos: list[EosResult],
    ref_elastic: list[ElasticResult],
    ref_melting: list[MeltingResult],
    ref_thexp: list[ThermalExpansionResult],
    ref_vacancy: list[VacancyResult],
) -> dict:
    devs: dict = {}

    ref_eos_map = {r.structure_id: r for r in ref_eos}
    mtp_eos_map = {r.structure_id: r for r in mtp_result.eos_results}
    for sid, ref in ref_eos_map.items():
        mtp = mtp_eos_map.get(sid)
        if mtp is None or not ref.fit_ok or not mtp.fit_ok:
            continue
        for prop in ("V0", "E0", "B0"):
            d = _pct(getattr(mtp, prop, None), getattr(ref, prop, None))
            if d is not None:
                devs[f"{prop}_{sid}"] = round(d, 2)

    ref_el_map = {r.structure_id: r for r in ref_elastic}
    mtp_el_map = {r.structure_id: r for r in mtp_result.elastic_results}
    for sid, ref in ref_el_map.items():
        mtp = mtp_el_map.get(sid)
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
            continue
        for prop in ("B_voigt", "G_voigt"):
            d = _pct(getattr(mtp, prop, None), getattr(ref, prop, None))
            if d is not None:
                devs[f"{prop}_{sid}"] = round(d, 2)
        if ref.C and mtp.C:
            for cij in ("C11", "C12", "C44", "C22", "C33"):
                ref_v = ref.C.get(cij)
                mtp_v = mtp.C.get(cij)
                d = _pct(mtp_v, ref_v)
                if d is not None:
                    devs[f"{cij}_{sid}"] = round(d, 2)

    ref_melt_map = {r.structure_id: r for r in ref_melting}
    mtp_melt_map = {r.structure_id: r for r in mtp_result.melting_results}
    for sid, ref in ref_melt_map.items():
        mtp = mtp_melt_map.get(sid)
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
            continue
        d = _pct(mtp.T_melt, ref.T_melt)
        if d is not None:
            devs[f"T_melt_{sid}"] = round(d, 2)

    ref_thexp_map = {r.structure_id: r for r in ref_thexp}
    mtp_thexp_map = {r.structure_id: r for r in mtp_result.thermal_expansion_results}
    for sid, ref in ref_thexp_map.items():
        mtp = mtp_thexp_map.get(sid)
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
            continue
        d = _pct(mtp.alpha, ref.alpha)
        if d is not None:
            devs[f"alpha_{sid}"] = round(d, 2)

    ref_vac_map = {r.structure_id: r for r in ref_vacancy}
    mtp_vac_map = {r.structure_id: r for r in mtp_result.vacancy_results}
    for sid, ref in ref_vac_map.items():
        mtp = mtp_vac_map.get(sid)
        if mtp is None or not ref.compute_ok or not ref.compute_ok:
            continue
        d = _pct(mtp.E_vac, ref.E_vac)
        if d is not None:
            devs[f"E_vac_{sid}"] = round(d, 2)

    return devs


# ------------------------------------------------------------------ #
# Deviation table printer                                               #
# ------------------------------------------------------------------ #

def print_classical_deviation_table(
    result: ClassicalReferenceResult,
    mtp_result: ValidationResult,
) -> None:
    """Print a clean MTP vs classical reference comparison table."""
    _banner(f"MTP vs {result.label}")

    ref_lbl = result.label
    col_w   = max(len(ref_lbl), 12)  # min width for reference column

    def _row(prop: str, mtp_v: float, ref_v: float, unit: str, dev_key: str) -> None:
        d    = result.deviations.get(dev_key)
        flag = "" if d is None else ("✓" if abs(d) < 10 else ("!" if abs(d) < 25 else "✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"  {prop:<8}  {mtp_v:>10.3f}  {ref_v:{col_w}.3f}  {d_str}  {unit}  {flag}")

    def _header(section: str) -> None:
        print(f"\n  ── {section} ")
        print(f"  {'Property':<8}  {'MTP':>10}  {ref_lbl:>{col_w}}  {'Δ%':>8}  Unit")
        print("  " + "─" * (10 + col_w + 32))

    # EOS
    for mtp_r in mtp_result.eos_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.eos_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.fit_ok or not ref_r.fit_ok:
            continue
        _header(f"EOS [{sid}]")
        for prop, unit in [("V0", "Å³/atom"), ("E0", "eV/atom"), ("B0", "GPa")]:
            mtp_v = getattr(mtp_r, prop, None)
            ref_v = getattr(ref_r, prop, None)
            if mtp_v is not None and ref_v is not None:
                _row(prop, mtp_v, ref_v, unit, f"{prop}_{sid}")

    # Elastic
    for mtp_r in mtp_result.elastic_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.elastic_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        _header(f"Elastic [{sid}]")
        for prop, unit in [
            ("C11", "GPa"), ("C12", "GPa"), ("C44", "GPa"),
            ("B_voigt", "GPa"), ("G_voigt", "GPa"),
        ]:
            mtp_v = (mtp_r.C or {}).get(prop) if prop.startswith("C") else getattr(mtp_r, prop, None)
            ref_v = (ref_r.C or {}).get(prop) if prop.startswith("C") else getattr(ref_r, prop, None)
            if mtp_v is not None and ref_v is not None:
                _row(prop, mtp_v, ref_v, unit, f"{prop}_{sid}")

    # Melting
    for mtp_r in mtp_result.melting_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.melting_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"T_melt_{sid}")
        flag  = "" if d is None else ("✓" if abs(d) < 5 else ("!" if abs(d) < 15 else "✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  ── Melting [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'Δ%':>8}")
        print(f"  {mtp_r.T_melt:>10.0f}  {ref_r.T_melt:{col_w}.0f}  {d_str}  K  {flag}")

    # Vacancy
    for mtp_r in mtp_result.vacancy_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.vacancy_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"E_vac_{sid}")
        flag  = "" if d is None else ("✓" if abs(d) < 15 else ("!" if abs(d) < 30 else "✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  ── Vacancy [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'Δ%':>8}")
        print(f"  {mtp_r.E_vac:>10.4f}  {ref_r.E_vac:{col_w}.4f}  {d_str}  eV  {flag}")

    # Thermal expansion
    for mtp_r in mtp_result.thermal_expansion_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.thermal_expansion_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"alpha_{sid}")
        flag  = "" if d is None else ("✓" if abs(d) < 15 else ("!" if abs(d) < 30 else "✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  ── Thermal expansion [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'Δ%':>8}")
        print(f"  {mtp_r.alpha*1e6:>10.2f}  {ref_r.alpha*1e6:{col_w}.2f}  {d_str}  ×10⁻⁶ K⁻¹  {flag}")

    print()


# ------------------------------------------------------------------ #
# Structure resolver (mirrors runner.py logic)                         #
# ------------------------------------------------------------------ #

def _resolve_structures_cl(step_cfg: dict, val_cfg: dict) -> list[dict]:
    """Return structure list, falling back to top-level structures block."""
    if "structures" in step_cfg:
        return step_cfg["structures"]
    top = val_cfg.get("structures", {})
    if isinstance(top, dict):
        result = []
        for sid, sdata in top.items():
            entry = dict(sdata)
            entry.setdefault("id", sid)
            if "data_file" in entry and "lammps_data" not in entry:
                entry["lammps_data"] = entry["data_file"]
            result.append(entry)
        return result
    if isinstance(top, list):
        return top
    return []


# ------------------------------------------------------------------ #
# Public runner                                                         #
# ------------------------------------------------------------------ #

def run_classical_reference(
    config: dict,
    pair_style: str,
    pair_coeff: str,
    out_dir: Path,
    mtp_result: ValidationResult,
    label: str = "classical",
    lammps_cmd: Optional[str] = None,
    mpi_command: Optional[str] = None,
    mpi_np: Optional[int] = None,
    cutoff: Optional[float] = None,
) -> ClassicalReferenceResult:
    val_cfg = config.get("validate", config)
    ref_dir = ensure_dir(out_dir)
    element = val_cfg.get("element", "Pb")

    if lammps_cmd is None:
        lammps_cmd = val_cfg.get("lammps_cmd") or val_cfg.get("lammps_command", "lmp_mpi")
    if mpi_command is None:
        mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    if mpi_np is None:
        mpi_np = val_cfg.get("mpi_np")
    if cutoff is None:
        cutoff = val_cfg.get("cutoff") or val_cfg.get("lammps_cutoff")

    _banner(f"Classical Reference: {label}")
    _step("setup", f"pair_style : {pair_style}")
    _step("setup", f"pair_coeff : {pair_coeff}")
    _step("setup", f"out_dir    : {ref_dir}")

    _dummy_model = Path(".")

    def _resolve(s: dict) -> Path:
        key = "lammps_data" if "lammps_data" in s else "data_file"
        data_path = Path(s[key])
        if not data_path.is_absolute():
            root = Path(config.get("project_root", "."))
            data_path = (root / data_path).resolve()
        return data_path

    eos_results:     list[EosResult]              = []
    elastic_results: list[ElasticResult]          = []
    melting_results: list[MeltingResult]          = []
    thexp_results:   list[ThermalExpansionResult] = []
    vacancy_results: list[VacancyResult]          = []
    plot_paths: dict = {}

    # ── EOS ──────────────────────────────────────────────────────────
    eos_cfg = val_cfg.get("eos", {})
    if eos_cfg.get("enabled", True):
        for s in _resolve_structures_cl(eos_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                _warn("ref-eos", f"{sid}: data not found")
                eos_results.append(EosResult(
                    structure_id=sid, volumes=[], energies=[],
                    fit_error=f"data file not found: {data_path}",
                ))
                continue
            res = run_eos(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                scale_min=eos_cfg.get("scale_min", 0.85),
                scale_max=eos_cfg.get("scale_max", 1.15),
                n_points=eos_cfg.get("n_points", 21),
                cutoff=cutoff,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            eos_results.append(res)
            p = plots.plot_eos(res, ref_dir)
            if p:
                plot_paths[f"eos_{sid}"] = p
            if res.fit_ok:
                _ok("ref-eos", f"{sid}: B₀={res.B0:.1f} GPa  V₀={res.V0:.3f} Å³")

    # ── Elastic ───────────────────────────────────────────────────────
    el_cfg = val_cfg.get("elastic", {})
    if el_cfg.get("enabled", True):
        for s in _resolve_structures_cl(el_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                _warn("ref-el", f"{sid}: data not found")
                elastic_results.append(ElasticResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_elastic(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                delta=el_cfg.get("delta", 0.01),
                cutoff=cutoff,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            elastic_results.append(res)
            if res.compute_ok:
                _ok("ref-el", f"{sid}: B={res.B_voigt:.1f}  G={res.G_voigt:.1f} GPa")

    # ── Melting ───────────────────────────────────────────────────────
    melt_cfg = val_cfg.get("melting", {})
    if melt_cfg.get("enabled", True):
        for s in _resolve_structures_cl(melt_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                _warn("ref-melt", f"{sid}: data not found")
                melting_results.append(MeltingResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_melting(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                T_start=melt_cfg.get("T_start", 550.0),
                T_end=melt_cfg.get("T_end", 700.0),
                T_step=melt_cfg.get("T_step", 10.0),
                supercell_repeat=melt_cfg.get("supercell_repeat", 5),
                n_equil=melt_cfg.get("n_equil", 5000),
                n_prod=melt_cfg.get("n_prod", 20000),
                dt=melt_cfg.get("dt", 0.002),
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            melting_results.append(res)
            if res.compute_ok:
                _ok("ref-melt", f"{sid}: T_melt = {res.T_melt:.0f} K")

    # ── Thermal expansion ─────────────────────────────────────────────
    thexp_cfg = val_cfg.get("thexp", val_cfg.get("thermal_expansion", {}))
    if thexp_cfg.get("enabled", True):
        for s in _resolve_structures_cl(thexp_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                _warn("ref-thexp", f"{sid}: data not found")
                thexp_results.append(ThermalExpansionResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_thermal_expansion(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                temperatures=thexp_cfg.get("temperatures"),
                T_ref=thexp_cfg.get("T_ref", 300.0),
                n_equil=thexp_cfg.get("n_equil", 5000),
                n_prod=thexp_cfg.get("n_prod", 10000),
                dt=thexp_cfg.get("dt", 0.002),
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            thexp_results.append(res)
            if res.compute_ok:
                _ok("ref-thexp", f"{sid}: α = {res.alpha*1e6:.2f}×10⁻⁶ K⁻¹")

    # ── Vacancy ───────────────────────────────────────────────────────
    vac_cfg = val_cfg.get("vacancy", {})
    if vac_cfg.get("enabled", True):
        for s in _resolve_structures_cl(vac_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                _warn("ref-vac", f"{sid}: data not found")
                vacancy_results.append(VacancyResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_vacancy(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                supercell_repeat=vac_cfg.get("supercell_repeat", 3),
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            vacancy_results.append(res)
            if res.compute_ok:
                _ok("ref-vac", f"{sid}: E_vac = {res.E_vac:.4f} eV")

    # Compute deviations
    devs = _collect_deviations(
        mtp_result,
        eos_results, elastic_results,
        melting_results, thexp_results, vacancy_results,
    )

    result = ClassicalReferenceResult(
        label=label,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
        eos_results=eos_results,
        elastic_results=elastic_results,
        melting_results=melting_results,
        thermal_expansion_results=thexp_results,
        vacancy_results=vacancy_results,
        deviations=devs,
        plot_paths=plot_paths,
    )

    print_classical_deviation_table(result, mtp_result)
    return result
