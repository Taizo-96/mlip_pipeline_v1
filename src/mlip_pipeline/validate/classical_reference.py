"""Classical potential reference runner.

Runs the same physics validation steps (EOS, elastic, melting, vacancy,
thermal expansion, RDF) using a classical interatomic potential (e.g. EAM,
MEAM) and collects result dataclasses for later comparison against MTP.

Design principle
----------------
This module is *computation only*: it calls ``run_*`` functions to produce
result dataclasses but does **not** call any plotting function itself.
Plotting is done by the caller (``runner.py``) after both MTP and reference
results are available, via ``plots.plot_comparison()``.

The module is potential-agnostic: the caller supplies ``pair_style`` and
``pair_coeff``, so it works for any LAMMPS-supported potential.

Typical usage
-------------
    from mlip_pipeline.validate.classical_reference import run_classical_reference

    ref_result = run_classical_reference(
        config=config,
        pair_style="meam",
        pair_coeff="* * /path/library.meam Pb /path/Pb.meam Pb",
        out_dir=val_dir / "classical_ref",
        label="Lee2003-MEAM",
        mtp_result=validation_result,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.models import (
    EosResult, ElasticResult, MeltingResult,
    ThermalExpansionResult, VacancyResult, RdfResult, ValidationResult,
)
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate.melting import run_melting
from mlip_pipeline.validate.thermal_expansion import run_thermal_expansion
from mlip_pipeline.validate.vacancy import run_vacancy
from mlip_pipeline.validate.rdf import run_rdf
from mlip_pipeline.validate._cli import banner, step, ok, warn


# ------------------------------------------------------------------ #
# Result dataclass                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ClassicalReferenceResult:
    """Outcome of a classical potential reference validation run."""
    label: str
    pair_style: str
    pair_coeff: str
    eos_results:               list[EosResult]              = field(default_factory=list)
    elastic_results:           list[ElasticResult]          = field(default_factory=list)
    melting_results:           list[MeltingResult]          = field(default_factory=list)
    thermal_expansion_results: list[ThermalExpansionResult] = field(default_factory=list)
    vacancy_results:           list[VacancyResult]          = field(default_factory=list)
    rdf_results:               list[RdfResult]              = field(default_factory=list)
    deviations: dict = field(default_factory=dict)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_manifest(self, out_dir: Path) -> Path:
        """Serialise all result data to ``<out_dir>/ref_manifest.json``.

        Saves enough data to reconstruct the object via :meth:`load_manifest`
        without re-running any LAMMPS calculations.
        """
        manifest = {
            "label":      self.label,
            "pair_style": self.pair_style,
            "pair_coeff": self.pair_coeff,
            "deviations": self.deviations,
            "eos": [
                {
                    "structure_id": r.structure_id,
                    "V0": r.V0, "E0": r.E0, "B0": r.B0, "B0p": r.B0p,
                    "volumes": r.volumes, "energies": r.energies,
                    "fit_ok": r.fit_ok, "fit_error": r.fit_error,
                }
                for r in self.eos_results
            ],
            "elastic": [
                {
                    "structure_id": r.structure_id,
                    "C": r.C, "B_voigt": r.B_voigt, "G_voigt": r.G_voigt,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.elastic_results
            ],
            "melting": [
                {
                    "structure_id": r.structure_id,
                    "T_melt": r.T_melt,
                    "T_bracket_lo": r.T_bracket_lo,
                    "T_bracket_hi": r.T_bracket_hi,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.melting_results
            ],
            "thermal_expansion": [
                {
                    "structure_id": r.structure_id,
                    "temperatures": r.temperatures, "volumes": r.volumes,
                    "alpha": r.alpha, "T_ref": r.T_ref,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.thermal_expansion_results
            ],
            "vacancy": [
                {
                    "structure_id": r.structure_id,
                    "E_vac": r.E_vac,
                    "n_atoms_perfect": r.n_atoms_perfect,
                    "n_atoms_vacancy": r.n_atoms_vacancy,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.vacancy_results
            ],
            "rdf": [
                {
                    "structure_id": r.structure_id,
                    "r": r.r, "g_r": r.g_r,
                    "first_peak_r": r.first_peak_r, "first_peak_g": r.first_peak_g,
                    "temperature": r.temperature,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.rdf_results
            ],
        }
        path = out_dir / "ref_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load_manifest(cls, manifest_path: Path) -> "ClassicalReferenceResult":
        """Reconstruct a ClassicalReferenceResult from a saved ref_manifest.json."""
        data = json.loads(manifest_path.read_text())

        eos_results = [
            EosResult(
                structure_id=r["structure_id"],
                volumes=r.get("volumes", []),
                energies=r.get("energies", []),
                V0=r.get("V0", 0.0),
                E0=r.get("E0", 0.0),
                B0=r.get("B0", 0.0),
                B0p=r.get("B0p", 0.0),
                fit_ok=r.get("fit_ok", False),
                fit_error=r.get("fit_error"),
            )
            for r in data.get("eos", [])
        ]
        elastic_results = [
            ElasticResult(
                structure_id=r["structure_id"],
                C=r.get("C") or {},
                B_voigt=r.get("B_voigt", 0.0),
                G_voigt=r.get("G_voigt", 0.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("elastic", [])
        ]
        melting_results = [
            MeltingResult(
                structure_id=r["structure_id"],
                T_melt=r.get("T_melt", 0.0),
                T_bracket_lo=r.get("T_bracket_lo", float("nan")),
                T_bracket_hi=r.get("T_bracket_hi", float("nan")),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("melting", [])
        ]
        thexp_results = [
            ThermalExpansionResult(
                structure_id=r["structure_id"],
                temperatures=r.get("temperatures", []),
                volumes=r.get("volumes", []),
                alpha=r.get("alpha", 0.0),
                T_ref=r.get("T_ref", 300.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("thermal_expansion", [])
        ]
        vacancy_results = [
            VacancyResult(
                structure_id=r["structure_id"],
                E_vac=r.get("E_vac", 0.0),
                n_atoms_perfect=r.get("n_atoms_perfect", 0),
                n_atoms_vacancy=r.get("n_atoms_vacancy", 0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("vacancy", [])
        ]
        rdf_results = [
            RdfResult(
                structure_id=r["structure_id"],
                r=r.get("r", []),
                g_r=r.get("g_r", []),
                first_peak_r=r.get("first_peak_r", 0.0),
                first_peak_g=r.get("first_peak_g", 0.0),
                temperature=r.get("temperature", 300.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("rdf", [])
        ]

        return cls(
            label=data["label"],
            pair_style=data["pair_style"],
            pair_coeff=data["pair_coeff"],
            eos_results=eos_results,
            elastic_results=elastic_results,
            melting_results=melting_results,
            thermal_expansion_results=thexp_results,
            vacancy_results=vacancy_results,
            rdf_results=rdf_results,
            deviations=data.get("deviations", {}),
        )


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
    ref_eos:     list[EosResult],
    ref_elastic: list[ElasticResult],
    ref_melting: list[MeltingResult],
    ref_thexp:   list[ThermalExpansionResult],
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

    ref_el_map  = {r.structure_id: r for r in ref_elastic}
    mtp_el_map  = {r.structure_id: r for r in mtp_result.elastic_results}
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
                d = _pct(mtp.C.get(cij), ref.C.get(cij))
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
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
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
    banner(f"MTP vs {result.label}")

    ref_lbl = result.label
    col_w   = max(len(ref_lbl), 12)

    def _row(prop: str, mtp_v: float, ref_v: float, unit: str, dev_key: str) -> None:
        d     = result.deviations.get(dev_key)
        flag  = "" if d is None else ("\u2713" if abs(d) < 10 else ("!" if abs(d) < 25 else "\u2717"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"  {prop:<8}  {mtp_v:>10.3f}  {ref_v:{col_w}.3f}  {d_str}  {unit}  {flag}")

    def _header(section: str) -> None:
        print(f"\n  \u2500\u2500 {section} ")
        print(f"  {'Property':<8}  {'MTP':>10}  {ref_lbl:>{col_w}}  {'\u0394%':>8}  Unit")
        print("  " + "\u2500" * (10 + col_w + 32))

    for mtp_r in mtp_result.eos_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.eos_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.fit_ok or not ref_r.fit_ok:
            continue
        _header(f"EOS [{sid}]")
        for prop, unit in [("V0", "\u00c5\u00b3/atom"), ("E0", "eV/atom"), ("B0", "GPa")]:
            mtp_v = getattr(mtp_r, prop, None)
            ref_v = getattr(ref_r, prop, None)
            if mtp_v is not None and ref_v is not None:
                _row(prop, mtp_v, ref_v, unit, f"{prop}_{sid}")

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
            mtp_v = (mtp_r.C or {}).get(prop) if prop.startswith("C") \
                    else getattr(mtp_r, prop, None)
            ref_v = (ref_r.C or {}).get(prop) if prop.startswith("C") \
                    else getattr(ref_r, prop, None)
            if mtp_v is not None and ref_v is not None:
                _row(prop, mtp_v, ref_v, unit, f"{prop}_{sid}")

    for mtp_r in mtp_result.melting_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.melting_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"T_melt_{sid}")
        flag  = "" if d is None else ("\u2713" if abs(d) < 5 else ("!" if abs(d) < 15 else "\u2717"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  \u2500\u2500 Melting [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'\u0394%':>8}")
        print(f"  {mtp_r.T_melt:>10.0f}  {ref_r.T_melt:{col_w}.0f}  {d_str}  K  {flag}")

    for mtp_r in mtp_result.vacancy_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.vacancy_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"E_vac_{sid}")
        flag  = "" if d is None else ("\u2713" if abs(d) < 15 else ("!" if abs(d) < 30 else "\u2717"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  \u2500\u2500 Vacancy [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'\u0394%':>8}")
        print(f"  {mtp_r.E_vac:>10.4f}  {ref_r.E_vac:{col_w}.4f}  {d_str}  eV  {flag}")

    for mtp_r in mtp_result.thermal_expansion_results:
        sid   = mtp_r.structure_id
        ref_r = next((r for r in result.thermal_expansion_results
                      if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d     = result.deviations.get(f"alpha_{sid}")
        flag  = "" if d is None else ("\u2713" if abs(d) < 15 else ("!" if abs(d) < 30 else "\u2717"))
        d_str = f"{d:>+7.1f}%" if d is not None else "    N/A"
        print(f"\n  \u2500\u2500 Thermal expansion [{sid}]")
        print(f"  {'MTP':>10}  {ref_lbl:>{col_w}}  {'\u0394%':>8}")
        print(f"  {mtp_r.alpha*1e6:>10.2f}  {ref_r.alpha*1e6:{col_w}.2f}  {d_str}  \u00d710\u207b\u2076 K\u207b\u00b9  {flag}")

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

    banner(f"Classical Reference: {label}")
    step("setup", f"pair_style : {pair_style}")
    step("setup", f"pair_coeff : {pair_coeff}")
    step("setup", f"out_dir    : {ref_dir}")

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
    rdf_results:     list[RdfResult]              = []

    # ── EOS ──────────────────────────────────────────────────────────
    eos_cfg = val_cfg.get("eos", {})
    if eos_cfg.get("enabled", True):
        for s in _resolve_structures_cl(eos_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-eos", f"{sid}: data not found")
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
            if res.fit_ok:
                ok("ref-eos", f"{sid}: B\u2080={res.B0:.1f} GPa  V\u2080={res.V0:.3f} \u00c5\u00b3")
            else:
                warn("ref-eos", f"{sid}: fit failed \u2014 {res.fit_error}")

    # ── Elastic ───────────────────────────────────────────────────────
    el_cfg = val_cfg.get("elastic", {})
    if el_cfg.get("enabled", True):
        for s in _resolve_structures_cl(el_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-el", f"{sid}: data not found")
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
                ok("ref-el", f"{sid}: B={res.B_voigt:.1f}  G={res.G_voigt:.1f} GPa")
            else:
                warn("ref-el", f"{sid}: failed \u2014 {res.error}")

    # ── Melting ───────────────────────────────────────────────────────
    melt_cfg = val_cfg.get("melting", {})
    if melt_cfg.get("enabled", True):
        for s in _resolve_structures_cl(melt_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-melt", f"{sid}: data not found")
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
                ok("ref-melt", f"{sid}: T_melt = {res.T_melt:.0f} K")
            else:
                warn("ref-melt", f"{sid}: failed \u2014 {res.error}")

    # ── Thermal expansion ─────────────────────────────────────────────
    thexp_cfg = val_cfg.get("thexp", val_cfg.get("thermal_expansion", {}))
    if thexp_cfg.get("enabled", True):
        for s in _resolve_structures_cl(thexp_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-thexp", f"{sid}: data not found")
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
                ok("ref-thexp", f"{sid}: \u03b1 = {res.alpha*1e6:.2f}\u00d710\u207b\u2076 K\u207b\u00b9")
            else:
                warn("ref-thexp", f"{sid}: failed \u2014 {res.error}")

    # ── Vacancy ───────────────────────────────────────────────────────
    vac_cfg = val_cfg.get("vacancy", {})
    if vac_cfg.get("enabled", True):
        for s in _resolve_structures_cl(vac_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-vac", f"{sid}: data not found")
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
                ok("ref-vac", f"{sid}: E_vac = {res.E_vac:.4f} eV")
            else:
                warn("ref-vac", f"{sid}: failed \u2014 {res.error}")

    # ── RDF ──────────────────────────────────────────────────────────
    rdf_cfg = val_cfg.get("rdf", {})
    if rdf_cfg.get("enabled", True):
        for s in _resolve_structures_cl(rdf_cfg, val_cfg):
            sid = s.get("id", "unknown")
            data_path = _resolve(s)
            if not data_path.exists():
                warn("ref-rdf", f"{sid}: data not found")
                rdf_results.append(RdfResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_rdf(
                sid, data_path, _dummy_model, ref_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                temperature=rdf_cfg.get("temperature", 300.0),
                r_max=rdf_cfg.get("r_max", 8.0),
                n_bins=rdf_cfg.get("n_bins", 200),
                n_equil=rdf_cfg.get("n_equil", 5000),
                n_prod=rdf_cfg.get("n_prod", 20000),
                dt=rdf_cfg.get("dt", 0.002),
                supercell_repeat=rdf_cfg.get("supercell_repeat", 3),
                pair_style=pair_style,
                pair_coeff=pair_coeff,
            )
            rdf_results.append(res)
            if res.compute_ok:
                ok("ref-rdf", f"{sid}: first peak at {res.first_peak_r:.3f} \u00c5")
            else:
                warn("ref-rdf", f"{sid}: failed \u2014 {res.error}")

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
        rdf_results=rdf_results,
        deviations=devs,
    )

    print_classical_deviation_table(result, mtp_result)
    result.save_manifest(ref_dir)
    return result
