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
from mlip_pipeline.validate.models import (
    EosResult, ElasticResult, MeltingResult,
    ThermalExpansionResult, VacancyResult, ValidationResult,
)
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate.melting import run_melting
from mlip_pipeline.validate.thermal_expansion import run_thermal_expansion
from mlip_pipeline.validate.vacancy import run_vacancy
from mlip_pipeline.validate import plots


# ------------------------------------------------------------------ #
# Result dataclass                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ClassicalReferenceResult:
    """Outcome of a classical potential reference validation run.

    Attributes
    ----------
    label
        Human-readable name of the reference potential (e.g. "Lee2003-MEAM").
    pair_style
        LAMMPS pair_style string used (e.g. "meam").
    pair_coeff
        LAMMPS pair_coeff string used.
    eos_results
        EOS results from the classical potential run.
    elastic_results
        Elastic constant results.
    melting_results
        Melting temperature bracket results.
    thermal_expansion_results
        Thermal expansion coefficient results.
    vacancy_results
        Vacancy formation energy results.
    deviations
        Flat dict of property deviations (MTP vs classical reference).
        Keys follow the pattern ``<property>_<structure_id>`` where
        property is one of: V0, E0, B0, C11, C12, C44, B_voigt, G_voigt,
        T_melt, alpha, E_vac.
        Values are percentage deviations: (MTP - REF) / |REF| * 100.
    plot_paths
        Dict of plot filenames written.
    """
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
    """Return percentage deviation (MTP - REF) / |REF| * 100, or None."""
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
    """Compare MTP results against classical reference results structure-by-structure.

    Returns a flat dict of percentage deviations keyed as
    ``<property>_<structure_id>``, e.g. ``V0_fcc``, ``C11_fcc``,
    ``T_melt_fcc``, ``E_vac_fcc``, ``alpha_fcc``.
    """
    devs: dict = {}

    # -- EOS --
    ref_eos_map  = {r.structure_id: r for r in ref_eos}
    mtp_eos_map  = {r.structure_id: r for r in mtp_result.eos_results}
    for sid, ref in ref_eos_map.items():
        mtp = mtp_eos_map.get(sid)
        if mtp is None or not ref.fit_ok or not mtp.fit_ok:
            continue
        for prop in ("V0", "E0", "B0"):
            d = _pct(getattr(mtp, prop, None), getattr(ref, prop, None))
            if d is not None:
                devs[f"{prop}_{sid}"] = round(d, 2)

    # -- Elastic --
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
        # Individual Cij
        if ref.C and mtp.C:
            for cij in ("C11", "C12", "C44", "C22", "C33"):
                ref_v = ref.C.get(cij)
                mtp_v = mtp.C.get(cij)
                d = _pct(mtp_v, ref_v)
                if d is not None:
                    devs[f"{cij}_{sid}"] = round(d, 2)

    # -- Melting --
    ref_melt_map = {r.structure_id: r for r in ref_melting}
    mtp_melt_map = {r.structure_id: r for r in mtp_result.melting_results}
    for sid, ref in ref_melt_map.items():
        mtp = mtp_melt_map.get(sid)
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
            continue
        d = _pct(mtp.T_melt, ref.T_melt)
        if d is not None:
            devs[f"T_melt_{sid}"] = round(d, 2)

    # -- Thermal expansion --
    ref_thexp_map = {r.structure_id: r for r in ref_thexp}
    mtp_thexp_map = {r.structure_id: r for r in mtp_result.thermal_expansion_results}
    for sid, ref in ref_thexp_map.items():
        mtp = mtp_thexp_map.get(sid)
        if mtp is None or not ref.compute_ok or not mtp.compute_ok:
            continue
        d = _pct(mtp.alpha, ref.alpha)
        if d is not None:
            devs[f"alpha_{sid}"] = round(d, 2)

    # -- Vacancy --
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
    """Print a formatted side-by-side comparison of MTP vs classical reference."""
    print(f"\n  ── vs. {result.label} (classical reference) ────────────────────")

    # EOS
    for mtp_r in mtp_result.eos_results:
        sid = mtp_r.structure_id
        ref_r = next((r for r in result.eos_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.fit_ok or not ref_r.fit_ok:
            continue
        print(f"\n  [EOS] {sid}")
        print(f"  {'Property':<8}  {'MTP':>10}  {result.label:>16}  {'Δ%':>8}  Unit")
        print("  " + "-" * 52)
        for prop, unit in [("V0", "Å³/atom"), ("E0", "eV/atom"), ("B0", "GPa")]:
            mtp_v = getattr(mtp_r, prop, None)
            ref_v = getattr(ref_r, prop, None)
            if mtp_v is None or ref_v is None:
                continue
            d = result.deviations.get(f"{prop}_{sid}")
            flag = "" if d is None else ("  ✓" if abs(d) < 10 else ("  !" if abs(d) < 25 else "  ✗"))
            d_str = f"{d:>+7.1f}%" if d is not None else "       N/A"
            print(f"  {prop:<8}  {mtp_v:>10.3f}  {ref_v:>16.3f}  {d_str}  {unit}{flag}")

    # Elastic
    for mtp_r in mtp_result.elastic_results:
        sid = mtp_r.structure_id
        ref_r = next((r for r in result.elastic_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        print(f"\n  [Elastic] {sid}")
        print(f"  {'Property':<8}  {'MTP':>10}  {result.label:>16}  {'Δ%':>8}  Unit")
        print("  " + "-" * 52)
        props = [(k, "GPa") for k in ("C11", "C12", "C44", "B_voigt", "G_voigt")]
        for prop, unit in props:
            mtp_v = (mtp_r.C or {}).get(prop) if prop.startswith("C") else getattr(mtp_r, prop, None)
            ref_v = (ref_r.C or {}).get(prop) if prop.startswith("C") else getattr(ref_r, prop, None)
            if mtp_v is None or ref_v is None:
                continue
            d = result.deviations.get(f"{prop}_{sid}")
            flag = "" if d is None else ("  ✓" if abs(d) < 10 else ("  !" if abs(d) < 25 else "  ✗"))
            d_str = f"{d:>+7.1f}%" if d is not None else "       N/A"
            print(f"  {prop:<8}  {mtp_v:>10.1f}  {ref_v:>16.1f}  {d_str}  {unit}{flag}")

    # Melting
    for mtp_r in mtp_result.melting_results:
        sid = mtp_r.structure_id
        ref_r = next((r for r in result.melting_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d = result.deviations.get(f"T_melt_{sid}")
        flag = "" if d is None else ("  ✓" if abs(d) < 5 else ("  !" if abs(d) < 15 else "  ✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "       N/A"
        print(f"\n  [Melting] {sid}: MTP={mtp_r.T_melt:.0f} K  "
              f"{result.label}={ref_r.T_melt:.0f} K  Δ={d_str}{flag}")

    # Vacancy
    for mtp_r in mtp_result.vacancy_results:
        sid = mtp_r.structure_id
        ref_r = next((r for r in result.vacancy_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d = result.deviations.get(f"E_vac_{sid}")
        flag = "" if d is None else ("  ✓" if abs(d) < 15 else ("  !" if abs(d) < 30 else "  ✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "       N/A"
        print(f"\n  [Vacancy] {sid}: MTP={mtp_r.E_vac:.4f} eV  "
              f"{result.label}={ref_r.E_vac:.4f} eV  Δ={d_str}{flag}")

    # Thermal expansion
    for mtp_r in mtp_result.thermal_expansion_results:
        sid = mtp_r.structure_id
        ref_r = next((r for r in result.thermal_expansion_results if r.structure_id == sid), None)
        if ref_r is None or not mtp_r.compute_ok or not ref_r.compute_ok:
            continue
        d = result.deviations.get(f"alpha_{sid}")
        flag = "" if d is None else ("  ✓" if abs(d) < 15 else ("  !" if abs(d) < 30 else "  ✗"))
        d_str = f"{d:>+7.1f}%" if d is not None else "       N/A"
        print(f"\n  [ThExp]   {sid}: MTP={mtp_r.alpha*1e6:.2f}e-6 K⁻¹  "
              f"{result.label}={ref_r.alpha*1e6:.2f}e-6 K⁻¹  Δ={d_str}{flag}")
    print()


# ------------------------------------------------------------------ #
# LAMMPS input patch                                                    #
# ------------------------------------------------------------------ #

def _patch_pair_style(
    script: str,
    pair_style: str,
    pair_coeff: str,
) -> str:
    """Replace MTP pair_style/pair_coeff lines in a LAMMPS script with
    classical potential lines.

    Handles scripts that use:
      pair_style  mlip  ...   (MLIP-2/3)
      pair_style  mliap ...   (MLIAP interface)
      pair_coeff  * *   ...
    """
    lines_out = []
    skip_next_pair_coeff = False
    for line in script.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("pair_style"):
            lines_out.append(f"pair_style  {pair_style}")
            skip_next_pair_coeff = False
            continue
        if stripped.startswith("pair_coeff"):
            if not skip_next_pair_coeff:
                lines_out.append(f"pair_coeff  {pair_coeff}")
                skip_next_pair_coeff = True  # only replace first pair_coeff
            continue
        lines_out.append(line)
    return "\n".join(lines_out)


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
    """Run all enabled validation steps with a classical potential and
    compare against the MTP *mtp_result*.

    Parameters
    ----------
    config
        Pipeline config dict (same as passed to run_validation).
    pair_style
        LAMMPS pair_style string, e.g. ``"meam"``.
    pair_coeff
        LAMMPS pair_coeff string (everything after ``pair_coeff``),
        e.g. ``"* * /path/library.meam Pb /path/Pb.meam Pb"``.
    out_dir
        Directory for classical reference outputs (separate from MTP outputs).
    mtp_result
        The ValidationResult from the MTP run to compare against.
    label
        Short name for the reference potential used in printed output and
        the deviation keys, e.g. ``"Lee2003-MEAM"``.
    lammps_cmd, mpi_command, mpi_np, cutoff
        LAMMPS execution options. If None, taken from config.
    """
    val_cfg = config.get("validate", config)
    ref_dir = ensure_dir(out_dir)
    element = val_cfg.get("element", "Pb")

    # Resolve LAMMPS execution options from config if not explicitly supplied
    if lammps_cmd is None:
        lammps_cmd = val_cfg.get("lammps_command", "lmp_mpi")
    if mpi_command is None:
        mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    if mpi_np is None:
        mpi_np = val_cfg.get("mpi_np")
    if cutoff is None:
        cutoff = val_cfg.get("cutoff") or val_cfg.get("lammps_cutoff")

    print(f"\n=== Classical Reference: {label} ===")
    print(f"  pair_style: {pair_style}")
    print(f"  pair_coeff: {pair_coeff}")
    print(f"  out_dir:    {ref_dir}")

    def _resolve(s: dict) -> Path:
        data_path = Path(s["lammps_data"])
        if not data_path.is_absolute():
            root = Path(config.get("project_root", "."))
            data_path = (root / data_path).resolve()
        return data_path

    # We monkey-patch pair style into lammps.py write functions by
    # temporarily overriding the module-level PAIR_STYLE / PAIR_COEFF.
    # Instead, we pass pair_style/pair_coeff via the classical_pair kwargs
    # added to each write_* function. For now we use the env-var injection
    # approach: set them in validate.lammps before calling runners.
    import mlip_pipeline.validate.lammps as _lammps_mod
    _orig_pair_style = getattr(_lammps_mod, "_CLASSICAL_PAIR_STYLE", None)
    _orig_pair_coeff = getattr(_lammps_mod, "_CLASSICAL_PAIR_COEFF", None)
    _lammps_mod._CLASSICAL_PAIR_STYLE = pair_style
    _lammps_mod._CLASSICAL_PAIR_COEFF = pair_coeff

    eos_results:   list[EosResult]              = []
    elastic_results: list[ElasticResult]        = []
    melting_results: list[MeltingResult]        = []
    thexp_results: list[ThermalExpansionResult] = []
    vacancy_results: list[VacancyResult]        = []
    plot_paths: dict = {}

    try:
        # ── EOS ──────────────────────────────────────────────────────
        eos_cfg = val_cfg.get("eos", {})
        if eos_cfg.get("enabled", True):
            for s in eos_cfg.get("structures", []):
                sid = s["id"]
                data_path = _resolve(s)
                if not data_path.exists():
                    print(f"  [ref-eos]     WARNING: data not found for {sid}")
                    eos_results.append(EosResult(
                        structure_id=sid, volumes=[], energies=[],
                        fit_error=f"data file not found: {data_path}",
                    ))
                    continue
                res = run_eos(
                    sid, data_path,
                    model_path=Path(pair_coeff.split()[2]) if len(pair_coeff.split()) > 2 else Path("."),
                    validate_dir=ref_dir,
                    element=element, lammps_cmd=lammps_cmd,
                    mpi_command=mpi_command, mpi_np=mpi_np,
                    scale_min=eos_cfg.get("scale_min", 0.90),
                    scale_max=eos_cfg.get("scale_max", 1.10),
                    n_points=eos_cfg.get("n_points", 21),
                    cutoff=cutoff,
                )
                eos_results.append(res)
                p = plots.plot_eos(res, ref_dir)
                if p:
                    plot_paths[f"eos_{sid}"] = p

        # ── Elastic ───────────────────────────────────────────────────
        el_cfg = val_cfg.get("elastic", {})
        if el_cfg.get("enabled", True):
            for s in el_cfg.get("structures", []):
                sid = s["id"]
                data_path = _resolve(s)
                if not data_path.exists():
                    print(f"  [ref-elastic] WARNING: data not found for {sid}")
                    elastic_results.append(ElasticResult(
                        structure_id=sid,
                        error=f"data file not found: {data_path}",
                    ))
                    continue
                res = run_elastic(
                    sid, data_path,
                    model_path=Path("."),
                    validate_dir=ref_dir,
                    element=element, lammps_cmd=lammps_cmd,
                    mpi_command=mpi_command, mpi_np=mpi_np,
                    delta=el_cfg.get("delta", 0.01),
                    cutoff=cutoff,
                )
                elastic_results.append(res)

        # ── Melting ───────────────────────────────────────────────────
        melt_cfg = val_cfg.get("melting", {})
        if melt_cfg.get("enabled", False):
            for s in melt_cfg.get("structures", []):
                sid = s["id"]
                data_path = _resolve(s)
                if not data_path.exists():
                    print(f"  [ref-melting] WARNING: data not found for {sid}")
                    melting_results.append(MeltingResult(
                        structure_id=sid,
                        error=f"data file not found: {data_path}",
                    ))
                    continue
                res = run_melting(
                    sid, data_path,
                    model_path=Path("."),
                    validate_dir=ref_dir,
                    element=element, lammps_cmd=lammps_cmd,
                    mpi_command=mpi_command, mpi_np=mpi_np,
                    cutoff=cutoff,
                    T_start=melt_cfg.get("T_start", 500.0),
                    T_end=melt_cfg.get("T_end", 750.0),
                    T_step=melt_cfg.get("T_step", 25.0),
                    supercell_repeat=melt_cfg.get("supercell_repeat", 4),
                    n_equil=melt_cfg.get("n_equil", 5000),
                    n_prod=melt_cfg.get("n_prod", 20000),
                    dt=melt_cfg.get("dt", 0.002),
                )
                melting_results.append(res)

        # ── Thermal expansion ─────────────────────────────────────────
        thexp_cfg = val_cfg.get("thermal_expansion", {})
        if thexp_cfg.get("enabled", False):
            for s in thexp_cfg.get("structures", []):
                sid = s["id"]
                data_path = _resolve(s)
                if not data_path.exists():
                    print(f"  [ref-thexp]   WARNING: data not found for {sid}")
                    thexp_results.append(ThermalExpansionResult(
                        structure_id=sid,
                        error=f"data file not found: {data_path}",
                    ))
                    continue
                res = run_thermal_expansion(
                    sid, data_path,
                    model_path=Path("."),
                    validate_dir=ref_dir,
                    element=element, lammps_cmd=lammps_cmd,
                    mpi_command=mpi_command, mpi_np=mpi_np,
                    cutoff=cutoff,
                    temperatures=thexp_cfg.get("temperatures"),
                    T_ref=thexp_cfg.get("T_ref", 300.0),
                    n_equil=thexp_cfg.get("n_equil", 5000),
                    n_prod=thexp_cfg.get("n_prod", 10000),
                    dt=thexp_cfg.get("dt", 0.002),
                )
                thexp_results.append(res)

        # ── Vacancy ───────────────────────────────────────────────────
        vac_cfg = val_cfg.get("vacancy", {})
        if vac_cfg.get("enabled", False):
            for s in vac_cfg.get("structures", []):
                sid = s["id"]
                data_path = _resolve(s)
                if not data_path.exists():
                    print(f"  [ref-vacancy] WARNING: data not found for {sid}")
                    vacancy_results.append(VacancyResult(
                        structure_id=sid,
                        error=f"data file not found: {data_path}",
                    ))
                    continue
                res = run_vacancy(
                    sid, data_path,
                    model_path=Path("."),
                    validate_dir=ref_dir,
                    element=element, lammps_cmd=lammps_cmd,
                    mpi_command=mpi_command, mpi_np=mpi_np,
                    cutoff=cutoff,
                    supercell_repeat=vac_cfg.get("supercell_repeat", 3),
                )
                vacancy_results.append(res)

    finally:
        # Always restore original lammps module state
        if _orig_pair_style is None:
            if hasattr(_lammps_mod, "_CLASSICAL_PAIR_STYLE"):
                delattr(_lammps_mod, "_CLASSICAL_PAIR_STYLE")
        else:
            _lammps_mod._CLASSICAL_PAIR_STYLE = _orig_pair_style
        if _orig_pair_coeff is None:
            if hasattr(_lammps_mod, "_CLASSICAL_PAIR_COEFF"):
                delattr(_lammps_mod, "_CLASSICAL_PAIR_COEFF")
        else:
            _lammps_mod._CLASSICAL_PAIR_COEFF = _orig_pair_coeff

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
