"""Orchestrator for physics validation.

Entry point:  run_validation(config, model_path, out_dir)

Design principle
----------------
Computation and plotting are kept strictly separate:
  1. All ``run_*`` calls collect result dataclasses.
  2. If a classical reference is configured, ``plot_comparison()`` produces
     side-by-side MTP vs reference PNG files.
  No standalone MTP-only plots are generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.models import (
    ValidationResult, EosResult, ElasticResult,
    MeltingResult, ThermalExpansionResult, VacancyResult, RdfResult,
)
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate.melting import run_melting
from mlip_pipeline.validate.thermal_expansion import run_thermal_expansion
from mlip_pipeline.validate.vacancy import run_vacancy
from mlip_pipeline.validate.rdf import run_rdf
from mlip_pipeline.validate import plots
from mlip_pipeline.validate._cli import banner, step, ok, warn
from mlip_pipeline.validate.classical_reference import run_classical_reference


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_val_config(config: dict) -> dict:
    return config.get("validate", config)


def _resolve_structures(step_cfg: dict, val_cfg: dict) -> list[dict]:
    """Return structure list for a step.

    Structures can be defined:
      1. Inside the step block:  eos.structures / elastic.structures / ...
      2. At the top level of the validate config under ``structures``
         as a mapping {id: {data_file: ...}} or list [{id:..., data_file:...}].

    The step-level definition takes precedence.
    """
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


def _resolve_data_path(s: dict, config: dict) -> Path:
    key = "lammps_data" if "lammps_data" in s else "data_file"
    data_path = Path(s[key])
    if not data_path.is_absolute():
        root = Path(config.get("project_root", "."))
        data_path = (root / data_path).resolve()
    return data_path


def _step_enabled(
    name: str,
    cfg_enabled: bool,
    only_steps: Optional[list[str]],
    skip_steps: list[str],
) -> bool:
    if only_steps is not None:
        return name in only_steps
    if name in skip_steps:
        return False
    return cfg_enabled


# ── Public runner ─────────────────────────────────────────────────────────────

def run_validation(
    config: dict,
    model_path: Path,
    out_dir: Path,
    only_steps: Optional[list[str]] = None,
    skip_steps: Optional[list[str]] = None,
) -> ValidationResult:
    val_cfg    = _resolve_val_config(config)
    val_dir    = ensure_dir(out_dir)
    model_path = Path(model_path).resolve()
    skip_steps = list(skip_steps or [])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    lammps_cmd  = (
        val_cfg.get("lammps_cmd")
        or val_cfg.get("lammps_command")
        or "lmp_mpi"
    )
    mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    mpi_np      = val_cfg.get("mpi_np")
    element     = val_cfg.get("element", "Fe")
    cutoff: Optional[float] = val_cfg.get("cutoff") or val_cfg.get("lammps_cutoff")

    banner("Validation")
    step("setup", f"model    : {model_path}")
    step("setup", f"out_dir  : {val_dir}")
    step("setup", f"lammps   : {lammps_cmd}")
    step("setup", f"element  : {element}")
    if only_steps is not None:
        step("setup", f"only     : {only_steps}")
    if skip_steps:
        step("setup", f"skip     : {skip_steps}")

    eos_results:     list[EosResult]              = []
    elastic_results: list[ElasticResult]          = []
    melting_results: list[MeltingResult]          = []
    thexp_results:   list[ThermalExpansionResult] = []
    vacancy_results: list[VacancyResult]          = []
    rdf_results:     list[RdfResult]              = []

    # ── Computation ──────────────────────────────────────────────────────────
    banner("Computation")

    # EOS
    eos_cfg = val_cfg.get("eos", {})
    if _step_enabled("eos", eos_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(eos_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("eos", f"{sid}: data not found \u2014 {data_path}")
                eos_results.append(EosResult(
                    structure_id=sid, volumes=[], energies=[],
                    fit_error=f"data file not found: {data_path}",
                ))
                continue
            res = run_eos(
                sid, data_path, model_path, val_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                scale_min=eos_cfg.get("scale_min", 0.85),
                scale_max=eos_cfg.get("scale_max", 1.15),
                n_points=eos_cfg.get("n_points", 21),
                cutoff=cutoff,
            )
            eos_results.append(res)
            if res.fit_ok:
                ok("eos", f"{sid}: B\u2080={res.B0:.1f} GPa  V\u2080={res.V0:.3f} \u00c5\u00b3")
            else:
                warn("eos", f"{sid}: fit failed \u2014 {res.fit_error}")

    # Elastic constants
    el_cfg = val_cfg.get("elastic", {})
    if _step_enabled("elastic", el_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(el_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("elastic", f"{sid}: data not found \u2014 {data_path}")
                elastic_results.append(ElasticResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_elastic(
                sid, data_path, model_path, val_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                delta=el_cfg.get("delta", 0.01),
                cutoff=cutoff,
            )
            elastic_results.append(res)
            if res.compute_ok:
                ok("elastic", f"{sid}: B={res.B_voigt:.1f}  G={res.G_voigt:.1f} GPa")
            else:
                warn("elastic", f"{sid}: failed \u2014 {res.error}")

    # Melting temperature
    melt_cfg = val_cfg.get("melting", {})
    if _step_enabled("melting", melt_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(melt_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("melting", f"{sid}: data not found \u2014 {data_path}")
                melting_results.append(MeltingResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_melting(
                sid, data_path, model_path, val_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                T_start=melt_cfg.get("T_start", 400.0),
                T_end=melt_cfg.get("T_end", 900.0),
                T_step=melt_cfg.get("T_step", 50.0),
                supercell_repeat=melt_cfg.get("supercell_repeat", 4),
                n_equil=melt_cfg.get("n_equil", 5000),
                n_prod=melt_cfg.get("n_prod", 20000),
                dt=melt_cfg.get("dt", 0.002),
            )
            melting_results.append(res)
            if res.compute_ok:
                ok("melting", f"{sid}: T_melt = {res.T_melt:.0f} K")
            else:
                warn("melting", f"{sid}: failed \u2014 {res.error}")

    # Thermal expansion
    thexp_cfg = val_cfg.get("thexp", val_cfg.get("thermal_expansion", {}))
    if _step_enabled("thexp", thexp_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(thexp_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("thexp", f"{sid}: data not found \u2014 {data_path}")
                thexp_results.append(ThermalExpansionResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_thermal_expansion(
                sid, data_path, model_path, val_dir,
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
            if res.compute_ok:
                ok("thexp", f"{sid}: \u03b1 = {res.alpha*1e6:.2f}\u00d710\u207b\u2076 K\u207b\u00b9")
            else:
                warn("thexp", f"{sid}: failed \u2014 {res.error}")

    # Vacancy formation energy
    vac_cfg = val_cfg.get("vacancy", {})
    if _step_enabled("vacancy", vac_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(vac_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("vacancy", f"{sid}: data not found \u2014 {data_path}")
                vacancy_results.append(VacancyResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_vacancy(
                sid, data_path, model_path, val_dir,
                element=element, lammps_cmd=lammps_cmd,
                mpi_command=mpi_command, mpi_np=mpi_np,
                cutoff=cutoff,
                supercell_repeat=vac_cfg.get("supercell_repeat", 3),
            )
            vacancy_results.append(res)
            if res.compute_ok:
                ok("vacancy", f"{sid}: E_vac = {res.E_vac:.4f} eV")
            else:
                warn("vacancy", f"{sid}: failed \u2014 {res.error}")

    # RDF
    rdf_cfg = val_cfg.get("rdf", {})
    if _step_enabled("rdf", rdf_cfg.get("enabled", True), only_steps, skip_steps):
        for s in _resolve_structures(rdf_cfg, val_cfg):
            sid = s.get("id", s.get("structure_id", "unknown"))
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                warn("rdf", f"{sid}: data not found \u2014 {data_path}")
                rdf_results.append(RdfResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_rdf(
                sid, data_path, model_path, val_dir,
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
            )
            rdf_results.append(res)
            if res.compute_ok:
                ok("rdf", f"{sid}: first peak at {res.first_peak_r:.3f} \u00c5")
            else:
                warn("rdf", f"{sid}: failed \u2014 {res.error}")

    # ── Manifest (computation results only, no standalone plots) ─────────────
    result = ValidationResult(
        model_path=model_path,
        validate_dir=val_dir,
        eos_results=eos_results,
        elastic_results=elastic_results,
        melting_results=melting_results,
        thermal_expansion_results=thexp_results,
        vacancy_results=vacancy_results,
        rdf_results=rdf_results,
        plot_paths={},
    )
    result.save_manifest()

    # ── Classical reference: compute + comparison plots ───────────────────────
    plot_paths: dict[str, Path] = {}
    cl_ref_cfg = val_cfg.get("classical_reference", {})
    if cl_ref_cfg.get("enabled", False):
        pair_style = cl_ref_cfg.get("pair_style")
        pair_coeff = cl_ref_cfg.get("pair_coeff")
        ref_label  = cl_ref_cfg.get("label", "classical")
        ref_dir    = val_dir / "classical_ref"
        if pair_style and pair_coeff:
            ref_result = run_classical_reference(
                config=config,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                out_dir=ref_dir,
                mtp_result=result,
                label=ref_label,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                cutoff=cutoff,
            )
            banner(f"Comparison plots (MTP vs {ref_label})")
            plot_paths = plots.plot_comparison(result, ref_result, val_dir)
            for k, p in plot_paths.items():
                ok("compare", f"{k} \u2192 {p.name}")
        else:
            warn("compare", "classical_reference enabled but pair_style/pair_coeff missing")

    banner("Done")
    step("output", f"manifest \u2192 {val_dir / 'validate_manifest.json'}")
    if plot_paths:
        step("output", f"plots    \u2192 {len(plot_paths)} comparison file(s) in {val_dir}")

    return result
