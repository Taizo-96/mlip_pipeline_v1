"""Orchestrator for physics validation.

Entry point:  run_validation(config, model_path, out_dir)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.validate.models import (
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
from mlip_pipeline.integrations.mp_reference import fetch_mp_reference, print_deviation_table


def _resolve_val_config(config: dict) -> dict:
    return config.get("validate", config)


def _resolve_data_path(s: dict, config: dict) -> Path:
    data_path = Path(s["lammps_data"])
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
    """Return True when a validation step should run.

    Priority order (highest first):
      1. --only  → run *only* the listed steps (ignores config enabled flag)
      2. --skip  → suppress listed steps (ignores config enabled flag)
      3. config  → respect the ``enabled`` key in the step's config block
    """
    if only_steps is not None:
        return name in only_steps
    if name in skip_steps:
        return False
    return cfg_enabled


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

    lammps_cmd  = val_cfg.get("lammps_command", "lmp_mpi")
    mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    mpi_np      = val_cfg.get("mpi_np")
    element     = val_cfg.get("element", "Fe")
    cutoff: Optional[float] = val_cfg.get("cutoff") or val_cfg.get("lammps_cutoff")

    print(f"\n=== Validation ===")
    print(f"  model:    {model_path}")
    print(f"  out_dir:  {val_dir}")
    print(f"  lammps:   {lammps_cmd}")
    print(f"  element:  {element}")
    if only_steps is not None:
        print(f"  only:     {only_steps}")
    if skip_steps:
        print(f"  skip:     {skip_steps}")

    eos_results:    list[EosResult]              = []
    elastic_results: list[ElasticResult]         = []
    melting_results: list[MeltingResult]         = []
    thexp_results:  list[ThermalExpansionResult] = []
    vacancy_results: list[VacancyResult]         = []
    rdf_results:    list[RdfResult]              = []
    plot_paths: dict = {}

    # ── EOS ──────────────────────────────────────────────────────────────
    eos_cfg = val_cfg.get("eos", {})
    if _step_enabled("eos", eos_cfg.get("enabled", True), only_steps, skip_steps):
        for s in eos_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [eos]     WARNING: data not found for {sid}: {data_path}")
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
            p = plots.plot_eos(res, val_dir)
            if p:
                plot_paths[f"eos_{sid}"] = p
                print(f"  [eos]     {sid}: plot -> {p.name}")

    # ── Elastic constants ─────────────────────────────────────────────────
    el_cfg = val_cfg.get("elastic", {})
    if _step_enabled("elastic", el_cfg.get("enabled", True), only_steps, skip_steps):
        for s in el_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [elastic] WARNING: data not found for {sid}: {data_path}")
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
        if elastic_results:
            p = plots.plot_elastic_bar(elastic_results, val_dir)
            if p:
                plot_paths["elastic_constants"] = p
                print(f"  [elastic] bar chart -> {p.name}")

    # ── Melting temperature ───────────────────────────────────────────────
    melt_cfg = val_cfg.get("melting", {})
    if _step_enabled("melting", melt_cfg.get("enabled", False), only_steps, skip_steps):
        for s in melt_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [melting] WARNING: data not found for {sid}: {data_path}")
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
        if melting_results:
            p = plots.plot_melting(melting_results, val_dir)
            if p:
                plot_paths["melting_temperature"] = p
                print(f"  [melting] bracket plot -> {p.name}")

    # ── Thermal expansion ─────────────────────────────────────────────────
    thexp_cfg = val_cfg.get("thermal_expansion", {})
    if _step_enabled("thermal_expansion", thexp_cfg.get("enabled", False), only_steps, skip_steps):
        for s in thexp_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [thexp]   WARNING: data not found for {sid}: {data_path}")
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
        if thexp_results:
            p = plots.plot_thermal_expansion(thexp_results, val_dir)
            if p:
                plot_paths["thermal_expansion"] = p
                print(f"  [thexp]   plot -> {p.name}")

    # ── Vacancy formation energy ──────────────────────────────────────────
    vac_cfg = val_cfg.get("vacancy", {})
    if _step_enabled("vacancy", vac_cfg.get("enabled", False), only_steps, skip_steps):
        for s in vac_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [vacancy] WARNING: data not found for {sid}: {data_path}")
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
        if vacancy_results:
            p = plots.plot_vacancy(vacancy_results, val_dir)
            if p:
                plot_paths["vacancy_formation"] = p
                print(f"  [vacancy] bar chart -> {p.name}")

    # ── RDF ───────────────────────────────────────────────────────────────
    rdf_cfg = val_cfg.get("rdf", {})
    if _step_enabled("rdf", rdf_cfg.get("enabled", False), only_steps, skip_steps):
        for s in rdf_cfg.get("structures", []):
            sid = s["id"]
            data_path = _resolve_data_path(s, config)
            if not data_path.exists():
                print(f"  [rdf]     WARNING: data not found for {sid}: {data_path}")
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
        if rdf_results:
            p = plots.plot_rdf(rdf_results, val_dir)
            if p:
                plot_paths["rdf"] = p
                print(f"  [rdf]     plot -> {p.name}")

    # ── MP reference comparison ───────────────────────────────────────────
    ref_cfg = val_cfg.get("reference", {})
    mp_id   = ref_cfg.get("mp_id")
    if mp_id:
        ref = fetch_mp_reference(
            mp_id, out_dir=val_dir,
            api_key_env=ref_cfg.get("api_key_env", "MP_API_KEY"),
        )
        if ref:
            mtp_vals: dict = {}
            for r in eos_results:
                if r.fit_ok and r.structure_id == "fcc":
                    mtp_vals.update({"V0": r.V0, "E0": r.E0, "B0": r.B0})
                    break
            for r in elastic_results:
                if r.structure_id == "fcc" and r.C:
                    mtp_vals.update({
                        "C11": r.C.get("C11"), "C12": r.C.get("C12"),
                        "C44": r.C.get("C44"), "G0": r.G_voigt,
                        "B0": mtp_vals.get("B0") or r.B_voigt,
                    })
                    break
            for r in melting_results:
                if r.compute_ok and r.structure_id == "fcc":
                    mtp_vals["T_melt"] = r.T_melt
                    break
            for r in vacancy_results:
                if r.compute_ok:
                    mtp_vals["E_vac"] = r.E_vac
                    break
            for r in thexp_results:
                if r.compute_ok:
                    mtp_vals["alpha"] = r.alpha
                    break
            print_deviation_table(mtp_vals, ref)

    # ── Manifest ──────────────────────────────────────────────────────────
    result = ValidationResult(
        model_path=model_path,
        validate_dir=val_dir,
        eos_results=eos_results,
        elastic_results=elastic_results,
        melting_results=melting_results,
        thermal_expansion_results=thexp_results,
        vacancy_results=vacancy_results,
        rdf_results=rdf_results,
        plot_paths=plot_paths,
    )
    result.save_manifest()
    return result
