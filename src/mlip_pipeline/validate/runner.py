"""Orchestrator for physics validation.

Entry point:  run_validation(config, model_path, out_dir)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.validate.models import ValidationResult, EosResult, ElasticResult
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate import plots
from mlip_pipeline.integrations.mp_reference import fetch_mp_reference, print_deviation_table


def _resolve_val_config(config: dict) -> dict:
    return config.get("validate", config)


def run_validation(
    config: dict,
    model_path: Path,
    out_dir: Path,
) -> ValidationResult:
    val_cfg  = _resolve_val_config(config)
    val_dir  = ensure_dir(out_dir)
    model_path = Path(model_path).resolve()

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
    if cutoff is not None:
        print(f"  cutoff:   {cutoff} Å  (used for MPI safety cap)")

    eos_results: list[EosResult] = []
    elastic_results: list[ElasticResult] = []
    plot_paths: dict = {}

    # ── EOS -----------------------------------------------------------------
    eos_cfg = val_cfg.get("eos", {})
    if eos_cfg.get("enabled", True):
        structures = eos_cfg.get("structures", [])
        if not structures:
            print("  [eos]     WARNING: no structures defined under validate.eos.structures")
        for s in structures:
            sid       = s["id"]
            data_path = Path(s["lammps_data"])
            if not data_path.is_absolute():
                root = Path(config.get("project_root", "."))
                data_path = (root / data_path).resolve()
            if not data_path.exists():
                print(f"  [eos]     WARNING: lammps_data not found for {sid}: {data_path}")
                eos_results.append(EosResult(
                    structure_id=sid, volumes=[], energies=[],
                    fit_error=f"data file not found: {data_path}",
                ))
                continue
            res = run_eos(
                sid, data_path, model_path, val_dir,
                element=element,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
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
    else:
        print("  [eos]     skipped (disabled in config)")

    # ── Elastic constants ---------------------------------------------------
    el_cfg = val_cfg.get("elastic", {})
    if el_cfg.get("enabled", True):
        structures = el_cfg.get("structures", [])
        if not structures:
            print("  [elastic] WARNING: no structures defined under validate.elastic.structures")
        for s in structures:
            sid       = s["id"]
            data_path = Path(s["lammps_data"])
            if not data_path.is_absolute():
                root = Path(config.get("project_root", "."))
                data_path = (root / data_path).resolve()
            if not data_path.exists():
                print(f"  [elastic] WARNING: lammps_data not found for {sid}: {data_path}")
                elastic_results.append(ElasticResult(
                    structure_id=sid,
                    error=f"data file not found: {data_path}",
                ))
                continue
            res = run_elastic(
                sid, data_path, model_path, val_dir,
                element=element,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                delta=el_cfg.get("delta", 0.01),
                cutoff=cutoff,
            )
            elastic_results.append(res)
    if elastic_results:
        p = plots.plot_elastic_bar(elastic_results, val_dir)
        if p:
            plot_paths["elastic_constants"] = p
            print(f"  [elastic] bar chart -> {p.name}")

    # ── MP reference comparison ---------------------------------------------
    ref_cfg = val_cfg.get("reference", {})
    mp_id   = ref_cfg.get("mp_id")
    ref     = None
    if mp_id:
        ref = fetch_mp_reference(
            mp_id,
            out_dir=val_dir,
            api_key_env=ref_cfg.get("api_key_env", "MP_API_KEY"),
        )

    if ref:
        # Collect MTP values from results
        mtp_vals: dict = {}
        for r in eos_results:
            if r.fit_ok and r.structure_id == "fcc":
                mtp_vals["V0"] = r.V0
                mtp_vals["E0"] = r.E0
                mtp_vals["B0"] = r.B0
                break
        for r in elastic_results:
            if r.structure_id == "fcc" and r.cij:
                mtp_vals["C11"] = r.cij.get("C11")
                mtp_vals["C12"] = r.cij.get("C12")
                mtp_vals["C44"] = r.cij.get("C44")
                mtp_vals["G0"]  = r.G_voigt
                break
        print_deviation_table(mtp_vals, ref)

    # ── Manifest ------------------------------------------------------------
    result = ValidationResult(
        model_path=model_path,
        validate_dir=val_dir,
        eos_results=eos_results,
        elastic_results=elastic_results,
        plot_paths=plot_paths,
    )
    manifest_path = result.save_manifest()
    print(f"Validation complete -- manifest -> {manifest_path}")
    return result
