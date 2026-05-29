"""Orchestrator for physics validation.

Entry point:  run_validation(config, model_path, out_dir)

The config dict may either be:
  - a full pipeline config (Fe_loop.yaml style) with a ``validate`` section, or
  - a standalone validation config (Fe_validate.yaml style).

Both styles resolve to the same flat ``validate`` dict used internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.validate.models import ValidationResult, EosResult, ElasticResult
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate import plots


def _resolve_val_config(config: dict) -> dict:
    """Return the flat validate config dict regardless of input style."""
    return config.get("validate", config)


def run_validation(
    config: dict,
    model_path: Path,
    out_dir: Path,
) -> ValidationResult:
    """Run all enabled validation tasks and return a ValidationResult.

    Parameters
    ----------
    config:
        Full pipeline config *or* a standalone validate config dict.
        The ``validate`` section (or the dict itself) controls what runs.
    model_path:
        Path to the trained ``.almtp`` file.
    out_dir:
        Directory where all validation output is written.
        Sub-directories are created automatically.
    """
    val_cfg  = _resolve_val_config(config)
    val_dir  = ensure_dir(out_dir)
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ---- shared LAMMPS settings -----------------------------------------
    lammps_cmd  = val_cfg.get("lammps_command", "lmp_mpi")
    mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    mpi_np      = val_cfg.get("mpi_np")
    element     = val_cfg.get("element", "Fe")   # primary element symbol
    # Pair potential cutoff used for MPI safety cap (box must be > cutoff).
    # Read from config; fall back to None (cap disabled) if not set.
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

    # ---- EOS ---------------------------------------------------------------
    eos_cfg = val_cfg.get("eos", {})
    if eos_cfg.get("enabled", True):
        structures = eos_cfg.get("structures", [])
        if not structures:
            print("  [eos]     WARNING: no structures defined under validate.eos.structures")
        for s in structures:
            sid          = s["id"]
            data_path    = Path(s["lammps_data"])
            if not data_path.is_absolute():
                # Resolve relative paths against project_root if present
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

            # Plot this EOS curve
            p = plots.plot_eos(res, val_dir)
            if p:
                plot_paths[f"eos_{sid}"] = p
                print(f"  [eos]     {sid}: plot -> {p.name}")
    else:
        print("  [eos]     skipped (disabled in config)")

    # ---- Elastic constants ------------------------------------------------
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

    # Plot elastic constants summary
    if elastic_results:
        p = plots.plot_elastic_bar(elastic_results, val_dir)
        if p:
            plot_paths["elastic_constants"] = p
            print(f"  [elastic] bar chart -> {p.name}")

    # ---- Persist manifest -------------------------------------------------
    result = ValidationResult(
        model_path=model_path,
        validate_dir=val_dir,
        eos_results=eos_results,
        elastic_results=elastic_results,
        plot_paths=plot_paths,
    )
    manifest_path = result.save_manifest()
    print(f"\nValidation complete -- manifest -> {manifest_path}")
    return result
