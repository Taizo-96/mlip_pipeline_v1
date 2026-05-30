from __future__ import annotations

from pathlib import Path


def infer_generated_structure_data(config: dict, resolved_paths: dict) -> Path | None:
    """Infer the LAMMPS data file path from the materials_project config block.

    Returns a path only when there is exactly one material_id and
    export_lammps_data is True.  For the multi-structure case use
    resolve_all_structure_data_paths instead.
    """
    mp_cfg = config.get("materials_project", {})
    if not mp_cfg.get("export_lammps_data", False):
        return None
    material_ids = mp_cfg.get("material_ids", [])
    if len(material_ids) != 1:
        return None
    lammps_subdir   = mp_cfg.get("lammps_output_subdir", "lammps_data")
    structures_root = resolved_paths["structures_root"]
    material_id     = material_ids[0]
    return Path(structures_root) / lammps_subdir / f"{material_id}.data"


def resolve_structure_data_path(config: dict, resolved_paths: dict) -> Path:
    """Return a single LAMMPS structure data Path (does not check existence).

    Kept for backwards compatibility with code that only needs one path.
    For multi-structure explore use resolve_all_structure_data_paths.
    """
    explore_cfg = config.get("explore", {})
    manual = explore_cfg.get("structure_data")
    if manual:
        return Path(manual).expanduser().resolve()
    inferred = infer_generated_structure_data(config, resolved_paths)
    if inferred is None:
        raise FileNotFoundError(
            "explore.structure_data is not set and no unique LAMMPS data file "
            "could be inferred from materials_project settings "
            "(need exactly one material_id with export_lammps_data: true)."
        )
    return inferred.resolve()


def resolve_all_structure_data_paths(config: dict, resolved_paths: dict) -> list[Path]:
    """Return ALL LAMMPS structure data paths to use for exploration.

    Resolution order (first that yields results wins):
    1. explore.structure_data  — explicit single path (backwards-compat)
    2. explore.structure_data_list — explicit list of paths
    3. materials_project.material_ids — auto-infer one path per id when
       export_lammps_data is True (supports any number of ids)

    Raises FileNotFoundError if no paths could be resolved.
    """
    explore_cfg = config.get("explore", {})

    # 1. Single explicit override
    manual = explore_cfg.get("structure_data")
    if manual:
        return [Path(manual).expanduser().resolve()]

    # 2. Explicit list override
    manual_list = explore_cfg.get("structure_data_list", [])
    if manual_list:
        return [Path(p).expanduser().resolve() for p in manual_list]

    # 3. Auto-infer from every material_id
    mp_cfg = config.get("materials_project", {})
    if mp_cfg.get("export_lammps_data", False):
        material_ids = mp_cfg.get("material_ids", [])
        if material_ids:
            lammps_subdir   = mp_cfg.get("lammps_output_subdir", "lammps_data")
            structures_root = resolved_paths["structures_root"]
            return [
                (Path(structures_root) / lammps_subdir / f"{mid}.data").resolve()
                for mid in material_ids
            ]

    raise FileNotFoundError(
        "explore.structure_data / explore.structure_data_list is not set and "
        "no LAMMPS data files could be inferred from materials_project settings "
        "(need at least one material_id with export_lammps_data: true)."
    )
