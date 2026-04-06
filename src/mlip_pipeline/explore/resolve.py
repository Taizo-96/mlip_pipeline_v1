from __future__ import annotations

from pathlib import Path


def infer_generated_structure_data(config: dict, resolved_paths: dict) -> Path | None:
    mp_cfg = config.get('materials_project', {})
    if not mp_cfg.get('export_lammps_data', False):
        return None
    material_ids = mp_cfg.get('material_ids', [])
    if len(material_ids) != 1:
        return None
    lammps_subdir = mp_cfg.get('lammps_output_subdir', 'lammps_data')
    structures_root = resolved_paths['structures_root']
    material_id = material_ids[0]
    return Path(structures_root) / lammps_subdir / f'{material_id}.data'


def resolve_structure_data_path(config: dict, resolved_paths: dict) -> Path:
    explore_cfg = config.get('explore', {})
    manual = explore_cfg.get('structure_data')
    if manual:
        return Path(manual).expanduser().resolve()
    inferred = infer_generated_structure_data(config, resolved_paths)
    if inferred is None:
        raise ValueError('explore.structure_data is not set and no unique generated LAMMPS data file could be inferred from materials_project settings')
    return inferred.resolve()
