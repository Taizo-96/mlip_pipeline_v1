from __future__ import annotations

from pathlib import Path

from mlip_pipeline.utils.fs import ensure_dir


def export_mp_structures_to_lammps(config: dict, resolved_paths: dict) -> list[Path]:
    mp_cfg = config.get('materials_project', {})
    if not mp_cfg.get('export_lammps_data', False):
        return []

    try:
        from pymatgen.core import Structure
        from pymatgen.io.lammps.data import LammpsData
    except ImportError as e:
        raise RuntimeError('pymatgen is required for LAMMPS export') from e

    raw_dir = ensure_dir(resolved_paths['structures_root'] / mp_cfg.get('output_subdir', 'mp_raw'))
    out_dir = ensure_dir(resolved_paths['structures_root'] / mp_cfg.get('lammps_output_subdir', 'lammps_data'))
    atom_style = mp_cfg.get('lammps_atom_style', 'atomic')
    ff_elements = mp_cfg.get('ff_elements')
    if not ff_elements:
        raise ValueError('materials_project.ff_elements must be set for LAMMPS export')

    exported: list[Path] = []
    for cif_path in sorted(raw_dir.glob('*.cif')):
        structure = Structure.from_file(cif_path)
        lammps_data = LammpsData.from_structure(structure=structure, atom_style=atom_style, ff_elements=ff_elements)
        out_path = out_dir / f'{cif_path.stem}.data'
        lammps_data.write_file(str(out_path))
        exported.append(out_path)

    return exported
