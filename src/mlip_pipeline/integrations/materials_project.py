from __future__ import annotations

from pathlib import Path

from mlip_pipeline.utils.fs import ensure_dir


def download_structures_from_mp(config: dict, resolved_paths: dict) -> list[Path]:
    mp_cfg = config.get('materials_project', {})
    material_ids = mp_cfg.get('material_ids', [])
    if not material_ids:
        return []

    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise RuntimeError('mp-api is required for Materials Project downloads') from e

    out_dir = ensure_dir(resolved_paths['structures_root'] / mp_cfg.get('output_subdir', 'mp_raw'))
    api_key = mp_cfg.get('api_key')
    downloaded: list[Path] = []

    with MPRester(api_key) as mpr:
        for material_id in material_ids:
            structure = mpr.get_structure_by_material_id(material_id)
            if mp_cfg.get('conventional_cell', False):
                structure = structure.to_conventional()
            out_path = out_dir / f'{material_id}.cif'
            structure.to(filename=str(out_path))
            downloaded.append(out_path)

    return downloaded
