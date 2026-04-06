from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def project_paths(config: dict) -> dict:
    project_root = Path(config.get('project_root', '.')).expanduser().resolve()
    data_root = Path(config['paths']['data_root']).expanduser().resolve()
    runs_root = (project_root / config['paths'].get('runs_root', 'runs')).resolve()
    structures_root = (project_root / config['paths'].get('structures_root', 'structures')).resolve()
    datasets_root = (project_root / config['paths'].get('datasets_root', 'datasets')).resolve()
    return {
        'project_root': project_root,
        'data_root': data_root,
        'runs_root': runs_root,
        'structures_root': structures_root,
        'datasets_root': datasets_root,
    }
