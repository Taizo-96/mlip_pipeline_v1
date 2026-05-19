from __future__ import annotations

import re
from pathlib import Path
import yaml


# ---------------------------------------------------------------------------
# Simple {{ key }} / {{ section.key }} template resolver
# ---------------------------------------------------------------------------

def _resolve_templates(config: dict) -> dict:
    """
    Expand {{ token }} placeholders in all string values.
    Supports:
      {{ project_root }}           -> top-level key
      {{ paths.runs_root }}        -> nested key (already resolved)
      {{ bins.mlp }}               -> nested key
    Two passes handle chains like {{ project_root }}/runs where
    project_root itself may contain a resolved value.
    """
    # Build a flat lookup of scalar values
    def _flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flat(v, key))
            elif isinstance(v, (str, int, float)):
                out[key] = str(v)
                # also register under short name for top-level keys
                if not prefix:
                    out[k] = str(v)
        return out

    def _expand(value: str, lookup: dict) -> str:
        def _sub(m):
            token = m.group(1).strip()
            return lookup.get(token, m.group(0))
        return re.sub(r'\{\{\s*([\w\.]+)\s*\}\}', _sub, value)

    def _walk(obj, lookup: dict):
        if isinstance(obj, dict):
            return {k: _walk(v, lookup) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v, lookup) for v in obj]
        if isinstance(obj, str):
            return _expand(obj, lookup)
        return obj

    # Two passes so that tokens that reference other tokens resolve correctly
    for _ in range(2):
        lookup = _flat(config)
        config = _walk(config, lookup)
    return config


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_templates(raw)


def project_paths(config: dict) -> dict:
    project_root = Path(config.get('project_root', '.')).expanduser().resolve()
    paths = config.get('paths', {})

    def _p(val: str, default: str) -> Path:
        """Resolve a path value: absolute stays absolute, relative is under project_root."""
        p = Path(val if val else default).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (project_root / p).resolve()

    return {
        'project_root': project_root,
        'data_root':      _p(paths.get('data_root', 'data'), 'data'),
        'runs_root':      _p(paths.get('runs_root', 'runs'), 'runs'),
        'structures_root':_p(paths.get('structures_root', 'structures'), 'structures'),
        'datasets_root':  _p(paths.get('datasets_root', 'datasets'), 'datasets'),
    }
