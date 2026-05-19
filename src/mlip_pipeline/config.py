from __future__ import annotations
import copy, re
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Matches {{ some.dotted.key }} or {{ simple_key }}
_TEMPLATE_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to dotted-key form.

    {"mpi": {"prefix": "mpirun"}, "project_root": "/x"}
    →  {"mpi.prefix": "mpirun", "project_root": "/x", "mpi": {...}}

    Top-level scalar values are also kept as plain keys so
    {{ project_root }} and {{ mpi.prefix }} both work.
    """
    out = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out[full] = v                       # keep the dict itself
            out.update(_flatten(v, full))       # recurse for dotted access
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[full] = v
    return out


def _resolve_value(value, context: dict, local_context: dict | None = None):
    """Render {{ }} templates in a value.

    Resolution order:
      1. local_context  – sibling keys in the same block (e.g. mtp_level inside fit:)
      2. context        – flattened whole-config keys

    Raises KeyError with a helpful message on unknown keys.
    """
    if isinstance(value, str):
        def _sub(m):
            key = m.group(1)
            # local first, then global
            if local_context and key in local_context:
                return str(local_context[key])
            if key in context:
                return str(context[key])
            available = sorted(set(list(context) + list(local_context or {})))
            raise KeyError(
                f"Template var '{key}' not found. Available: {available}"
            )
        return _TEMPLATE_RE.sub(_sub, value)

    if isinstance(value, dict):
        # Build a local context from the scalar siblings in this block
        local = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
        return {k: _resolve_value(v, context, local) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_value(v, context, local_context) for v in value]

    return value


def render_config(config: dict) -> dict:
    """Render all {{ }} template references in a config dict.

    Supports:
      - Top-level scalars:   {{ project_root }}
      - Dotted nested paths: {{ mpi.prefix }}, {{ paths.datasets_root }}
      - Sibling keys:        {{ mtp_level }} inside the same block

    Rendering is done in two passes so that a ref that itself contains a
    template (e.g. paths.runs_root = "{{ project_root }}/runs") resolves
    correctly before being used by downstream refs.
    """
    cfg = copy.deepcopy(config)
    # Two passes handle chained refs (A → B → C)
    for _ in range(2):
        context = _flatten(cfg)
        cfg = _resolve_value(cfg, context)
    return cfg


def project_paths(config: dict) -> dict:
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    p = config["paths"]
    return {
        "project_root":    root,
        "data_root":       Path(p["data_root"]).expanduser().resolve(),
        "runs_root":       Path(p.get("runs_root", str(root / "runs"))).expanduser().resolve(),
        "structures_root": Path(p.get("structures_root", str(root / "structures"))).expanduser().resolve(),
        "datasets_root":   Path(p.get("datasets_root", str(root / "datasets"))).expanduser().resolve(),
    }


def gen_paths(config: dict, resolved: dict) -> dict:
    gen = config.get("generation")
    if gen is None:
        return resolved
    gen_tag = f"gen_{str(gen).zfill(2)}"
    gen_dir = resolved["runs_root"] / gen_tag
    return {
        **resolved,
        "gen_dir":     gen_dir,
        "fit_dir":     gen_dir / "fit",
        "explore_dir": gen_dir / "explore",
        "select_dir":  gen_dir / "select",
        "label_dir":   gen_dir / "label",
        "convert_dir": gen_dir / "convert",
    }


def build_gen_config(base_config: dict, generation: int) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["generation"] = str(generation).zfill(2)
    return render_config(cfg)


def load_and_render(path: str | Path) -> dict:
    return render_config(load_yaml(path))
