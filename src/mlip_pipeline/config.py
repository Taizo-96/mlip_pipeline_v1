from __future__ import annotations
import copy, os, re
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_TEMPLATE_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

def _render_value(value, context: dict):
    if isinstance(value, str):
        def _sub(m):
            key = m.group(1)
            if key not in context:
                raise KeyError(f"Template var '{key}' not found. Available: {list(context)}")
            return str(context[key])
        return _TEMPLATE_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _render_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, context) for v in value]
    return value

def render_config(config: dict) -> dict:
    context = {k: v for k, v in config.items() if isinstance(v, (str, int, float))}
    return _render_value(copy.deepcopy(config), context)

def project_paths(config: dict) -> dict:
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    p    = config["paths"]
    return {
        "project_root":    root,
        "data_root":       Path(p["data_root"]).expanduser().resolve(),
        "runs_root":       (root / p.get("runs_root", "runs")).resolve(),
        "structures_root": (root / p.get("structures_root", "structures")).resolve(),
        "datasets_root":   (root / p.get("datasets_root", "datasets")).resolve(),
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
    }

def build_gen_config(base_config: dict, generation: int) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["generation"] = str(generation).zfill(2)
    return render_config(cfg)

def load_and_render(path: str | Path) -> dict:
    return render_config(load_yaml(path))