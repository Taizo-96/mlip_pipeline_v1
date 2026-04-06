from __future__ import annotations

from pathlib import Path
import json


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_if_exists(src: str | Path, dst: str | Path) -> Path:
    from shutil import copy2

    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(f"Source file not found: {src_p}")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_p, dst_p)
    return dst_p


def write_json(path: str | Path, data: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p