from __future__ import annotations
import json, shutil
from pathlib import Path

def safe_mkdir(
    path: str | Path,
    *,
    label: str = "",
    force: bool = False,
) -> Path:
    """Create directory. Prompts the user if it already contains data."""
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        step_info = f"  Step  : {label}\n" if label else ""
        print(
            f"\n⚠️  Output directory already exists and is not empty:\n"
            f"  Path  : {p}\n"
            f"{step_info}"
            f"  Items : {len(list(p.iterdir()))} file(s)/dir(s)\n"
        )
        if not force:
            answer = input(
                "Proceed anyway? Existing data may be overwritten. [y/N]: "
            ).strip().lower()
            if answer != "y":
                raise SystemExit("Aborted by user — no data was overwritten.")
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) silently. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    """Copy src → dst if it exists. Returns True if copied."""
    src = Path(src)
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False

def write_json(path: str | Path, data: dict, *, indent: int = 2) -> Path:
    """Serialise data to path as JSON, creating parents. Returns Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    return p