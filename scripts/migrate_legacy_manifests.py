#!/usr/bin/env python3
"""Migrate legacy gen 0-15 manifests to the current pipeline schema.

What this script does
---------------------
For each generation directory under <runs_root> that is missing a
``fit_manifest.json`` it:

1. Reads the old ``metadata.json`` (written by the pre-pipeline fit step)
   and writes a proper ``fit_manifest.json`` that FitResult.load_manifest()
   can consume.  ``n_train_cfgs`` is obtained by counting BEGIN_CFG blocks
   in the referenced ``train_cfg`` file.

2. Reads the old-style selection manifest (which used varied field names
   across eras) and rewrites it so that SelectionResult.load_manifest()
   can consume it.  The original file is backed up as
   ``selection_manifest.json.bak`` before overwriting.

The script is **idempotent** — if ``fit_manifest.json`` already exists the
generation is skipped entirely.

Usage
-----
    python scripts/migrate_legacy_manifests.py [--runs-root /path/to/runs] [--dry-run]

Defaults
--------
    --runs-root   ./runs   (relative to the repo root)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _count_begin_cfg(cfg_path: Path) -> int:
    """Count BEGIN_CFG blocks in a .cfg file."""
    if not cfg_path or not cfg_path.exists():
        return 0
    count = 0
    with cfg_path.open("r", errors="replace") as fh:
        for line in fh:
            if line.strip() == "BEGIN_CFG":
                count += 1
    return count


def _write_json(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY-RUN] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  wrote {path}")


def _find_gen_dirs(runs_root: Path) -> list[Path]:
    """Return sorted list of gen_NN directories that look like legacy gens."""
    pattern = re.compile(r"^gen_\d+$")
    dirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )
    return dirs


# ---------------------------------------------------------------------------
# Era detection helpers
# ---------------------------------------------------------------------------

def _find_metadata(gen_dir: Path) -> Path | None:
    """Locate metadata.json for a legacy gen.

    Legacy gens can be in two layouts:
      - new layout : <runs_root>/gen_NN/fit/metadata.json
      - old layout : <runs_root>/Pb_gen_NN_fit/  (sibling of gen_NN)
    We prefer the new layout; fall back to the old sibling.
    """
    # New layout (gen 08+)
    new = gen_dir / "fit" / "metadata.json"
    if new.exists():
        return new

    # Old layout (gen 00-07): sibling directory named Pb_gen_NN_fit
    gen_num = gen_dir.name.split("_")[1]  # e.g. "00"
    sibling = gen_dir.parent / f"Pb_gen_{gen_num}_fit" / "metadata.json"
    if sibling.exists():
        return sibling

    return None


def _find_model_path(meta: dict, gen_dir: Path) -> Path | None:
    """Resolve the trained model path from metadata."""
    # Prefer explicit result_model field
    if meta.get("result_model"):
        p = Path(meta["result_model"])
        if p.exists():
            return p

    # Fall back: look for any *.almtp in run_dir
    run_dir = Path(meta["run_dir"]) if meta.get("run_dir") else None
    if run_dir and run_dir.exists():
        candidates = list(run_dir.glob("*.almtp"))
        if candidates:
            return candidates[0]

    # New-layout fallback: gen_NN/fit/*.almtp
    fit_dir = gen_dir / "fit"
    if fit_dir.exists():
        candidates = list(fit_dir.glob("*.almtp"))
        if candidates:
            return candidates[0]

    return None


def _find_log_path(meta: dict, gen_dir: Path) -> Path | None:
    run_dir = Path(meta["run_dir"]) if meta.get("run_dir") else None
    if run_dir and run_dir.exists():
        for name in ("train.log", "fit.log"):
            p = run_dir / name
            if p.exists():
                return p
    fit_dir = gen_dir / "fit"
    if fit_dir.exists():
        for name in ("train.log", "fit.log"):
            p = fit_dir / name
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# fit_manifest migration
# ---------------------------------------------------------------------------

def _migrate_fit_manifest(gen_dir: Path, dry_run: bool) -> bool:
    """Write fit_manifest.json from metadata.json.  Returns True on success."""
    target = gen_dir / "fit" / "fit_manifest.json"
    if target.exists():
        print(f"  fit_manifest.json already exists — skipping fit migration")
        return True

    meta_path = _find_metadata(gen_dir)
    if meta_path is None:
        print(f"  WARNING: no metadata.json found — cannot migrate fit manifest")
        return False

    meta = json.loads(meta_path.read_text())

    model_path = _find_model_path(meta, gen_dir)
    if model_path is None:
        print(f"  WARNING: could not locate trained model — fit manifest will have null model_path")

    log_path = _find_log_path(meta, gen_dir)

    train_cfg_str = meta.get("train_cfg")
    train_cfg = Path(train_cfg_str) if train_cfg_str else None
    n_train_cfgs = _count_begin_cfg(train_cfg)
    if n_train_cfgs == 0:
        print(f"  WARNING: could not count configs in {train_cfg} — n_train_cfgs will be 0")

    # run_dir in the new schema is always gen_NN/fit/
    new_run_dir = gen_dir / "fit"

    manifest = {
        "step":          "fit",
        "run_dir":       str(new_run_dir),
        "model_path":    str(model_path) if model_path else None,
        "log_path":      str(log_path) if log_path else None,
        "train_cfg":     str(train_cfg) if train_cfg else None,
        "n_train_cfgs":  n_train_cfgs,
        "init_template": meta.get("init_template"),
        "mlp_command":   meta.get("mlp_command", "mlp"),
        "mpi_prefix":    meta.get("mpi_prefix"),
        "command":       meta.get("command", []),
        # Preserve original fit timestamp if available
        "completed_at":  meta.get("timestamp", _now()),
        "_migrated_from": str(meta_path),
    }

    _write_json(target, manifest, dry_run)
    return True


# ---------------------------------------------------------------------------
# selection_manifest migration
# ---------------------------------------------------------------------------

# All known key variants across eras -> canonical key
_SELECT_KEY_MAP = {
    # old free-form keys
    "selected_cfg_path":    "selected_cfg_path",  # kept for reference, not in schema
    "training_cfg":         "training_cfg",        # kept for reference
    "merged_candidates_path": "merged_candidates_path",
    # canonical keys (already present in new schema)
    "step":                 "step",
    "converged":            "converged",
    "selected_count":       "selected_count",
    "model_path":           "model_path",
    "candidate_sources":    "candidate_sources",
    "selected_block_files": "selected_block_files",
    "completed_at":         "completed_at",
    "mpi_np":               "mpi_np",
}


def _migrate_selection_manifest(gen_dir: Path, dry_run: bool) -> bool:
    """Normalise selection_manifest.json to the current schema."""
    target = gen_dir / "select" / "selection_manifest.json"

    # Also handle old path variant (gen 00-07 used Pb_gen_NN_select)
    gen_num = gen_dir.name.split("_")[1]
    old_sibling_manifest = (
        gen_dir.parent / f"Pb_gen_{gen_num}_select" / "selection_manifest.json"
    )

    if not target.exists() and old_sibling_manifest.exists():
        # Copy it into the new location first
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_sibling_manifest, target)
            print(f"  copied {old_sibling_manifest} -> {target}")
        else:
            print(f"  [DRY-RUN] would copy {old_sibling_manifest} -> {target}")

    if not target.exists():
        print(f"  WARNING: no selection_manifest.json found — skipping select migration")
        return False

    raw = json.loads(target.read_text())

    # Check if it already has the canonical 'step' key — already migrated
    if raw.get("step") == "select" and "selected_block_files" in raw:
        print(f"  selection_manifest.json already in current schema — skipping")
        return True

    # Backup original
    bak = target.with_suffix(".json.bak")
    if not bak.exists():
        if not dry_run:
            shutil.copy2(target, bak)
            print(f"  backed up original to {bak}")
        else:
            print(f"  [DRY-RUN] would backup {target} -> {bak}")

    # Build normalised manifest
    normalised = {
        "step":                 "select",
        "converged":            raw.get("converged", False),
        "selected_count":       raw.get("selected_count", 0),
        "model_path":           raw.get("model_path"),
        "candidate_sources":    raw.get("candidate_sources", []),
        # selected_block_files: use existing value or derive from selected_cfg_path
        "selected_block_files": _resolve_block_files(raw, gen_dir),
        "completed_at":         raw.get("completed_at", _now()),
        "_migrated": True,
    }
    # Preserve any extra keys not in the canonical schema
    for k, v in raw.items():
        if k not in normalised:
            normalised[f"_legacy_{k}"] = v

    _write_json(target, normalised, dry_run)
    return True


def _resolve_block_files(raw: dict, gen_dir: Path) -> list[str]:
    """Return selected_block_files list, deriving it if absent."""
    if raw.get("selected_block_files"):
        return raw["selected_block_files"]

    # Try gen_NN/select/selected_blocks/
    blocks_dir = gen_dir / "select" / "selected_blocks"
    if blocks_dir.exists():
        files = sorted(blocks_dir.glob("selected_*.cfg"))
        if files:
            return [str(f) for f in files]

    # Try old sibling dir
    gen_num = gen_dir.name.split("_")[1]
    old_blocks = gen_dir.parent / f"Pb_gen_{gen_num}_select" / "selected_blocks"
    if old_blocks.exists():
        files = sorted(old_blocks.glob("selected_*.cfg"))
        if files:
            return [str(f) for f in files]

    # Last resort: the single selected.cfg listed in selected_cfg_path
    if raw.get("selected_cfg_path"):
        return [raw["selected_cfg_path"]]

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def migrate(runs_root: Path, dry_run: bool) -> None:
    gen_dirs = _find_gen_dirs(runs_root)
    if not gen_dirs:
        print(f"No gen_NN directories found under {runs_root}")
        return

    print(f"Found {len(gen_dirs)} generation directories under {runs_root}")
    if dry_run:
        print("DRY-RUN mode — no files will be written\n")

    for gen_dir in gen_dirs:
        gen = gen_dir.name
        fit_manifest = gen_dir / "fit" / "fit_manifest.json"

        if fit_manifest.exists():
            print(f"[{gen}] fit_manifest.json present — skipping")
            continue

        print(f"[{gen}] migrating...")
        _migrate_fit_manifest(gen_dir, dry_run)
        _migrate_selection_manifest(gen_dir, dry_run)
        print()

    print("Migration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Path to the runs directory (default: ./runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.exists():
        raise SystemExit(f"runs-root does not exist: {runs_root}")

    migrate(runs_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
