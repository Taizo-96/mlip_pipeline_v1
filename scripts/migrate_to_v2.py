#!/usr/bin/env python3
"""
migrate_to_v2.py
────────────────
One-shot migration from the old flat layout to the new hierarchical layout.

Old layout:
  runs/Pb_gen_NN_fit/
  runs/Pb_gen_NN_explore/
  runs/Pb_gen_NN_select/
  runs/Pb_gen_NN_label/

  datasets/converted_cfg/run_NN/
  datasets/pb_cfg/

New layout:
  runs/gen_NN/fit/
  runs/gen_NN/explore/
  runs/gen_NN/select/
  runs/gen_NN/label/
  runs/gen_NN/state.json          ← synthesized from what's on disk
  runs/gen_NN/config_snapshot.yaml ← copied from original YAML if found

  datasets/converted_cfg/run_NN/  ← unchanged (already per-run subdirs)
  datasets/pb_cfg/                ← unchanged

Usage:
  python scripts/migrate_to_v2.py --project-root /path/to/project [--dry-run]

Options:
  --dry-run     Print what would happen without moving anything.
  --prefix      Folder prefix to strip from run dirs (default: "Pb_gen_")
  --config-dir  Directory containing the old per-gen YAMLs (default: project root)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

STEP_SUFFIXES = ("fit", "explore", "select", "label")
STEP_ORDER    = ("fit", "explore", "select", "label", "convert")


def _detect_generations(runs_root: Path, prefix: str) -> list[str]:
    """Find all generation numbers present in the old flat layout."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_({'|'.join(STEP_SUFFIXES)})$")
    gens = set()
    for d in runs_root.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                gens.add(m.group(1))
    return sorted(gens)


def _find_model_path(gen_dir_new: Path) -> str | None:
    for almtp in gen_dir_new.rglob("*.almtp"):
        return str(almtp)
    return None


def _find_merged_cfg(datasets_root: Path, gen_str: str) -> str | None:
    """
    After migration the merged train.cfg in datasets/converted_cfg/ is the
    authoritative cumulative set — return its path.
    """
    merged = datasets_root / "converted_cfg" / "train.cfg"
    if merged.exists():
        return str(merged)
    return None


def _infer_completed_steps(gen_dir_new: Path, runs_root: Path, prefix: str, gen_str: str) -> list[str]:
    """
    Decide which steps are 'done' based on what files exist in the new location.
    Order matters: label implies select implies explore implies fit.
    """
    completed = []

    # fit: trained .almtp file present
    fit_dir = gen_dir_new / "fit"
    if fit_dir.exists() and list(fit_dir.rglob("*.almtp")):
        completed.append("fit")

    # explore: any preselected.cfg or lammps output
    explore_dir = gen_dir_new / "explore"
    if explore_dir.exists() and (
        list(explore_dir.rglob("preselected.cfg"))
        or list(explore_dir.rglob("log.lammps"))
        or list(explore_dir.rglob("*.log"))
    ):
        completed.append("explore")

    # select: selected.cfg or selected_blocks/
    select_dir = gen_dir_new / "select"
    if select_dir.exists() and (
        (select_dir / "selected.cfg").exists()
        or (select_dir / "selected_blocks").exists()
        or list(select_dir.glob("*.cfg"))
    ):
        completed.append("select")

    # label: task.* subdirs with POSCARs
    label_dir = gen_dir_new / "label"
    if label_dir.exists() and list(label_dir.glob("task.*")):
        completed.append("label")

    # convert: OUTCARs in label/task.* → means labeling ran on HPC and results are back
    if "label" in completed:
        outcars = list((gen_dir_new / "label").rglob("OUTCAR"))
        if outcars:
            completed.append("convert")

    return completed


def _synthesize_state(
    gen_str: str,
    gen_dir_new: Path,
    completed_steps: list[str],
    model_path: str | None,
    merged_cfg: str | None,
) -> dict:
    all_done = set(STEP_ORDER) == set(completed_steps) or "convert" in completed_steps
    return {
        "generation":      gen_str,
        "status":          "completed" if all_done else "partial",
        "completed_steps": completed_steps,
        "current_step":    None,
        "error":           None,
        "started_at":      _now(),          # approximate — real time unknown
        "completed_at":    _now() if all_done else None,
        "model_path":      model_path,
        "merged_cfg":      merged_cfg,
        "_migrated":       True,
        "_migration_note": "State synthesized by migrate_to_v2.py from existing files.",
    }


def _find_yaml_for_gen(config_dir: Path, gen_str: str, prefix: str) -> Path | None:
    """Try several naming conventions for the old per-gen YAML."""
    candidates = [
        config_dir / f"{prefix}{gen_str}.yaml",
        config_dir / f"{prefix}{gen_str}-2.yaml",   # e.g. Pb_gen_00-2.yaml
        config_dir / f"gen_{gen_str}.yaml",
        config_dir / f"config_gen_{gen_str}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    # glob fallback
    matches = list(config_dir.glob(f"*{gen_str}*.yaml"))
    return matches[0] if matches else None


# ── main migration ─────────────────────────────────────────────────────────────

def migrate(
    project_root: Path,
    prefix: str = "Pb_gen_",
    config_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    runs_root     = project_root / "runs"
    datasets_root = project_root / "datasets"
    config_dir    = config_dir or project_root

    if not runs_root.exists():
        print(f"ERROR: runs/ not found at {runs_root}")
        sys.exit(1)

    generations = _detect_generations(runs_root, prefix)
    if not generations:
        print(f"No folders matching '{prefix}<NN>_<step>' found in {runs_root}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(generations)} generation(s) to migrate: {generations}")
    if dry_run:
        print("\n[DRY RUN] No files will be moved.\n")

    for gen_str in generations:
        gen_tag = f"gen_{gen_str}"
        gen_dir = runs_root / gen_tag
        print(f"\n{'─'*60}")
        print(f"  gen_{gen_str}  →  {gen_dir.relative_to(project_root)}/")

        if not dry_run:
            gen_dir.mkdir(parents=True, exist_ok=True)

        # ── move step directories ──────────────────────────────────────────────
        for step in STEP_SUFFIXES:
            old_name = f"{prefix}{gen_str}_{step}"
            old_path = runs_root / old_name
            new_path = gen_dir / step

            if not old_path.exists():
                print(f"    [SKIP]  {old_name}/ not found")
                continue

            if new_path.exists():
                print(f"    [SKIP]  {new_path.relative_to(project_root)}/ already exists")
                continue

            print(f"    MOVE  {old_path.relative_to(project_root)}")
            print(f"       →  {new_path.relative_to(project_root)}")
            if not dry_run:
                shutil.move(str(old_path), str(new_path))

        if dry_run:
            continue   # skip state synthesis in dry-run

        # ── synthesize state.json ─────────────────────────────────────────────
        state_path = gen_dir / "state.json"
        if state_path.exists():
            print(f"    [SKIP]  state.json already exists")
        else:
            completed = _infer_completed_steps(gen_dir, runs_root, prefix, gen_str)
            model_path = _find_model_path(gen_dir)
            merged_cfg = _find_merged_cfg(datasets_root, gen_str)
            state      = _synthesize_state(gen_str, gen_dir, completed, model_path, merged_cfg)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"    WROTE state.json  (completed_steps: {completed})")

        # ── copy config snapshot ──────────────────────────────────────────────
        snapshot_path = gen_dir / "config_snapshot.yaml"
        if snapshot_path.exists():
            print(f"    [SKIP]  config_snapshot.yaml already exists")
        else:
            yaml_src = _find_yaml_for_gen(config_dir, gen_str, prefix)
            if yaml_src:
                shutil.copy2(yaml_src, snapshot_path)
                print(f"    COPIED config_snapshot.yaml  ← {yaml_src.name}")
            else:
                print(f"    WARN   no YAML found for gen_{gen_str} — config_snapshot.yaml not written")

    print(f"\n{'─'*60}")
    print("Migration complete." if not dry_run else "Dry run complete — nothing was changed.")
    print()
    print("Next steps:")
    print("  1. Verify the new layout under runs/gen_NN/")
    print("  2. Check state.json in each gen (status, completed_steps)")
    print("  3. If any state is wrong, edit state.json by hand or re-run individual steps")
    print("  4. Continue with:  mlip-pipeline run-loop configs/base.yaml --start-gen 4 --end-gen N")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Migrate mlip-pipeline runs/ to the new hierarchical layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project_root",
        help="Absolute path to the project root (contains runs/, datasets/).",
    )
    parser.add_argument(
        "--prefix",
        default="Pb_gen_",
        help="Folder prefix to strip (default: 'Pb_gen_'). "
             "Set to 'gen_' if you already partially renamed things.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory containing the old per-gen YAMLs (default: project root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving anything.",
    )
    args = parser.parse_args()

    migrate(
        project_root=Path(args.project_root).expanduser().resolve(),
        prefix=args.prefix,
        config_dir=Path(args.config_dir).expanduser().resolve() if args.config_dir else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()