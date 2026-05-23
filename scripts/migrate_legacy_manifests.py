#!/usr/bin/env python3
"""Migrate legacy gen 0-15 manifests to the current pipeline schema.

What this script does
---------------------
For each generation directory under <runs_root> that is missing a
``fit_manifest.json`` it:

1. Reads the old ``metadata.json`` and writes ``fit_manifest.json``.
   ``n_train_cfgs`` is determined by the following priority chain:

   a) Count BEGIN_CFG blocks in a *generation-specific* training cfg
      (a file that was NOT shared across all gens, e.g. pb_cfg_gen_NN/).
      If the per-gen file exists and its size differs from the global
      merged cfg, it is used directly.

   b) Reconstruct from cumulative ``selected_count`` values read from
      every generation's selection manifest, anchored to the initial
      dataset size (gen 0's training cfg BEGIN_CFG count or a manually
      provided ``--initial-dataset-size``).  This handles the common
      case where a single ``train.cfg`` was overwritten each generation.

2. Normalises ``selection_manifest.json`` to the current schema.

3. Parses ``train.log`` (MLIP-2 and MLIP-3 formats) and writes
   ``gen_NN/fit/eval/metrics.csv`` so ``replot-eval`` can populate
   RMSE annotations without re-running ``mlp calculate_efs``.

The script is **idempotent** — if ``fit_manifest.json`` AND
``eval/metrics.csv`` both exist the generation is skipped.

Usage
-----
    python scripts/migrate_legacy_manifests.py \\
        --runs-root /path/to/runs \\
        [--initial-dataset-size N] \\
        [--dry-run]

    # or, using the pipeline config to resolve runs_root automatically:
    python scripts/migrate_legacy_manifests.py \\
        --config configs/Pb_loop.yaml \\
        [--initial-dataset-size N] \\
        [--dry-run]

Defaults
--------
    --runs-root              ./runs
    --initial-dataset-size   auto (counted from gen_00's training cfg)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _count_begin_cfg(cfg_path: Path | None) -> int:
    """Count BEGIN_CFG blocks in a .cfg file. Returns 0 if absent."""
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
    pattern = re.compile(r"^gen_\d+$")
    return sorted(
        [d for d in runs_root.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )


# ---------------------------------------------------------------------------
# Metadata / path resolution
# ---------------------------------------------------------------------------

def _find_metadata(gen_dir: Path) -> Path | None:
    new = gen_dir / "fit" / "metadata.json"
    if new.exists():
        return new
    gen_num = gen_dir.name.split("_")[1]
    sibling = gen_dir.parent / f"Pb_gen_{gen_num}_fit" / "metadata.json"
    if sibling.exists():
        return sibling
    return None


def _find_global_train_cfg(meta: dict) -> Path | None:
    """Return the train_cfg path from metadata (may be the global/shared one)."""
    t = meta.get("train_cfg")
    if t:
        p = Path(t)
        if p.exists():
            return p
    return None


def _find_per_gen_train_cfg(gen_dir: Path, runs_root: Path) -> Path | None:
    """Search for a generation-specific training cfg that is NOT the global one.

    Returns None if only the global (shared) file is found.
    """
    gen_num = int(gen_dir.name.split("_")[1])
    project_root = runs_root.parent

    for datasets_root in (project_root / "datasets", runs_root.parent / "datasets"):
        for subdir_name in (
            f"pb_cfg_gen_{gen_num:02d}",
            f"pb_cfg_gen_{gen_num}",
            f"Pb_gen_{gen_num:02d}",
            f"Pb_gen_{gen_num}",
        ):
            for cfg_name in ("train.cfg", "merged_train.cfg", "training.cfg"):
                p = datasets_root / subdir_name / cfg_name
                if p.exists():
                    return p
    return None


def _find_model_path(meta: dict, gen_dir: Path) -> Path | None:
    if meta.get("result_model"):
        p = Path(meta["result_model"])
        if p.exists():
            return p
    run_dir = Path(meta["run_dir"]) if meta.get("run_dir") else None
    if run_dir and run_dir.exists():
        candidates = list(run_dir.glob("*.almtp"))
        if candidates:
            return candidates[0]
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
# n_train_cfgs reconstruction from selection manifests
# ---------------------------------------------------------------------------

def _read_selected_count(gen_dir: Path) -> int:
    """Read selected_count from the selection manifest, or -1 if absent."""
    for candidate in (
        gen_dir / "select" / "selection_manifest.json",
        gen_dir / "select" / "selection_manifest.json.bak",
    ):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
                v = data.get("selected_count", -1)
                if isinstance(v, int) and v >= 0:
                    return v
            except Exception:
                pass
    return -1


def build_cumulative_n_train(
    gen_dirs: list[Path],
    initial_size: int,
) -> dict[int, int]:
    """Return mapping {gen_number -> n_train_cfgs} by cumulatively summing
    selected_count values.

    Gen 0 used the initial dataset (initial_size configs).
    Gen N's training set = initial_size + sum(selected_count for gen 0..N-1).
    """
    cumulative: dict[int, int] = {}
    running = initial_size
    for gen_dir in gen_dirs:
        gen_num = int(gen_dir.name.split("_")[1])
        cumulative[gen_num] = running
        selected = _read_selected_count(gen_dir)
        if selected >= 0:
            running += selected
        # If selected_count unknown, leave running unchanged (best guess)
    return cumulative


# ---------------------------------------------------------------------------
# RMSE parsing (MLIP-2 and MLIP-3 formats)
# ---------------------------------------------------------------------------

_MLIP3_RE = re.compile(
    r"(Energy|Forces|Virial stresses)[^\n]*\n"
    r"(?:[^\n]*\n){0,6}?"
    r"[^\n]*RMS\s+absolute difference\s*=\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)
_MLIP2_ITER_RE = re.compile(
    r"RMSE_E=([\d.eE+\-]+).*?RMSE_F=([\d.eE+\-]+).*?RMSE_S=([\d.eE+\-]+)",
    re.IGNORECASE,
)
_MLIP2_SUMMARY_RE = re.compile(
    r"EFS[- ]train RMSE[^\n]*\n"
    r"\s*energy\s+([\d.eE+\-]+)[^\n]*\n"
    r"\s*forces\s+([\d.eE+\-]+)[^\n]*\n"
    r"\s*stress(?:es)?\s+([\d.eE+\-]+)",
    re.IGNORECASE,
)


def _parse_log_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    results: dict[str, float] = {}
    _KEY_MAP = {"energy": "rmse_e", "force": "rmse_f", "stress": "rmse_s", "virial": "rmse_s"}

    for m in _MLIP3_RE.finditer(text):
        section = m.group(1).lower()
        value   = float(m.group(2))
        for fragment, key in _KEY_MAP.items():
            if fragment in section:
                results[key] = value
                break
    if results:
        return results

    m = _MLIP2_SUMMARY_RE.search(text)
    if m:
        return {"rmse_e": float(m.group(1)), "rmse_f": float(m.group(2)), "rmse_s": float(m.group(3))}

    last = None
    for m in _MLIP2_ITER_RE.finditer(text):
        last = m
    if last:
        return {"rmse_e": float(last.group(1)), "rmse_f": float(last.group(2)), "rmse_s": float(last.group(3))}

    return {}


def _write_metrics_csv(metrics: dict, dest: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY-RUN] would write {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in ("rmse_e", "rmse_f", "rmse_s"):
            if key in metrics:
                w.writerow([key, metrics[key]])
    print(f"  wrote {dest}")


# ---------------------------------------------------------------------------
# Per-generation migration steps
# ---------------------------------------------------------------------------

def _migrate_fit_manifest(
    gen_dir: Path,
    runs_root: Path,
    n_train_cfgs: int,
    dry_run: bool,
) -> tuple[bool, Path | None]:
    """Write fit_manifest.json. Returns (success, log_path)."""
    target = gen_dir / "fit" / "fit_manifest.json"
    if target.exists():
        print(f"  fit_manifest.json already exists — skipping")
        return True, None

    meta_path = _find_metadata(gen_dir)
    if meta_path is None:
        print(f"  WARNING: no metadata.json found — cannot migrate fit manifest")
        return False, None

    meta       = json.loads(meta_path.read_text())
    model_path = _find_model_path(meta, gen_dir)
    log_path   = _find_log_path(meta, gen_dir)
    train_cfg  = _find_global_train_cfg(meta)  # stored for reference only

    print(f"  n_train_cfgs={n_train_cfgs}")

    manifest = {
        "step":          "fit",
        "run_dir":       str(gen_dir / "fit"),
        "model_path":    str(model_path) if model_path else None,
        "log_path":      str(log_path) if log_path else None,
        "train_cfg":     str(train_cfg) if train_cfg else None,
        "n_train_cfgs":  n_train_cfgs,
        "init_template": meta.get("init_template"),
        "mlp_command":   meta.get("mlp_command", "mlp"),
        "mpi_prefix":    meta.get("mpi_prefix"),
        "command":       meta.get("command", []),
        "completed_at":  meta.get("timestamp", _now()),
        "_migrated_from": str(meta_path),
    }
    _write_json(target, manifest, dry_run)
    return True, log_path


def _migrate_metrics_csv(gen_dir: Path, log_path: Path | None, dry_run: bool) -> None:
    dest = gen_dir / "fit" / "eval" / "metrics.csv"
    if dest.exists():
        print(f"  eval/metrics.csv already exists — skipping")
        return

    if log_path is None or not log_path.exists():
        for name in ("train.log", "fit.log"):
            for base in (gen_dir / "fit", gen_dir):
                p = base / name
                if p.exists():
                    log_path = p
                    break
            if log_path and log_path.exists():
                break

    if log_path is None or not log_path.exists():
        print(f"  WARNING: no train.log found — cannot write metrics.csv")
        return

    metrics = _parse_log_metrics(log_path)
    if not metrics:
        print(f"  WARNING: no RMSE metrics found in {log_path} — metrics.csv not written")
        return

    e = metrics.get("rmse_e", float("nan"))
    f = metrics.get("rmse_f", float("nan"))
    s = metrics.get("rmse_s", float("nan"))
    print(f"  RMSE: E={e:.6g}  F={f:.6g}  S={s:.6g}  (from {log_path.name})")
    _write_metrics_csv(metrics, dest, dry_run)


def _migrate_selection_manifest(gen_dir: Path, dry_run: bool) -> bool:
    target = gen_dir / "select" / "selection_manifest.json"
    gen_num = gen_dir.name.split("_")[1]
    old_sibling = gen_dir.parent / f"Pb_gen_{gen_num}_select" / "selection_manifest.json"

    if not target.exists() and old_sibling.exists():
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_sibling, target)
            print(f"  copied {old_sibling} -> {target}")
        else:
            print(f"  [DRY-RUN] would copy {old_sibling} -> {target}")

    if not target.exists():
        print(f"  WARNING: no selection_manifest.json — skipping select migration")
        return False

    raw = json.loads(target.read_text())
    if raw.get("step") == "select" and "selected_block_files" in raw:
        print(f"  selection_manifest.json already in current schema — skipping")
        return True

    bak = target.with_suffix(".json.bak")
    if not bak.exists():
        if not dry_run:
            shutil.copy2(target, bak)
            print(f"  backed up original to {bak}")
        else:
            print(f"  [DRY-RUN] would backup {target} -> {bak}")

    normalised = {
        "step":                 "select",
        "converged":            raw.get("converged", False),
        "selected_count":       raw.get("selected_count", 0),
        "model_path":           raw.get("model_path"),
        "candidate_sources":    raw.get("candidate_sources", []),
        "selected_block_files": _resolve_block_files(raw, gen_dir),
        "completed_at":         raw.get("completed_at", _now()),
        "_migrated": True,
    }
    for k, v in raw.items():
        if k not in normalised:
            normalised[f"_legacy_{k}"] = v

    _write_json(target, normalised, dry_run)
    return True


def _resolve_block_files(raw: dict, gen_dir: Path) -> list[str]:
    if raw.get("selected_block_files"):
        return raw["selected_block_files"]
    blocks_dir = gen_dir / "select" / "selected_blocks"
    if blocks_dir.exists():
        files = sorted(blocks_dir.glob("selected_*.cfg"))
        if files:
            return [str(f) for f in files]
    gen_num = gen_dir.name.split("_")[1]
    old_blocks = gen_dir.parent / f"Pb_gen_{gen_num}_select" / "selected_blocks"
    if old_blocks.exists():
        files = sorted(old_blocks.glob("selected_*.cfg"))
        if files:
            return [str(f) for f in files]
    if raw.get("selected_cfg_path"):
        return [raw["selected_cfg_path"]]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def migrate(runs_root: Path, initial_dataset_size: int | None, dry_run: bool) -> None:
    gen_dirs = _find_gen_dirs(runs_root)
    if not gen_dirs:
        print(f"No gen_NN directories found under {runs_root}")
        return

    print(f"Found {len(gen_dirs)} generation directories under {runs_root}")
    if dry_run:
        print("DRY-RUN mode — no files will be written\n")

    # ── Determine initial dataset size for gen 0 ──────────────────────────
    if initial_dataset_size is None:
        # Try to count from gen_00's metadata train_cfg
        gen0_meta_path = _find_metadata(gen_dirs[0]) if gen_dirs else None
        gen0_meta      = json.loads(gen0_meta_path.read_text()) if gen0_meta_path else {}
        gen0_cfg       = _find_global_train_cfg(gen0_meta)
        initial_dataset_size = _count_begin_cfg(gen0_cfg)
        if initial_dataset_size > 0:
            print(f"Initial dataset size (gen_00): {initial_dataset_size}  "
                  f"(from {gen0_cfg})")
        else:
            print("WARNING: could not determine initial dataset size; "
                  "use --initial-dataset-size N to set it manually.")
            initial_dataset_size = 0

    # ── Build cumulative n_train_cfgs for every gen ───────────────────────
    cumulative = build_cumulative_n_train(gen_dirs, initial_dataset_size)
    print("Cumulative training set sizes:")
    for gen_dir in gen_dirs:
        gen_num = int(gen_dir.name.split("_")[1])
        sel     = _read_selected_count(gen_dir)
        print(f"  gen_{gen_num:02d}: {cumulative[gen_num]:>6d} configs"
              f"  (selected_count={sel if sel>=0 else '?'})")
    print()

    # ── Migrate each generation ───────────────────────────────────────────
    for gen_dir in gen_dirs:
        gen     = gen_dir.name
        gen_num = int(gen.split("_")[1])
        fit_manifest = gen_dir / "fit" / "fit_manifest.json"
        metrics_csv  = gen_dir / "fit" / "eval" / "metrics.csv"

        if fit_manifest.exists() and metrics_csv.exists():
            print(f"[{gen}] fully migrated — skipping")
            continue

        print(f"[{gen}] migrating...")
        ok, log_path = _migrate_fit_manifest(
            gen_dir, runs_root, cumulative[gen_num], dry_run
        )
        if ok:
            _migrate_metrics_csv(gen_dir, log_path, dry_run)
        _migrate_selection_manifest(gen_dir, dry_run)
        print()

    print("Migration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config", type=Path,
        help="Path to pipeline config yaml (used to resolve runs_root automatically)",
    )
    group.add_argument(
        "--runs-root", type=Path, default=Path("runs"),
        help="Path to the runs directory (default: ./runs)",
    )
    parser.add_argument(
        "--initial-dataset-size", type=int, default=None,
        help="Number of configs in the initial (gen_00) training set. "
             "Auto-detected from gen_00's train.cfg if not provided.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    if args.config:
        args.config = args.config.expanduser().resolve()
        if not args.config.exists():
            raise SystemExit(f"config not found: {args.config}")
        import sys
        try:
            import yaml
            cfg       = yaml.safe_load(args.config.read_text())
            runs_root = Path(cfg.get("project", {}).get("runs_root", "runs"))
            if not runs_root.is_absolute():
                runs_root = args.config.parent / runs_root
        except ImportError:
            runs_root = args.config.parent.parent / "runs"
            print(f"WARNING: PyYAML not available, guessing runs_root={runs_root}", file=sys.stderr)
    else:
        runs_root = args.runs_root.expanduser().resolve()

    if not runs_root.exists():
        raise SystemExit(f"runs_root does not exist: {runs_root}")

    migrate(runs_root, args.initial_dataset_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
