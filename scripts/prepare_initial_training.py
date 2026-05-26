#!/usr/bin/env python3
"""
scripts/prepare_initial_training.py

One-time utility: convert a collection of DeePMD-format system directories
(containing set.000/coord.npy, box.npy, energy.npy, force.npy and type.raw)
into MLIP-3 .cfg files, then merge them into a single train.cfg.

This is NOT part of the active-learning loop.  Run it once to prepare your
initial training set before calling ``mlip-pipeline fit``.

Usage
-----
    # Recommended — read output paths directly from your pipeline config:
    python scripts/prepare_initial_training.py \\
        --config configs/Fe_loop.yaml \\
        --data-root /path/to/deepmd/systems \\
        --type-map-elements Fe Pb

    # Legacy — explicit output directory (no config needed):
    python scripts/prepare_initial_training.py \\
        --data-root /path/to/deepmd/systems \\
        --output-dir /path/to/datasets/fe_initial \\
        --merge-name train.cfg

When --config is supplied the script produces:

    <datasets_root>/
    ├── fe_initial/               ← per-system .cfg files + train.cfg + manifest
    │   ├── fe_Fe16eq.cfg
    │   ├── fepb_Fe125Pb3.cfg
    │   ├── ...
    │   ├── train.cfg
    │   └── manifest.json
    ├── converted_cfg/
    │   └── train.cfg             ← copy; this is what the pipeline reads
    └── vasp_template/
        └── type_map.raw

The script walks the full directory tree under --data-root and automatically
finds every valid DeePMD system (any folder containing set.000/ with the
required .npy files and a type.raw).  No --glob argument is needed.

The script is self-contained and does not depend on the mlip_pipeline package
beyond numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# DeePMD helpers
# ---------------------------------------------------------------------------

def _is_valid_system(directory: Path) -> bool:
    """Return True if directory looks like a complete DeePMD system."""
    base = directory / "set.000"
    required = [
        base / "coord.npy",
        base / "box.npy",
        base / "energy.npy",
        base / "force.npy",
        directory / "type.raw",
    ]
    return all(p.exists() for p in required)


def _find_all_systems(data_root: Path) -> list[Path]:
    """Walk data_root recursively and return every valid DeePMD system folder."""
    systems: list[Path] = []
    for dirpath, dirnames, _ in os.walk(data_root):
        candidate = Path(dirpath)
        if _is_valid_system(candidate):
            systems.append(candidate)
            dirnames.clear()
    return sorted(systems)


def _write_cfg(system_dir: Path, out_path: Path) -> int:
    """Write a single .cfg file from a DeePMD system directory.

    Returns the number of frames written.
    """
    base = system_dir / "set.000"

    coord  = np.load(base / "coord.npy")
    box    = np.load(base / "box.npy")
    energy = np.load(base / "energy.npy")
    force  = np.load(base / "force.npy")
    types  = np.loadtxt(system_dir / "type.raw", dtype=int)

    nframes = coord.shape[0]
    natoms  = len(types)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(nframes):
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"   {natoms}\n")
            f.write(" Supercell\n")
            cell = box[i].reshape(3, 3)
            for row in cell:
                f.write(
                    "  "
                    + "  ".join(f"{float(x):20.12f}" for x in row)
                    + "\n"
                )
            f.write(
                " AtomData:  id type       cartes_x      cartes_y      cartes_z"
                "           fx          fy          fz\n"
            )
            xyz = coord[i].reshape(natoms, 3)
            frc = force[i].reshape(natoms, 3)
            for j in range(natoms):
                atype = int(types[j])
                f.write(
                    f"{j+1:13d}{atype:5d}"
                    + "".join(f"{float(x):14.6f}" for x in xyz[j])
                    + "".join(f"{float(x):12.6f}" for x in frc[j])
                    + "\n"
                )
            e = float(
                energy[i, 0] if getattr(energy[i], "shape", ()) else energy[i]
            )
            f.write(" Energy\n")
            f.write(f"{e:20.12f}\n")
            f.write(" PlusStress:  xx yy zz yz xz xy\n")
            f.write("  0.0 0.0 0.0 0.0 0.0 0.0\n")
            f.write(" Feature   conf_id\n")
            f.write(f"   {i}\n")
            f.write("END_CFG\n\n")

    return nframes


# ---------------------------------------------------------------------------
# vasp_template scaffold
# ---------------------------------------------------------------------------

def _create_vasp_template(datasets_root: Path, elements: list[str]) -> Path:
    """Create datasets_root/vasp_template/ and write type_map.raw."""
    tmpl_dir = datasets_root / "vasp_template"
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    type_map_path = tmpl_dir / "type_map.raw"
    type_map_path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    print(f"vasp_template created : {tmpl_dir}/")
    print(f"type_map.raw          : {type_map_path}  ({' '.join(elements)})")

    for fname in ("POTCAR", "KPOINTS"):
        if not (tmpl_dir / fname).exists():
            print(f"  [REMINDER] copy your {fname} into {tmpl_dir}/")

    return tmpl_dir


# ---------------------------------------------------------------------------
# Config reader (minimal — no jinja2 required)
# ---------------------------------------------------------------------------

def _resolve_datasets_root(config_path: Path) -> Path:
    """Parse the YAML config just enough to resolve paths.datasets_root."""
    try:
        import yaml  # type: ignore
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        project_root  = raw.get("project_root", "")
        datasets_root = raw.get("paths", {}).get("datasets_root", "")
        datasets_root = datasets_root.replace("{{ project_root }}", project_root)
        return Path(datasets_root)
    except Exception:
        pass

    # Fallback: naive line scan
    project_root  = ""
    datasets_root = ""
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("project_root:"):
                project_root = line.split(":", 1)[1].strip().strip('"').strip("'")
            if "datasets_root:" in line:
                datasets_root = line.split(":", 1)[1].strip().strip('"').strip("'")
                datasets_root = datasets_root.replace("{{ project_root }}", project_root)
    if not datasets_root:
        raise ValueError(f"Could not resolve paths.datasets_root from {config_path}")
    return Path(datasets_root)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def prepare(
    data_root: str | Path,
    initial_dir: str | Path,
    merge_name: str = "train.cfg",
    manifest_name: str = "manifest.json",
) -> Path:
    """Convert all DeePMD systems, write per-system cfgs + merged train.cfg
    into initial_dir.

    Returns the path to the merged train.cfg inside initial_dir.
    """
    data_root   = Path(data_root)
    initial_dir = Path(initial_dir)
    initial_dir.mkdir(parents=True, exist_ok=True)

    systems = _find_all_systems(data_root)
    if not systems:
        raise FileNotFoundError(
            f"No valid DeePMD system directories found under: {data_root}"
        )

    print(f"Found {len(systems)} system(s) under {data_root}\n")

    generated:  list[Path]     = []
    cfg_counts: dict[str, int] = {}
    total_cfgs: int            = 0

    for system_dir in systems:
        rel_name = str(system_dir.relative_to(data_root)).replace(os.sep, "_")
        out_path = initial_dir / f"{rel_name}.cfg"
        n_cfgs   = _write_cfg(system_dir, out_path)
        generated.append(out_path)
        cfg_counts[rel_name] = n_cfgs
        total_cfgs          += n_cfgs
        print(f"  {rel_name}: {n_cfgs} frames → {out_path}")

    merged_cfg = initial_dir / merge_name
    with open(merged_cfg, "w", encoding="utf-8") as merged:
        for cfg in generated:
            text = cfg.read_text(encoding="utf-8")
            merged.write(text)
            if not text.endswith("\n"):
                merged.write("\n")
    print(f"\nMerged {total_cfgs} frames → {merged_cfg}")

    manifest_path = initial_dir / manifest_name
    manifest = {
        "total_cfgs":     total_cfgs,
        "per_system":     cfg_counts,
        "merged_cfg":     str(merged_cfg),
        "generated_cfgs": [str(p) for p in generated],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest          → {manifest_path}")

    return merged_cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert all DeePMD systems to MLIP-3 .cfg files.\n\n"
            "Pass --config to derive all output paths from your pipeline YAML.\n"
            "Pass --output-dir to specify the initial directory explicitly (legacy)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-root", required=True,
        help="Root directory to search recursively for DeePMD systems",
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to pipeline YAML config (e.g. configs/Fe_loop.yaml). "
            "Derives datasets_root; writes to <datasets_root>/fe_initial/ "
            "and copies train.cfg to <datasets_root>/converted_cfg/."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Initial output directory for per-system cfgs + train.cfg "
            "(overrides config default of <datasets_root>/fe_initial)."
        ),
    )
    parser.add_argument(
        "--merge-name", default="train.cfg",
        help="Filename for merged cfg (default: train.cfg)",
    )
    parser.add_argument(
        "--manifest", default="manifest.json",
        help="Filename for manifest JSON (default: manifest.json)",
    )
    parser.add_argument(
        "--type-map-elements", nargs="+", default=None,
        metavar="ELEMENT",
        help="Ordered element list, e.g. --type-map-elements Fe Pb",
    )

    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    datasets_root: Path | None = None

    if args.config:
        datasets_root = _resolve_datasets_root(Path(args.config))
        initial_dir   = Path(args.output_dir) if args.output_dir else datasets_root / "fe_initial"
        converted_dir = datasets_root / "converted_cfg"
        print(f"Config          : {args.config}")
        print(f"datasets_root   : {datasets_root}")
        print(f"initial_dir     : {initial_dir}")
        print(f"converted_cfg   : {converted_dir}\n")
    elif args.output_dir:
        initial_dir   = Path(args.output_dir)
        converted_dir = None
    else:
        parser.error("Provide either --config or --output-dir.")

    # ── vasp_template ──────────────────────────────────────────────────────
    if datasets_root is not None:
        elements = args.type_map_elements
        if not elements:
            try:
                import yaml  # type: ignore
                with open(args.config, encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                elements = raw.get("materials_project", {}).get("ff_elements")
            except Exception:
                pass
        if not elements:
            parser.error(
                "--type-map-elements is required when using --config "
                "(e.g. --type-map-elements Fe Pb)"
            )
        _create_vasp_template(datasets_root, elements)
        print()

    # ── Convert + merge into fe_initial/ ───────────────────────────────────
    merged_cfg = prepare(
        data_root     = args.data_root,
        initial_dir   = initial_dir,
        merge_name    = args.merge_name,
        manifest_name = args.manifest,
    )

    # ── Copy train.cfg to converted_cfg/ (pipeline input) ──────────────────
    if converted_dir is not None:
        converted_dir.mkdir(parents=True, exist_ok=True)
        dest = converted_dir / args.merge_name
        shutil.copy2(merged_cfg, dest)
        print(f"Copied train.cfg  → {dest}  (pipeline input)")
