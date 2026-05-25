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
    python scripts/prepare_initial_training.py \
        --data-root /path/to/deepmd/systems \
        --output-dir /path/to/datasets/fe_cfg \
        --merge-name train.cfg

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
from pathlib import Path

import numpy as np


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
            # Don't descend further into a valid system folder
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


def prepare(
    data_root: str | Path,
    output_dir: str | Path = "fe_cfg",
    merge_name: str = "train.cfg",
    manifest_name: str = "manifest.json",
) -> Path:
    """Main entry-point for the script.

    Returns the path to the merged .cfg file.
    """
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = _find_all_systems(data_root)
    if not systems:
        raise FileNotFoundError(
            f"No valid DeePMD system directories found under: {data_root}"
        )

    print(f"Found {len(systems)} system(s) under {data_root}\n")

    generated:  list[Path]       = []
    cfg_counts: dict[str, int]   = {}
    total_cfgs: int              = 0

    for system_dir in systems:
        # Use the relative path as a unique name to avoid collisions
        rel_name = str(system_dir.relative_to(data_root)).replace(os.sep, "_")
        out_path = output_dir / f"{rel_name}.cfg"
        n_cfgs   = _write_cfg(system_dir, out_path)
        generated.append(out_path)
        cfg_counts[rel_name] = n_cfgs
        total_cfgs          += n_cfgs
        print(f"  {rel_name}: {n_cfgs} frames → {out_path}")

    merged_cfg: Path | None = None
    if merge_name:
        merged_cfg = output_dir / merge_name
        with open(merged_cfg, "w", encoding="utf-8") as merged:
            for cfg in generated:
                text = Path(cfg).read_text(encoding="utf-8")
                merged.write(text)
                if not text.endswith("\n"):
                    merged.write("\n")
        print(f"\nMerged {total_cfgs} frames into {merged_cfg}")

    manifest_path = output_dir / manifest_name
    manifest = {
        "total_cfgs":      total_cfgs,
        "per_system":      cfg_counts,
        "merged_cfg":      str(merged_cfg) if merged_cfg is not None else None,
        "generated_cfgs":  [str(p) for p in generated],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    return merged_cfg or output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all DeePMD systems (nested or flat) to a single MLIP-3 train.cfg."
    )
    parser.add_argument("--data-root",  required=True, help="Root directory to search recursively")
    parser.add_argument("--output-dir", default="fe_cfg", help="Output directory for .cfg files")
    parser.add_argument("--merge-name", default="train.cfg", help="Filename for merged .cfg")
    parser.add_argument("--manifest",   default="manifest.json", help="Filename for manifest JSON")
    args = parser.parse_args()

    prepare(
        data_root     = args.data_root,
        output_dir    = args.output_dir,
        merge_name    = args.merge_name,
        manifest_name = args.manifest,
    )
