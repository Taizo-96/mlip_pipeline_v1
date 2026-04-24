from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from mlip_pipeline.models import LabelResult, ScalingPolicy
from mlip_pipeline.label.incar_writer import write_incar
from mlip_pipeline.label.job_writer import write_job_sh
from mlip_pipeline.select.runner import split_cfg_blocks
from mlip_pipeline.utils.fs import ensure_dir, copy_if_exists, write_json


# ---------------------------------------------------------------------------
# MLIP/MTP .cfg parser
# ---------------------------------------------------------------------------

def _parse_cfg_block(block: str) -> dict:
    """
    Parse a single BEGIN_CFG ... END_CFG block from an MLIP/MTP cfg file.

    Returns a dict with keys:
        cell       : (3,3) float array  (row = lattice vector)
        atom_types : (N,) int array     (0-based MLIP type indices)
        positions  : (N,3) float array  (Cartesian)
    """
    READ_NONE  = 0
    READ_SIZE  = 1
    READ_CELL  = 2
    READ_ATOMS = 3
    state = READ_NONE

    cell       = []
    atom_types = []
    positions  = []

    for line in block.splitlines():
        s = line.strip()
        if not s or s in ("BEGIN_CFG", "END_CFG"):
            continue

        if s == "Size":
            state = READ_SIZE; continue
        if s == "Supercell":
            state = READ_CELL; continue
        if s.startswith("AtomData:"):
            state = READ_ATOMS; continue
        if s.startswith(("Energy", "PlusStress", "Stress", "Feature")):
            state = READ_NONE; continue

        if state == READ_SIZE:
            state = READ_NONE

        elif state == READ_CELL:
            row = [float(x) for x in s.split()]
            if len(row) == 3:
                cell.append(row)
            if len(cell) == 3:
                state = READ_NONE

        elif state == READ_ATOMS:
            cols = s.split()
            atom_types.append(int(cols[1]))
            positions.append([float(cols[2]), float(cols[3]), float(cols[4])])

    if len(cell) != 3 or not atom_types:
        raise ValueError(
            f"Malformed cfg block — got {len(cell)} cell rows "
            f"and {len(atom_types)} atoms.\n"
            f"Block preview:\n{block[:300]}"
        )

    return {
        "cell":       np.array(cell),
        "atom_types": np.array(atom_types),
        "positions":  np.array(positions),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_type_map(type_map_path: Path) -> list[str]:
    """Load ordered element symbols from a DeePMD type_map.raw file."""
    return list(np.loadtxt(type_map_path, dtype=str).reshape(-1))


def _cfg_block_to_poscar(block_path: Path, type_map: list[str], task_dir: Path) -> int:
    """Parse a single MLIP/MTP .cfg block, write POSCAR, return n_atoms."""
    data = _parse_cfg_block(block_path.read_text())
    symbols = [type_map[t] for t in data["atom_types"]]
    atoms = Atoms(
        symbols=symbols,
        positions=data["positions"],
        cell=data["cell"],
        pbc=True,
    )
    ensure_dir(task_dir)
    write(str(task_dir / "POSCAR"), atoms, format="vasp", vasp5=True, sort=True)
    return len(atoms)


def _count_kpoints(kpoints_path: Path) -> int:
    """
    Read n_kpoints from a VASP KPOINTS file.
    Handles Gamma/Monkhorst-Pack meshes (returns product of subdivisions)
    and explicit k-point lists (returns count on line 2).
    """
    lines = [l.strip() for l in kpoints_path.read_text().splitlines() if l.strip()]
    mode = lines[2].upper()[0]
    if mode in ("G", "M"):
        subdivisions = [int(x) for x in lines[3].split()[:3]]
        return subdivisions[0] * subdivisions[1] * subdivisions[2]
    return int(lines[1])


def _scale_kpoints_file(src: Path, dest: Path, n_atoms: int, policy: ScalingPolicy):
    lines = src.read_text().splitlines()
    if len(lines) < 4:
        copy_if_exists(src, dest)
        return

    # Check if it's Gamma/Monkhorst (Line 3)
    mode = lines[2].strip().upper()[0]
    if mode in ("G", "M"):
        kpts = [int(x) for x in lines[3].split()[:3]]
        scaled = policy.scale_kpoints(n_atoms, kpts)
        lines[3] = f" {scaled[0]} {scaled[1]} {scaled[2]}"
        dest.write_text("\n".join(lines) + "\n")
    else:
        # Explicit list: too complex to auto-scale safely, just copy
        copy_if_exists(src, dest)

def _write_vasp_inputs(template_dir: Path, task_dir: Path, n_atoms: int, label_cfg: dict):
    policy = ScalingPolicy(label_cfg.get("scaling_rules", {}))
    sys_cfg = policy.get_config(n_atoms)

    # 1. KPOINTS with scaling
    _scale_kpoints_file(template_dir / "KPOINTS", task_dir / "KPOINTS", n_atoms, policy)
    copy_if_exists(template_dir / "POTCAR", task_dir / "POTCAR")

    # 2. INCAR (merge base tags with the NCORE/KPAR from our policy)
    write_incar(task_dir, label_cfg["incar"], sys_cfg.to_incar_dict())

    # 3. Job Script (Pass the sys_cfg object instead of just a dict)
    write_job_sh(task_dir, n_atoms, slurm_cfg=label_cfg.get("slurm"), sys_override=sys_cfg)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_labeling(
    config: dict,
    resolved_paths: dict,
    selection_result=None,
) -> LabelResult:
    """
    Convert selected MLIP/MTP .cfg structures to VASP single-point input directories.

    For each structure:
      1. Parses the MLIP cfg block and writes POSCAR.
      2. Copies KPOINTS + POTCAR from template_dir.
      3. Generates INCAR from config['label']['incar'] + auto parallelization.
      4. Generates job.sh from config['label']['slurm'] + auto parallelization.
      5. Writes label_manifest.json.
    """
    label_cfg    = config["label"]
    runs_root    = resolved_paths["runs_root"]
    project_root = resolved_paths["project_root"]

    select_root = runs_root / label_cfg["input_subdir"]
    label_root  = ensure_dir(runs_root / label_cfg["output_subdir"])

    # ── type_map ──────────────────────────────────────────────────────────────
    type_map_path = Path(label_cfg["type_map"])
    if not type_map_path.is_absolute():
        type_map_path = project_root / type_map_path
    if not type_map_path.exists():
        raise FileNotFoundError(f"type_map not found: {type_map_path}")
    type_map = _load_type_map(type_map_path)

    # ── VASP template dir (KPOINTS + POTCAR only) ─────────────────────────────
    template_dir = Path(label_cfg["template_dir"])
    if not template_dir.is_absolute():
        template_dir = project_root / template_dir
    if not template_dir.exists():
        raise FileNotFoundError(f"VASP template dir not found: {template_dir}")

    # ── locate cfg block files ────────────────────────────────────────────────
    if selection_result is not None and selection_result.selected_cfg_paths:
        cfg_paths = selection_result.selected_cfg_paths
    else:
        blocks_dir        = select_root / "selected_blocks"
        selected_filename = label_cfg.get("selected_filename", "selected.cfg")

        if blocks_dir.exists():
            cfg_paths = sorted(blocks_dir.glob("*.cfg"))
        else:
            selected_cfg = select_root / selected_filename
            if not selected_cfg.exists():
                raise FileNotFoundError(
                    f"Neither {blocks_dir} nor {selected_cfg} found. "
                    "Run the select step first."
                )
            blocks     = split_cfg_blocks(selected_cfg.read_text())
            blocks_dir = ensure_dir(select_root / "selected_blocks")
            cfg_paths  = []
            for i, block in enumerate(blocks):
                p = blocks_dir / f"selected_{i:05d}.cfg"
                p.write_text(block.strip() + "\n")
                cfg_paths.append(p)

    if not cfg_paths:
        raise FileNotFoundError(f"No cfg block files found under {select_root}.")

    # ── convert + write inputs ────────────────────────────────────────────────
    task_dirs: list[Path] = []
    for i, cfg_path in enumerate(cfg_paths):
        task_dir = label_root / f"task.{i:06d}"
        n_atoms  = _cfg_block_to_poscar(cfg_path, type_map, task_dir)
        _write_vasp_inputs(template_dir, task_dir, n_atoms, label_cfg)
        task_dirs.append(task_dir)
        print(f"  [{i+1:>4d}/{len(cfg_paths)}]  {cfg_path.name}  →  {task_dir.name}/  ({n_atoms} atoms)")

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest_path = write_json(label_root / "label_manifest.json", {
        "input_subdir":  str(select_root),
        "output_subdir": str(label_root),
        "type_map":      type_map,
        "template_dir":  str(template_dir),
        "task_count":    len(task_dirs),
        "task_dirs":     [str(t) for t in task_dirs],
    })

    print(f"\nDone. {len(task_dirs)} VASP task(s) written to {label_root}/")
    print(f"Manifest: {manifest_path}")

    return LabelResult(
        label_root=label_root,
        task_dirs=task_dirs,
        task_count=len(task_dirs),
        manifest_path=manifest_path,
    )