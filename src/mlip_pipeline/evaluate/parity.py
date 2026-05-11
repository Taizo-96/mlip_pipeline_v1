from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def run_calculate_efs(
    mlp_command: str,
    model_path: Path,
    input_cfg: Path,
    output_cfg: Path,
) -> None:
    """
    mlp calculate_efs takes exactly 2 positional args:
        mlp calculate_efs model.almtp configs.cfg
    It overwrites configs.cfg in-place, adding MTP-predicted EFS alongside
    the existing DFT values.

    Strategy: copy input_cfg → output_cfg first, then run on the copy.
    That way input_cfg (DFT reference) is untouched and output_cfg holds
    the MTP-predicted version for parity comparison.
    """
    shutil.copy2(input_cfg, output_cfg)

    cmd = [
        mlp_command,
        "calculate_efs",
        str(model_path),
        str(output_cfg),       # overwritten in-place by mlp
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"calculate_efs failed.\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    if not output_cfg.exists():
        raise RuntimeError(f"calculate_efs ran but output not found: {output_cfg}")


def parse_cfg_efs(cfg_path: Path) -> list[dict]:
    """
    Parse BEGIN_CFG...END_CFG blocks, extracting:
      - energy per atom
      - all force components (fx, fy, fz) per atom
      - stress components if present (PlusStress block)
    Returns list of dicts, one per config.
    """
    text = cfg_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=BEGIN_CFG)", text)
    records = []

    for block in blocks:
        if "BEGIN_CFG" not in block:
            continue

        # natoms
        size_m = re.search(r"Size\s+(\d+)", block)
        natoms = int(size_m.group(1)) if size_m else None

        # energy — take the LAST Energy line in the block so MTP value
        # (written after DFT) is used when parsing predicted_cfg
        e_matches = list(re.finditer(r"Energy\s+([\d.eE+\-]+)", block))
        energy = float(e_matches[-1].group(1)) if e_matches else None

        # forces from AtomData block: cols are id type x y z fx fy fz ...
        forces: list[float] = []
        in_atoms = False
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("AtomData:"):
                in_atoms = True
                continue
            if in_atoms:
                if not s or any(
                    s.startswith(kw)
                    for kw in ("Energy", "PlusStress", "Stress", "Feature", "END_CFG")
                ):
                    in_atoms = False
                    continue
                cols = s.split()
                if len(cols) >= 8:
                    forces.extend([float(cols[5]), float(cols[6]), float(cols[7])])

        # stress from PlusStress block (six components on one or more lines)
        stress: list[float] | None = None
        ps_m = re.search(
            r"PlusStress[^\n]*\n([\s\S]*?)(?=\n\s*(?:Feature|END_CFG))",
            block,
        )
        if ps_m:
            stress_vals = [float(v) for v in ps_m.group(1).split()]
            if len(stress_vals) >= 6:
                stress = stress_vals[:6]

        if energy is not None and natoms:
            records.append({
                "energy_per_atom": energy / natoms,
                "forces": forces,
                "stress": stress,
            })

    return records


def build_parity_data(ref_records: list[dict], pred_records: list[dict]) -> dict:
    """
    Align reference and predicted records into flat lists for plotting.
    """
    n = min(len(ref_records), len(pred_records))
    energies_ref  = [ref_records[i]["energy_per_atom"] for i in range(n)]
    energies_pred = [pred_records[i]["energy_per_atom"] for i in range(n)]
    forces_ref    = [f for i in range(n) for f in ref_records[i]["forces"]]
    forces_pred   = [f for i in range(n) for f in pred_records[i]["forces"]]

    stress_ref, stress_pred = None, None
    if ref_records[0].get("stress") and pred_records[0].get("stress"):
        stress_ref  = [s for i in range(n) for s in (ref_records[i]["stress"] or [])]
        stress_pred = [s for i in range(n) for s in (pred_records[i]["stress"] or [])]

    return {
        "energies_ref":  energies_ref,
        "energies_pred": energies_pred,
        "forces_ref":    forces_ref,
        "forces_pred":   forces_pred,
        "stress_ref":    stress_ref,
        "stress_pred":   stress_pred,
    }
