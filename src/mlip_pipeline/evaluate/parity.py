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
    *,
    mpi_command: str | None = None,
    mpi_np: int | None = None,
) -> None:
    """
    mlp calculate_efs takes exactly 2 positional args:
        mlp calculate_efs model.almtp configs.cfg
    It overwrites configs.cfg in-place, adding MTP-predicted EFS alongside
    the existing DFT values.

    Strategy: copy input_cfg -> output_cfg first, then run on the copy.
    That way input_cfg (DFT reference) is untouched and output_cfg holds
    the MTP-predicted version for parity comparison.

    MPI: if mpi_command is provided (e.g. "mpirun" or "srun"), the call
    is wrapped as::

        <mpi_command> [-n <mpi_np>] mlp calculate_efs model cfg

    mpi_np is optional; omit it to let the MPI launcher use its default
    process count (e.g. all available cores, or the SLURM allocation).
    """
    shutil.copy2(input_cfg, output_cfg)

    cmd: list[str] = []
    if mpi_command:
        cmd.append(mpi_command)
        if mpi_np is not None:
            cmd.extend(["-n", str(mpi_np)])

    cmd.extend([
        mlp_command,
        "calculate_efs",
        str(model_path),
        str(output_cfg),       # overwritten in-place by mlp
    ])

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
    Only keep matched data and enforce equal x/y lengths for every channel.
    """
    n = min(len(ref_records), len(pred_records))

    energies_ref, energies_pred = [], []
    forces_ref, forces_pred = [], []
    stress_ref, stress_pred = [], []

    for i in range(n):
        ref = ref_records[i]
        pred = pred_records[i]

        # Energy: only if both are present
        if ref.get("energy_per_atom") is not None and pred.get("energy_per_atom") is not None:
            energies_ref.append(ref["energy_per_atom"])
            energies_pred.append(pred["energy_per_atom"])

        # Forces: trim per-config to matched component count
        ref_forces = ref.get("forces") or []
        pred_forces = pred.get("forces") or []
        nf = min(len(ref_forces), len(pred_forces))
        if nf > 0:
            forces_ref.extend(ref_forces[:nf])
            forces_pred.extend(pred_forces[:nf])

        # Stress: only include configs where both sides have stress
        ref_stress = ref.get("stress") or []
        pred_stress = pred.get("stress") or []
        ns = min(len(ref_stress), len(pred_stress))
        if ns > 0:
            stress_ref.extend(ref_stress[:ns])
            stress_pred.extend(pred_stress[:ns])

    return {
        "energies_ref": energies_ref,
        "energies_pred": energies_pred,
        "forces_ref": forces_ref,
        "forces_pred": forces_pred,
        "stress_ref": stress_ref if stress_ref else None,
        "stress_pred": stress_pred if stress_pred else None,
    }
