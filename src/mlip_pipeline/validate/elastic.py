"""Elastic constant runner (finite-difference stress method)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import ElasticResult
from mlip_pipeline.validate.lammps import write_elastic_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir

# 1 bar = 0.0001 GPa
_BAR_TO_GPA = 1e-4


# ------------------------------------------------------------------ #
# Output parser                                                         #
# ------------------------------------------------------------------ #

def _parse_stress_output(out_file: Path) -> dict[int, list[float]]:
    """Parse the LAMMPS elastic stress output.

    Each non-comment line: strain_id sxx syy szz sxy sxz syz (in bar)

    strain_id=-1 is the reference (zero-strain) state.

    Returns dict mapping strain_id → [sxx, syy, szz, sxy, sxz, syz] in GPa.
    """
    stresses: dict[int, list[float]] = {}
    for line in out_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            sid = int(parts[0])
            s = [float(x) * _BAR_TO_GPA for x in parts[1:7]]
            stresses[sid] = s
        except ValueError:
            continue
    return stresses


# ------------------------------------------------------------------ #
# Elastic constant extraction (Voigt notation)                          #
# ------------------------------------------------------------------ #

def _compute_elastic_constants(
    stresses: dict[int, list[float]],
    delta: float,
) -> dict[str, float]:
    """Extract Cij from finite-difference stresses.

    Cij = (sigma_i_strained - sigma_i_ref) / epsilon_j

    Strain indices (as written by write_elastic_input):
        id -1 → reference (no strain)
        id  0 → exx = delta
        id  1 → eyy = delta
        id  2 → ezz = delta
        id  3 → exy = delta  (engineering shear)
        id  4 → exz = delta
        id  5 → eyz = delta

    Stress vector components: [sxx, syy, szz, sxy, sxz, syz]
    """
    ref = stresses.get(-1, [0.0] * 6)

    def _ds(strain_id: int, component: int) -> float:
        """(sigma_strained - sigma_ref)[component] / delta."""
        s = stresses.get(strain_id, [0.0] * 6)
        return (s[component] - ref[component]) / delta

    C: dict[str, float] = {}
    # Normal strains
    C["C11"] = _ds(0, 0)  # dsxx / dexx
    C["C22"] = _ds(1, 1)  # dsyy / deyy
    C["C33"] = _ds(2, 2)  # dszz / dezz
    C["C12"] = _ds(0, 1)  # dsyy / dexx
    C["C13"] = _ds(0, 2)  # dszz / dexx
    C["C23"] = _ds(1, 2)  # dszz / deyy
    # Shear strains (engineering strain = delta exactly, by construction in lammps.py)
    C["C44"] = _ds(5, 5)  # dsyz / deyz
    C["C55"] = _ds(4, 4)  # dsxz / dexz
    C["C66"] = _ds(3, 3)  # dsxy / dexy

    return C


def _voigt_moduli(C: dict[str, float]) -> tuple[float, float]:
    """Compute Voigt bulk and shear moduli from Cij (GPa).

    Voigt B = (C11+C22+C33 + 2*(C12+C13+C23)) / 9
    Voigt G = (C11+C22+C33 - C12-C13-C23 + 3*(C44+C55+C66)) / 15
    """
    c11, c22, c33 = C.get("C11", 0), C.get("C22", 0), C.get("C33", 0)
    c12, c13, c23 = C.get("C12", 0), C.get("C13", 0), C.get("C23", 0)
    c44, c55, c66 = C.get("C44", 0), C.get("C55", 0), C.get("C66", 0)

    B = (c11 + c22 + c33 + 2 * (c12 + c13 + c23)) / 9.0
    G = (c11 + c22 + c33 - c12 - c13 - c23 + 3 * (c44 + c55 + c66)) / 15.0
    return B, G


# ------------------------------------------------------------------ #
# Public runner                                                         #
# ------------------------------------------------------------------ #

def run_elastic(
    structure_id: str,
    lammps_data: Path,
    model_path: Path,
    validate_dir: Path,
    *,
    element: str,
    lammps_cmd: str = "lmp_mpi",
    mpi_command: Optional[str] = None,
    mpi_np: Optional[int] = None,
    delta: float = 0.01,
    cutoff: Optional[float] = None,
) -> ElasticResult:
    """Compute elastic constants and return an ElasticResult."""
    work_dir = ensure_dir(validate_dir / "elastic" / structure_id)
    out_file = work_dir / "stress_data.txt"

    if out_file.exists():
        out_file.unlink()

    print(f"  [elastic] {structure_id}: writing LAMMPS input (delta={delta}) ...")
    script = write_elastic_input(
        lammps_data, model_path, out_file,
        element=element,
        delta=delta,
    )

    try:
        print(f"  [elastic] {structure_id}: running LAMMPS ...")
        run_lammps(
            script, work_dir,
            lammps_cmd=lammps_cmd,
            mpi_command=mpi_command,
            mpi_np=mpi_np,
            log_file=work_dir / "lammps.log",
            lammps_data=lammps_data,
            cutoff=cutoff,
        )
    except RuntimeError as exc:
        print(f"  [elastic] WARNING: LAMMPS failed for {structure_id}: {exc}")
        return ElasticResult(structure_id=structure_id, error=str(exc))

    stresses = _parse_stress_output(out_file)
    # Expect reference (-1) + 6 strain states
    if len(stresses) < 7 or -1 not in stresses:
        msg = f"only {len(stresses)}/7 stress states parsed (missing reference?)"
        print(f"  [elastic] WARNING: {structure_id}: {msg}")
        return ElasticResult(structure_id=structure_id, error=msg)

    try:
        C = _compute_elastic_constants(stresses, delta)
        B, G = _voigt_moduli(C)
        print(
            f"  [elastic] {structure_id}: "
            + "  ".join(f"{k}={v:.1f}" for k, v in sorted(C.items()))
            + f"  B_V={B:.1f} GPa  G_V={G:.1f} GPa"
        )
        return ElasticResult(
            structure_id=structure_id,
            C=C,
            B_voigt=B,
            G_voigt=G,
            compute_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [elastic] WARNING: Cij extraction failed for {structure_id}: {exc}")
        return ElasticResult(structure_id=structure_id, error=str(exc))
