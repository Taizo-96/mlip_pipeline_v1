"""elastic_lammps.py -- Finite-difference elastic constants from LAMMPS.

Implements Eq. (Cij_finite_diff):
    C_ij = -(p_i^strained - p_i^ref) / delta

where p_i are LAMMPS pressure components (= -sigma_i) in bar,
converted to GPa on output.

Usage
-----
Standalone (generates LAMMPS input + runs if lammps_cmd given)::

    from mlip_pipeline.evaluate.elastic_lammps import compute_elastic_lammps
    results = compute_elastic_lammps(
        lammps_cmd="lmp_mpi",
        potential_file="MTP.mtp",
        poscar="POSCAR",
        output_dir="elastic_lammps",
    )

Or call parse_elastic_from_logs() directly if LAMMPS runs are already done.
"""

from __future__ import annotations

import logging
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Bar -> GPa
_BAR_TO_GPA = 1e-4

# Voigt index labels (LAMMPS order: xx yy zz xy xz yz)
VOIGT_LABELS = ["xx", "yy", "zz", "xy", "xz", "yz"]

# Six independent strain modes (Voigt): [exx, eyy, ezz, exy, exz, eyz]
STRAIN_MODES: dict[str, np.ndarray] = {
    "e1": np.array([1, 0, 0, 0, 0, 0], dtype=float),
    "e2": np.array([0, 1, 0, 0, 0, 0], dtype=float),
    "e3": np.array([0, 0, 1, 0, 0, 0], dtype=float),
    "e4": np.array([0, 0, 0, 1, 0, 0], dtype=float),
    "e5": np.array([0, 0, 0, 0, 1, 0], dtype=float),
    "e6": np.array([0, 0, 0, 0, 0, 1], dtype=float),
}


# ---------------------------------------------------------------------------
# POSCAR reader (minimal, reuses logic from elastic_constants.py)
# ---------------------------------------------------------------------------

def _read_poscar(path: str | Path) -> dict:
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].strip())
    lattice = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale
    try:
        counts = list(map(int, lines[5].split()))
        species = None
        idx = 5
    except ValueError:
        species = lines[5].split()
        counts = list(map(int, lines[6].split()))
        idx = 6
    coord_type = lines[idx + 1].strip()
    n_atoms = sum(counts)
    coords = np.array([list(map(float, lines[idx + 2 + i].split()[:3])) for i in range(n_atoms)])
    return dict(lattice=lattice, species=species, counts=counts,
                coord_type=coord_type, coords=coords, scale=1.0, comment=lines[0])


def _apply_voigt_strain(lattice: np.ndarray, voigt: np.ndarray, delta: float) -> np.ndarray:
    e = np.zeros((3, 3))
    e[0, 0] = voigt[0] * delta
    e[1, 1] = voigt[1] * delta
    e[2, 2] = voigt[2] * delta
    e[0, 1] = e[1, 0] = 0.5 * voigt[3] * delta
    e[0, 2] = e[2, 0] = 0.5 * voigt[4] * delta
    e[1, 2] = e[2, 1] = 0.5 * voigt[5] * delta
    return lattice @ (np.eye(3) + e).T


# ---------------------------------------------------------------------------
# LAMMPS input generator
# ---------------------------------------------------------------------------

def _write_lammps_data(poscar: dict, path: Path) -> None:
    """Write a minimal LAMMPS data file (atomic style) from a POSCAR dict."""
    lat = poscar["lattice"]
    coords = poscar["coords"]
    counts = poscar["counts"]
    n_atoms = sum(counts)
    is_direct = poscar["coord_type"].lower().startswith("d")

    if is_direct:
        cart = coords @ lat
    else:
        cart = coords

    # Triclinic parameters
    a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
    ax = float(np.linalg.norm(a_vec))
    bx = float(np.dot(a_vec, b_vec) / ax)
    by = float(np.sqrt(max(np.dot(b_vec, b_vec) - bx**2, 0)))
    cx = float(np.dot(a_vec, c_vec) / ax)
    cy = float((np.dot(b_vec, c_vec) - bx * cx) / by) if by > 1e-12 else 0.0
    cz = float(np.sqrt(max(np.dot(c_vec, c_vec) - cx**2 - cy**2, 0)))

    n_types = len(counts)
    atom_types = []
    for i, cnt in enumerate(counts):
        atom_types.extend([i + 1] * cnt)

    with path.open("w") as f:
        f.write("LAMMPS data file written by mlip_pipeline\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"0.0 {ax:.10f} xlo xhi\n")
        f.write(f"0.0 {by:.10f} ylo yhi\n")
        f.write(f"0.0 {cz:.10f} zlo zhi\n")
        f.write(f"{bx:.10f} {cx:.10f} {cy:.10f} xy xz yz\n\n")
        f.write("Masses\n\n")
        for i in range(n_types):
            f.write(f"{i+1} 1.0\n")  # placeholder mass
        f.write("\nAtoms\n\n")
        for i, (atype, xyz) in enumerate(zip(atom_types, cart), 1):
            f.write(f"{i} {atype} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def _write_lammps_input(data_file: Path, potential_file: Path, out_log: Path,
                        potential_type: str = "mlip") -> Path:
    """Write a minimal LAMMPS input for a static stress evaluation."""
    inp = data_file.parent / "in.elastic"
    pot_str = (
        f"pair_style mlip load_from={potential_file.resolve()}\n"
        f"pair_coeff * *"
    )
    inp.write_text(f"""# Static stress evaluation for elastic constants
units metal
atom_style atomic
boundary p p p

read_data {data_file.resolve()}

{pot_str}

thermo 1
thermo_style custom step pxx pyy pzz pxy pxz pyz pe

run 0
""")
    return inp


# ---------------------------------------------------------------------------
# LAMMPS log parser
# ---------------------------------------------------------------------------

def parse_pressures_from_log(log_path: str | Path) -> np.ndarray | None:
    """
    Parse the final thermo line from a LAMMPS log.
    Returns 6-vector [pxx, pyy, pzz, pxy, pxz, pyz] in GPa.
    LAMMPS reports in bar (metal units); converted here.
    """
    text = Path(log_path).read_text()
    # Find last thermo block
    header_pat = re.compile(r"Step\s+Pxx\s+Pyy\s+Pzz\s+Pxy\s+Pxz\s+Pyz", re.IGNORECASE)
    match = None
    for m in header_pat.finditer(text):
        match = m
    if match is None:
        logger.warning("No thermo header found in %s", log_path)
        return None
    after = text[match.end():]
    # First data line after header
    for line in after.splitlines():
        line = line.strip()
        if re.match(r"^\d+", line):
            vals = line.split()
            if len(vals) >= 7:
                p_bar = np.array(list(map(float, vals[1:7])))
                return p_bar * _BAR_TO_GPA  # bar -> GPa
    logger.warning("No thermo data line found in %s", log_path)
    return None


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def generate_lammps_runs(
    poscar: str | Path,
    potential_file: str | Path,
    output_dir: str | Path,
    delta: float = 0.005,
) -> list[Path]:
    """Create one LAMMPS run folder per strain mode (6 modes + 1 reference)."""
    poscar_dict = _read_poscar(poscar)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    potential_file = Path(potential_file)

    folders = []
    # Reference (unstrained)
    for tag, mode_vec in {"ref": np.zeros(6), **STRAIN_MODES}.items():
        folder = output_dir / tag
        folder.mkdir(exist_ok=True)
        strained = dict(poscar_dict)
        if tag != "ref":
            strained["lattice"] = _apply_voigt_strain(poscar_dict["lattice"], mode_vec, delta)
        data_path = folder / "structure.lammps"
        _write_lammps_data(strained, data_path)
        _write_lammps_input(data_path, potential_file, folder / "log.lammps")
        folders.append(folder)
        logger.debug("Created LAMMPS folder: %s", folder)

    return folders


def run_lammps_jobs(folders: list[Path], lammps_cmd: str = "lmp") -> None:
    """Run LAMMPS in each folder sequentially."""
    for folder in folders:
        inp = folder / "in.elastic"
        log = folder / "log.lammps"
        cmd = f"{lammps_cmd} -in {inp} -log {log}"
        logger.info("Running: %s", cmd)
        result = subprocess.run(cmd, shell=True, cwd=folder,
                                capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("LAMMPS failed in %s:\n%s", folder, result.stderr[-500:])


def parse_elastic_from_logs(
    output_dir: str | Path,
    delta: float = 0.005,
) -> dict:
    """
    Read LAMMPS log files and compute Cij via finite differences.

    C_ij = -(p_i^strained - p_i^ref) / delta

    Returns dict with C_matrix (6x6, GPa), C11/C12/C44/C33 (cubic check),
    B_voigt, G_voigt, and raw pressures.
    """
    root = Path(output_dir)
    ref_log = root / "ref" / "log.lammps"
    p_ref = parse_pressures_from_log(ref_log)
    if p_ref is None:
        raise RuntimeError(f"Could not parse reference pressures from {ref_log}")

    C = np.zeros((6, 6))
    raw: dict = {"ref": p_ref.tolist()}

    for j, mode_name in enumerate(STRAIN_MODES):
        log_path = root / mode_name / "log.lammps"
        p_strained = parse_pressures_from_log(log_path)
        if p_strained is None:
            logger.warning("Missing pressures for mode %s", mode_name)
            continue
        raw[mode_name] = p_strained.tolist()
        # Eq. (Cij_finite_diff): C_ij = -(p_i^strained - p_i^ref) / delta
        C[:, j] = -(p_strained - p_ref) / delta

    results = _build_results(C)
    results["raw_pressures"] = raw
    results["delta"] = delta
    return results


def _build_results(C: np.ndarray) -> dict:
    results: dict = {"C_matrix": C, "voigt_labels": VOIGT_LABELS}
    results["C11"] = float(C[0, 0])
    results["C22"] = float(C[1, 1])
    results["C33"] = float(C[2, 2])
    results["C12"] = float(C[0, 1])
    results["C13"] = float(C[0, 2])
    results["C23"] = float(C[1, 2])
    results["C44"] = float(C[3, 3])
    results["C55"] = float(C[4, 4])
    results["C66"] = float(C[5, 5])
    # Voigt averages (Eqs. voigt_bv, voigt_gv)
    results["B_voigt"] = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9
    results["G_voigt"] = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[0,2]+C[1,2])
                          + 3*(C[3,3]+C[4,4]+C[5,5])) / 15
    return results


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def compute_elastic_lammps(
    lammps_cmd: str,
    potential_file: str | Path,
    poscar: str | Path,
    output_dir: str | Path = "elastic_lammps",
    delta: float = 0.005,
    reuse_existing: bool = False,
) -> dict:
    """Generate runs, execute LAMMPS, parse results. Returns Cij dict."""
    output_dir = Path(output_dir)
    ref_log = output_dir / "ref" / "log.lammps"

    if not reuse_existing or not ref_log.exists():
        folders = generate_lammps_runs(poscar, potential_file, output_dir, delta)
        run_lammps_jobs(folders, lammps_cmd)
    else:
        logger.info("Reusing existing LAMMPS logs in %s", output_dir)

    return parse_elastic_from_logs(output_dir, delta)
