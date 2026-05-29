"""Low-level LAMMPS helpers for the validate module.

Only builds input scripts and runs LAMMPS — no physics logic here.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# Utilities                                                            #
# ------------------------------------------------------------------ #

def _count_atoms(lammps_data: Path) -> Optional[int]:
    """Return the number of atoms declared in a LAMMPS data file.

    Looks for a line of the form ``<int> atoms`` near the top of the file.
    Returns None if the line is not found.
    """
    try:
        with lammps_data.open() as fh:
            for line in fh:
                m = re.match(r"^\s*(\d+)\s+atoms\s*$", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def _safe_mpi_np(
    requested: Optional[int],
    lammps_data: Optional[Path] = None,
) -> Optional[int]:
    """Return a safe MPI rank count for the given structure.

    LAMMPS domain decomposition requires that each sub-domain contains at
    least one atom.  For tiny unit cells (e.g. 4-atom FCC primitive cell)
    running with more ranks than atoms triggers MPI_ABORT.

    The returned value is ``min(requested, n_atoms)`` when the atom count
    can be determined, otherwise ``requested`` is returned unchanged.
    """
    if requested is None or requested <= 1:
        return requested
    if lammps_data is not None:
        n_atoms = _count_atoms(lammps_data)
        if n_atoms is not None and requested > n_atoms:
            return n_atoms
    return requested


# ------------------------------------------------------------------ #
# LAMMPS input builders                                                #
# ------------------------------------------------------------------ #

def write_eos_input(
    lammps_data: Path,
    model_path: Path,
    out_file: Path,
    *,
    element: str,
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    n_points: int = 21,
) -> Path:
    """Write a LAMMPS input script that computes energy at scaled volumes.

    For each volume point the lattice is uniformly scaled by a factor
    ``s`` in [scale_min, scale_max] and a single-point energy computed.
    Output is written to ``out_file`` as whitespace-separated lines::

        scale  vol_per_atom  energy_per_atom

    Returns the path of the written input script.

    Notes
    -----
    The MLIP-3 LAMMPS interface (interface-lammps-mlip-3) requires::

        pair_style  mlip load_from=<path>
        pair_coeff  * *

    The model path is passed via the ``load_from=`` keyword on the
    ``pair_style`` line; ``pair_coeff`` takes no element arguments.

    The undo scale step uses a named variable ``sinv = 1/s`` rather than
    the inline ``$(1/${s})`` expression, which is not valid in all LAMMPS
    builds (triggers "Invalid syntax in variable formula").
    """
    script_path = out_file.parent / "eos.in"
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {lammps_data}",
        "",
        f"pair_style      mlip load_from={model_path}",
        "pair_coeff      * *",
        "",
        "thermo_style    custom step vol pe",
        "thermo          1",
        "",
        f"variable        n_steps equal {n_points - 1}",
        f"variable        s_min   equal {scale_min}",
        f"variable        s_max   equal {scale_max}",
        "variable        ds      equal (v_s_max-v_s_min)/v_n_steps",
        "",
        f"print           \"# scale vol_per_atom energy_per_atom\" file {out_file} screen no",
        "",
        "variable        i       loop 0 ${n_steps}",
        "label           loop_start",
        "  variable      s       equal v_s_min+v_i*v_ds",
        "  variable      sinv    equal 1.0/v_s",
        "  change_box    all     x scale ${s} y scale ${s} z scale ${s} remap",
        "  run           0",
        "  variable      vpat    equal vol/atoms",
        "  variable      epat    equal pe/atoms",
        f" print         \"${{s}} ${{vpat}} ${{epat}}\" append {out_file} screen no",
        "  change_box    all     x scale ${sinv} y scale ${sinv} z scale ${sinv} remap",
        "next            i",
        "jump            SELF    loop_start",
    ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def write_elastic_input(
    lammps_data: Path,
    model_path: Path,
    out_file: Path,
    *,
    element: str,
    delta: float = 0.01,
) -> Path:
    """Write a LAMMPS input script that computes the stress tensor for
    six finite-difference strain states.

    Each strain state is applied, the stress tensor read, then the
    strain is reversed.  The six states are written to ``out_file`` as::

        strain_id  sxx  syy  szz  sxy  sxz  syz   (all in bar)

    where strain_id encodes the deformation (0=exx, 1=eyy, ..., 5=eyz).
    Elastic constants are computed in ``elastic.py`` from these stresses.

    Returns the path of the written input script.

    Notes
    -----
    The MLIP-3 LAMMPS interface (interface-lammps-mlip-3) requires::

        pair_style  mlip load_from=<path>
        pair_coeff  * *

    Inline ``$(1/(1+v_delta))`` is replaced by a named variable
    ``sinv = 1/(1+delta)`` for the same reason as in write_eos_input.
    """
    script_path = out_file.parent / "elastic.in"
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {lammps_data}",
        "",
        f"pair_style      mlip load_from={model_path}",
        "pair_coeff      * *",
        "",
        "minimize        1e-10 1e-12 10000 100000",
        "",
        "thermo_style    custom step pxx pyy pzz pxy pxz pyz",
        "thermo          1",
        "",
        f"variable        delta   equal {delta}",
        "variable        sinv    equal 1.0/(1.0+v_delta)",
        f"print           \"# strain_id sxx syy szz sxy sxz syz\" file {out_file} screen no",
        "",
    ]
    # Six strain states: exx, eyy, ezz, exy, exz, eyz
    # Shear strains (3-5) use +/- delta*lx/ly directly — no division needed.
    strains = [
        ("0", "x scale ${sdelta} remap",        "x scale ${sinv} remap"),
        ("1", "y scale ${sdelta} remap",        "y scale ${sinv} remap"),
        ("2", "z scale ${sdelta} remap",        "z scale ${sinv} remap"),
        ("3", "xy delta ${shear_d} remap",      "xy delta ${neg_shear_d} remap"),
        ("4", "xz delta ${shear_d} remap",      "xz delta ${neg_shear_d} remap"),
        ("5", "yz delta ${shear_dy} remap",     "yz delta ${neg_shear_dy} remap"),
    ]
    # Pre-compute helper variables before the strain loop
    lines += [
        "variable        sdelta       equal 1.0+v_delta",
        "variable        shear_d      equal v_delta*lx",
        "variable        neg_shear_d  equal -v_delta*lx",
        "variable        shear_dy     equal v_delta*ly",
        "variable        neg_shear_dy equal -v_delta*ly",
        "",
    ]
    for sid, apply_strain, undo_strain in strains:
        lines += [
            f"change_box      all {apply_strain}",
            "run             0",
            f"print           \"{sid} ${{pxx}} ${{pyy}} ${{pzz}} ${{pxy}} ${{pxz}} ${{pyz}}\""
            f" append {out_file} screen no",
            f"change_box      all {undo_strain}",
            "",
        ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


# ------------------------------------------------------------------ #
# Runner                                                               #
# ------------------------------------------------------------------ #

def run_lammps(
    script: Path,
    work_dir: Path,
    lammps_cmd: str = "lmp_mpi",
    mpi_command: Optional[str] = None,
    mpi_np: Optional[int] = None,
    log_file: Optional[Path] = None,
    lammps_data: Optional[Path] = None,
) -> None:
    """Execute LAMMPS for the given input script.

    Parameters
    ----------
    script:
        Path to the LAMMPS input file.
    work_dir:
        Working directory for the subprocess.
    lammps_cmd:
        LAMMPS executable name or path.
    mpi_command:
        MPI launcher (e.g. ``"mpirun"``).  None means run without MPI.
    mpi_np:
        Number of MPI ranks requested.  Automatically capped to the number
        of atoms in ``lammps_data`` to prevent domain-decomposition failures
        on small unit cells (e.g. 4-atom FCC primitive cell with a cutoff
        larger than the box).
    log_file:
        Path for the LAMMPS log file (passed via ``-log``).
    lammps_data:
        Path to the LAMMPS data file being simulated.  Used only to count
        atoms for the ``mpi_np`` safety cap; may be None.

    Raises RuntimeError if the process exits with a non-zero return code.
    """
    safe_np = _safe_mpi_np(mpi_np, lammps_data)

    cmd: list[str] = []
    if mpi_command:
        cmd += mpi_command.split()
        if safe_np is not None:
            cmd += ["-n", str(safe_np)]
    cmd += [lammps_cmd, "-in", str(script)]
    if log_file:
        cmd += ["-log", str(log_file)]

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:] if result.stderr else ''}"
        )
