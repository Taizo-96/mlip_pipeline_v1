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
    """Return the number of atoms declared in a LAMMPS data file."""
    try:
        with lammps_data.open() as fh:
            for line in fh:
                m = re.match(r"^\s*(\d+)\s+atoms\s*$", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def _box_lengths(lammps_data: Path) -> Optional[tuple[float, float, float]]:
    """Return (Lx, Ly, Lz) box lengths from a LAMMPS data file."""
    lengths: list[float] = []
    pattern = re.compile(
        r"^\s*(-?[\d.eE+\-]+)\s+(-?[\d.eE+\-]+)\s+(xlo xhi|ylo yhi|zlo zhi)"
    )
    try:
        with lammps_data.open() as fh:
            for line in fh:
                m = pattern.match(line)
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    lengths.append(hi - lo)
                    if len(lengths) == 3:
                        return (lengths[0], lengths[1], lengths[2])
    except OSError:
        pass
    return None


def _safe_mpi_np(
    requested: Optional[int],
    lammps_data: Optional[Path] = None,
    cutoff: Optional[float] = None,
) -> int:
    """Return a safe MPI rank count for the given structure."""
    if requested is None or requested <= 1:
        return requested if requested is not None else 1

    cap = requested

    if lammps_data is not None:
        n_atoms = _count_atoms(lammps_data)
        if n_atoms is not None:
            cap = min(cap, n_atoms)

        if cutoff is not None and cutoff > 0:
            box = _box_lengths(lammps_data)
            if box is not None:
                L_min = min(box)
                if L_min < cutoff:
                    cap = 1

    return max(1, cap)


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
    """Write a LAMMPS input script that computes energy at scaled volumes."""
    script_path = out_file.parent / "eos.in"
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    abs_out   = Path(out_file).resolve()
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
        f"pair_style      mlip load_from={abs_model}",
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
        f"print           \"# scale vol_per_atom energy_per_atom\" file {abs_out} screen no",
        "",
        "variable        i       loop 0 ${n_steps}",
        "label           loop_start",
        "  variable      s       equal v_s_min+v_i*v_ds",
        "  variable      sinv    equal 1.0/v_s",
        "  change_box    all     x scale ${s} y scale ${s} z scale ${s} remap",
        "  run           0",
        "  variable      vpat    equal vol/atoms",
        "  variable      epat    equal pe/atoms",
        f"  print         \"${{s}} ${{vpat}} ${{epat}}\" append {abs_out} screen no",
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
    """Write a LAMMPS input script for elastic constants via finite-difference stress.

    Strategy
    --------
    1. Read structure, set up potential.
    2. Minimize to true equilibrium.
    3. Convert box to triclinic (required for xy/xz/yz shear strains).
    4. Declare ALL variables (must precede any change_box that uses them).
    5. Apply each of the 6 strain states, run 0, print stress, undo strain.

    The 'change_box all triclinic' command converts an orthogonal box
    (as read from a standard LAMMPS data file) to triclinic form so that
    shear tilt components (xy, xz, yz) can be manipulated.  Without this,
    LAMMPS aborts with "Cannot change box to orthogonal" or similar.
    """
    script_path = out_file.parent / "elastic.in"
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    abs_out   = Path(out_file).resolve()

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
        f"pair_style      mlip load_from={abs_model}",
        "pair_coeff      * *",
        "",
        # Minimize to true equilibrium before straining
        "minimize        1e-10 1e-12 10000 100000",
        "",
        # Convert orthogonal box -> triclinic so shear change_box commands work
        "change_box      all triclinic",
        "",
        "thermo_style    custom step pxx pyy pzz pxy pxz pyz",
        "thermo          1",
        "",
        # --- ALL variable declarations BEFORE any change_box strain ---
        f"variable        delta        equal {delta}",
        "variable        sdelta       equal 1.0+v_delta",
        "variable        sinv_normal  equal 1.0/(1.0+v_delta)",
        # Shear displacement = delta * current box length (equal-style = live)
        "variable        shear_d      equal v_delta*lx",
        "variable        neg_shear_d  equal -v_delta*lx",
        "variable        shear_dy     equal v_delta*ly",
        "variable        neg_shear_dy equal -v_delta*ly",
        "",
        f"print           \"# strain_id sxx syy szz sxy sxz syz\" file {abs_out} screen no",
        "",
    ]

    # Six Voigt strain states
    strains = [
        ("0", "x scale ${sdelta} remap",         "x scale ${sinv_normal} remap"),
        ("1", "y scale ${sdelta} remap",         "y scale ${sinv_normal} remap"),
        ("2", "z scale ${sdelta} remap",         "z scale ${sinv_normal} remap"),
        ("3", "xy delta ${shear_d} remap",       "xy delta ${neg_shear_d} remap"),
        ("4", "xz delta ${shear_d} remap",       "xz delta ${neg_shear_d} remap"),
        ("5", "yz delta ${shear_dy} remap",      "yz delta ${neg_shear_dy} remap"),
    ]

    for sid, apply_strain, undo_strain in strains:
        lines += [
            f"change_box      all {apply_strain}",
            "run             0",
            f"print           \"{sid} ${{pxx}} ${{pyy}} ${{pzz}} ${{pxy}} ${{pxz}} ${{pyz}}\""
            f" append {abs_out} screen no",
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
    cutoff: Optional[float] = None,
) -> None:
    """Execute LAMMPS for the given input script."""
    safe_np = _safe_mpi_np(mpi_np, lammps_data, cutoff=cutoff)

    cmd: list[str] = []
    if mpi_command:
        cmd += mpi_command.split()
        if safe_np is not None:
            cmd += ["-n", str(safe_np)]
    cmd += [lammps_cmd, "-in", str(Path(script).resolve())]
    if log_file:
        cmd += ["-log", str(Path(log_file).resolve())]

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
