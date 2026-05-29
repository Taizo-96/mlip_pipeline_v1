"""Low-level LAMMPS helpers for the validate module.

Only builds input scripts and runs LAMMPS — no physics logic here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


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
    """
    script_path = out_file.parent / "eos.in"
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {lammps_data}",
        "",
        f"pair_style      mlip {model_path}",
        f"pair_coeff      * * {element}",
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
        "  change_box    all     x scale ${s} y scale ${s} z scale ${s} remap",
        "  run           0",
        "  variable      vpat    equal vol/atoms",
        "  variable      epat    equal pe/atoms",
        f" print         \"${{s}} ${{vpat}} ${{epat}}\" append {out_file} screen no",
        "  change_box    all     x scale $(1/${s}) y scale $(1/${s}) z scale $(1/${s}) remap",
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
    """
    script_path = out_file.parent / "elastic.in"
    bars_per_gpa = 10000.0
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {lammps_data}",
        "",
        f"pair_style      mlip {model_path}",
        f"pair_coeff      * * {element}",
        "",
        "minimize        1e-10 1e-12 10000 100000",
        "",
        "thermo_style    custom step pxx pyy pzz pxy pxz pyz",
        "thermo          1",
        "",
        f"variable        delta   equal {delta}",
        f"print           \"# strain_id sxx syy szz sxy sxz syz\" file {out_file} screen no",
        "",
    ]
    # Six strain states: exx, eyy, ezz, exy, exz, eyz
    strains = [
        ("0", "x scale $(1+v_delta) remap",   "x scale $(1/(1+v_delta)) remap"),
        ("1", "y scale $(1+v_delta) remap",   "y scale $(1/(1+v_delta)) remap"),
        ("2", "z scale $(1+v_delta) remap",   "z scale $(1/(1+v_delta)) remap"),
        ("3", "xy delta $(v_delta*lx) remap", "xy delta $(-v_delta*lx) remap"),
        ("4", "xz delta $(v_delta*lx) remap", "xz delta $(-v_delta*lx) remap"),
        ("5", "yz delta $(v_delta*ly) remap", "yz delta $(-v_delta*ly) remap"),
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
) -> None:
    """Execute LAMMPS for the given input script.

    Raises RuntimeError if the process exits with a non-zero return code.
    """
    cmd: list[str] = []
    if mpi_command:
        cmd += mpi_command.split()
        if mpi_np is not None:
            cmd += ["-n", str(mpi_np)]
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
