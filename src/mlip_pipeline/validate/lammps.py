"""Low-level LAMMPS helpers for the validate module."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# Utilities                                                            #
# ------------------------------------------------------------------ #

def _count_atoms(lammps_data: Path) -> Optional[int]:
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
    supercell_repeat: int = 1,
) -> int:
    """Return safe MPI rank count, accounting for supercell replication."""
    if requested is None or requested <= 1:
        return requested if requested is not None else 1
    cap = requested
    rep = max(1, supercell_repeat)
    if lammps_data is not None:
        n_atoms = _count_atoms(lammps_data)
        if n_atoms is not None:
            n_sim = n_atoms * rep ** 2 * (rep * 2)  # NxNx2N for melting, else N^3
            cap = min(cap, n_sim)
        if cutoff is not None and cutoff > 0:
            box = _box_lengths(lammps_data)
            if box is not None:
                scaled_box = tuple(l * rep for l in box)
                if min(scaled_box) < cutoff:
                    cap = 1
    return max(1, cap)


# ------------------------------------------------------------------ #
# LAMMPS input builders                                                #
# ------------------------------------------------------------------ #

def _pair_block(model_path: Path) -> list[str]:
    return [
        f"pair_style      mlip load_from={model_path.resolve()}",
        "pair_coeff      * *",
    ]


def write_eos_input(
    lammps_data: Path, model_path: Path, out_file: Path,
    *, element: str, scale_min: float = 0.85, scale_max: float = 1.15,
    n_points: int = 21,
) -> Path:
    script_path = out_file.parent / "eos.in"
    abs_data  = Path(lammps_data).resolve()
    abs_out   = Path(out_file).resolve()
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
    ] + _pair_block(model_path) + [
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
    lammps_data: Path, model_path: Path, out_file: Path,
    *, element: str, delta: float = 0.01,
) -> Path:
    script_path = out_file.parent / "elastic.in"
    abs_data = Path(lammps_data).resolve()
    abs_out  = Path(out_file).resolve()
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
    ] + _pair_block(model_path) + [
        "",
        "minimize        1e-10 1e-12 10000 100000",
        "",
        "change_box      all triclinic",
        "",
        "thermo_style    custom step pxx pyy pzz pxy pxz pyz",
        "thermo          1",
        "",
        f"variable        delta        equal {delta}",
        "variable        sdelta       equal 1.0+v_delta",
        "variable        sinv_normal  equal 1.0/(1.0+v_delta)",
        "variable        shear_dxy    equal v_delta*ly",
        "variable        neg_shear_dxy equal -v_delta*ly",
        "variable        shear_dxz    equal v_delta*lz",
        "variable        neg_shear_dxz equal -v_delta*lz",
        "variable        shear_dyz    equal v_delta*lz",
        "variable        neg_shear_dyz equal -v_delta*lz",
        "",
        "variable        v_pxx        equal pxx",
        "variable        v_pyy        equal pyy",
        "variable        v_pzz        equal pzz",
        "variable        v_pxy        equal pxy",
        "variable        v_pxz        equal pxz",
        "variable        v_pyz        equal pyz",
        "",
        f"print           \"# strain_id sxx syy szz sxy sxz syz\" file {abs_out} screen no",
        "",
        "run             0",
        f"print           \"-1 ${{v_pxx}} ${{v_pyy}} ${{v_pzz}}"
        f" ${{v_pxy}} ${{v_pxz}} ${{v_pyz}}\""
        f" append {abs_out} screen no",
        "",
    ]
    strains = [
        ("0", "x scale ${sdelta} remap",        "x scale ${sinv_normal} remap"),
        ("1", "y scale ${sdelta} remap",        "y scale ${sinv_normal} remap"),
        ("2", "z scale ${sdelta} remap",        "z scale ${sinv_normal} remap"),
        ("3", "xy delta ${shear_dxy} remap",    "xy delta ${neg_shear_dxy} remap"),
        ("4", "xz delta ${shear_dxz} remap",    "xz delta ${neg_shear_dxz} remap"),
        ("5", "yz delta ${shear_dyz} remap",    "yz delta ${neg_shear_dyz} remap"),
    ]
    for sid, apply_s, undo_s in strains:
        lines += [
            f"change_box      all {apply_s}",
            "run             0",
            f"print           \"{sid} ${{v_pxx}} ${{v_pyy}} ${{v_pzz}}"
            f" ${{v_pxy}} ${{v_pxz}} ${{v_pyz}}\""
            f" append {abs_out} screen no",
            f"change_box      all {undo_s}",
            "",
        ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def write_melting_input(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperature: float,
    n_solid: int,
    n_liquid: int,
    supercell_repeat: int = 5,
    dt: float = 0.002,
    n_equil: int = 5000,
    n_prod: int = 20000,
    seed: int = 12345,
) -> Path:
    """Write a two-phase coexistence LAMMPS input script.

    Disordering protocol
    --------------------
    The liquid half is disordered using NVE + fix temp/rescale rather than
    a tight NVT thermostat.  temp/rescale injects kinetic energy every N steps
    without fighting the MTP forces, which prevents the force spikes and atom
    ejections seen with NVT ramping near T_melt.

    Stage 1 – Liquid disordering (dt_heat = dt/20, n_disorder steps):
      * Solid half: fix nvt at T (standard thermostat).
      * Liquid half: fix nve + fix temp/rescale every 10 steps up to T+50 K.
        The tiny timestep (0.0001 ps) and velocity rescaling keep forces
        manageable while gently disordering the liquid region.
      * thermo_modify lost ignore throughout disordering.
      * Adaptive timestep (fix dt/reset) as a further safety net.

    Stage 2 – Whole-system NVT equilibration (dt_heat, n_equil steps):
      * Both halves in a single NVT fix at T.
      * Still at reduced timestep; lost ignore still active.

    Stage 3 – Production NVT (full dt, n_prod steps):
      * reset_atoms id (new LAMMPS >=22Jul2023 syntax) to close any ID gaps
        left by lost ignore before velocity reinitialisation.
      * velocity all create T seed+1.
      * thermo_modify lost ignore kept (never switch to lost error for
        two-phase runs — any remaining ejections simply reflect a potential
        that cannot hold the liquid at this T, which is useful information).
      * fix nvt on the whole cell; thermo + file output every 50 steps.

    Atom-loss sentinel
    ------------------
    After production the script writes the final atom count to
    ``atom_count.txt``.  The Python caller reads this and emits a warning
    if >10 % of atoms were lost, treating the temperature point as failed.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    thermo_out  = (work_dir / "coex_thermo.txt").resolve()
    natom_out   = (work_dir / "atom_count.txt").resolve()
    script_path = work_dir / "melting.in"

    # Timestep for disordering: 10x smaller than production dt.
    dt_heat   = dt / 20.0          # 0.0001 ps when dt=0.002
    T_liq     = temperature + 50.0  # target for liquid rescaling (mild overshoot)
    n_dis     = max(1000, n_equil // 2)   # disordering steps
    n_eq      = max(500,  n_equil // 4)   # whole-system equilibration steps

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat * 2}",
        "",
        f"pair_style      mlip load_from={abs_model}",
        "pair_coeff      * *",
        "",
        "# Increase neighbour list capacity for dense/disordered configs.",
        "neigh_modify    one 4000 page 100000",
        "",
        "# Minimise before any dynamics to remove any residual forces.",
        "minimize        1e-8 1e-10 5000 50000",
        "",
        "# Split into solid (lo-z) and liquid (hi-z) halves.",
        "# Epsilon offset prevents atoms sitting exactly on the boundary.",
        "variable        Lz     equal lz",
        "variable        zmid   equal (zlo+zhi)/2.0 + 1e-6*v_Lz",
        "region          solid_region  block INF INF INF INF INF ${zmid} units box",
        "region          liquid_region block INF INF INF INF ${zmid} INF units box",
        "group           solid_atoms   region solid_region",
        "group           liquid_atoms  region liquid_region",
        "",
        "# Record the initial atom count for the loss check after production.",
        "variable        N0 equal atoms",
        "",
        f"timestep        {dt_heat}",
        "thermo_modify   flush yes lost ignore",
        "thermo          200",
        "",
        "# ----------------------------------------------------------------",
        "# Stage 1: Liquid disordering via NVE + temp/rescale.",
        "# temp/rescale injects KE every Nevery steps without fighting",
        "# MTP forces, avoiding the runaway ejections seen with NVT ramps.",
        "# ----------------------------------------------------------------",
        f"velocity        all create {temperature:.1f} {seed} dist gaussian",
        "",
        "# Adaptive timestep safety net (shrinks dt if any atom moves >0.02 Ang).",
        f"fix             fxDT all dt/reset 1 {dt_heat*0.1:.6f} {dt_heat:.6f} 0.02 units box",
        "",
        "# Solid half: standard NVT at target T.",
        f"fix             fxS_dis solid_atoms nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        "# Liquid half: NVE dynamics + velocity rescaling every 10 steps.",
        "# temp/rescale args: Nevery T_start T_stop T_window fraction",
        "#   T_window = 20 K tolerance band, fraction = 1.0 (rescale all KE)",
        f"fix             fxL_nve  liquid_atoms nve",
        f"fix             fxL_rsc  liquid_atoms temp/rescale 10 {T_liq:.1f} {T_liq:.1f} 20.0 1.0",
        f"run             {n_dis}",
        "unfix           fxS_dis",
        "unfix           fxL_nve",
        "unfix           fxL_rsc",
        "unfix           fxDT",
        "",
        "# ----------------------------------------------------------------",
        "# Stage 2: Whole-system NVT equilibration at target T.",
        "# ----------------------------------------------------------------",
        f"fix             fxEQ all nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        f"run             {n_eq}",
        "unfix           fxEQ",
        "",
        "# ----------------------------------------------------------------",
        "# Stage 3: Production NVT.",
        "# reset_atoms id closes ID gaps left by lost ignore before velocity",
        "# reinitialisation (required for `velocity all create ... loop all`).",
        "# ----------------------------------------------------------------",
        "reset_atoms     id",
        "",
        f"velocity        all create {temperature:.1f} {seed + 1} dist gaussian",
        "",
        f"timestep        {dt}",
        "# Keep lost ignore in production: ejections signal instability at",
        "# this T, which the Python caller detects via atom_count.txt.",
        "thermo_modify   flush yes lost ignore",
        "",
        f"fix             fxNVT all nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        "",
        "thermo_style    custom step temp vol pe atoms",
        "thermo          50",
        "",
        f"print           \"# step temp vol pe\" file {thermo_out} screen no",
        f"fix             fxPrint all print 50 "
        f"\"$(step) $(temp) $(vol) $(pe)\" append {thermo_out} screen no",
        "",
        f"run             {n_prod}",
        "unfix           fxNVT",
        "unfix           fxPrint",
        "",
        "# Write final atom count so Python can detect excessive atom loss.",
        f"print           \"${{atoms}}\" file {natom_out} screen no",
    ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def write_thermal_expansion_input(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperatures: list[float],
    dt: float = 0.002,
    n_equil: int = 5000,
    n_prod: int = 10000,
    seed: int = 42,
) -> Path:
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    out_file  = (work_dir / "thexp_output.txt").resolve()
    script_path = work_dir / "thexp.in"

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
        "minimize        1e-10 1e-12 10000 100000",
        "",
        f"timestep        {dt}",
        "thermo          100",
        "",
        f"print           \"# T vol_per_atom\" file {out_file} screen no",
        "",
    ]
    for i, T in enumerate(temperatures):
        lines += [
            f"# --- T = {T:.1f} K ---",
            f"velocity        all create {T:.1f} {seed + i} dist gaussian",
            f"fix             fxNPT all npt temp {T:.1f} {T:.1f} $(100*dt) "
            f"iso 0.0 0.0 $(1000*dt)",
            f"run             {n_equil}",
            f"run             {n_prod}",
            "variable        vpat equal vol/atoms",
            f"print           \"{T:.1f} ${{vpat}}\" append {out_file} screen no",
            "unfix           fxNPT",
            "",
        ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def write_vacancy_inputs(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    supercell_repeat: int = 3,
) -> tuple[Path, Path]:
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()

    def _write(name: str, remove_atom: bool) -> Path:
        out_e = (work_dir / f"{name}_energy.txt").resolve()
        script = work_dir / f"{name}.in"
        lines = [
            "units           metal",
            "atom_style      atomic",
            "boundary        p p p",
            "",
            f"read_data       {abs_data}",
            f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat}",
            "",
            f"pair_style      mlip load_from={abs_model}",
            "pair_coeff      * *",
            "",
        ]
        if remove_atom:
            lines += [
                "# remove atom with lowest ID (creates vacancy)",
                "group           vac_atom id 1",
                "delete_atoms    group vac_atom",
                "",
            ]
        lines += [
            "minimize        1e-10 1e-12 10000 100000",
            "",
            "variable        etot equal pe",
            "variable        natoms equal atoms",
            f"print           \"${{etot}} ${{natoms}}\" file {out_e} screen no",
        ]
        script.write_text("\n".join(lines) + "\n")
        return script

    return _write("perfect", False), _write("vacancy", True)


def write_rdf_input(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperature: float = 300.0,
    dt: float = 0.002,
    n_equil: int = 5000,
    n_prod: int = 20000,
    r_max: float = 8.0,
    n_bins: int = 200,
    seed: int = 99,
    supercell_repeat: int = 3,
) -> Path:
    """NVT MD run followed by compute rdf accumulation.

    ave/time parameters:
      Nevery  = 10          sample every 10 steps
      Nfreq   = n_prod//10  output every n_prod//10 steps (10 output blocks)
      Nrepeat = Nfreq//10   average Nrepeat samples into each output block

    This ensures Nevery * Nrepeat <= Nfreq (LAMMPS requirement) and that
    the compute fires multiple times during the production run.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    out_file  = (work_dir / "rdf_output.txt").resolve()
    script_path = work_dir / "rdf.in"

    n_every  = 10
    n_freq   = max(n_every * 2, n_prod // 10)   # output 10 times during production
    n_repeat = n_freq // n_every                  # Nevery * Nrepeat == Nfreq

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat}",
        "",
        f"pair_style      mlip load_from={abs_model}",
        "pair_coeff      * *",
        "",
        "minimize        1e-10 1e-12 10000 100000",
        "",
        f"timestep        {dt}",
        "thermo_modify   flush yes",
        "",
        f"velocity        all create {temperature:.1f} {seed} dist gaussian",
        f"fix             fxNVT all nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        "",
        "# --- equilibration ---",
        f"run             {n_equil}",
        "",
        "# --- RDF accumulation ---",
        f"compute         rdf_c all rdf {n_bins}",
        f"fix             rdf_avg all ave/time {n_every} {n_repeat} {n_freq} "
        f"c_rdf_c[*] file {out_file} mode vector",
        f"run             {n_prod}",
        "unfix           fxNVT",
        "unfix           rdf_avg",
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
    supercell_repeat: int = 1,
) -> None:
    safe_np = _safe_mpi_np(
        mpi_np, lammps_data, cutoff=cutoff, supercell_repeat=supercell_repeat
    )
    cmd: list[str] = []
    if mpi_command:
        cmd += mpi_command.split()
        if safe_np is not None:
            cmd += ["-n", str(safe_np)]
    cmd += [lammps_cmd, "-in", str(Path(script).resolve())]
    if log_file:
        cmd += ["-log", str(Path(log_file).resolve())]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:] if result.stderr else ''}"
        )
