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
            n_sim = n_atoms * rep ** 2 * (rep * 2)
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
        ("0", "x scale ${sdelta} remap",      "x scale ${sinv_normal} remap"),
        ("1", "y scale ${sdelta} remap",      "y scale ${sinv_normal} remap"),
        ("2", "z scale ${sdelta} remap",      "z scale ${sinv_normal} remap"),
        ("3", "xy delta ${shear_dxy} remap",  "xy delta ${neg_shear_dxy} remap"),
        ("4", "xz delta ${shear_dxz} remap",  "xz delta ${neg_shear_dxz} remap"),
        ("5", "yz delta ${shear_dyz} remap",  "yz delta ${neg_shear_dyz} remap"),
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

    Protocol (Belonoshko-style two-phase coexistence)
    -------------------------------------------------
    Stage 1 — Liquid disordering at 2×T_target (NVT, liquid group only):
      The liquid half is heated to 2×T_target, which is well above T_melt
      for any reasonable scan temperature. This guarantees the liquid half
      is truly disordered regardless of T_target. The solid half is held
      at T_target with its own NVT thermostat throughout.
      Timestep is reduced to dt/5 during disordering for stability.

    Stage 2 — Quench and equilibrate at T_target (NVT, whole cell):
      The liquid half is cooled back to T_target. A whole-system NVT
      equilibration allows the interface to form cleanly.

    Stage 3 — Production NVT at T_target:
      The PE slope over the production run is the classifier signal.
      Under NVT (constant volume), PE rises if liquid grows and falls
      if solid grows. Volume is logged for diagnostics only.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    thermo_out  = (work_dir / "coex_thermo.txt").resolve()
    natom_out   = (work_dir / "atom_count.txt").resolve()
    script_path = work_dir / "melting.in"

    dt_heat  = dt / 5.0            # 0.0004 ps — safe for high-T disordering
    T_dis    = 2.0 * temperature   # guaranteed to disorder for any T_target
    n_dis    = max(2000, n_equil)  # enough steps to fully disorder at 2×T
    n_eq     = max(1000, n_equil // 2)  # quench + interface equilibration

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
        "neigh_modify    one 4000 page 100000",
        "",
        "minimize        1e-8 1e-10 5000 50000",
        "",
        "# Split into solid (lo-z) and liquid (hi-z) halves.",
        "variable        zmid   equal (zlo+zhi)/2.0",
        "region          solid_region  block INF INF INF INF INF ${zmid} units box",
        "region          liquid_region block INF INF INF INF ${zmid} INF units box",
        "group           solid_atoms   region solid_region",
        "group           liquid_atoms  region liquid_region",
        "",
        f"timestep        {dt_heat}",
        "thermo_modify   flush yes lost ignore",
        "thermo          500",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage 1: Disorder liquid half at 2*T = {T_dis:.0f} K.",
        "# ----------------------------------------------------------------",
        f"velocity        all create {temperature:.1f} {seed} dist gaussian",
        f"fix             fxS solid_atoms nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        f"fix             fxL liquid_atoms nvt temp {T_dis:.1f} {T_dis:.1f} $(100*dt)",
        f"run             {n_dis}",
        "unfix           fxS",
        "unfix           fxL",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage 2: Quench + equilibrate whole cell at T = {temperature:.1f} K.",
        "# ----------------------------------------------------------------",
        f"fix             fxEQ all nvt temp {T_dis:.1f} {temperature:.1f} $(100*dt)",
        f"run             {n_eq}",
        "unfix           fxEQ",
        "",
        "# ----------------------------------------------------------------",
        "# Stage 3: Production NVT. Classify via PE slope.",
        "# ----------------------------------------------------------------",
        f"velocity        all create {temperature:.1f} {seed + 1} dist gaussian",
        "",
        f"timestep        {dt}",
        "thermo_style    custom step temp vol pe atoms",
        "thermo          50",
        "thermo_modify   flush yes lost ignore",
        "",
        f"fix             fxNVT all nvt temp {temperature:.1f} {temperature:.1f} $(100*dt)",
        f"print           \"# step temp vol pe\" file {thermo_out} screen no",
        f"fix             fxPrint all print 50 "
        f"\"$(step) $(temp) $(vol) $(pe)\" append {thermo_out} screen no",
        "",
        f"run             {n_prod}",
        "unfix           fxNVT",
        "unfix           fxPrint",
        "",
        f"print           \"$(atoms)\" file {natom_out} screen no",
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
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    out_file  = (work_dir / "rdf_output.txt").resolve()
    script_path = work_dir / "rdf.in"

    n_every  = 10
    n_freq   = max(n_every * 2, n_prod // 10)
    n_repeat = n_freq // n_every

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
        f"run             {n_equil}",
        "",
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
