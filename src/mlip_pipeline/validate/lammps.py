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


def _resolve_pair_coeff(pair_coeff: str) -> str:
    """Resolve any relative file paths inside a pair_coeff string to absolute.

    LAMMPS is always launched from a per-run subdirectory, so relative paths
    embedded in pair_coeff (common for EAM/MEAM potential files) would fail.
    Any token that looks like an existing file path is replaced with its
    resolved absolute equivalent; all other tokens (wildcards, element names,
    numbers) are left untouched.
    """
    tokens = pair_coeff.split()
    resolved = []
    for tok in tokens:
        p = Path(tok)
        # Only resolve tokens that are neither a wildcard nor a bare word/number
        # and that point to an existing file on disk.
        if not p.is_absolute() and p.suffix and p.exists():
            resolved.append(str(p.resolve()))
        else:
            resolved.append(tok)
    return " ".join(resolved)


# ------------------------------------------------------------------ #
# LAMMPS input builders                                                #
# ------------------------------------------------------------------ #

def _pair_block(
    model_path: Path,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> list[str]:
    """Return pair_style + pair_coeff lines.

    When pair_style/pair_coeff are provided (classical reference mode),
    they are used verbatim — except that any relative file paths inside
    pair_coeff are resolved to absolute so LAMMPS can find them regardless
    of the working directory it is launched from.
    Otherwise the default MTP mlip block is used.
    """
    if pair_style is not None and pair_coeff is not None:
        return [
            f"pair_style      {pair_style}",
            f"pair_coeff      {_resolve_pair_coeff(pair_coeff)}",
        ]
    return [
        f"pair_style      mlip load_from={model_path.resolve()}",
        "pair_coeff      * *",
    ]


def write_eos_input(
    lammps_data: Path, model_path: Path, out_file: Path,
    *, element: str, scale_min: float = 0.85, scale_max: float = 1.15,
    n_points: int = 21,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
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
    ] + _pair_block(model_path, pair_style, pair_coeff) + [
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
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
    ] + _pair_block(model_path, pair_style, pair_coeff) + [
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    """Write a two-phase coexistence LAMMPS input script.

    Ensemble choices
    ----------------
    - Stage 1 (liquid disordering): NVT with a reduced timestep (dt/5).
      The liquid half is heated to min(1.3*T, T+300) K to avoid excessive
      over-disordering in low-melting-point systems (Pb, K, Na, ...).
    - Stage 2 (quench + equilibration): NVT ramp back to T_target.
    - Stage 3 (production / phase classification): NPH at P=0 (iso).
      NPH is the physically correct ensemble for two-phase coexistence:
      it allows the cell volume to relax, removing the artificial pressure
      build-up that NVT would introduce as one phase grows into the other.
      Velocities are carried forward from Stage 2 (no velocity reset).

    Group partitioning
    ------------------
    The solid (lo-z) and liquid (hi-z) halves are defined with open-face
    regions so that every atom belongs to exactly one group:

      solid_region : open 6  (z-hi face is open/exclusive)
      liquid_region: open 5  (z-lo face is open/exclusive)

    This guarantees no atom is double-counted regardless of its z position,
    eliminating the "time integrated more than once" crash that would occur
    with the default closed regions where atoms at z=zmid fall into both.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    thermo_out  = (work_dir / "coex_thermo.txt").resolve()
    natom_out   = (work_dir / "atom_count.txt").resolve()
    script_path = work_dir / "melting.in"

    # Reduced timestep for the disordering stage (avoids instabilities).
    dt_heat  = round(dt / 5.0, 6)
    # Cap disordering temperature: aggressive heating can cause instabilities
    # for low-Tm systems (Pb ~600 K, K ~336 K).  Use min(1.3*T, T+300) K.
    T_dis    = min(1.3 * temperature, temperature + 300.0)
    n_dis    = max(2000, n_equil)
    n_eq     = max(1000, n_equil // 2)
    # Explicit Nose-Hoover damping: tdamp=100*dt, pdamp=1000*dt (ps)
    tdamp    = round(100 * dt, 6)
    pdamp    = round(1000 * dt, 6)

    pair_lines = _pair_block(abs_model, pair_style, pair_coeff)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat * 2}",
        "",
    ] + pair_lines + [
        "",
        "neigh_modify    one 4000 page 100000",
        "",
        "minimize        1e-8 1e-10 5000 50000",
        "",
        "# ----------------------------------------------------------------",
        "# Split into solid (lo-z) and liquid (hi-z) halves.",
        "# open 6 = z-hi face exclusive on solid_region",
        "# open 5 = z-lo face exclusive on liquid_region",
        "# Together these guarantee every atom belongs to exactly one group.",
        "# ----------------------------------------------------------------",
        "variable        zmid   equal (zlo+zhi)/2.0",
        "region          solid_region  block INF INF INF INF INF ${zmid} units box open 6",
        "region          liquid_region block INF INF INF INF ${zmid} INF  units box open 5",
        "group           solid_atoms   region solid_region",
        "group           liquid_atoms  region liquid_region",
        "",
        f"timestep        {dt_heat}",
        "thermo_modify   flush yes lost warn",
        "thermo          500",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage 1: Disorder liquid half at {T_dis:.0f} K (= min(1.3*T, T+300)).",
        f"#          Reduced timestep {dt_heat} ps used for stability.",
        "# ----------------------------------------------------------------",
        f"velocity        all create {temperature:.1f} {seed} dist gaussian",
        f"fix             fxS solid_atoms nvt temp {temperature:.1f} {temperature:.1f} {tdamp}",
        f"fix             fxL liquid_atoms nvt temp {T_dis:.1f} {T_dis:.1f} {tdamp}",
        f"run             {n_dis}",
        "unfix           fxS",
        "unfix           fxL",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage 2: Quench + equilibrate whole cell at T = {temperature:.1f} K.",
        f"#          Switch back to production timestep {dt} ps.",
        "# ----------------------------------------------------------------",
        f"timestep        {dt}",
        f"fix             fxEQ all nvt temp {T_dis:.1f} {temperature:.1f} {tdamp}",
        f"run             {n_eq}",
        "unfix           fxEQ",
        "",
        "reset_atoms     id",
        "",
        "# ----------------------------------------------------------------",
        "# Stage 3: Production NPH (iso, P=0).  Correct ensemble for TPC:",
        "#          constant pressure allows the cell to adjust as one phase",
        "#          grows into the other, avoiding artificial pressure buildup.",
        "#          Velocities are inherited from Stage 2 (no velocity reset).",
        "# ----------------------------------------------------------------",
        f"timestep        {dt}",
        "thermo_style    custom step temp vol pe atoms",
        "thermo          50",
        "thermo_modify   flush yes lost warn",
        "",
        f"fix             fxNPH all nph iso 0.0 0.0 {pdamp}",
        f"print           \"# step temp vol pe\" file {thermo_out} screen no",
        f"fix             fxPrint all print 50 "
        f"\"$(step) $(temp) $(vol) $(pe)\" append {thermo_out} screen no",
        "",
        f"run             {n_prod}",
        "unfix           fxNPH",
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    out_file  = (work_dir / "thexp_output.txt").resolve()
    script_path = work_dir / "thexp.in"

    # Explicit damping constants: tdamp=100*dt, pdamp=1000*dt (ps)
    tdamp = round(100 * dt, 6)
    pdamp = round(1000 * dt, 6)

    pair_lines = _pair_block(abs_model, pair_style, pair_coeff)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
    ] + pair_lines + [
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
            f"fix             fxNPT all npt temp {T:.1f} {T:.1f} {tdamp} "
            f"iso 0.0 0.0 {pdamp}",
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
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
        ] + _pair_block(abs_model, pair_style, pair_coeff) + [
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    out_file  = (work_dir / "rdf_output.txt").resolve()
    script_path = work_dir / "rdf.in"

    n_every  = 10
    n_freq   = max(n_every * 2, n_prod // 10)
    n_repeat = n_freq // n_every

    # Explicit damping
    tdamp = round(100 * dt, 6)

    pair_lines = _pair_block(abs_model, pair_style, pair_coeff)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat}",
        "",
    ] + pair_lines + [
        "",
        "minimize        1e-10 1e-12 10000 100000",
        "",
        f"timestep        {dt}",
        "thermo_modify   flush yes",
        "",
        f"velocity        all create {temperature:.1f} {seed} dist gaussian",
        f"fix             fxNVT all nvt temp {temperature:.1f} {temperature:.1f} {tdamp}",
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
