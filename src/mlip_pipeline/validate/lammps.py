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
    tokens = pair_coeff.split()
    resolved = []
    for tok in tokens:
        p = Path(tok)
        if not p.is_absolute() and p.suffix and p.exists():
            resolved.append(str(p.resolve()))
        else:
            resolved.append(tok)
    return " ".join(resolved)


def _pair_block(
    model_path: Path,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> list[str]:
    if pair_style is not None and pair_coeff is not None:
        return [
            f"pair_style      {pair_style}",
            f"pair_coeff      {_resolve_pair_coeff(pair_coeff)}",
        ]
    return [
        f"pair_style      mlip load_from={model_path.resolve()}",
        "pair_coeff      * *",
    ]


# ------------------------------------------------------------------ #
# TPC preparation scripts                                              #
# ------------------------------------------------------------------ #

def write_solid_eq_input(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperature: float,
    supercell_repeat: int = 5,
    dt: float = 0.002,
    n_equil: int = 5000,
    seed: int = 12345,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    """Equilibrate the whole supercell as solid at T_target, P=0.

    The reference data file has 0 K DFT lattice parameters.  After
    replicate the MTP pressure is ~20 kbar.

    Safe two-stage protocol
    -----------------------
    Stage 0  Cold NPT at 1 K (500 steps, dt/5):
             Barostat decompresses ~20 kbar -> ~0 with negligible
             kinetic energy.  The box is now at the correct P=0
             volume AND aspect ratio for this potential.
             No change_box is needed — iso NPT preserves the 5x5x10
             shape because it scales all three axes equally.

    Stage 1  NVT at T_target (n_equil steps, full dt):
             Velocities assigned at T_target after the box is at the
             correct volume.  NVT is appropriate because volume is
             already relaxed.  No pressure kick, no overshoot.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    dump_out  = (work_dir / "solid_final.dump").resolve()
    script_path = work_dir / "solid_eq.in"

    dt_slow = round(dt / 5.0, 6)
    tdamp   = round(100 * dt, 6)
    pdamp   = round(1000 * dt, 6)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat * 2}",
        "",
    ] + _pair_block(abs_model, pair_style, pair_coeff) + [
        "",
        "neigh_modify    one 4000 page 100000",
        "",
        "minimize        1e-8 1e-10 5000 50000",
        "",
        "thermo          500",
        "thermo_modify   flush yes",
        "",
        "# ---------------------------------------------------------------",
        "# Stage 0: cold NPT at 1 K.",
        "# iso barostat decompresses ~20 kbar -> ~0 with negligible KE.",
        "# The box reaches the correct P=0 volume in the right 5x5x10",
        "# aspect ratio.  No change_box needed after this.",
        "# ---------------------------------------------------------------",
        f"timestep        {dt_slow}",
        f"velocity        all create 1.0 {seed} dist gaussian",
        f"fix             fxS0 all npt temp 1.0 1.0 {tdamp} iso 0.0 0.0 {pdamp}",
        "run             500",
        "unfix           fxS0",
        "",
        "# ---------------------------------------------------------------",
        f"# Stage 1: NVT equilibration at T_target = {temperature:.1f} K.",
        "# Velocities assigned NOW at the correct volume -> no pressure kick.",
        "# ---------------------------------------------------------------",
        f"timestep        {dt}",
        f"velocity        all create {temperature:.1f} {seed + 1} dist gaussian",
        f"fix             fxS1 all nvt temp {temperature:.1f} {temperature:.1f} {tdamp}",
        f"run             {n_equil}",
        "unfix           fxS1",
        "",
        f"write_dump      all custom {dump_out} id type x y z modify sort id",
    ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def write_liquid_eq_input(
    lammps_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperature: float,
    supercell_repeat: int = 5,
    dt: float = 0.002,
    dt_heat: Optional[float] = None,
    n_equil: int = 5000,
    seed: int = 54321,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    """Melt the whole supercell and equilibrate as liquid at T_dis, P=0.

    T_dis = min(1.3 * T_target, T_target + 300).

    Stage 0  Cold NPT at 1 K (500 steps, dt/5) — decompress cell.
    Stage 1  NVT at T_dis (n_equil steps, dt) — velocities assigned
             at T_dis after Stage 0 has set the correct volume.
    """
    abs_data  = Path(lammps_data).resolve()
    abs_model = Path(model_path).resolve()
    dump_out  = (work_dir / "liquid_final.dump").resolve()
    script_path = work_dir / "liquid_eq.in"

    T_dis    = min(1.3 * temperature, temperature + 300.0)
    dt_slow  = dt_heat if dt_heat is not None else round(dt / 5.0, 6)
    tdamp    = round(100 * dt, 6)
    pdamp    = round(1000 * dt, 6)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        f"replicate       {supercell_repeat} {supercell_repeat} {supercell_repeat * 2}",
        "",
    ] + _pair_block(abs_model, pair_style, pair_coeff) + [
        "",
        "neigh_modify    one 4000 page 100000",
        "",
        "minimize        1e-8 1e-10 5000 50000",
        "",
        "thermo          500",
        "thermo_modify   flush yes",
        "",
        "# ---------------------------------------------------------------",
        "# Stage 0: cold NPT at 1 K — decompress cell before adding heat.",
        "# iso barostat preserves the 5x5x10 aspect ratio.",
        "# ---------------------------------------------------------------",
        f"timestep        {dt_slow}",
        f"velocity        all create 1.0 {seed} dist gaussian",
        f"fix             fxL0 all npt temp 1.0 1.0 {tdamp} iso 0.0 0.0 {pdamp}",
        "run             500",
        "unfix           fxL0",
        "",
        "# ---------------------------------------------------------------",
        f"# Stage 1: NVT at T_dis = {T_dis:.1f} K.",
        "# Velocities assigned at T_dis after box is at correct volume.",
        "# ---------------------------------------------------------------",
        f"timestep        {dt}",
        f"velocity        all create {T_dis:.1f} {seed + 1} dist gaussian",
        f"fix             fxL1 all nvt temp {T_dis:.1f} {T_dis:.1f} {tdamp}",
        f"run             {n_equil}",
        "unfix           fxL1",
        "",
        f"write_dump      all custom {dump_out} id type x y z modify sort id",
    ]
    script_path.write_text("\n".join(lines) + "\n")
    return script_path


def splice_tpc_cell(
    solid_dump: Path,
    liquid_dump: Path,
    out_data: Path,
    atom_type: int = 1,
) -> None:
    """Assemble a two-phase cell from two single-phase LAMMPS dump files.

    Takes the lo-z half of *solid_dump* and the hi-z half of *liquid_dump*
    and writes a combined LAMMPS data file to *out_data*.

    Both dumps must have the same box dimensions (guaranteed when both
    use NPT at P=0 with the same supercell_repeat; the equilibrium
    volumes will be very close).  The solid box is used as the reference.

    The split plane is z = (zlo + zhi) / 2.  Atoms exactly on the plane
    go to the solid half.
    """
    def _read_dump(path: Path) -> tuple[dict, list[tuple[int, float, float, float]]]:
        """Return (box_bounds, [(id, x, y, z), ...]) sorted by atom id."""
        box: dict[str, tuple[float, float]] = {}
        atoms: list[tuple[int, float, float, float]] = []
        with path.open() as fh:
            lines = fh.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "ITEM: BOX BOUNDS pp pp pp":
                for key in ("x", "y", "z"):
                    i += 1
                    lo, hi = map(float, lines[i].split())
                    box[key] = (lo, hi)
            elif line.startswith("ITEM: ATOMS"):
                i += 1
                while i < len(lines) and not lines[i].startswith("ITEM:"):
                    parts = lines[i].split()
                    atoms.append((int(parts[0]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
                    i += 1
                continue
            i += 1
        atoms.sort(key=lambda a: a[0])
        return box, atoms

    box_s, solid_atoms  = _read_dump(solid_dump)
    box_l, liquid_atoms = _read_dump(liquid_dump)

    if len(solid_atoms) != len(liquid_atoms):
        raise ValueError(
            f"splice_tpc_cell: atom count mismatch "
            f"({len(solid_atoms)} solid vs {len(liquid_atoms)} liquid). "
            f"Both equilibration runs must use identical supercell_repeat."
        )

    xlo, xhi = box_s["x"]
    ylo, yhi = box_s["y"]
    zlo, zhi = box_s["z"]
    zmid = (zlo + zhi) / 2.0

    lx_scale = (xhi - xlo) / (box_l["x"][1] - box_l["x"][0])
    ly_scale = (yhi - ylo) / (box_l["y"][1] - box_l["y"][0])
    lz_scale = (zhi - zlo) / (box_l["z"][1] - box_l["z"][0])
    lxlo = box_l["x"][0]
    lylo = box_l["y"][0]
    lzlo = box_l["z"][0]

    combined: list[tuple[float, float, float]] = []
    for (_, sx, sy, sz), (_, lx, ly, lz) in zip(solid_atoms, liquid_atoms):
        if sz <= zmid:
            combined.append((sx, sy, sz))
        else:
            rx = xlo + (lx - lxlo) * lx_scale
            ry = ylo + (ly - lylo) * ly_scale
            rz = zlo + (lz - lzlo) * lz_scale
            combined.append((rx, ry, rz))

    n_atoms = len(combined)
    out_lines = [
        "LAMMPS data file - TPC splice",
        "",
        f"{n_atoms} atoms",
        "",
        "1 atom types",
        "",
        f"{xlo:.8f} {xhi:.8f} xlo xhi",
        f"{ylo:.8f} {yhi:.8f} ylo yhi",
        f"{zlo:.8f} {zhi:.8f} zlo zhi",
        "",
        "Masses",
        "",
        f"{atom_type} 1.0",
        "",
        "Atoms  # atomic",
        "",
    ]
    for i, (x, y, z) in enumerate(combined, start=1):
        out_lines.append(f"{i} {atom_type} {x:.8f} {y:.8f} {z:.8f}")

    out_data.write_text("\n".join(out_lines) + "\n")


def write_coex_input(
    tpc_data: Path,
    model_path: Path,
    work_dir: Path,
    *,
    temperature: float,
    dt: float = 0.002,
    n_equil: int = 5000,
    n_prod: int = 20000,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> Path:
    """Write the interface equilibration + NPH production script.

    Stage A - NVT at T_target: relaxes the solid/liquid interface.
               No groups.  Single thermostat on all atoms.
    Stage B - NPH iso 0 0 pdamp: production run.
               Correct ensemble for TPC — constant pressure lets the
               cell volume relax as one phase grows into the other.
               Velocities inherited from Stage A (no reset).
    """
    abs_data  = Path(tpc_data).resolve()
    abs_model = Path(model_path).resolve()
    thermo_out = (work_dir / "coex_thermo.txt").resolve()
    natom_out  = (work_dir / "atom_count.txt").resolve()
    script_path = work_dir / "coex.in"

    tdamp = round(100 * dt, 6)
    pdamp = round(1000 * dt, 6)

    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "",
        f"read_data       {abs_data}",
        "",
    ] + _pair_block(abs_model, pair_style, pair_coeff) + [
        "",
        "neigh_modify    one 4000 page 100000",
        "",
        f"timestep        {dt}",
        "thermo          500",
        "thermo_modify   flush yes lost warn",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage A: NVT interface relaxation at T = {temperature:.1f} K.",
        "# ----------------------------------------------------------------",
        f"velocity        all create {temperature:.1f} 99999 dist gaussian",
        f"fix             fxEQ all nvt temp {temperature:.1f} {temperature:.1f} {tdamp}",
        f"run             {n_equil}",
        "unfix           fxEQ",
        "",
        "reset_atoms     id",
        "",
        "# ----------------------------------------------------------------",
        f"# Stage B: NPH production at P = 0 bar.",
        "#          Velocities inherited from Stage A — no reset.",
        "# ----------------------------------------------------------------",
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


# ------------------------------------------------------------------ #
# Legacy single-script entry point (kept for non-melting callers)     #
# ------------------------------------------------------------------ #

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
    """Deprecated single-script TPC writer. Delegates to write_coex_input."""
    tpc_data = work_dir / "tpc_start.lammps"
    return write_coex_input(
        tpc_data, model_path, work_dir,
        temperature=temperature,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )


# ------------------------------------------------------------------ #
# Other input writers (unchanged)                                      #
# ------------------------------------------------------------------ #

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
