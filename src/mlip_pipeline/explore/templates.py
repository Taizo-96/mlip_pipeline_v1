from __future__ import annotations


def render_mlip_ini(model_path: str) -> str:
    return "\n".join([
        f"mtp-filename {model_path}",
        "calculate_efs TRUE",
        "select FALSE",
        "write_cfgs FALSE",
        "atoms 1 Pb",           # ← ADD THIS
        "basis radial 6.0 10",   # ← ADD THESE
        "basis angular 4.0 12",
    ])


def render_lammps_input(*, temp: int, structure_file: str, nsteps: int, timestep: float, dump_every: int, ensemble: str, pressure_bar: float, seed: int, mlip_ini_name: str = "mlip.ini") -> str:
    fix_line = (
        f"fix             1 all npt temp {temp} {temp} 0.1 iso {pressure_bar} {pressure_bar} 1.0"
        if ensemble.lower() == "npt"
        else f"fix             1 all nvt temp {temp} {temp} 0.1"
    )
    return f'''units           metal
atom_style      atomic
boundary        p p p

read_data       {structure_file}
replicate       3 3 3

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

pair_style      mlip config_file={mlip_ini_name}
pair_coeff      * *

timestep        {timestep}
thermo          100
thermo_style    custom step temp pe ke etotal press

velocity        all create {temp} {seed} dist gaussian
{fix_line}

dump            1 all cfg {dump_every} preselected_{temp}.*.cfg mass type x y z fx fy fz
dump_modify     1 element Pb

run             {nsteps}

unfix           1
undump          1
'''
