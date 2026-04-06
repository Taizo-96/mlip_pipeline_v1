from __future__ import annotations

from pathlib import Path
import shutil

from mlip_pipeline.explore.resolve import resolve_structure_data_path
from mlip_pipeline.explore.templates import render_mlip_ini
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.models import FitResult


def build_lammps_input(
    config: dict,
    resolved_paths: dict,
    temperature: int,
    potential_path: str,
) -> str:
    explore_cfg = config["explore"]
    structure_data = resolve_structure_data_path(config, resolved_paths)

    ensemble = explore_cfg.get("ensemble", "npt").lower()
    timestep = explore_cfg.get("timestep", 0.001)
    nsteps = explore_cfg.get("nsteps", 20000)
    dump_every = explore_cfg.get("dump_every", 100)
    pressure_bar = explore_cfg.get("pressure_bar", 0.0)
    seed_base = int(explore_cfg.get("seed_base", 492845))
    seed = seed_base + int(temperature)

    replicate = explore_cfg.get("replicate")
    tdamp = explore_cfg["tdamp"] if "tdamp" in explore_cfg else 0.1
    pdamp = explore_cfg["pdamp"] if "pdamp" in explore_cfg else 1.0
    neighbor_skin = explore_cfg.get("neighbor_skin")
    thermo_every = explore_cfg.get("thermo_every")
    nvt_pre_steps = explore_cfg.get("nvt_pre_steps")

    stab_cfg = explore_cfg.get("stabilization", {})
    neigh_mod_cfg = stab_cfg.get("neigh_modify", {})
    velocity_scale = stab_cfg.get("velocity_scale", False)
    momentum_cfg = stab_cfg.get("momentum_fix", {})

    rep_line = ""
    if replicate and any(r > 1 for r in replicate):
        rep_line = f"replicate {replicate[0]} {replicate[1]} {replicate[2]}\n"

    neighbor_line = ""
    if neighbor_skin is not None:
        neighbor_line = f"neighbor {neighbor_skin} bin\n"

    neigh_modify_line = ""
    if neigh_mod_cfg:
        every = neigh_mod_cfg.get("every", 1)
        delay = neigh_mod_cfg.get("delay", 0)
        check = neigh_mod_cfg.get("check", "yes")

        if isinstance(check, bool):
            check = "yes" if check else "no"
        else:
            check = str(check).lower()

        neigh_modify_line = (
            f"neigh_modify every {every} delay {delay} check {check}\n"
        )

    thermo_line = ""
    if thermo_every is not None:
        thermo_line = f"thermo {thermo_every}\n"

    velocity_scale_line = ""
    if velocity_scale:
        velocity_scale_line = f"velocity all scale {temperature}\n"

    momentum_fix_line = ""
    if momentum_cfg:
        mom_every = momentum_cfg.get("every", 1)
        linear = momentum_cfg.get("linear", [1, 1, 1])
        rescale = momentum_cfg.get("rescale", False)

        momentum_fix_line = (
            f"fix MOM all momentum {mom_every} linear "
            f"{linear[0]} {linear[1]} {linear[2]}"
        )
        if rescale:
            momentum_fix_line += " rescale"
        momentum_fix_line += "\n"

    pre_nvt_block = ""
    if nvt_pre_steps is not None and nvt_pre_steps > 0:
        pre_nvt_block = (
            f"fix PRE all nvt temp {temperature} {temperature} {tdamp}\n"
            f"run {nvt_pre_steps}\n"
            f"unfix PRE\n"
        )

    if ensemble == "npt":
        fix_line = (
            f"fix 1 all npt temp {temperature} {temperature} {tdamp} "
            f"iso {pressure_bar} {pressure_bar} {pdamp}"
        )
    elif ensemble == "nvt":
        fix_line = f"fix 1 all nvt temp {temperature} {temperature} {tdamp}"
    else:
        raise ValueError(f"unsupported ensemble: {ensemble}")

    return f"""units metal
atom_style atomic
read_data {structure_data}
{rep_line}pair_style mlip load_from={potential_path}
pair_coeff * *

{neighbor_line}{neigh_modify_line}timestep {timestep}
velocity all create {temperature} {seed} mom yes rot no dist gaussian
{velocity_scale_line}{thermo_line}thermo_style custom step temp pe ke etotal press vol
{momentum_fix_line}{pre_nvt_block}{fix_line}

dump 1 all custom {dump_every} traj_{temperature}.lammpstrj id type x y z
run {nsteps}
"""


def create_exploration_runs(config: dict, resolved_paths: dict, fit_result: FitResult) -> Path:
    explore_root = ensure_dir(resolved_paths['runs_root'] / config['explore']['output_subdir'])
    explore_cfg = config['explore']
    temperatures = explore_cfg['temperatures']

    for temp in temperatures:
        temp_dir = ensure_dir(explore_root / f"T{temp}K")

        # 1. Copy potential from fit phase
        potential_path = fit_result.model_path.name
        shutil.copy(fit_result.model_path, temp_dir / potential_path)

        # 2. Generate LAMMPS input
        lammps_input = build_lammps_input(
            config=config,
            resolved_paths=resolved_paths,
            temperature=temp,
            potential_path=potential_path,
        )
        (temp_dir / "in.mlip.pb").write_text(lammps_input)

    return explore_root