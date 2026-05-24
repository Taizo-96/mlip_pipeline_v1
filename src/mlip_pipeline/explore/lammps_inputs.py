from __future__ import annotations

from pathlib import Path

from mlip_pipeline.explore.resolve import resolve_structure_data_path
from mlip_pipeline.utils.fs import ensure_dir, copy_if_exists
from mlip_pipeline.models import FitResult


def build_lammps_input(
    config: dict,
    resolved_paths: dict,
    temperature: int,
    potential_path: str,
    pressure: float = 0.0,
) -> str:
    explore_cfg = config["explore"]
    structure_data = resolve_structure_data_path(config, resolved_paths)

    ensemble      = explore_cfg.get("ensemble", "npt").lower()
    timestep      = explore_cfg.get("timestep", 0.001)
    nsteps        = explore_cfg.get("nsteps", 20000)
    dump_every    = explore_cfg.get("dump_every", 100)
    pressure_bar  = pressure
    seed_base     = int(explore_cfg.get("seed_base", 492845))
    seed          = seed_base + int(temperature) + int(abs(pressure))

    replicate      = explore_cfg.get("replicate")
    tdamp          = explore_cfg.get("tdamp", 0.1)
    pdamp          = explore_cfg.get("pdamp", 1.0)
    neighbor_skin  = explore_cfg.get("neighbor_skin")
    thermo_every   = explore_cfg.get("thermo_every")
    nvt_pre_steps  = explore_cfg.get("nvt_pre_steps")

    stab_cfg       = explore_cfg.get("stabilization", {})
    neigh_mod_cfg  = stab_cfg.get("neigh_modify", {})
    velocity_scale = stab_cfg.get("velocity_scale", False)
    momentum_cfg   = stab_cfg.get("momentum_fix", {})

    dump_cfg    = explore_cfg.get("dump", {})
    dump_attrs  = dump_cfg.get("attributes", ["id", "type", "x", "y", "z", "fx", "fy", "fz"])
    dump_sort   = str(dump_cfg.get("sort",   "id")).lower()
    dump_pbc    = str(dump_cfg.get("pbc",    "no")).lower()
    dump_time   = str(dump_cfg.get("time",   "yes")).lower()
    dump_units  = str(dump_cfg.get("units",  "yes")).lower()
    dump_unwrap = str(dump_cfg.get("unwrap", "no")).lower()

    al_cfg               = explore_cfg.get("active_learning", {})
    al_enabled           = bool(al_cfg.get("enabled", False))
    threshold_save       = al_cfg.get("threshold_save", 2.1)
    threshold_break      = al_cfg.get("threshold_break", 10.0)
    save_extrapolative_to = al_cfg.get("save_extrapolative_to", "preselected.cfg")

    # ── optional LAMMPS lines ──────────────────────────────────────────────────
    rep_line = (
        f"replicate {replicate[0]} {replicate[1]} {replicate[2]}\n"
        if replicate and any(r > 1 for r in replicate)
        else ""
    )
    neighbor_line = f"neighbor {neighbor_skin} bin\n" if neighbor_skin is not None else ""

    if neigh_mod_cfg:
        every = neigh_mod_cfg.get("every", 1)
        delay = neigh_mod_cfg.get("delay", 0)
        check = neigh_mod_cfg.get("check", "yes")
        check = "yes" if check is True else ("no" if check is False else str(check).lower())
        neigh_modify_line = f"neigh_modify every {every} delay {delay} check {check}\n"
    else:
        neigh_modify_line = ""

    thermo_line        = f"thermo {thermo_every}\n" if thermo_every is not None else ""
    velocity_scale_line = f"velocity all scale {temperature}\n" if velocity_scale else ""

    if momentum_cfg:
        mom_every = momentum_cfg.get("every", 1)
        linear    = momentum_cfg.get("linear", [1, 1, 1])
        rescale   = momentum_cfg.get("rescale", False)
        momentum_fix_line = (
            f"fix MOM all momentum {mom_every} linear "
            f"{linear[0]} {linear[1]} {linear[2]}"
            + (" rescale" if rescale else "") + "\n"
        )
    else:
        momentum_fix_line = ""

    pre_nvt_block = (
        f"fix PRE all nvt temp {temperature} {temperature} {tdamp}\n"
        f"run {nvt_pre_steps}\n"
        f"unfix PRE\n"
        if nvt_pre_steps and nvt_pre_steps > 0
        else ""
    )

    if ensemble == "npt":
        fix_line = (
            f"fix 1 all npt temp {temperature} {temperature} {tdamp} "
            f"iso {pressure_bar} {pressure_bar} {pdamp}"
        )
    elif ensemble == "nvt":
        fix_line = f"fix 1 all nvt temp {temperature} {temperature} {tdamp}"
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble!r}. Use 'npt' or 'nvt'.")

    pair_style_line = f"pair_style mlip load_from={potential_path}"
    if al_enabled:
        pair_style_line += (
            " extrapolation_control=true"
            f" threshold_save={threshold_save}"
            f" threshold_break={threshold_break}"
            f" save_extrapolative_to={save_extrapolative_to}"
        )

    dump_line = (
        f"dump 1 all custom {dump_every} traj_{temperature}K_P{int(pressure)}bar.lammpstrj "
        + " ".join(dump_attrs)
    )
    dump_modify_parts = []
    if dump_sort in {"off", "id"} or dump_sort.lstrip("-").isdigit():
        dump_modify_parts.append(f"dump_modify 1 sort {dump_sort}")
    if dump_pbc    in {"yes", "no"}: dump_modify_parts.append(f"dump_modify 1 pbc {dump_pbc}")
    if dump_time   in {"yes", "no"}: dump_modify_parts.append(f"dump_modify 1 time {dump_time}")
    if dump_units  in {"yes", "no"}: dump_modify_parts.append(f"dump_modify 1 units {dump_units}")
    if dump_unwrap in {"yes", "no"}: dump_modify_parts.append(f"dump_modify 1 unwrap {dump_unwrap}")
    dump_modify_block = ("\n".join(dump_modify_parts) + "\n") if dump_modify_parts else ""

    return (
        f"units metal\n"
        f"atom_style atomic\n"
        f"read_data {structure_data}\n"
        f"{rep_line}"
        f"{pair_style_line}\n"
        f"pair_coeff * *\n\n"
        f"{neighbor_line}"
        f"{neigh_modify_line}"
        f"timestep {timestep}\n"
        f"velocity all create {temperature} {seed} mom yes rot no dist gaussian\n"
        f"{velocity_scale_line}"
        f"{thermo_line}"
        f"thermo_style custom step temp pe ke etotal press vol\n"
        f"{momentum_fix_line}"
        f"{pre_nvt_block}"
        f"{fix_line}\n\n"
        f"{dump_line}\n"
        f"{dump_modify_block}"
        f"run {nsteps}\n"
    )


def create_exploration_runs(
    config: dict, resolved_paths: dict, fit_result: FitResult
) -> Path:
    explore_cfg  = config["explore"]
    explore_root = ensure_dir(
        resolved_paths["runs_root"] / explore_cfg["output_subdir"]
    )

    # Validate model exists before staging into any run dir
    if not fit_result.model_path.exists():
        raise FileNotFoundError(
            f"Trained potential not found: {fit_result.model_path}"
        )

    # Validate structure data exists before writing any LAMMPS inputs
    structure_data = resolve_structure_data_path(config, resolved_paths)
    if not structure_data.exists():
        raise FileNotFoundError(
            f"Structure data file not found: {structure_data}\n"
            f"Run 'prepare-structures' or set explore.structure_data in config."
        )

    temperatures  = explore_cfg["temperatures"]
    pressures_raw = explore_cfg.get("pressure_bar", 0.0)
    pressures     = pressures_raw if isinstance(pressures_raw, list) else [pressures_raw]

    for temp in temperatures:
        for pressure in pressures:
            run_dir        = ensure_dir(explore_root / f"T{temp}K_P{int(pressure)}bar")
            potential_name = fit_result.model_path.name
            copy_if_exists(fit_result.model_path, run_dir / potential_name)

            lammps_input = build_lammps_input(
                config=config,
                resolved_paths=resolved_paths,
                temperature=temp,
                pressure=pressure,
                potential_path=potential_name,
            )
            (run_dir / "in.mlip.pb").write_text(lammps_input)

    return explore_root
