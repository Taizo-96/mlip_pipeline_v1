from __future__ import annotations

from pathlib import Path

from mlip_pipeline.utils.shell import run_command
from mlip_pipeline.utils.logging import warn


def run_exploration_runs(
    config: dict, resolved_paths: dict, explore_root: Path
) -> None:
    """Run all LAMMPS exploration jobs found under explore_root.

    Non-zero exit codes are warned about but do not abort the loop —
    some runs may diverge (e.g. near the melt) and that is expected.
    The select step downstream will simply find fewer preselected.cfg
    candidates from those runs.
    """
    explore_cfg    = config["explore"]
    lammps_command = explore_cfg.get("lammps_command", "lmp")
    mpi_prefix     = explore_cfg.get("mpi_prefix", "")

    input_files = sorted(explore_root.rglob("in.mlip.pb"))
    if not input_files:
        raise FileNotFoundError(f"No exploration inputs found in {explore_root}")

    n_ok   = 0
    n_fail = 0
    for input_file in input_files:
        command: list[str] = []
        if mpi_prefix:
            command.extend(mpi_prefix.split())
        command.extend([lammps_command, "-in", input_file.name])

        log_path    = input_file.with_suffix(input_file.suffix + ".log")
        return_code = run_command(command, cwd=input_file.parent, log_file=log_path)

        if return_code != 0:
            warn(
                f"LAMMPS run {input_file.parent.name} exited {return_code} — "
                f"continuing (see {log_path.name})"
            )
            n_fail += 1
        else:
            n_ok += 1

    if n_fail:
        warn(f"Exploration complete: {n_ok} ok, {n_fail} failed (non-fatal).")
