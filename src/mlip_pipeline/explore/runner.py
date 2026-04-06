from __future__ import annotations

from pathlib import Path

from mlip_pipeline.utils.shell import run_command


def run_exploration_runs(config: dict, resolved_paths: dict, explore_root: Path) -> Path:
    explore_cfg = config['explore']
    lammps_command = explore_cfg.get('lammps_command', 'lmp')
    mpi_prefix = explore_cfg.get('mpi_prefix')

    input_files = sorted(explore_root.rglob('in.mlip.pb'))
    if not input_files:
        raise FileNotFoundError(f'no exploration inputs found in {explore_root}')

    for input_file in input_files:
        command = []
        if mpi_prefix:
            command.extend(mpi_prefix.split())
        command.extend([lammps_command, '-in', input_file.name])

        log_path = input_file.with_suffix(input_file.suffix + '.log')
        return_code = run_command(command, cwd=input_file.parent, log_file=log_path)

        # This is the only change you need:
        if return_code != 0:
            print(f"WARN: LAMMPS run for {input_file} returned {return_code}; continuing anyway...")
        # Do NOT raise RuntimeError

    return explore_root