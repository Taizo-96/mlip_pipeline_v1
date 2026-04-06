from __future__ import annotations

import argparse

from mlip_pipeline.config import load_yaml, project_paths
from mlip_pipeline.data.training_data import prepare_training_cfgs
from mlip_pipeline.fit.trainer import train_potential
from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
from mlip_pipeline.integrations.materials_project import download_structures_from_mp
from mlip_pipeline.integrations.lammps_export import export_mp_structures_to_lammps
from mlip_pipeline.explore.runner import run_exploration_runs


def main() -> None:
    parser = argparse.ArgumentParser(prog='mlip-pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    for name in ['prepare-train', 'fit', 'explore', 'run', 'download-structures']:
        sub = subparsers.add_parser(name)
        sub.add_argument('--config', required=True)

    args = parser.parse_args()
    config = load_yaml(args.config)
    resolved_paths = project_paths(config)

    if args.command == 'download-structures':
        downloaded = download_structures_from_mp(config, resolved_paths)
        exported = export_mp_structures_to_lammps(config, resolved_paths)
        print(f'downloaded={len(downloaded)} exported={len(exported)}')
        return

    if args.command == 'prepare-train':
        result = prepare_training_cfgs(config, resolved_paths)
        print(result.merged_cfg or result.output_dir)
        return

    if args.command == 'fit':
        result = train_potential(config, resolved_paths)
        print(result.model_path)
        return

    if args.command == 'explore':
        from mlip_pipeline.models import FitResult
        from pathlib import Path

        fit_dir = resolved_paths['runs_root'] / config['fit']['output_subdir']
        fit_model = fit_dir / config['fit']['trained_potential_name']
        fit_log = fit_dir / 'train.log'

        fit_result = FitResult(
            run_dir=fit_dir,
            model_path=fit_model,
            log_path=fit_log
        )
        explore_root = create_exploration_runs(config, resolved_paths, fit_result)
        result = run_exploration_runs(config, resolved_paths, explore_root)
        print(result)
        return


    download_structures_from_mp(config, resolved_paths)
    export_mp_structures_to_lammps(config, resolved_paths)
    prepare_training_cfgs(config, resolved_paths)
    fit_result = train_potential(config, resolved_paths)
    result = create_exploration_runs(config, resolved_paths)
    print(result)


if __name__ == '__main__':
    main()
