from __future__ import annotations

import argparse

from mlip_pipeline.config import load_yaml, project_paths
from mlip_pipeline.data.training_data import prepare_training_cfgs
from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
from mlip_pipeline.explore.runner import run_exploration_runs
from mlip_pipeline.fit.trainer import train_potential
from mlip_pipeline.integrations.lammps_export import export_mp_structures_to_lammps
from mlip_pipeline.integrations.materials_project import download_structures_from_mp
from mlip_pipeline.models import FitResult
from mlip_pipeline.select.runner import run_selection


def build_fit_result(config: dict, resolved_paths: dict) -> FitResult:
    fit_dir = resolved_paths["runs_root"] / config["fit"]["output_subdir"]
    fit_model = fit_dir / config["fit"]["trained_potential_name"]
    fit_log = fit_dir / "train.log"

    return FitResult(
        run_dir=fit_dir,
        model_path=fit_model,
        log_path=fit_log,
    )


def cmd_download_structures(config: dict, resolved_paths: dict) -> None:
    downloaded = download_structures_from_mp(config, resolved_paths)
    exported = export_mp_structures_to_lammps(config, resolved_paths)
    print(f"downloaded={len(downloaded)} exported={len(exported)}")


def cmd_prepare_train(config: dict, resolved_paths: dict) -> None:
    result = prepare_training_cfgs(config, resolved_paths)
    print(result.merged_cfg or result.output_dir)


def cmd_fit(config: dict, resolved_paths: dict) -> None:
    result = train_potential(config, resolved_paths)
    print(result.model_path)


def cmd_explore(config: dict, resolved_paths: dict) -> None:
    fit_result = build_fit_result(config, resolved_paths)
    explore_root = create_exploration_runs(config, resolved_paths, fit_result)
    result = run_exploration_runs(config, resolved_paths, explore_root)
    print(result)


def cmd_select(config: dict, resolved_paths: dict) -> None:
    fit_result = build_fit_result(config, resolved_paths)
    result = run_selection(config, resolved_paths, fit_result)
    print(f"Selected {result.selected_count} configurations in {result.select_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlip-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = [
        "download-structures",
        "prepare-train",
        "fit",
        "explore",
        "select",
    ]

    for name in commands:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_yaml(args.config)
    resolved_paths = project_paths(config)

    if args.command == "download-structures":
        cmd_download_structures(config, resolved_paths)
        return

    if args.command == "prepare-train":
        cmd_prepare_train(config, resolved_paths)
        return

    if args.command == "fit":
        cmd_fit(config, resolved_paths)
        return

    if args.command == "explore":
        cmd_explore(config, resolved_paths)
        return

    if args.command == "select":
        cmd_select(config, resolved_paths)
        return

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()