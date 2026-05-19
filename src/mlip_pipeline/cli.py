from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlip_pipeline.config import load_yaml, project_paths
from mlip_pipeline.data.training_data import prepare_training_cfgs
from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
from mlip_pipeline.explore.runner import run_exploration_runs
from mlip_pipeline.fit.trainer import train_potential
from mlip_pipeline.integrations.lammps_export import export_mp_structures_to_lammps
from mlip_pipeline.integrations.materials_project import download_structures_from_mp
from mlip_pipeline.models import FitResult, LabelResult
from mlip_pipeline.select.runner import run_selection
from mlip_pipeline.label.runner import run_labeling
from mlip_pipeline.label.local_runner import run_vasp_local
from mlip_pipeline.io.dardel import (
    submit_label_jobs,
    sync_inputs_to_dardel,
    submit_jobs_on_dardel,
    watch_queue,
    sync_outputs_from_dardel,
)
from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
from mlip_pipeline.evaluate.runner import run_evaluation


# ---------------------------------------------------------------------------
# Valid step names (used for reset-steps validation)
# ---------------------------------------------------------------------------
ALL_STEPS = ["fit", "explore", "select", "label", "label_hpc", "convert", "evaluate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_path(runs_root: Path, gen: int) -> Path:
    return runs_root / f"gen_{gen:02d}" / "state.json"


def _load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"state.json not found: {path}")
    return json.loads(path.read_text())


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def build_fit_result(config: dict, resolved_paths: dict) -> FitResult:
    fit_dir = resolved_paths["runs_root"] / config["fit"]["output_subdir"]
    fit_model = fit_dir / config["fit"]["trained_potential_name"]
    fit_log = fit_dir / "train.log"
    return FitResult(
        run_dir=fit_dir,
        model_path=fit_model,
        log_path=fit_log,
    )


# ---------------------------------------------------------------------------
# Step commands
# ---------------------------------------------------------------------------

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


def cmd_label(config: dict, resolved_paths: dict) -> None:
    result = run_labeling(config, resolved_paths)
    print(f"Labelled {result.task_count} task(s) in {result.label_root}")


def cmd_label_run(config: dict, resolved_paths: dict) -> None:
    """Run already-labeled VASP tasks locally using mpirun (no scheduler)."""
    label_cfg  = config["label"]
    runs_root  = resolved_paths["runs_root"]
    label_root = runs_root / label_cfg["output_subdir"]

    if not label_root.exists():
        raise FileNotFoundError(f"Label output not found: {label_root}. Run 'label' first.")

    task_dirs = sorted(label_root.glob("task.*"))
    if not task_dirs:
        raise FileNotFoundError(f"No task dirs found in {label_root}")

    label_result = LabelResult(
        label_root=label_root,
        task_dirs=task_dirs,
        task_count=len(task_dirs),
        manifest_path=label_root / "label_manifest.json",
    )
    run_vasp_local(label_result, config)


def cmd_sync_to_remote(label_result: LabelResult, config: dict, resolved_paths: dict):
    label_cfg = config["label"]
    d_cfg = label_cfg["dardel"]
    runs_root = resolved_paths["runs_root"]
    remote_dir = f"{d_cfg['remote_root']}/{runs_root.name}/{label_cfg['output_subdir']}"
    sync_inputs_to_dardel(
        label_result.label_root,
        d_cfg["user"],
        d_cfg.get("host", "dardel.pdc.kth.se"),
        remote_dir,
    )


def cmd_submit_and_watch(label_result: LabelResult, config: dict, resolved_paths: dict):
    label_cfg = config["label"]
    d_cfg = label_cfg["dardel"]
    user = d_cfg["user"]
    host = d_cfg.get("host", "dardel.pdc.kth.se")
    runs_root = resolved_paths["runs_root"]
    remote_dir = f"{d_cfg['remote_root']}/{runs_root.name}/{label_cfg['output_subdir']}"
    print(f"Submitting jobs in {remote_dir}...")
    submit_jobs_on_dardel(user, host, remote_dir)
    finished_cleanly = watch_queue(user, host, poll_interval=d_cfg.get("poll_interval", 60))
    if finished_cleanly:
        sync_outputs_from_dardel(label_result.label_root, user, host, remote_dir)


def cmd_convert_cfg(config: dict, resolved_paths: dict) -> None:
    label_root = resolved_paths["runs_root"] / config["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)
    merged = convert_outcars_to_cfg(label_result, config, resolved_paths)
    print(merged)


def cmd_evaluate(config: dict, resolved_paths: dict) -> None:
    fit_result = build_fit_result(config, resolved_paths)
    result = run_evaluation(config, resolved_paths, fit_result)
    for p in result.plot_paths.values():
        print(p)


# ---------------------------------------------------------------------------
# reset-steps command
# ---------------------------------------------------------------------------

def cmd_reset_steps(args: argparse.Namespace, config: dict, resolved_paths: dict) -> None:
    """
    Remove specific steps from state.json completed_steps so the loop
    will re-run them on the next invocation.

    Examples:
      mlip-pipeline reset-steps --config cfg.yaml --gen 8 --steps label_hpc convert
      mlip-pipeline reset-steps --config cfg.yaml --gen 8 --steps all
    """
    runs_root = resolved_paths["runs_root"]
    gen = args.gen
    steps_to_reset = args.steps

    state_file = _state_path(runs_root, gen)
    state = _load_state(state_file)

    before = list(state.get("completed_steps", []))

    if steps_to_reset == ["all"]:
        state["completed_steps"] = []
    else:
        # Validate step names
        unknown = [s for s in steps_to_reset if s not in ALL_STEPS]
        if unknown:
            raise ValueError(
                f"Unknown step(s): {unknown}. Valid steps: {ALL_STEPS}"
            )
        state["completed_steps"] = [
            s for s in before if s not in steps_to_reset
        ]

    state["status"] = "running"
    state["current_step"] = None

    _save_state(state_file, state)

    after = state["completed_steps"]
    removed = [s for s in before if s not in after]

    print(f"\ngen_{gen:02d} state.json updated:")
    print(f"  Before : {before}")
    print(f"  Removed: {removed}")
    print(f"  After  : {after}")
    print(f"\nRe-run the loop to execute the reset step(s).")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlip-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- standard single-step commands ---
    single_step_commands = [
        "download-structures",
        "prepare-train",
        "fit",
        "explore",
        "select",
        "label",
        "label-run",
        "sync-to-remote",
        "submit-remote",
        "watch-remote",
        "sync-from-remote",
        "convert-cfg",
        "evaluate",
    ]
    for name in single_step_commands:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)

    # --- run-loop ---
    loop_sub = subparsers.add_parser(
        "run-loop",
        help="Run multiple generations end-to-end.",
    )
    loop_sub.add_argument("config", help="Path to loop YAML config")
    loop_sub.add_argument("--start-gen", type=int, required=True)
    loop_sub.add_argument("--end-gen",   type=int, required=True)
    loop_sub.add_argument(
        "--local",
        action="store_true",
        default=False,
        help=(
            "Run VASP locally via mpirun instead of submitting to Dardel. "
            "Skips label_hpc and uses label-run in its place."
        ),
    )

    # --- reset-steps ---
    reset_sub = subparsers.add_parser(
        "reset-steps",
        help="Remove completed steps from state.json so the loop re-runs them.",
    )
    reset_sub.add_argument("--config", required=True)
    reset_sub.add_argument(
        "--gen", type=int, required=True,
        help="Generation number to reset (e.g. 8)",
    )
    reset_sub.add_argument(
        "--steps", nargs="+", required=True,
        metavar="STEP",
        help=(
            f"Steps to un-complete. One or more of: {ALL_STEPS}. "
            "Use 'all' to reset every step."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # reset-steps has its own config loading path (no resolved_paths needed
    # beyond runs_root, but we load fully for consistency)
    config_path = getattr(args, "config", None)
    if config_path is None:
        parser.print_help()
        return

    config = load_yaml(config_path)
    resolved_paths = project_paths(config)

    label_root = resolved_paths["runs_root"] / config["label"]["output_subdir"]

    if args.command == "download-structures":
        cmd_download_structures(config, resolved_paths)

    elif args.command == "prepare-train":
        cmd_prepare_train(config, resolved_paths)

    elif args.command == "fit":
        cmd_fit(config, resolved_paths)

    elif args.command == "explore":
        cmd_explore(config, resolved_paths)

    elif args.command == "select":
        cmd_select(config, resolved_paths)

    elif args.command == "label":
        cmd_label(config, resolved_paths)

    elif args.command == "label-run":
        cmd_label_run(config, resolved_paths)

    elif args.command == "sync-to-remote":
        result = LabelResult.load_from_dir(label_root)
        cmd_sync_to_remote(result, config, resolved_paths)

    elif args.command == "submit-remote":
        d_cfg = config["label"]["dardel"]
        remote_dir = f"{d_cfg['remote_root']}/{resolved_paths['runs_root'].name}/{config['label']['output_subdir']}"
        submit_jobs_on_dardel(d_cfg["user"], d_cfg.get("host"), remote_dir)

    elif args.command == "watch-remote":
        d_cfg = config["label"]["dardel"]
        watch_queue(d_cfg["user"], d_cfg.get("host"), poll_interval=d_cfg.get("poll_interval", 60))

    elif args.command == "sync-from-remote":
        result = LabelResult.load_from_dir(label_root)
        d_cfg = config["label"]["dardel"]
        remote_dir = f"{d_cfg['remote_root']}/{resolved_paths['runs_root'].name}/{config['label']['output_subdir']}"
        sync_outputs_from_dardel(result.label_root, d_cfg["user"], d_cfg.get("host"), remote_dir)

    elif args.command == "convert-cfg":
        cmd_convert_cfg(config, resolved_paths)

    elif args.command == "evaluate":
        cmd_evaluate(config, resolved_paths)

    elif args.command == "run-loop":
        # import here to avoid circular imports at module level
        from mlip_pipeline.loop import run_loop
        run_loop(config, resolved_paths, args.start_gen, args.end_gen, local=args.local)

    elif args.command == "reset-steps":
        cmd_reset_steps(args, config, resolved_paths)

    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
