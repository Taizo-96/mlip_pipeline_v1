from __future__ import annotations

import json
import typer
from typing_extensions import Annotated
from pathlib import Path

from mlip_pipeline.config import load_yaml, project_paths
from mlip_pipeline.data.training_data import prepare_training_cfgs
from mlip_pipeline.explore.lammps_inputs import create_exploration_runs
from mlip_pipeline.explore.runner import run_exploration_runs
from mlip_pipeline.fit.trainer import train_potential
from mlip_pipeline.integrations.lammps_export import export_mp_structures_to_lammps
from mlip_pipeline.integrations.materials_project import download_structures_from_mp
from mlip_pipeline.models import FitResult, LabelResult, STEPS
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

# Initialize Typer App cleanly
app = typer.Typer(
    help="MLIP Pipeline Command Line Interface",
    pretty_exceptions_show_locals=False
)


def build_fit_result(config: dict, resolved_paths: dict) -> FitResult:
    fit_dir = resolved_paths["runs_root"] / config["fit"]["output_subdir"]
    fit_model = fit_dir / config["fit"]["trained_potential_name"]
    fit_log = fit_dir / "train.log"
    return FitResult(run_dir=fit_dir, model_path=fit_model, log_path=fit_log)


def get_paths(config_path: str) -> tuple[dict, dict]:
    config = load_yaml(config_path)
    resolved_paths = project_paths(config)
    return config, resolved_paths


# --- PIPELINE STEP COMMANDS ---

@app.command("download-structures")
def download_structures(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    downloaded = download_structures_from_mp(cfg, paths)
    exported = export_mp_structures_to_lammps(cfg, paths)
    print(f"downloaded={len(downloaded)} exported={len(exported)}")


@app.command("prepare-train")
def prepare_train(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    result = prepare_training_cfgs(cfg, paths)
    print(result.merged_cfg or result.output_dir)


@app.command("fit")
def fit(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    result = train_potential(cfg, paths)
    print(result.model_path)


@app.command("explore")
def explore(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    fit_result = build_fit_result(cfg, paths)
    explore_root = create_exploration_runs(cfg, paths, fit_result)
    result = run_exploration_runs(cfg, paths, explore_root)
    print(result)


@app.command("select")
def select(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    fit_result = build_fit_result(cfg, paths)
    result = run_selection(cfg, paths, fit_result)
    print(f"Selected {result.selected_count} configurations in {result.select_root}")


@app.command("label")
def label(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    result = run_labeling(cfg, paths)
    print(f"Labelled {result.task_count} task(s) in {result.label_root}")


@app.command("label-run")
def label_run(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    """Run already-labeled VASP tasks locally using mpirun (no scheduler)."""
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]

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
    run_vasp_local(label_result, cfg)


@app.command("sync-to-remote")
def sync_to_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)

    d_cfg = cfg["label"]["dardel"]
    remote_dir = f"{d_cfg['remote_root']}/{paths['runs_root'].name}/{cfg['label']['output_subdir']}"
    sync_inputs_to_dardel(
        label_result.label_root,
        d_cfg["user"],
        d_cfg.get("host", "dardel.pdc.kth.se"),
        remote_dir,
    )


@app.command("submit-remote")
def submit_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    d_cfg = cfg["label"]["dardel"]
    remote_dir = f"{d_cfg['remote_root']}/{paths['runs_root'].name}/{cfg['label']['output_subdir']}"
    submit_jobs_on_dardel(d_cfg["user"], d_cfg.get("host", "dardel.pdc.kth.se"), remote_dir)


@app.command("watch-remote")
def watch_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    d_cfg = cfg["label"]["dardel"]
    watch_queue(d_cfg["user"], d_cfg.get("host", "dardel.pdc.kth.se"), poll_interval=d_cfg.get("poll_interval", 60))


@app.command("sync-from-remote")
def sync_from_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)
    d_cfg = cfg["label"]["dardel"]
    remote_dir = f"{d_cfg['remote_root']}/{paths['runs_root'].name}/{cfg['label']['output_subdir']}"
    sync_outputs_from_dardel(label_result.label_root, d_cfg["user"], d_cfg.get("host", "dardel.pdc.kth.se"), remote_dir)


@app.command("convert-cfg")
def convert_cfg(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)
    merged = convert_outcars_to_cfg(label_result, cfg, paths)
    print(merged)


@app.command("evaluate")
def evaluate(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    fit_result = build_fit_result(cfg, paths)
    result = run_evaluation(cfg, paths, fit_result)
    for p in result.plot_paths.values():
        print(p)


# --- STATE MANAGEMENT ---

@app.command("reset-steps")
def reset_steps(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gen: Annotated[int, typer.Option(..., "--gen", help="Generation number")],
    steps: Annotated[str, typer.Option(..., "--steps", help="Comma-separated steps to reset")],
):
    """
    Remove one or more steps from a generation's completed_steps so the loop
    will re-run them on the next invocation.

    Example:
        mlip-pipeline reset-steps --config configs/Pb_loop.yaml --gen 8 --steps label,label_local,convert
    """
    valid = set(STEPS)
    requested = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise typer.BadParameter(
            f"Unknown step(s): {unknown}. Valid steps: {sorted(valid)}",
            param_hint="--steps",
        )

    cfg, paths = get_paths(config)
    gen_tag = f"gen_{gen:02d}"
    state_file = paths["runs_root"] / gen_tag / "state.json"

    if not state_file.exists():
        typer.echo(f"No state.json found at {state_file} — nothing to reset.", err=True)
        raise typer.Exit(1)

    state = json.loads(state_file.read_text())
    before = list(state.get("completed_steps", []))
    state["completed_steps"] = [s for s in before if s not in requested]

    # If we reset any steps, the generation is no longer "completed"
    if state["completed_steps"] != before:
        if state.get("status") == "completed":
            state["status"] = "running"
        state["error"] = None
        state["completed_at"] = None
        state_file.write_text(json.dumps(state, indent=2))
        removed = [s for s in before if s in requested]
        typer.echo(f"Reset step(s) {removed} for {gen_tag}.")
    else:
        typer.echo(f"None of {requested} were in completed_steps for {gen_tag} — no changes made.")


@app.command("show-state")
def show_state(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gen: Annotated[int, typer.Option(..., "--gen", help="Generation number")],
):
    """Print the current state.json for a generation."""
    cfg, paths = get_paths(config)
    gen_tag = f"gen_{gen:02d}"
    state_file = paths["runs_root"] / gen_tag / "state.json"

    if not state_file.exists():
        typer.echo(f"No state.json found at {state_file}.", err=True)
        raise typer.Exit(1)

    typer.echo(state_file.read_text())


# --- ACTIVE LEARNING LOOPS ---

@app.command("run-loop")
def run_loop(
        config: Annotated[str, typer.Argument(help="Path to base.yaml")],
        start_gen: Annotated[int, typer.Option("--start-gen")] = 0,
        end_gen: Annotated[int, typer.Option("--end-gen")] = ...,
        force: Annotated[bool, typer.Option("--force")] = False,
        skip: Annotated[str, typer.Option("--skip", help="Comma-separated steps to skip")] = "",
        only: Annotated[str, typer.Option("--only", help="Comma-separated steps to run")] = "",
        stop_on_failure: Annotated[bool, typer.Option("--stop-on-failure/--continue-on-failure")] = True,
):
    """Run multiple generations automatically (fit→explore→select→label→convert)."""
    from mlip_pipeline.loop.runner import run_loop as _run_loop
    skip_steps = [s.strip() for s in skip.split(",") if s.strip()]
    only_steps = [s.strip() for s in only.split(",") if s.strip()] or None
    _run_loop(
        config, start_gen, end_gen,
        force=force,
        skip_steps=skip_steps,
        only_steps=only_steps,
        stop_on_failure=stop_on_failure,
    )


@app.command("run-gen")
def run_gen(
        config: Annotated[str, typer.Argument(help="Path to base.yaml")],
        gen: Annotated[int, typer.Argument(help="Generation number")],
        force: Annotated[bool, typer.Option("--force")] = False,
        skip: Annotated[str, typer.Option("--skip")] = "",
        only: Annotated[str, typer.Option("--only")] = "",
):
    """Run (or resume) a single generation."""
    from mlip_pipeline.loop.runner import run_single_generation
    base = load_yaml(config)
    skip_steps = [s.strip() for s in skip.split(",") if s.strip()]
    only_steps = [s.strip() for s in only.split(",") if s.strip()] or None
    run_single_generation(base, gen, force=force, skip_steps=skip_steps, only_steps=only_steps)


main = app

if __name__ == "__main__":
    app()
