from __future__ import annotations

import json
import typer
from typing import List, Optional
from typing_extensions import Annotated
from pathlib import Path

from mlip_pipeline.config import load_yaml, project_paths, gen_paths
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
    HpcJobState,
    submit_label_jobs,
    sync_inputs_to_dardel,
    submit_jobs_on_dardel,
    watch_queue,
    sync_outputs_from_dardel,
)
from mlip_pipeline.data.outcar_to_cfg import convert_outcars_to_cfg
from mlip_pipeline.evaluate.runner import run_evaluation, replot_evaluation

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


def _remote_dir(cfg: dict, paths: dict) -> str:
    d_cfg = cfg["label"]["dardel"]
    return f"{d_cfg['remote_root']}/{paths['runs_root'].name}/{cfg['label']['output_subdir']}"


def _resolve_gens(runs_root: Path, gens_str: str, require_cache: bool = False) -> list[int]:
    """Parse --gens value into a sorted list of generation numbers.

    'all'              → every gen_NN directory (optionally filtered to those
                         that have a cached predicted_train.cfg when
                         require_cache=True)
    '0,3,7'           → explicit list
    """
    if gens_str.strip().lower() == "all":
        gen_dirs = sorted(runs_root.glob("gen_*"))
        if require_cache:
            gen_dirs = [
                d for d in gen_dirs
                if (d / "fit" / "eval" / "predicted_train.cfg").exists()
            ]
        if not gen_dirs:
            return []
        return [int(d.name.split("_")[1]) for d in gen_dirs]
    try:
        return sorted({int(g.strip()) for g in gens_str.split(",") if g.strip()})
    except ValueError:
        raise typer.BadParameter(f"Invalid --gens value: {gens_str!r}. Use comma-separated integers or 'all'.")


def _runs_root_from_config(config_path: str) -> Path:
    from mlip_pipeline.loop.runner import _config_for_gen
    base_cfg = load_yaml(config_path)
    cfg0 = _config_for_gen(base_cfg, 0)
    return project_paths(cfg0)["runs_root"]


# ---------------------------------------------------------------------------
# PIPELINE STEP COMMANDS
# ---------------------------------------------------------------------------

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
    sync_inputs_to_dardel(
        label_result.label_root,
        d_cfg["user"],
        d_cfg.get("host", "dardel.pdc.kth.se"),
        _remote_dir(cfg, paths),
    )


@app.command("submit-remote")
def submit_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    """
    Submit VASP jobs on Dardel and persist the resulting Slurm job IDs to
    slurm_job_ids.json inside the label directory so that watch-remote can
    be re-attached independently.
    """
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    d_cfg = cfg["label"]["dardel"]
    job_ids = submit_jobs_on_dardel(
        d_cfg["user"],
        d_cfg.get("host", "dardel.pdc.kth.se"),
        _remote_dir(cfg, paths),
        local_label_dir=label_root,
    )
    print(f"Submitted {len(job_ids)} job(s). IDs saved to {label_root}/slurm_job_ids.json")


@app.command("watch-remote")
def watch_remote(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    evaluate: Annotated[bool, typer.Option("--evaluate/--no-evaluate",
        help="Run evaluation locally while waiting for HPC jobs")] = True,
):
    """
    Watch the Dardel queue for jobs submitted by submit-remote.
    """
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    d_cfg = cfg["label"]["dardel"]
    user = d_cfg["user"]
    host = d_cfg.get("host", "dardel.pdc.kth.se")
    poll = d_cfg.get("poll_interval", 60)

    job_ids: list[str] | None = None
    if HpcJobState.exists(label_root):
        try:
            hpc_state = HpcJobState.load(label_root)
            job_ids = hpc_state.job_ids
            print(f"Resuming watch for {len(job_ids)} job(s) from {hpc_state.path}")
        except Exception as exc:
            typer.echo(f"WARNING: could not load slurm_job_ids.json ({exc}) — falling back to user-scoped squeue", err=True)
    else:
        typer.echo("WARNING: slurm_job_ids.json not found — falling back to user-scoped squeue. "
                   "Run submit-remote to persist job IDs next time.", err=True)

    eval_thread = None
    if evaluate:
        import threading
        fit_result = build_fit_result(cfg, paths)
        def _run_eval():
            print("\n[evaluate]  Starting evaluation in background...")
            try:
                result = run_evaluation(cfg, paths, fit_result)
                print(f"[evaluate]  Done — {len(result.plot_paths)} plot(s) in {result.eval_dir}/")
            except Exception as exc:
                print(f"[evaluate]  WARNING: evaluation failed: {exc}")
        eval_thread = threading.Thread(target=_run_eval, daemon=True, name="evaluate")
        eval_thread.start()

    watch_queue(user, host, poll_interval=poll, job_ids=job_ids)

    if eval_thread is not None:
        print("[evaluate]  Waiting for evaluation thread to finish...")
        eval_thread.join(timeout=600)
        if eval_thread.is_alive():
            print("[evaluate]  WARNING: evaluation thread still running after 10 min — not waiting further.")


@app.command("sync-from-remote")
def sync_from_remote(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)
    d_cfg = cfg["label"]["dardel"]
    sync_outputs_from_dardel(
        label_result.label_root,
        d_cfg["user"],
        d_cfg.get("host", "dardel.pdc.kth.se"),
        _remote_dir(cfg, paths),
    )


@app.command("convert-cfg")
def convert_cfg(config: Annotated[str, typer.Option(..., help="Path to config yaml")]):
    cfg, paths = get_paths(config)
    label_root = paths["runs_root"] / cfg["label"]["output_subdir"]
    label_result = LabelResult.load_from_dir(label_root)
    merged = convert_outcars_to_cfg(label_result, cfg, paths)
    print(merged)


# ---------------------------------------------------------------------------
# EVALUATION COMMANDS
#
# Two commands, clear separation of concerns:
#
#   evaluate      — run mlp calculate_efs (expensive MTP inference) + plot.
#                   Skips inference if predicted_train.cfg already exists,
#                   unless --force is given.
#                   Accepts --gen N (single gen) or --gens 0,3,all (multi).
#
#   plot-eval     — regenerate plots only from an existing predicted_train.cfg.
#                   Fast; never re-runs calculate_efs.
#                   Accepts --gens 0,3,all.
#
#   compare-parity — side-by-side parity subplots across chosen gens.
#                   Reads cached predicted_train.cfg; no MTP inference.
#                   Accepts --gens 0,17 and --quantities energy,forces,stress.
# ---------------------------------------------------------------------------

@app.command("evaluate")
def evaluate(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gen: Annotated[Optional[int], typer.Option(
        "--gen",
        help="Single generation number. Mutually exclusive with --gens.",
    )] = None,
    gens: Annotated[str, typer.Option(
        "--gens",
        help="Comma-separated generation numbers or 'all'. Mutually exclusive with --gen.",
    )] = "",
    force: Annotated[bool, typer.Option(
        "--force",
        help="Re-run mlp calculate_efs even when predicted_train.cfg already exists.",
    )] = False,
    update_state: Annotated[bool, typer.Option(
        "--update-state/--no-update-state",
        help="Update state.json evaluation fields after completion (default: true).",
    )] = True,
):
    """
    Run evaluation (mlp calculate_efs + plots) for one or more generations.

    By default, calculate_efs is skipped when predicted_train.cfg already
    exists — pass --force to always re-run inference.

    Examples
    --------
    # Single gen (current-loop style):
        mlip-pipeline evaluate --config configs/Pb_loop.yaml --gen 5

    # All gens (replaces old regenerate-eval):
        mlip-pipeline evaluate --config configs/Pb_loop.yaml --gens all

    # Force-rerun even if cache exists:
        mlip-pipeline evaluate --config configs/Pb_loop.yaml --gens all --force

    # Subset:
        mlip-pipeline evaluate --config configs/Pb_loop.yaml --gens 0,3,7
    """
    if gen is not None and gens:
        raise typer.BadParameter("--gen and --gens are mutually exclusive.")

    from mlip_pipeline.loop.runner import _config_for_gen, _resolve_fit_result
    from mlip_pipeline.models import GenerationState

    base_cfg = load_yaml(config)

    # Build generation list
    if gen is not None:
        gen_list = [gen]
    elif gens:
        runs_root = _runs_root_from_config(config)
        gen_list = _resolve_gens(runs_root, gens)
    else:
        # No --gen or --gens: behave like old single-gen `evaluate` using
        # non-loop paths (backward compat for non-loop workflows).
        cfg, paths = get_paths(config)
        fit_result = build_fit_result(cfg, paths)
        if force:
            predicted = paths["runs_root"] / cfg["fit"]["output_subdir"] / "eval" / "predicted_train.cfg"
            predicted.unlink(missing_ok=True)
        result = run_evaluation(cfg, paths, fit_result)
        for p in result.plot_paths.values():
            print(p)
        return

    if not gen_list:
        typer.echo("No generations found to evaluate.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Evaluating generation(s): {gen_list}{' [--force: cache cleared]' if force else ''}")

    n_ok = n_fail = 0
    for gen_num in gen_list:
        gen_tag = f"gen_{gen_num:02d}"
        cfg = _config_for_gen(base_cfg, gen_num)
        paths = project_paths(cfg)
        paths = {**paths, **gen_paths(cfg, paths)}
        gen_dir = paths["runs_root"] / gen_tag

        if not gen_dir.exists():
            typer.echo(f"  {gen_tag}: directory not found — skipping.", err=True)
            n_fail += 1
            continue

        try:
            state_file = gen_dir / "state.json"
            state = GenerationState.load(gen_dir) if state_file.exists() else GenerationState.init(str(gen_num), gen_dir)

            try:
                fit_result = _resolve_fit_result(cfg, paths, state)
            except RuntimeError as e:
                typer.echo(f"  {gen_tag}: {e} — skipping.", err=True)
                n_fail += 1
                continue

            if force:
                predicted = gen_dir / "fit" / "eval" / "predicted_train.cfg"
                predicted.unlink(missing_ok=True)

            typer.echo(f"  {gen_tag}: running evaluate (model={fit_result.model_path}) ...")
            result = run_evaluation(cfg, paths, fit_result)
            typer.echo(f"  {gen_tag}: {len(result.plot_paths)} plot(s) → {result.eval_dir}/")

            if update_state and state_file.exists():
                state_data = json.loads(state_file.read_text())
                state_data["evaluation_done"]     = True
                state_data["evaluation_failed"]   = False
                state_data["evaluation_manifest"] = str(result.eval_dir / "eval_manifest.json")
                state_file.write_text(json.dumps(state_data, indent=2))

            n_ok += 1
        except Exception as exc:
            typer.echo(f"  {gen_tag}: FAILED — {exc}", err=True)
            n_fail += 1

    typer.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        raise typer.Exit(1)


@app.command("plot-eval")
def plot_eval(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gens: Annotated[str, typer.Option(
        "--gens",
        help=(
            "Comma-separated generation numbers, or 'all' to process every "
            "gen_NN directory that has a cached predicted_train.cfg."
        ),
    )] = "all",
    update_state: Annotated[bool, typer.Option(
        "--update-state/--no-update-state",
        help="Update state.json evaluation_manifest path after replotting (default: true).",
    )] = True,
):
    """
    Regenerate evaluation plots from cached data WITHOUT re-running mlp calculate_efs.

    Use this after changing plot styles, adding new plot types, or fixing a
    missing PNG — without paying the MTP inference cost.

    Requires that 'evaluate' has been run at least once so that
    predicted_train.cfg exists in gen_NN/fit/eval/.

    Examples
    --------
    # Replot all gens that have cached data:
        mlip-pipeline plot-eval --config configs/Pb_loop.yaml

    # Replot a subset:
        mlip-pipeline plot-eval --config configs/Pb_loop.yaml --gens 0,3,7
    """
    from mlip_pipeline.loop.runner import _config_for_gen, _resolve_fit_result
    from mlip_pipeline.models import GenerationState

    base_cfg = load_yaml(config)
    runs_root = _runs_root_from_config(config)
    gen_list = _resolve_gens(runs_root, gens, require_cache=True)

    if not gen_list:
        typer.echo(
            "No gen_NN/fit/eval/predicted_train.cfg files found.\n"
            "Run 'evaluate' first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Replotting generation(s): {gen_list}")

    n_ok = n_fail = 0
    for gen_num in gen_list:
        gen_tag = f"gen_{gen_num:02d}"
        gen_dir = runs_root / gen_tag

        if not gen_dir.exists():
            typer.echo(f"  {gen_tag}: directory not found — skipping.", err=True)
            n_fail += 1
            continue

        try:
            cfg   = _config_for_gen(base_cfg, gen_num)
            paths = project_paths(cfg)
            paths = {**paths, **gen_paths(cfg, paths)}

            state_file = gen_dir / "state.json"
            state = GenerationState.load(gen_dir) if state_file.exists() else GenerationState.init(str(gen_num), gen_dir)

            fit_result = _resolve_fit_result(cfg, paths, state)

            typer.echo(f"  {gen_tag}: replotting from cache ...")
            result = replot_evaluation(cfg, paths, fit_result)
            typer.echo(f"  {gen_tag}: {len(result.plot_paths)} plot(s) → {result.eval_dir}/")

            if update_state and state_file.exists():
                state_data = json.loads(state_file.read_text())
                state_data["evaluation_manifest"] = str(result.eval_dir / "eval_manifest.json")
                state_file.write_text(json.dumps(state_data, indent=2))

            n_ok += 1
        except FileNotFoundError as exc:
            typer.echo(f"  {gen_tag}: {exc} — skipping.", err=True)
            n_fail += 1
        except Exception as exc:
            typer.echo(f"  {gen_tag}: FAILED — {exc}", err=True)
            n_fail += 1

    typer.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        raise typer.Exit(1)


@app.command("compare-parity")
def compare_parity(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gens: Annotated[str, typer.Option(
        "--gens",
        help="Comma-separated generation numbers to compare, e.g. '0,17'. Order is preserved.",
    )] = "0",
    quantities: Annotated[str, typer.Option(
        "--quantities",
        help="Comma-separated quantities to plot: energy, forces, stress. Default: energy,forces",
    )] = "energy,forces",
    output: Annotated[Optional[str], typer.Option(
        "--output",
        help="Output directory for PNG files. Defaults to runs_root/loop_summary/compare_parity/",
    )] = None,
):
    """
    Produce side-by-side parity comparison plots across chosen generations.

    Reads cached predicted_train.cfg files — no MTP inference is run.
    Each requested quantity gets its own PNG with one subplot per generation.

    Examples
    --------
    # Compare gen 0 and gen 17 for energy and forces (default):
        mlip-pipeline compare-parity --config configs/Pb_loop.yaml --gens 0,17

    # Energy only:
        mlip-pipeline compare-parity --config configs/Pb_loop.yaml --gens 0,5,10,17 --quantities energy

    # All three quantities:
        mlip-pipeline compare-parity --config configs/Pb_loop.yaml --gens 0,17 --quantities energy,forces,stress

    # Custom output directory:
        mlip-pipeline compare-parity --config configs/Pb_loop.yaml --gens 0,17 --output runs/my_comparisons/
    """
    from mlip_pipeline.loop.runner import _config_for_gen, _resolve_fit_result
    from mlip_pipeline.evaluate.parity import parse_cfg_efs, build_parity_data
    from mlip_pipeline.evaluate.plots import plot_parity_comparison
    from mlip_pipeline.models import GenerationState

    base_cfg  = load_yaml(config)
    runs_root = _runs_root_from_config(config)

    # Parse gen list (order preserved, no sorting — user controls display order)
    try:
        gen_list = [int(g.strip()) for g in gens.split(",") if g.strip()]
    except ValueError:
        raise typer.BadParameter(f"Invalid --gens value: {gens!r}. Use comma-separated integers.")

    if not gen_list:
        typer.echo("No generations specified.", err=True)
        raise typer.Exit(1)

    qty_list = [q.strip().lower() for q in quantities.split(",") if q.strip()]
    valid_qtys = {"energy", "forces", "stress"}
    unknown_qtys = [q for q in qty_list if q not in valid_qtys]
    if unknown_qtys:
        raise typer.BadParameter(
            f"Unknown quantities: {unknown_qtys}. Valid: energy, forces, stress"
        )

    dest_dir = Path(output) if output else runs_root / "loop_summary" / "compare_parity"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load parity data for each gen from cache
    gen_parities: list[tuple[int, dict]] = []
    for gen_num in gen_list:
        gen_tag   = f"gen_{gen_num:02d}"
        gen_dir   = runs_root / gen_tag
        eval_dir  = gen_dir / "fit" / "eval"
        pred_cfg  = eval_dir / "predicted_train.cfg"

        if not pred_cfg.exists():
            typer.echo(
                f"  {gen_tag}: predicted_train.cfg not found at {pred_cfg} — skipping."
                " Run 'evaluate' first.",
                err=True,
            )
            continue

        try:
            cfg   = _config_for_gen(base_cfg, gen_num)
            paths = project_paths(cfg)
            paths = {**paths, **gen_paths(cfg, paths)}

            # Resolve train_cfg the same way runner.py does
            from mlip_pipeline.evaluate.runner import _resolve_train_cfg
            train_cfg = _resolve_train_cfg(cfg, paths)

            ref_records  = parse_cfg_efs(train_cfg)
            pred_records = parse_cfg_efs(pred_cfg)
            parity       = build_parity_data(ref_records, pred_records)
            gen_parities.append((gen_num, parity))
            typer.echo(f"  {gen_tag}: {len(ref_records)} configs loaded.")
        except Exception as exc:
            typer.echo(f"  {gen_tag}: FAILED to load parity data — {exc}", err=True)

    if not gen_parities:
        typer.echo("No parity data could be loaded. Aborting.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generating comparison plots for quantities: {qty_list} ...")
    written = plot_parity_comparison(gen_parities, qty_list, dest_dir)

    for p in written:
        typer.echo(str(p))

    typer.echo(f"\n{len(written)} plot(s) written to {dest_dir}/")


# ---------------------------------------------------------------------------
# PHYSICS VALIDATION
# ---------------------------------------------------------------------------

@app.command("validate")
def validate(
    config: Annotated[str, typer.Option(..., help="Path to validate yaml (e.g. configs/Pb_validate.yaml)")],
    model: Annotated[str, typer.Option(..., "--model", help="Path to trained .almtp potential")],
    outdir: Annotated[Optional[str], typer.Option(
        "--outdir",
        help="Output directory for plots and manifest. Defaults to <model_dir>/validate/",
    )] = None,
):
    """
    Run physics validation (EOS + elastic constants) for a trained MTP potential.

    Runs LAMMPS to compute E(V) curves and finite-difference elastic constants,
    fits a Birch-Murnaghan EOS, and writes plots + a JSON manifest.

    Examples
    --------
    # Pb after generation 16:
        mlip-pipeline validate \\
            --config configs/Pb_validate.yaml \\
            --model  runs/gen_16/fit/Pb16.almtp

    # Custom output directory:
        mlip-pipeline validate \\
            --config configs/Pb_validate.yaml \\
            --model  runs/gen_16/fit/Pb16.almtp \\
            --outdir runs/gen_16/validate
    """
    from mlip_pipeline.validate.runner import run_validation

    model_path = Path(model)
    if not model_path.exists():
        typer.echo(f"Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    out_path = Path(outdir) if outdir else model_path.parent / "validate"

    cfg = load_yaml(config)
    result = run_validation(cfg, model_path, out_path)

    manifest = result.validate_dir / "validate_manifest.json"
    typer.echo(f"Manifest: {manifest}")

    if result.eos_results:
        for eos in result.eos_results:
            if eos.fit_ok:
                typer.echo(
                    f"  EOS [{eos.structure_id}]  V0={eos.V0:.3f} Å³  B0={eos.B0:.1f} GPa  "
                    f"B0'={eos.B0p:.2f}  E0={eos.E0:.4f} eV"
                )
            else:
                typer.echo(f"  EOS [{eos.structure_id}]  fit failed: {eos.fit_error}")

    if result.elastic_results:
        for el in result.elastic_results:
            if el.compute_ok:
                typer.echo(
                    f"  Elastic [{el.structure_id}]  B={el.B_voigt:.1f} GPa  "
                    f"G={el.G_voigt:.1f} GPa  Cij={el.C}"
                )
            else:
                typer.echo(f"  Elastic [{el.structure_id}]  failed: {el.error}")

    for name, p in result.plot_paths.items():
        typer.echo(f"  Plot [{name}]: {p}")


# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------

_LABEL_ADJACENT_STEPS = {"label", "label_local", "label_hpc", "convert"}


@app.command("reset-steps")
def reset_steps(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    gen: Annotated[int, typer.Option(..., "--gen", help="Generation number")],
    steps: Annotated[str, typer.Option(..., "--steps", help="Comma-separated steps to reset")],
):
    """
    Remove one or more steps from a generation's completed_steps so the loop
    will re-run them on the next invocation.
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

    removed = [s for s in before if s in requested]
    dirty = bool(removed)

    if "evaluate" in requested:
        state["evaluation_done"] = False
        state["evaluation_failed"] = False
        state["evaluation_manifest"] = None
        if state.get("status") == "completed":
            state["status"] = "running"
            state["error"] = None
            state["completed_at"] = None
        dirty = True

    if dirty:
        if removed and "evaluate" not in requested and state.get("status") == "completed":
            state["status"] = "running"
            state["error"] = None
            state["completed_at"] = None

        if _LABEL_ADJACENT_STEPS.intersection(requested):
            state["label_prepared"] = False

        state_file.write_text(json.dumps(state, indent=2))
        msg_parts = []
        if removed:
            msg_parts.append(f"Removed step(s) {removed} from completed_steps.")
        if "evaluate" in requested:
            msg_parts.append("Cleared evaluation_done/failed/manifest and reopened status.")
        typer.echo(f"{gen_tag}: " + " ".join(msg_parts))
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


# ---------------------------------------------------------------------------
# ACTIVE LEARNING LOOPS
# ---------------------------------------------------------------------------

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


@app.command("plot-loop")
def plot_loop(
    config: Annotated[str, typer.Option(..., help="Path to config yaml")],
    start_gen: Annotated[int, typer.Option("--start-gen", help="First generation to include")] = 0,
    end_gen: Annotated[Optional[int], typer.Option("--end-gen", help="Last generation to include (inclusive)")] = None,
):
    """
    Collect per-generation evaluation records and produce loop-summary plots.
    """
    from mlip_pipeline.evaluate.loop_plots import collect_loop_records, plot_loop_summary

    cfg, paths = get_paths(config)
    runs_root = paths["runs_root"]

    if end_gen is None:
        gen_dirs = sorted(runs_root.glob("gen_*"))
        if not gen_dirs:
            typer.echo("No generation directories found.", err=True)
            raise typer.Exit(1)
        end_gen = int(gen_dirs[-1].name.split("_")[1])

    generations = list(range(start_gen, end_gen + 1))
    records = collect_loop_records(runs_root, generations)

    if not records:
        typer.echo("No records collected — ensure generations have completed state.json files.", err=True)
        raise typer.Exit(1)

    output_dir = runs_root / "loop_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_loop_summary(records, output_dir)
    for p in plot_paths:
        typer.echo(str(p))


main = app

if __name__ == "__main__":
    app()
