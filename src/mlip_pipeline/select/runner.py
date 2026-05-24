from __future__ import annotations

from pathlib import Path

from mlip_pipeline.models import ExploreResult, SelectionResult
from mlip_pipeline.utils.fs import ensure_dir, copy_if_exists
from mlip_pipeline.utils.shell import run_command
from mlip_pipeline.utils.logging import warn, info


def run_selection(
    config: dict,
    resolved_paths: dict,
    fit_result,
) -> SelectionResult:
    select_cfg  = config["select"]
    runs_root   = resolved_paths["runs_root"]

    explore_root = runs_root / select_cfg["input_subdir"]
    select_root  = ensure_dir(runs_root / select_cfg["output_subdir"])
    manifest_path = select_root / "selection_manifest.json"

    candidate_filename         = select_cfg.get("candidate_filename", "preselected.cfg")
    merged_candidates_filename = select_cfg.get("merged_candidates_filename", "candidates_merged.cfg")
    selected_filename          = select_cfg.get("selected_filename", "selected.cfg")
    mlip_command               = select_cfg.get("mlip_command", "mlp")
    mpi_np                     = select_cfg.get("mpi_np", None)
    mpi_command                = select_cfg.get("mpi_command", "mpirun")

    training_cfg = Path(select_cfg["training_cfg"])
    if not training_cfg.is_absolute():
        training_cfg = resolved_paths["project_root"] / training_cfg
    if not training_cfg.exists():
        raise FileNotFoundError(f"Training cfg not found: {training_cfg}")

    # ── Resolve candidate paths from explore manifest, fall back to disk scan ──
    explore_manifest = explore_root / "explore_manifest.json"
    if explore_manifest.exists():
        explore_result = ExploreResult.load_manifest(explore_root)
        candidate_paths = [
            p for p in explore_result.preselected_cfgs if p.exists()
        ]
        if explore_result.n_failed:
            warn(
                f"select: {explore_result.n_failed} explore run(s) failed — "
                f"their candidates are excluded. "
                f"Failed dirs: {[str(p) for p in explore_result.failed_runs]}"
            )
        info(
            f"select: loaded {len(candidate_paths)} candidate file(s) "
            f"from explore_manifest.json "
            f"({explore_result.n_ok}/{explore_result.n_runs} runs ok)"
        )
    else:
        # Fallback for runs that pre-date the manifest
        warn(
            f"select: explore_manifest.json not found under {explore_root} — "
            "falling back to disk scan for candidate files."
        )
        candidate_paths = sorted(explore_root.rglob(candidate_filename))

    # ── No candidates = converged ────────────────────────────────────────────
    if not candidate_paths:
        warn(
            f"No {candidate_filename!r} files found under {explore_root}. "
            "The potential did not extrapolate — generation is likely converged. "
            "Skipping select / label / convert."
        )
        result = SelectionResult(
            select_root=select_root,
            manifest_path=manifest_path,
            selected_cfg_paths=[],
            selected_count=0,
            converged=True,
        )
        result.save_manifest()
        return result

    # ── Normal path ───────────────────────────────────────────────────────────
    info(f"Found {len(candidate_paths)} candidate file(s) — running mlp select_add.")

    merged_candidates_path = select_root / merged_candidates_filename
    selected_cfg_path      = select_root / selected_filename

    model_name = fit_result.model_path.name
    model_path = select_root / model_name
    copy_if_exists(fit_result.model_path, model_path)

    merge_cfg_files(candidate_paths, merged_candidates_path)

    mlp_args = [
        mlip_command, "select_add",
        model_path.name,
        str(training_cfg),
        merged_candidates_path.name,
        selected_cfg_path.name,
    ]
    command = ([mpi_command, "-np", str(mpi_np)] + mlp_args) if mpi_np is not None else mlp_args

    log_path    = select_root / "select_add.log"
    return_code = run_command(command, cwd=select_root, log_file=log_path)
    if return_code != 0:
        raise RuntimeError(
            f"mlp select_add failed (exit {return_code}); see {log_path}"
        )

    if not selected_cfg_path.exists():
        raise FileNotFoundError(f"selected cfg not created: {selected_cfg_path}")

    selected_blocks = split_cfg_blocks(selected_cfg_path.read_text())
    selected_dir    = ensure_dir(select_root / "selected_blocks")
    selected_cfg_paths: list[Path] = []

    for i, block in enumerate(selected_blocks):
        out_path = selected_dir / f"selected_{i:05d}.cfg"
        out_path.write_text(block.strip() + "\n")
        selected_cfg_paths.append(out_path)

    candidate_sources = [
        {"source_cfg": str(p), "temperature_dir": p.parent.name}
        for p in candidate_paths
    ]

    result = SelectionResult(
        select_root=select_root,
        manifest_path=manifest_path,
        selected_cfg_paths=selected_cfg_paths,
        selected_count=len(selected_cfg_paths),
        model_path=model_path,
        candidate_sources=candidate_sources,
        converged=False,
    )
    result.save_manifest()
    return result


def merge_cfg_files(input_paths: list[Path], output_path: Path) -> None:
    with output_path.open("w") as fout:
        for path in input_paths:
            text = path.read_text().strip()
            if not text:
                continue
            fout.write(text)
            fout.write("\n")


def split_cfg_blocks(text: str) -> list[str]:
    blocks, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BEGIN_CFG"):
            current = [line]
        elif stripped.startswith("END_CFG"):
            current.append(line)
            blocks.append("\n".join(current))
            current = []
        elif current:
            current.append(line)
    return blocks


def build_source_manifest(candidate_paths: list[Path]) -> list[dict]:
    return [
        {"source_cfg": str(p), "temperature_dir": p.parent.name}
        for p in candidate_paths
    ]
