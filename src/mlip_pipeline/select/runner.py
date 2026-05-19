from __future__ import annotations

import json
from pathlib import Path

from mlip_pipeline.models import SelectionResult
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

    candidate_filename         = select_cfg.get("candidate_filename", "preselected.cfg")
    merged_candidates_filename = select_cfg.get("merged_candidates_filename", "candidates_merged.cfg")
    selected_filename          = select_cfg.get("selected_filename", "selected.cfg")
    mlip_command               = select_cfg.get("mlip_command", "mlp")
    mpi_np                     = select_cfg.get("mpi_np", None)
    mpi_command                = select_cfg.get("mpi_command", "mpirun")
    write_manifest             = bool(select_cfg.get("write_manifest", True))

    training_cfg = Path(select_cfg["training_cfg"])
    if not training_cfg.is_absolute():
        training_cfg = resolved_paths["project_root"] / training_cfg
    if not training_cfg.exists():
        raise FileNotFoundError(f"Training cfg not found: {training_cfg}")

    # ── Check for candidates ───────────────────────────────────────────────
    candidate_paths = sorted(explore_root.rglob(candidate_filename))

    # No candidates = potential did not extrapolate on any run.
    # This is a convergence signal, not an error. Return empty result so
    # the loop can skip label+convert for this generation.
    if not candidate_paths:
        warn(
            f"No {candidate_filename!r} files found under {explore_root}. "
            "The potential did not extrapolate — generation is likely converged. "
            "Skipping select / label / convert."
        )
        manifest_path = select_root / "selection_manifest.json"
        if write_manifest:
            manifest_path.write_text(json.dumps({
                "strategy": "mlip_select_add",
                "converged": True,
                "selected_count": 0,
                "note": "No extrapolative structures found during explore.",
            }, indent=2))
        return SelectionResult(
            select_root=select_root,
            manifest_path=manifest_path,
            selected_cfg_paths=[],
            selected_count=0,
        )

    # ── Normal path ───────────────────────────────────────────────────────
    info(f"Found {len(candidate_paths)} candidate file(s) — running mlp select_add.")

    merged_candidates_path = select_root / merged_candidates_filename
    selected_cfg_path      = select_root / selected_filename
    manifest_path          = select_root / "selection_manifest.json"

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

    if write_manifest:
        manifest_path.write_text(json.dumps({
            "strategy": "mlip_select_add",
            "converged": False,
            "model_path": str(model_path),
            "training_cfg": str(training_cfg),
            "merged_candidates_path": str(merged_candidates_path),
            "selected_cfg_path": str(selected_cfg_path),
            "selected_count": len(selected_cfg_paths),
            "candidate_sources": build_source_manifest(candidate_paths),
            "selected_block_files": [str(p) for p in selected_cfg_paths],
            "mpi_np": mpi_np,
        }, indent=2))
    else:
        manifest_path.touch()

    return SelectionResult(
        select_root=select_root,
        manifest_path=manifest_path,
        selected_cfg_paths=selected_cfg_paths,
        selected_count=len(selected_cfg_paths),
    )


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
