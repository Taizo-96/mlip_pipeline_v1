from __future__ import annotations

import json
import shutil
from pathlib import Path

from mlip_pipeline.models import SelectionResult
from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.utils.shell import run_command


def run_selection(
    config: dict,
    resolved_paths: dict,
    fit_result,
) -> SelectionResult:
    select_cfg = config["select"]
    runs_root = resolved_paths["runs_root"]

    explore_root = runs_root / select_cfg["input_subdir"]
    select_root = ensure_dir(runs_root / select_cfg["output_subdir"])

    candidate_filename = select_cfg.get("candidate_filename", "preselected.cfg")
    merged_candidates_filename = select_cfg.get(
        "merged_candidates_filename", "candidates_merged.cfg"
    )
    selected_filename = select_cfg.get("selected_filename", "selected.cfg")
    mlip_command = select_cfg.get("mlip_command", "mlp")
    write_manifest = bool(select_cfg.get("write_manifest", True))

    training_cfg = Path(select_cfg["training_cfg"])
    if not training_cfg.is_absolute():
        training_cfg = resolved_paths["project_root"] / training_cfg

    if not training_cfg.exists():
        raise FileNotFoundError(f"training cfg not found: {training_cfg}")

    candidate_paths = sorted(explore_root.rglob(candidate_filename))
    if not candidate_paths:
        raise FileNotFoundError(
            f"no {candidate_filename} files found under {explore_root}"
        )

    merged_candidates_path = select_root / merged_candidates_filename
    selected_cfg_path = select_root / selected_filename
    manifest_path = select_root / "selection_manifest.json"

    model_name = fit_result.model_path.name
    model_path = select_root / model_name
    shutil.copy(fit_result.model_path, model_path)

    merge_cfg_files(candidate_paths, merged_candidates_path)

    command = [
        mlip_command,
        "select_add",
        model_path.name,
        str(training_cfg),
        merged_candidates_path.name,
        selected_cfg_path.name,
    ]

    log_path = select_root / "select_add.log"
    return_code = run_command(command, cwd=select_root, log_file=log_path)
    if return_code != 0:
        raise RuntimeError(
            f"mlp select_add failed with return code {return_code}; see {log_path}"
        )

    if not selected_cfg_path.exists():
        raise FileNotFoundError(f"selected cfg not created: {selected_cfg_path}")

    selected_blocks = split_cfg_blocks(selected_cfg_path.read_text())
    selected_dir = ensure_dir(select_root / "selected_blocks")
    selected_cfg_paths: list[Path] = []

    for i, block in enumerate(selected_blocks):
        out_path = selected_dir / f"selected_{i:05d}.cfg"
        out_path.write_text(block.strip() + "\n")
        selected_cfg_paths.append(out_path)

    if write_manifest:
        source_map = build_source_manifest(candidate_paths)
        manifest = {
            "strategy": "mlip_select_add",
            "model_path": str(model_path),
            "training_cfg": str(training_cfg),
            "merged_candidates_path": str(merged_candidates_path),
            "selected_cfg_path": str(selected_cfg_path),
            "selected_count": len(selected_cfg_paths),
            "candidate_sources": source_map,
            "selected_block_files": [str(p) for p in selected_cfg_paths],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
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
    lines = text.splitlines()
    blocks = []
    current = []

    for line in lines:
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
    items = []
    for path in candidate_paths:
        items.append(
            {
                "source_cfg": str(path),
                "temperature_dir": path.parent.name,
            }
        )
    return items