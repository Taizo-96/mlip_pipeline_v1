from __future__ import annotations

import subprocess
from pathlib import Path

from mlip_pipeline.models import LabelResult
from mlip_pipeline.utils.fs import ensure_dir


def _write_merged(sources: list[Path], dest: Path) -> None:
    with open(dest, "w", encoding="utf-8") as out:
        for cfg in sources:
            text = cfg.read_text(encoding="utf-8")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")


def convert_outcars_to_cfg(
    label_result: LabelResult,
    config: dict,
    resolved_paths: dict,
) -> Path:
    convert_cfg = config.get("convert", {})
    base_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    run_name    = convert_cfg.get("run_name", "run_00")
    merge_name  = convert_cfg.get("merge_name", "train.cfg")

    # Bug 2 fix: derive origin_cfg explicitly from training block, no fallback string
    training_cfg = config.get("convert", {}).get("origin_cfg")
    if training_cfg:
        origin_cfg = (resolved_paths["datasets_root"] / training_cfg).resolve()
    else:
        training_block = config["training"]
        origin_cfg = (
            resolved_paths["datasets_root"]
            / training_block["output_subdir"]
            / training_block["merge_name"]
        ).resolve()

    if not origin_cfg.exists():
        raise FileNotFoundError(
            f"Origin cfg not found: {origin_cfg}. Run 'prepare-train' first."
        )

    base_dir = (resolved_paths["datasets_root"] / base_subdir).resolve()
    run_dir  = ensure_dir(base_dir / run_name)

    # ── 1. Convert each OUTCAR → per-task .cfg ────────────────────────────────
    this_run_cfgs: list[Path] = []
    for task_dir in sorted(label_result.task_dirs):
        outcar = task_dir / "OUTCAR"
        if not outcar.exists():
            print(f"  [SKIP] No OUTCAR in {task_dir.name}")
            continue

        out_cfg = run_dir / f"{task_dir.name}.cfg"
        result  = subprocess.run(
            ["mlp", "convert-cfg", str(outcar), str(out_cfg)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mlp convert-cfg failed for {task_dir.name}:\n{result.stderr}"
            )
        this_run_cfgs.append(out_cfg)
        print(f"  [{len(this_run_cfgs):>4d}]  {task_dir.name}/OUTCAR  →  {out_cfg.name}")

    if not this_run_cfgs:
        raise FileNotFoundError(f"No OUTCARs found under {label_result.label_root}")

    # ── 2. Bug 1 fix: collect all run cfgs by scanning subdirs explicitly ─────
    # Sorted subdirs ensures run_00 < run_01 < run_02, etc.
    all_run_cfgs: list[Path] = []
    for run_subdir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        all_run_cfgs.extend(sorted(run_subdir.glob("task.*.cfg")))

    # ── 3. Rebuild accumulated train.cfg: origin first, then all run cfgs ─────
    merged_cfg = base_dir / merge_name
    _write_merged([origin_cfg] + all_run_cfgs, merged_cfg)

    print(f"\nDone. {len(this_run_cfgs)} new cfg(s) added to {run_dir.name}/")
    print(f"Accumulated {merge_name}: origin({origin_cfg.name}) + {len(all_run_cfgs)} run cfg(s) → {merged_cfg}")
    return merged_cfg