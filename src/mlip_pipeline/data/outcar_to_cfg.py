from __future__ import annotations

import subprocess
from pathlib import Path

from mlip_pipeline.models import LabelResult, ConvertResult
from mlip_pipeline.utils.fs import ensure_dir


def _count_cfg_blocks(path: Path) -> int:
    """Count BEGIN_CFG markers in a cfg file.  O(file-size), no parsing."""
    return path.read_text(encoding="utf-8").count("BEGIN_CFG")


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
) -> ConvertResult:
    convert_cfg = config.get("convert", {})
    base_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    run_name    = convert_cfg.get("run_name", "run_00")
    merge_name  = convert_cfg.get("merge_name", "train.cfg")
    mlp_command = convert_cfg.get("mlp_command", config.get("fit", {}).get("mlp_command", "mlp"))

    # Derive origin_cfg from training block (e.g. datasets/pb_cfg/train.cfg)
    origin_cfg_key = convert_cfg.get("origin_cfg")
    if origin_cfg_key:
        origin_cfg = (resolved_paths["datasets_root"] / origin_cfg_key).resolve()
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

    # ── 1. Snapshot the block count BEFORE we add anything ───────────────────
    merged_cfg = base_dir / merge_name
    prev_block_count = _count_cfg_blocks(merged_cfg) if merged_cfg.exists() else 0

    # ── 2. Convert each OUTCAR → per-task .cfg ────────────────────────────────
    this_run_cfgs: list[Path] = []
    for task_dir in sorted(label_result.task_dirs):
        outcar = task_dir / "OUTCAR"
        if not outcar.exists():
            print(f"  [SKIP] No OUTCAR in {task_dir.name}")
            continue

        out_cfg = run_dir / f"{task_dir.name}.cfg"
        cmd = [mlp_command, "convert", str(outcar), str(out_cfg), "--input_format=outcar"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"mlp convert failed for {task_dir.name}:\n"
                f"  cmd:    {' '.join(cmd)}\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}"
            )

        if not out_cfg.exists() or out_cfg.stat().st_size == 0:
            raise RuntimeError(
                f"mlp convert exited 0 but produced no output for {task_dir.name}.\n"
                f"  cmd:    {' '.join(cmd)}\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  Expected output: {out_cfg}"
            )

        this_run_cfgs.append(out_cfg)
        print(f"  [{len(this_run_cfgs):>4d}]  {task_dir.name}/OUTCAR  →  {out_cfg.name}")

    if not this_run_cfgs:
        raise FileNotFoundError(f"No OUTCARs found under {label_result.label_root}")

    # ── 3. Count blocks contributed by the new cfgs ───────────────────────────
    new_cfg_block_count = sum(_count_cfg_blocks(p) for p in this_run_cfgs)

    # ── 4. Collect ALL run cfgs (historical + this run) in sorted order ───────
    all_run_cfgs: list[Path] = []
    for run_subdir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        all_run_cfgs.extend(sorted(run_subdir.glob("task.*.cfg")))

    # ── 5. Rebuild accumulated train.cfg ──────────────────────────────────────
    _write_merged([origin_cfg] + all_run_cfgs, merged_cfg)

    # ── 6. Verify the written file has the expected number of blocks ──────────
    total_block_count = _count_cfg_blocks(merged_cfg)
    origin_block_count = _count_cfg_blocks(origin_cfg)
    expected_total = origin_block_count + sum(
        _count_cfg_blocks(p)
        for run_subdir in sorted(d for d in base_dir.iterdir() if d.is_dir())
        for p in sorted(run_subdir.glob("task.*.cfg"))
    )

    if total_block_count != expected_total:
        raise RuntimeError(
            f"Block count mismatch in {merged_cfg.name} after convert-cfg:\n"
            f"  Expected : {expected_total}  "
            f"(origin {origin_block_count} + all run cfgs)\n"
            f"  Actual   : {total_block_count}\n"
            f"  This run : {len(this_run_cfgs)} files, {new_cfg_block_count} blocks\n"
            f"  File     : {merged_cfg}\n"
            f"\n"
            f"Do NOT run 'fit' — the training set is incomplete.  "
            f"Re-run 'convert-cfg' or inspect {run_dir} for partial/empty cfgs."
        )

    print(f"\nDone. {len(this_run_cfgs)} new cfg(s) added to {run_dir.name}/")
    print(
        f"Accumulated {merge_name}: "
        f"origin({origin_block_count}) + all run cfgs → "
        f"{total_block_count} blocks total  [{merged_cfg}]"
    )

    return ConvertResult(
        merged_cfg=merged_cfg,
        prev_block_count=prev_block_count,
        new_cfg_count=len(this_run_cfgs),
        total_block_count=total_block_count,
        run_dir=run_dir,
    )
