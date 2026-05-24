from __future__ import annotations

import subprocess
from pathlib import Path

from mlip_pipeline.models import ConvertResult, LabelResult
from mlip_pipeline.utils.fs import ensure_dir


def _count_cfg_blocks(path: Path) -> int:
    """Count BEGIN_CFG markers in a cfg file.  O(file-size), no AST parsing."""
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
    """
    Convert OUTCARs → .cfg files and accumulate them into a merged train.cfg.

    Directory layout
    ----------------
    Per-gen storage (flat, one dir per generation inside accum_dir):
        datasets/<accum_subdir>/<gen_tag>/task.000000.cfg
        e.g.  datasets/converted_cfg/gen_11/task.000000.cfg

    Accumulation root (fixed across all gens):
        datasets/<accum_subdir>/train.cfg
        e.g.  datasets/converted_cfg/train.cfg

    The accumulation root is what `fit` reads via config.fit.train_cfg.

    After writing, the block count is verified against the sum of all
    sources.  A RuntimeError is raised if the counts do not match so that
    `fit` is never launched against a stale or incomplete training set.
    """
    convert_cfg  = config.get("convert", {})
    accum_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    gen_subdir   = convert_cfg.get("gen_subdir", "gen_00")
    merge_name   = convert_cfg.get("merge_name", "train.cfg")
    mlp_command  = convert_cfg.get(
        "mlp_command", config.get("fit", {}).get("mlp_command", "mlp")
    )

    datasets_root = resolved_paths["datasets_root"]

    # ── Origin cfg (the pre-loop training set, e.g. datasets/pb_cfg/train.cfg)
    origin_cfg_key = convert_cfg.get("origin_cfg")
    if origin_cfg_key:
        origin_cfg = (datasets_root / origin_cfg_key).resolve()
    else:
        training_block = config["training"]
        origin_cfg = (
            datasets_root
            / training_block["output_subdir"]
            / training_block["merge_name"]
        ).resolve()

    if not origin_cfg.exists():
        raise FileNotFoundError(
            f"Origin cfg not found: {origin_cfg}. Run 'prepare-train' first."
        )

    # ── Accumulation dir: fixed root that train.cfg lives in
    accum_dir  = ensure_dir((datasets_root / accum_subdir).resolve())
    gen_dir    = ensure_dir((accum_dir / gen_subdir).resolve())
    merged_cfg = accum_dir / merge_name

    # ── Snapshot block count BEFORE anything is written ─────────────────────
    prev_block_count = _count_cfg_blocks(merged_cfg) if merged_cfg.exists() else 0

    # ── 1. Convert each OUTCAR → per-task .cfg ───────────────────────────────
    this_run_cfgs: list[Path] = []
    for task_dir in sorted(label_result.task_dirs):
        outcar = task_dir / "OUTCAR"
        if not outcar.exists():
            print(f"  [SKIP] No OUTCAR in {task_dir.name}")
            continue

        out_cfg = gen_dir / f"{task_dir.name}.cfg"
        cmd = [mlp_command, "convert", str(outcar), str(out_cfg),
               "--input_format=outcar"]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"mlp convert failed for {task_dir.name}:\n"
                f"  cmd:    {' '.join(cmd)}\n"
                f"  stdout: {proc.stdout.strip()}\n"
                f"  stderr: {proc.stderr.strip()}"
            )

        if not out_cfg.exists() or out_cfg.stat().st_size == 0:
            raise RuntimeError(
                f"mlp convert exited 0 but produced no output for {task_dir.name}.\n"
                f"  cmd:    {' '.join(cmd)}\n"
                f"  stdout: {proc.stdout.strip()}\n"
                f"  stderr: {proc.stderr.strip()}\n"
                f"  Expected output: {out_cfg}"
            )

        this_run_cfgs.append(out_cfg)
        print(f"  [{len(this_run_cfgs):>4d}]  {task_dir.name}/OUTCAR  →  {out_cfg.relative_to(datasets_root)}")

    if not this_run_cfgs:
        raise FileNotFoundError(f"No OUTCARs found under {label_result.label_root}")

    # ── 2. Count blocks contributed by this run before rebuilding ─────────────
    new_block_count = sum(_count_cfg_blocks(p) for p in this_run_cfgs)

    # ── 3. Collect ALL converted cfgs across every gen subdir ─────────────────
    all_run_cfgs: list[Path] = sorted(accum_dir.rglob("task.*.cfg"))

    # ── 4. Write accumulated train.cfg ────────────────────────────────────────
    _write_merged([origin_cfg] + all_run_cfgs, merged_cfg)

    # ── 5. Verify written block count == expected ────────────────────────────
    total_block_count  = _count_cfg_blocks(merged_cfg)
    origin_block_count = _count_cfg_blocks(origin_cfg)
    expected_total     = origin_block_count + sum(
        _count_cfg_blocks(p) for p in all_run_cfgs
    )

    if total_block_count != expected_total:
        raise RuntimeError(
            f"Block count mismatch in {merged_cfg.name} after convert-cfg:\n"
            f"  Expected : {expected_total}  "
            f"(origin {origin_block_count} + {len(all_run_cfgs)} run cfgs)\n"
            f"  Actual   : {total_block_count}\n"
            f"  This gen : {len(this_run_cfgs)} files, {new_block_count} blocks\n"
            f"  File     : {merged_cfg}\n"
            f"\nDo NOT run 'fit' — the training set is incomplete.  "
            f"Re-run 'convert-cfg' or inspect {gen_dir} for partial/empty cfgs."
        )

    print(f"\nDone. {len(this_run_cfgs)} new cfg(s) written to {gen_dir.relative_to(datasets_root)}/")
    print(
        f"Accumulated {merge_name}: origin({origin_block_count})"
        f" + {len(all_run_cfgs)} run cfg(s) → {total_block_count} blocks total"
        f"  [{merged_cfg}]"
    )

    result = ConvertResult(
        merged_cfg=merged_cfg,
        run_dir=gen_dir,
        n_new_cfgs=len(this_run_cfgs),
        n_total_cfgs=len(all_run_cfgs),
        prev_block_count=prev_block_count,
        new_block_count=new_block_count,
        total_block_count=total_block_count,
    )
    result.save_manifest()
    return result
