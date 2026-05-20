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
    """
    Convert OUTCARs → .cfg files and accumulate them into a merged train.cfg.

    Directory layout
    ----------------
    Per-gen storage  (new cfgs only, one subdir per generation, INSIDE accum_dir):
        datasets/<accum_subdir>/<gen_subdir>/run_00/task.000000.cfg
        e.g.  datasets/converted_cfg/gen_11/convert/run_00/

    Accumulation root  (fixed, always the same dir across gens):
        datasets/<accum_subdir>/train.cfg
        e.g.  datasets/converted_cfg/train.cfg

    The accumulation root is what `fit` reads via config.fit.train_cfg.
    """
    convert_cfg  = config.get("convert", {})
    # Fixed accumulation root — must match config.fit.train_cfg
    accum_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    # Per-gen subdir injected by runner.py (e.g. "gen_11/convert").
    # Resolved *inside* accum_dir so cfgs are always co-located with train.cfg.
    gen_subdir   = convert_cfg.get("gen_subdir", accum_subdir)
    run_name     = convert_cfg.get("run_name", "run_00")
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
    accum_dir = ensure_dir((datasets_root / accum_subdir).resolve())

    # ── Per-gen dir: always nested INSIDE accum_dir so the scan below finds it
    gen_dir = ensure_dir((accum_dir / gen_subdir / run_name).resolve())

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
        print(f"  [{len(this_run_cfgs):>4d}]  {task_dir.name}/OUTCAR  →  {out_cfg.relative_to(datasets_root)}")

    if not this_run_cfgs:
        raise FileNotFoundError(f"No OUTCARs found under {label_result.label_root}")

    # ── 2. Collect ALL previously converted cfgs from the accumulation dir ────
    #    Recursive glob for task.*.cfg so every nested run_00/ subdir is
    #    included regardless of how deep gen_subdir nests.
    all_run_cfgs: list[Path] = sorted(accum_dir.rglob("task.*.cfg"))

    # ── 3. Write accumulated train.cfg into the fixed accumulation root ────────
    merged_cfg = accum_dir / merge_name
    _write_merged([origin_cfg] + all_run_cfgs, merged_cfg)

    print(f"\nDone. {len(this_run_cfgs)} new cfg(s) written to {gen_dir.relative_to(datasets_root)}/")
    print(
        f"Accumulated {merge_name}: origin({origin_cfg.name})"
        f" + {len(all_run_cfgs)} run cfg(s) → {merged_cfg}"
    )
    return merged_cfg
