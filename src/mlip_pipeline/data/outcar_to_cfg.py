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
    Per-gen storage  (new cfgs only, one subdir per generation):
        datasets/<gen_subdir>/run_00/task.000000.cfg  ...
        e.g.  datasets/gen_08/convert/run_00/

    Accumulation root  (fixed, always the same dir across gens):
        datasets/<output_subdir>/train.cfg
        datasets/<output_subdir>/gen_08/run_00/task.000000.cfg  (symlinked scan)
        e.g.  datasets/converted_cfg/train.cfg

    The accumulation root is what `fit` reads via config.fit.train_cfg.
    The per-gen subdir keeps each generation's raw cfgs isolated.
    """
    convert_cfg     = config.get("convert", {})
    # Fixed accumulation root — must match config.fit.train_cfg
    accum_subdir    = convert_cfg.get("output_subdir", "converted_cfg")
    # Per-gen subdir injected by runner.py (e.g. "gen_08/convert")
    gen_subdir      = convert_cfg.get("gen_subdir", accum_subdir)
    run_name        = convert_cfg.get("run_name", "run_00")
    merge_name      = convert_cfg.get("merge_name", "train.cfg")
    mlp_command     = convert_cfg.get(
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

    # ── Per-gen dir: where this generation's cfgs are stored
    gen_dir = ensure_dir((datasets_root / gen_subdir / run_name).resolve())

    # ── Accumulation dir: fixed root that train.cfg lives in
    accum_dir = ensure_dir((datasets_root / accum_subdir).resolve())

    # ── 1. Convert each OUTCAR → per-task .cfg ────────────────────────────────
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
    #    Scan every subdir (one per generation) in sorted order so the
    #    accumulated train.cfg is deterministic and append-only.
    all_run_cfgs: list[Path] = []
    for run_subdir in sorted(d for d in accum_dir.iterdir() if d.is_dir()):
        all_run_cfgs.extend(sorted(run_subdir.glob("task.*.cfg")))

    # Also include cfgs from the current gen dir if it is outside accum_dir
    # (i.e. gen_subdir != accum_subdir).
    if gen_dir.parent != accum_dir:
        for cfg in sorted(gen_dir.glob("task.*.cfg")):
            if cfg not in all_run_cfgs:
                all_run_cfgs.append(cfg)

    # ── 3. Write accumulated train.cfg into the fixed accumulation root ────────
    merged_cfg = accum_dir / merge_name
    _write_merged([origin_cfg] + all_run_cfgs, merged_cfg)

    print(f"\nDone. {len(this_run_cfgs)} new cfg(s) written to {gen_dir.relative_to(datasets_root)}/")
    print(
        f"Accumulated {merge_name}: origin({origin_cfg.name})"
        f" + {len(all_run_cfgs)} run cfg(s) → {merged_cfg}"
    )
    return merged_cfg
