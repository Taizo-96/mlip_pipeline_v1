from __future__ import annotations

import os
import shutil
from pathlib import Path

from mlip_pipeline.utils.shell import run_command
from mlip_pipeline.models import LabelResult


def _resolve_vasp_bin(config: dict) -> Path:
    """
    Resolve the VASP binary with this priority:
      1. config.label.local.vasp_bin  (explicit override in the local block)
      2. config.bins.vasp             (shared bins section)
      3. 'vasp_std'                   (last-resort default)

    For each candidate:
      - If it looks like an absolute path or a path that exists as-is, use it directly.
      - Otherwise call shutil.which() so bare names like 'vasp_std' are found via PATH.
    """
    local_cfg = config.get("label", {}).get("local", {})
    bins_cfg  = config.get("bins", {})

    candidates = [
        local_cfg.get("vasp_bin"),   # highest priority
        bins_cfg.get("vasp"),        # shared bins section
        "vasp_std",                  # fallback
    ]

    for raw in candidates:
        if not raw:
            continue
        p = Path(raw).expanduser()
        # Absolute or relative path that actually exists on disk
        if p.is_absolute() or p.exists():
            if p.exists():
                return p
            raise FileNotFoundError(
                f"VASP binary not found at explicit path: {p}"
            )
        # Bare name (e.g. 'vasp_std') — search PATH
        found = shutil.which(str(raw))
        if found:
            return Path(found)

    raise FileNotFoundError(
        f"VASP binary not found. Tried: {[c for c in candidates if c]}. "
        f"PATH={os.environ.get('PATH', '(not set)')}"
    )


def run_vasp_local(label_result: LabelResult, config: dict) -> None:
    vasp_bin  = _resolve_vasp_bin(config)
    local_cfg = config.get("label", {}).get("local", {})
    np         = local_cfg.get("np", config.get("mpi", {}).get("np", 16))
    source_env = local_cfg.get("source_env")
    parallel   = local_cfg.get("parallel", 1)  # number of tasks to run concurrently

    print(f"VASP binary : {vasp_bin}")
    print(f"MPI ranks   : {np}")
    print(f"Tasks       : {len(label_result.task_dirs)}")
    if source_env:
        print(f"Env script  : {source_env}")

    for idx, task_dir in enumerate(label_result.task_dirs, 1):
        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)

        if source_env:
            full_cmd = ["bash", "-lc",
                        f"source {source_env} && mpirun -np {np} {vasp_bin}"]
        else:
            full_cmd = ["mpirun", "-np", str(np), str(vasp_bin)]

        print(f"  [{idx:>4d}/{len(label_result.task_dirs)}]  {task_dir.name} -> vasp.log")

        exit_code = run_command(
            command=full_cmd,
            cwd=task_dir,
            log_file=output_dir / "vasp.log",
        )

        if exit_code != 0:
            print(f"  FAILED: {task_dir.name} (exit code {exit_code})")
