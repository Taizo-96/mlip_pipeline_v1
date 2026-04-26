from __future__ import annotations

import subprocess
from pathlib import Path

from mlip_pipeline.models import LabelResult


def run_vasp_local(label_result: LabelResult, config: dict) -> None:
    local_cfg = config["label"].get("local", {})
    vasp_bin = Path(local_cfg.get("vasp_bin", "/opt/vasp/vasp.6.5.1_cpu/bin/vasp_std")).expanduser()
    np = local_cfg.get("np", 16)
    source_env = local_cfg.get("source_env", None)

    if not vasp_bin.exists():
        raise FileNotFoundError(f"VASP binary not found: {vasp_bin}")

    print(f"Running {len(label_result.task_dirs)} VASP task(s) | np={np}")

    for idx, task_dir in enumerate(label_result.task_dirs, 1):
        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Build the command
        # If source_env is provided, we wrap it in bash -c to preserve the environment
        if source_env:
            full_cmd = ["bash", "-lc", f"source {source_env} && mpirun -np {np} {vasp_bin}"]
        else:
            full_cmd = ["mpirun", "-np", str(np), str(vasp_bin)]

        print(f"  [{idx:>4d}/{len(label_result.task_dirs)}]  {task_dir.name} -> vasp.log")

        # Standardize execution using the shell utility
        exit_code = run_command(
            command=full_cmd,
            cwd=task_dir,
            log_file=output_dir / "vasp.log"
        )

        if exit_code != 0:
            print(f"  FAILED: {task_dir.name} (Exit code: {exit_code})")