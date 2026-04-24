from __future__ import annotations

import subprocess
from pathlib import Path

from mlip_pipeline.models import LabelResult


def run_vasp_local(
    label_result: LabelResult,
    config: dict,
) -> None:
    local_cfg  = config["label"].get("local", {})
    vasp_bin   = local_cfg.get("vasp_bin",   "/opt/vasp/vasp.6.5.1_cpu/bin/vasp_std")
    np         = local_cfg.get("np",         16)
    parallel   = local_cfg.get("parallel",   1)
    source_env = local_cfg.get("source_env", None)

    vasp_bin = Path(vasp_bin).expanduser()
    if not vasp_bin.exists():
        raise FileNotFoundError(f"VASP binary not found: {vasp_bin}")

    script = None
    if source_env:
        script = Path(source_env).expanduser()
        if not script.exists():
            raise FileNotFoundError(f"source_env script not found: {script}")

    task_dirs = label_result.task_dirs
    total     = len(task_dirs)

    print(f"Running {total} VASP task(s) | np={np} | parallel={parallel}")
    if script:
        print(f"  sourcing environment from: {script}")

    for chunk_start in range(0, total, parallel):
        chunk = task_dirs[chunk_start : chunk_start + parallel]
        procs = []

        for offset, task_dir in enumerate(chunk, start=1):
            output_dir = task_dir / "output"
            output_dir.mkdir(exist_ok=True)
            log_path = output_dir / "vasp.log"

            idx = chunk_start + offset
            print(f"  [{idx:>4d}/{total}]  {task_dir.name}/output/vasp.log")

            if script:
                cmd = f"source {script} && mpirun -np {np} {vasp_bin}"
                popen_args = ["bash", "-lc", cmd]
            else:
                popen_args = ["mpirun", "-np", str(np), str(vasp_bin)]

            proc = subprocess.Popen(
                popen_args,
                cwd=task_dir,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
            )
            procs.append((task_dir, proc))

        for task_dir, proc in procs:
            returncode = proc.wait()
            if returncode != 0:
                print(f"  WARNING: {task_dir.name} failed (exit {returncode}) — check output/vasp.log")
            else:
                print(f"  OK: {task_dir.name}")

    print("\nDone. Outputs are in each task_dir/output/")