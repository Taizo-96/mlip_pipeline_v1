from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Sequence
from mlip_pipeline.utils.logging import info

def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    log_file: str | Path | None = None,
    live_tail: bool = False,
    env: dict | None = None,
) -> int:
    cwd = Path(cwd) if cwd else None
    info(f"$ {' '.join(str(a) for a in command)}")
    lf = open(log_file, "w", encoding="utf-8") if log_file else None
    try:
        proc = subprocess.Popen(
            [str(a) for a in command],
            cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, env=env,
        )
        for line in proc.stdout:
            if lf:
                lf.write(line)
            if live_tail:
                print(f"  │ {line.rstrip()}", flush=True)
        proc.wait()
        return proc.returncode
    finally:
        if lf:
            lf.close()