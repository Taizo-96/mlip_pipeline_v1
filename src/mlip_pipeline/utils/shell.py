from __future__ import annotations

from pathlib import Path
import subprocess


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    log_file: str | Path | None = None,
) -> int:
    cwd_path = Path(cwd).resolve() if cwd is not None else None

    if log_file is None:
        result = subprocess.run(command, cwd=cwd_path)
        return result.returncode

    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as fh:
        result = subprocess.run(
            command,
            cwd=cwd_path,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode