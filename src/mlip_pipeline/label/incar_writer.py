from __future__ import annotations
from pathlib import Path


def write_incar(
    task_dir: Path,
    base_incar: dict,
    parallel_overrides: dict[str, int],
) -> Path:
    """
    Write a complete INCAR to task_dir.

    Tag priority (highest last, so it wins):
        base_incar (from YAML)  <  parallel_overrides (NCORE, KPAR)

    This means the user can never accidentally override parallelization
    tags from the YAML — the code always has the final word on those.
    """
    tags = {**base_incar, **parallel_overrides}
    incar_path = task_dir / "INCAR"
    incar_path.write_text(
        "\n".join(f"{k} = {v}" for k, v in tags.items()) + "\n"
    )
    return incar_path