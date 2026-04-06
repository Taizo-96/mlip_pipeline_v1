from __future__ import annotations


def build_mlip_train_command(
    mlp_command: str,
    initial_model: str,          # template first
    train_cfg: str,              # cfg second
    trained_name: str | None = None,
    mpi_prefix: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    base = []
    if mpi_prefix:
        base.extend(mpi_prefix.split())
    base.extend([mlp_command, "train", initial_model, train_cfg])
    if trained_name:
        base.append(f"--save_to={trained_name}")
    if extra_args:
        base.extend(extra_args)
    return base