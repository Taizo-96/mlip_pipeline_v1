from __future__ import annotations

from pathlib import Path
from datetime import datetime

from mlip_pipeline.models import FitResult
from mlip_pipeline.fit.mlip3 import build_mlip_train_command
from mlip_pipeline.utils.fs import ensure_dir, copy_if_exists, write_json
from mlip_pipeline.utils.shell import run_command


def resolve_template_path(fit_cfg: dict) -> Path:
    if "init_template" in fit_cfg:
        return Path(fit_cfg["init_template"]).expanduser().resolve()
    template_dir = Path(fit_cfg["template_dir"]).expanduser().resolve()
    return template_dir / f"{int(fit_cfg['mtp_level']):02d}.almtp"


def resolve_train_cfg(fit_cfg: dict, config: dict, resolved_paths: dict) -> Path:
    """Single authoritative resolver for the training cfg path."""
    if "train_cfg" in fit_cfg:
        return Path(fit_cfg["train_cfg"]).expanduser().resolve()
    # Fall back to convert accumulation root
    convert_cfg  = config.get("convert", {})
    accum_subdir = convert_cfg.get("output_subdir", "converted_cfg")
    merge_name   = convert_cfg.get("merge_name", "train.cfg")
    return (resolved_paths["datasets_root"] / accum_subdir / merge_name).resolve()


def train_potential(config: dict, resolved_paths: dict) -> FitResult:
    fit_cfg      = config["fit"]
    mlp_command  = fit_cfg.get("mlp_command", "mlp")
    mpi_prefix   = fit_cfg.get("mpi_prefix")
    init_template = resolve_template_path(fit_cfg)
    train_cfg    = resolve_train_cfg(fit_cfg, config, resolved_paths)
    trained_name = fit_cfg.get(
        "trained_potential_name",
        f"pb{fit_cfg.get('mtp_level', 'xx')}.almtp",
    )
    extra_args   = fit_cfg.get("extra_args", [])
    run_dir      = ensure_dir(
        resolved_paths["runs_root"] / fit_cfg.get("output_subdir", "fit")
    )

    if not init_template.exists():
        raise FileNotFoundError(f"Initial MTP template not found: {init_template}")
    if not train_cfg.exists():
        raise FileNotFoundError(f"Training cfg not found: {train_cfg}")

    log_path        = run_dir / "train.log"
    staged_template = run_dir / init_template.name
    staged_cfg      = run_dir / train_cfg.name
    copy_if_exists(init_template, staged_template)
    copy_if_exists(train_cfg, staged_cfg)

    command = build_mlip_train_command(
        mlp_command,
        staged_template.name,
        staged_cfg.name,
        trained_name=trained_name,
        mpi_prefix=mpi_prefix,
        extra_args=extra_args,
    )
    print("DEBUG COMMAND:", command)
    return_code = run_command(command, cwd=run_dir, log_file=log_path)
    if return_code != 0:
        raise RuntimeError(
            f"mlp train failed (exit {return_code}). "
            f"See {log_path} for details."
        )

    model_path = run_dir / trained_name
    if not model_path.exists():
        raise RuntimeError(
            f"mlp train exited 0 but model file was not written: {model_path}\n"
            f"Check {log_path} for details."
        )

    # Count cfgs for traceability (fast line scan)
    n_cfgs = sum(
        1 for line in train_cfg.open(encoding="utf-8") if line.strip() == "BEGIN_CFG"
    )

    write_json(
        run_dir / "metadata.json",
        {
            "stage":          "fit",
            "timestamp":      datetime.utcnow().isoformat() + "Z",
            "mlp_command":    mlp_command,
            "mpi_prefix":     mpi_prefix,
            "init_template":  str(init_template),
            "train_cfg":      str(train_cfg),
            "n_train_cfgs":   n_cfgs,
            "run_dir":        str(run_dir),
            "result_model":   str(model_path),
            "command":        command,
        },
    )
    return FitResult(run_dir=run_dir, model_path=model_path, log_path=log_path,
                     train_cfg=train_cfg)
