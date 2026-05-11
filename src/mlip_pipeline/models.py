# src/mlip_pipeline/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PrepareTrainResult:
    output_dir: Path
    merged_cfg: Path | None
    generated_cfgs: Sequence[Path]


@dataclass
class FitResult:
    run_dir: Path
    model_path: Path
    log_path: Path


@dataclass
class SelectionResult:
    select_root: Path
    manifest_path: Path
    selected_cfg_paths: list[Path]
    selected_count: int


@dataclass(frozen=True)
class VaspParallelConfig:
    ncore: int
    kpar: int
    nodes: int
    ntasks: int
    partition: str
    omp_num_threads: int = 1

    def to_incar_dict(self) -> dict[str, int]:
        return {"NCORE": self.ncore, "KPAR": self.kpar}


class ScalingPolicy:
    def __init__(self, yaml_cfg: dict):
        self.threshold = yaml_cfg.get("size_threshold", 150)
        self.k_scaling = yaml_cfg.get("kpoint_scaling", 0.5)
        self.small_cfg = yaml_cfg.get("small_system", {})
        self.large_cfg = yaml_cfg.get("large_system", {})

    def get_config(self, n_atoms: int) -> VaspParallelConfig:
        cfg = self.large_cfg if n_atoms > self.threshold else self.small_cfg
        return VaspParallelConfig(
            ncore=cfg.get("ncore", 1),
            kpar=cfg.get("kpar", 1),
            nodes=cfg.get("nodes", 1),
            ntasks=cfg.get("ntasks", 32),
            partition=cfg.get("partition", "shared")
        )

    def scale_kpoints(self, n_atoms: int, original_k: list[int]) -> list[int]:
        if n_atoms > self.threshold:
            return [max(1, int(k * self.k_scaling)) for k in original_k]
        return original_k


@dataclass
class LabelResult:
    label_root: Path
    task_dirs: list[Path]
    task_count: int
    manifest_path: Path

    @classmethod
    def load_from_dir(cls, label_root: Path) -> LabelResult:
        manifest_path = label_root / "label_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest found at {manifest_path}. Did the labeling step finish?"
            )
        import json
        data = json.loads(manifest_path.read_text())
        return cls(
            label_root=label_root,
            task_dirs=[Path(d) for d in data["task_dirs"]],
            task_count=data["task_count"],
            manifest_path=manifest_path,
        )


@dataclass
class EvaluateResult:
    eval_dir: Path
    metrics_csv: Path
    plots: list[Path] = field(default_factory=list)
