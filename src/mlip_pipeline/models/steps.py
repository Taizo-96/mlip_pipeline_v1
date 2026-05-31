"""Per-step result dataclasses for the active-learning pipeline.

Each dataclass captures the output of one pipeline step and can
serialise / deserialise itself via save_manifest / load_manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class PrepareTrainResult:
    output_dir: Path
    merged_cfg: Optional[Path]
    generated_cfgs: list[Path] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    completed_at: str = field(default_factory=_now)


@dataclass
class FitResult:
    run_dir: Path
    model_path: Path
    log_path: Optional[Path] = None
    train_cfg: Optional[Path] = None
    n_train_cfgs: int = 0
    init_template: Optional[Path] = None
    mlp_command: str = "mlp"
    mpi_prefix: Optional[str] = None
    command: list = field(default_factory=list)
    completed_at: str = field(default_factory=_now)

    def resolve_model_path(self) -> Path:
        """Return the actual model path on disk, tolerating case mismatches."""
        if self.model_path.exists():
            return self.model_path

        parent = self.model_path.parent
        if not parent.exists():
            parent = self.run_dir

        stored_lower = self.model_path.name.lower()
        candidates = list(parent.glob("*.almtp"))

        for c in candidates:
            if c.name.lower() == stored_lower:
                return c

        if candidates:
            return sorted(candidates)[0]

        return self.model_path

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.run_dir / "fit_manifest.json", {
            "step":          "fit",
            "run_dir":       str(self.run_dir),
            "model_path":    str(self.model_path),
            "log_path":      str(self.log_path) if self.log_path else None,
            "train_cfg":     str(self.train_cfg) if self.train_cfg else None,
            "n_train_cfgs":  self.n_train_cfgs,
            "init_template": str(self.init_template) if self.init_template else None,
            "mlp_command":   self.mlp_command,
            "mpi_prefix":    self.mpi_prefix,
            "command":       self.command,
            "completed_at":  self.completed_at,
        })

    @classmethod
    def load_manifest(cls, run_dir: Path) -> "FitResult":
        d = json.loads((run_dir / "fit_manifest.json").read_text())
        return cls(
            run_dir=run_dir,
            model_path=Path(d["model_path"]),
            log_path=Path(d["log_path"]) if d.get("log_path") else None,
            train_cfg=Path(d["train_cfg"]) if d.get("train_cfg") else None,
            n_train_cfgs=d.get("n_train_cfgs", 0),
            init_template=Path(d["init_template"]) if d.get("init_template") else None,
            mlp_command=d.get("mlp_command", "mlp"),
            mpi_prefix=d.get("mpi_prefix"),
            command=d.get("command", []),
            completed_at=d.get("completed_at", ""),
        )


@dataclass
class ExploreResult:
    explore_root: Path
    run_dirs: list[Path]
    n_runs: int
    n_ok: int = 0
    n_failed: int = 0
    replicate: list[int] = field(default_factory=lambda: [1, 1, 1])
    run_records: list[dict] = field(default_factory=list)
    preselected_cfgs: list[Path] = field(default_factory=list)
    failed_runs: list[Path] = field(default_factory=list)
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.explore_root / "explore_manifest.json", {
            "step":             "explore",
            "replicate":        self.replicate,
            "n_runs":           self.n_runs,
            "n_ok":             self.n_ok,
            "n_failed":         self.n_failed,
            "runs":             self.run_records,
            "preselected_cfgs": [str(p) for p in self.preselected_cfgs],
            "failed_runs":      [str(p) for p in self.failed_runs],
            "completed_at":     self.completed_at,
        })

    @classmethod
    def load_manifest(cls, explore_root: Path) -> "ExploreResult":
        d = json.loads((explore_root / "explore_manifest.json").read_text())
        return cls(
            explore_root=explore_root,
            run_dirs=[],
            n_runs=d["n_runs"],
            n_ok=d.get("n_ok", 0),
            n_failed=d.get("n_failed", 0),
            replicate=d.get("replicate", [1, 1, 1]),
            run_records=d.get("runs", []),
            preselected_cfgs=[Path(p) for p in d.get("preselected_cfgs", [])],
            failed_runs=[Path(p) for p in d.get("failed_runs", [])],
            completed_at=d.get("completed_at", ""),
        )


@dataclass
class SelectionResult:
    select_root: Path
    manifest_path: Path
    selected_cfg_paths: list[Path] = field(default_factory=list)
    selected_count: int = 0
    model_path: Optional[Path] = None
    candidate_sources: list[dict] = field(default_factory=list)
    converged: bool = False
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.manifest_path, {
            "step":                 "select",
            "converged":            self.converged,
            "selected_count":       self.selected_count,
            "model_path":           str(self.model_path) if self.model_path else None,
            "candidate_sources":    self.candidate_sources,
            "selected_block_files": [str(p) for p in self.selected_cfg_paths],
            "completed_at":         self.completed_at,
        })

    @classmethod
    def load_manifest(cls, select_root: Path) -> "SelectionResult":
        d = json.loads((select_root / "selection_manifest.json").read_text())
        return cls(
            select_root=select_root,
            manifest_path=select_root / "selection_manifest.json",
            selected_cfg_paths=[Path(p) for p in d.get("selected_block_files", [])],
            selected_count=d.get("selected_count", 0),
            model_path=Path(d["model_path"]) if d.get("model_path") else None,
            candidate_sources=d.get("candidate_sources", []),
            converged=d.get("converged", False),
            completed_at=d.get("completed_at", ""),
        )


@dataclass
class LabelResult:
    label_root: Path
    task_dirs: list[Path]
    task_count: int
    manifest_path: Optional[Path] = None
    type_map: list[str] = field(default_factory=list)
    template_dir: Optional[Path] = None
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        manifest_path = self.label_root / "label_manifest.json"
        write_json(manifest_path, {
            "step":         "label",
            "task_count":   self.task_count,
            "task_dirs":    [str(t) for t in self.task_dirs],
            "type_map":     self.type_map,
            "template_dir": str(self.template_dir) if self.template_dir else None,
            "completed_at": self.completed_at,
        })
        self.manifest_path = manifest_path
        return manifest_path

    @classmethod
    def load_manifest(cls, label_root: Path) -> "LabelResult":
        d = json.loads((label_root / "label_manifest.json").read_text())
        return cls(
            label_root=label_root,
            task_dirs=[Path(t) for t in d["task_dirs"]],
            task_count=d["task_count"],
            manifest_path=label_root / "label_manifest.json",
            type_map=d.get("type_map", []),
            template_dir=Path(d["template_dir"]) if d.get("template_dir") else None,
            completed_at=d.get("completed_at", ""),
        )

    @classmethod
    def load_from_dir(cls, label_root: Path) -> "LabelResult":
        return cls.load_manifest(label_root)


@dataclass
class ConvertResult:
    merged_cfg: Path
    run_dir: Path
    n_new_cfgs: int
    n_total_cfgs: int
    prev_block_count: int = 0
    new_block_count: int = 0
    total_block_count: int = 0
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.run_dir / "convert_manifest.json", {
            "step":              "convert",
            "merged_cfg":        str(self.merged_cfg),
            "n_new_cfgs":        self.n_new_cfgs,
            "n_total_cfgs":      self.n_total_cfgs,
            "prev_block_count":  self.prev_block_count,
            "new_block_count":   self.new_block_count,
            "total_block_count": self.total_block_count,
            "completed_at":      self.completed_at,
        })

    @classmethod
    def load_manifest(cls, run_dir: Path) -> "ConvertResult":
        d = json.loads((run_dir / "convert_manifest.json").read_text())
        return cls(
            merged_cfg=Path(d["merged_cfg"]),
            run_dir=run_dir,
            n_new_cfgs=d.get("n_new_cfgs", 0),
            n_total_cfgs=d.get("n_total_cfgs", 0),
            prev_block_count=d.get("prev_block_count", 0),
            new_block_count=d.get("new_block_count", 0),
            total_block_count=d.get("total_block_count", 0),
            completed_at=d.get("completed_at", ""),
        )


@dataclass
class EvaluationResult:
    rmse_energy: float
    rmse_forces: float
    rmse_stress: float
    eval_dir: Path
    plot_paths: dict = field(default_factory=dict)
    mean_gamma: float = float("nan")
    max_gamma: float = float("nan")
    frac_above_save: float = float("nan")
    frac_above_break: float = float("nan")
    gamma_grades_path: Optional[Path] = None
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.eval_dir / "eval_manifest.json", {
            "step":              "evaluate",
            "rmse_energy":       self.rmse_energy,
            "rmse_forces":       self.rmse_forces,
            "rmse_stress":       self.rmse_stress,
            "mean_gamma":        self.mean_gamma,
            "max_gamma":         self.max_gamma,
            "frac_above_save":   self.frac_above_save,
            "frac_above_break":  self.frac_above_break,
            "gamma_grades_path": str(self.gamma_grades_path) if self.gamma_grades_path else None,
            "plot_paths":        {k: str(v) for k, v in self.plot_paths.items()},
            "completed_at":      self.completed_at,
        })

    @classmethod
    def load_manifest(cls, eval_dir: Path) -> "EvaluationResult":
        d = json.loads((eval_dir / "eval_manifest.json").read_text())
        return cls(
            rmse_energy=d.get("rmse_energy", float("nan")),
            rmse_forces=d.get("rmse_forces", float("nan")),
            rmse_stress=d.get("rmse_stress", float("nan")),
            mean_gamma=d.get("mean_gamma", float("nan")),
            max_gamma=d.get("max_gamma", float("nan")),
            frac_above_save=d.get("frac_above_save", float("nan")),
            frac_above_break=d.get("frac_above_break", float("nan")),
            gamma_grades_path=Path(d["gamma_grades_path"]) if d.get("gamma_grades_path") else None,
            eval_dir=eval_dir,
            plot_paths={k: Path(v) for k, v in d.get("plot_paths", {}).items()},
            completed_at=d.get("completed_at", ""),
        )


__all__ = [
    "PrepareTrainResult",
    "FitResult",
    "ExploreResult",
    "SelectionResult",
    "LabelResult",
    "ConvertResult",
    "EvaluationResult",
]
