from __future__ import annotations
import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ── VASP scaling ──────────────────────────────────────────────────────────────

@dataclass
class VaspParallelConfig:
    """Unified parallelisation config — covers both INCAR tags and SLURM job fields."""
    # INCAR tags
    ncore: int = 4
    kpar: int = 1
    npar: Optional[int] = None
    additional_incar: dict = field(default_factory=dict)

    # SLURM job fields
    partition: str = "shared"
    nodes: int = 1
    ntasks: int = 32
    omp_num_threads: int = 1

    def to_incar_dict(self) -> dict:
        d = {"NCORE": self.ncore, "KPAR": self.kpar}
        if self.npar is not None:
            d["NPAR"] = self.npar
        d.update(self.additional_incar)
        return d


@dataclass
class ScalingPolicy:
    rules: dict = field(default_factory=dict)

    def get_config(self, n_atoms: int) -> VaspParallelConfig:
        for threshold, rule in sorted(
            ((int(k), v) for k, v in self.rules.items()), reverse=True
        ):
            if n_atoms >= threshold:
                return VaspParallelConfig(
                    ncore=rule.get("ncore", 4),
                    kpar=rule.get("kpar", 1),
                    npar=rule.get("npar"),
                    additional_incar=rule.get("additional_incar", {}),
                    partition=rule.get("partition", "shared"),
                    nodes=rule.get("nodes", 1),
                    ntasks=rule.get("ntasks", 32),
                    omp_num_threads=rule.get("omp_num_threads", 1),
                )
        return VaspParallelConfig()

    def scale_kpoints(self, n_atoms: int, base: list[int]) -> list[int]:
        factor = max(1, self.get_config(n_atoms).kpar)
        return [max(1, round(k / factor)) for k in base]

# ── Step results ──────────────────────────────────────────────────────────────

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
    train_cfg: Optional[Path] = None   # path to train.cfg used for this fit
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.run_dir / "fit_manifest.json", {
            "step": "fit",
            "model_path": str(self.model_path),
            "log_path": str(self.log_path) if self.log_path else None,
            "train_cfg": str(self.train_cfg) if self.train_cfg else None,
            "completed_at": self.completed_at,
        })

    @classmethod
    def load_manifest(cls, run_dir: Path) -> "FitResult":
        d = json.loads((run_dir / "fit_manifest.json").read_text())
        return cls(
            run_dir=run_dir,
            model_path=Path(d["model_path"]),
            log_path=Path(d["log_path"]) if d.get("log_path") else None,
            train_cfg=Path(d["train_cfg"]) if d.get("train_cfg") else None,
            completed_at=d.get("completed_at", ""),
        )

@dataclass
class ExploreResult:
    explore_root: Path
    run_dirs: list[Path]
    n_runs: int
    preselected_cfgs: list[Path] = field(default_factory=list)
    failed_runs: list[Path] = field(default_factory=list)
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.explore_root / "explore_manifest.json", {
            "step": "explore",
            "n_runs": self.n_runs,
            "preselected_cfgs": [str(p) for p in self.preselected_cfgs],
            "failed_runs": [str(p) for p in self.failed_runs],
            "completed_at": self.completed_at,
        })

    @classmethod
    def load_manifest(cls, explore_root: Path) -> "ExploreResult":
        d = json.loads((explore_root / "explore_manifest.json").read_text())
        return cls(
            explore_root=explore_root,
            run_dirs=[],
            n_runs=d["n_runs"],
            preselected_cfgs=[Path(p) for p in d["preselected_cfgs"]],
            failed_runs=[Path(p) for p in d["failed_runs"]],
            completed_at=d.get("completed_at", ""),
        )

@dataclass
class SelectionResult:
    select_root: Path
    manifest_path: Path
    selected_cfg_paths: list[Path] = field(default_factory=list)
    selected_count: int = 0
    completed_at: str = field(default_factory=_now)

@dataclass
class LabelResult:
    label_root: Path
    task_dirs: list[Path]
    task_count: int
    manifest_path: Optional[Path] = None
    completed_at: str = field(default_factory=_now)

    @classmethod
    def load_manifest(cls, label_root: Path) -> "LabelResult":
        d = json.loads((label_root / "label_manifest.json").read_text())
        return cls(
            label_root=label_root,
            task_dirs=[Path(t) for t in d["task_dirs"]],
            task_count=d["task_count"],
            manifest_path=label_root / "label_manifest.json",
        )

    @classmethod
    def load_from_dir(cls, label_root: Path) -> "LabelResult":
        """Alias for load_manifest — used by cli.py sync/submit commands."""
        return cls.load_manifest(label_root)

@dataclass
class ConvertResult:
    merged_cfg: Path
    run_dir: Path
    n_new_cfgs: int
    n_total_cfgs: int
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.run_dir / "convert_manifest.json", {
            "step": "convert",
            "merged_cfg": str(self.merged_cfg),
            "n_new_cfgs": self.n_new_cfgs,
            "n_total_cfgs": self.n_total_cfgs,
            "completed_at": self.completed_at,
        })

@dataclass
class EvaluationResult:
    rmse_energy: float
    rmse_forces: float
    rmse_stress: float
    eval_dir: Path
    plot_paths: dict = field(default_factory=dict)
    completed_at: str = field(default_factory=_now)


# ── Generation state (automation) ─────────────────────────────────────────────

# HPC mode: label prepares inputs, label_hpc submits to Dardel and waits.
# Local mode: label prepares inputs, label_local runs VASP via mpirun.
STEPS = ("fit", "explore", "select", "label", "label_hpc", "label_local", "convert")

@dataclass
class GenerationState:
    generation: str
    gen_dir: Path
    status: str = "pending"
    completed_steps: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    merged_cfg: Optional[str] = None

    @property
    def _state_path(self) -> Path:
        return self.gen_dir / "state.json"

    def save(self) -> None:
        from mlip_pipeline.utils.fs import write_json
        write_json(self._state_path, {
            "generation": self.generation,
            "status": self.status,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_path": self.model_path,
            "merged_cfg": self.merged_cfg,
        })

    @classmethod
    def load(cls, gen_dir: Path) -> "GenerationState":
        d = json.loads((gen_dir / "state.json").read_text())
        known = {f.name for f in fields(cls)}
        return cls(gen_dir=gen_dir, **{k: v for k, v in d.items() if k in known and k != "gen_dir"})

    @classmethod
    def init(cls, generation: str, gen_dir: Path) -> "GenerationState":
        gen_dir.mkdir(parents=True, exist_ok=True)
        s = cls(generation=generation, gen_dir=gen_dir)
        s.save()
        return s

    def mark_step_start(self, step: str) -> None:
        if self.started_at is None:
            self.started_at = _now()
            self.status = "running"
        self.current_step = step
        self.save()

    def mark_step_done(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.current_step = None
        self.save()

    def mark_done(self) -> None:
        self.status = "completed"
        self.current_step = None
        self.completed_at = _now()
        self.save()

    def mark_failed(self, exc: Exception) -> None:
        self.status = "failed"
        self.error = str(exc)
        self.save()

    def is_step_done(self, step: str) -> bool:
        return step in self.completed_steps
