from __future__ import annotations
import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ── VASP scaling ───────────────────────────────────────────────────────────────────

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

# ── Step results ──────────────────────────────────────────────────────────────────

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
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.run_dir / "fit_manifest.json", {
            "step":          "fit",
            "model_path":    str(self.model_path),
            "log_path":      str(self.log_path) if self.log_path else None,
            "train_cfg":     str(self.train_cfg) if self.train_cfg else None,
            "n_train_cfgs":  self.n_train_cfgs,
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
            "step":                  "select",
            "converged":             self.converged,
            "selected_count":        self.selected_count,
            "model_path":            str(self.model_path) if self.model_path else None,
            "candidate_sources":     self.candidate_sources,
            "selected_block_files":  [str(p) for p in self.selected_cfg_paths],
            "completed_at":          self.completed_at,
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
    # Gamma statistics (NaN when no grade data was found)
    mean_gamma: float = float("nan")
    max_gamma: float = float("nan")
    frac_above_save: float = float("nan")
    frac_above_break: float = float("nan")
    completed_at: str = field(default_factory=_now)

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        return write_json(self.eval_dir / "eval_manifest.json", {
            "step":             "evaluate",
            "rmse_energy":      self.rmse_energy,
            "rmse_forces":      self.rmse_forces,
            "rmse_stress":      self.rmse_stress,
            "mean_gamma":       self.mean_gamma,
            "max_gamma":        self.max_gamma,
            "frac_above_save":  self.frac_above_save,
            "frac_above_break": self.frac_above_break,
            "plot_paths":       {k: str(v) for k, v in self.plot_paths.items()},
            "completed_at":     self.completed_at,
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
            eval_dir=eval_dir,
            plot_paths={k: Path(v) for k, v in d.get("plot_paths", {}).items()},
            completed_at=d.get("completed_at", ""),
        )


# ── Generation state (automation) ──────────────────────────────────────────────

# Fix #1: added "evaluate" to STEPS after "convert"
STEPS = ("fit", "explore", "select", "label", "label_hpc", "label_local", "convert", "evaluate")

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
    replicate_tier: int = 0
    converged_replicate_tier: int = 0
    label_prepared: bool = False
    jobs_submitted: bool = False   # True once SLURM jobs have been submitted
    # Fix #3: evaluation state fields
    evaluation_done: bool = False
    evaluation_failed: bool = False
    evaluation_manifest: Optional[str] = None  # path to eval_manifest.json

    @property
    def _state_path(self) -> Path:
        return self.gen_dir / "state.json"

    def save(self) -> None:
        from mlip_pipeline.utils.fs import write_json
        write_json(self._state_path, {
            "generation":               self.generation,
            "status":                   self.status,
            "completed_steps":          self.completed_steps,
            "current_step":             self.current_step,
            "error":                    self.error,
            "started_at":               self.started_at,
            "completed_at":             self.completed_at,
            "model_path":               self.model_path,
            "merged_cfg":               self.merged_cfg,
            "replicate_tier":           self.replicate_tier,
            "converged_replicate_tier": self.converged_replicate_tier,
            "label_prepared":           self.label_prepared,
            "jobs_submitted":           self.jobs_submitted,
            "evaluation_done":          self.evaluation_done,
            "evaluation_failed":        self.evaluation_failed,
            "evaluation_manifest":      self.evaluation_manifest,
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
        """Finalise the generation, locking in converged_replicate_tier."""
        self.converged_replicate_tier = self.replicate_tier
        self.status = "completed"
        self.current_step = None
        self.completed_at = _now()
        self.save()

    def save_convergence_report(
        self,
        schedule: list[list[int]],
    ) -> Path:
        from mlip_pipeline.utils.fs import write_json
        tier = self.converged_replicate_tier
        replicate = schedule[min(tier, len(schedule) - 1)]
        report = {
            "generation":               self.generation,
            "converged_replicate_tier": tier,
            "converged_replicate":      replicate,
            "schedule":                 schedule,
            "completed_at":             self.completed_at,
        }
        path = self.gen_dir / "convergence.json"
        write_json(path, report)
        return path

    def mark_failed(self, exc: Exception) -> None:
        self.status = "failed"
        self.error = str(exc)
        self.save()

    def is_step_done(self, step: str) -> bool:
        return step in self.completed_steps
