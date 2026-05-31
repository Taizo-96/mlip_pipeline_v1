"""Active-learning loop state and summary models.

GenerationState  — mutable per-generation fault-tolerance record, written
                   throughout a generation's lifetime to support crash recovery.
LoopResult       — immutable summary written once when run_loop() exits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


#: Ordered pipeline steps recognised by GenerationState.
STEPS = ("fit", "explore", "select", "label", "label_hpc", "label_local", "convert", "evaluate")

#: Valid values for LoopResult.stop_reason.
STOP_REASONS = (
    "completed",   # requested end_gen was reached
    "converged",   # potential found 0 new structures at all replicate tiers
    "failed",      # a generation failed and stop_on_failure=True
    "interrupted", # KeyboardInterrupt or unexpected exception in run_loop
)


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
    jobs_submitted: bool = False
    evaluation_done: bool = False
    evaluation_failed: bool = False
    evaluation_manifest: Optional[str] = None

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

    def save_convergence_report(self, schedule: list[list[int]]) -> Path:
        from mlip_pipeline.utils.fs import write_json
        tier = self.converged_replicate_tier
        replicate = schedule[min(tier, len(schedule) - 1)]
        path = self.gen_dir / "convergence.json"
        write_json(path, {
            "generation":               self.generation,
            "converged_replicate_tier": tier,
            "converged_replicate":      replicate,
            "schedule":                 schedule,
            "completed_at":             self.completed_at,
        })
        return path

    def mark_failed(self, exc: Exception) -> None:
        self.status = "failed"
        self.error = str(exc)
        self.save()

    def is_step_done(self, step: str) -> bool:
        return step in self.completed_steps


@dataclass
class LoopResult:
    """Immutable summary of a completed (or interrupted) active-learning loop.

    Written to <runs_root>/loop_manifest.json by run_loop() on exit.
    """
    runs_root: Path
    start_gen: int
    requested_end_gen: int
    actual_end_gen: int
    stop_reason: str
    n_gens_completed: int
    n_gens_failed: int
    failed_gens: list[int] = field(default_factory=list)
    generations: list[dict] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = field(default_factory=_now)

    @classmethod
    def from_states(
        cls,
        runs_root: Path,
        states: list["GenerationState"],
        start_gen: int,
        requested_end_gen: int,
        stop_reason: str,
        started_at: str,
    ) -> "LoopResult":
        """Build a LoopResult by reading step-manifest files for each state."""
        from mlip_pipeline.evaluate.loop_plots import collect_loop_records

        processed_gens = [int(s.generation) for s in states]
        actual_end_gen = max(processed_gens) if processed_gens else start_gen
        failed_gens = [int(s.generation) for s in states if s.status == "failed"]
        completed_states = [s for s in states if s.status in ("completed", "converged")]
        gen_rows = collect_loop_records(runs_root, processed_gens)

        return cls(
            runs_root=runs_root,
            start_gen=start_gen,
            requested_end_gen=requested_end_gen,
            actual_end_gen=actual_end_gen,
            stop_reason=stop_reason,
            n_gens_completed=len(completed_states),
            n_gens_failed=len(failed_gens),
            failed_gens=failed_gens,
            generations=gen_rows,
            started_at=started_at,
        )

    def save(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        import math

        def _clean(v):
            if isinstance(v, float) and not math.isfinite(v):
                return None
            if isinstance(v, dict):
                return {k: _clean(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_clean(x) for x in v]
            return v

        path = self.runs_root / "loop_manifest.json"
        write_json(path, _clean({
            "start_gen":         self.start_gen,
            "requested_end_gen": self.requested_end_gen,
            "actual_end_gen":    self.actual_end_gen,
            "stop_reason":       self.stop_reason,
            "n_gens_completed":  self.n_gens_completed,
            "n_gens_failed":     self.n_gens_failed,
            "failed_gens":       self.failed_gens,
            "generations":       self.generations,
            "started_at":        self.started_at,
            "completed_at":      self.completed_at,
        }))
        return path

    @classmethod
    def load(cls, runs_root: Path) -> "LoopResult":
        d = json.loads((runs_root / "loop_manifest.json").read_text())
        return cls(
            runs_root=runs_root,
            start_gen=d["start_gen"],
            requested_end_gen=d["requested_end_gen"],
            actual_end_gen=d["actual_end_gen"],
            stop_reason=d["stop_reason"],
            n_gens_completed=d["n_gens_completed"],
            n_gens_failed=d["n_gens_failed"],
            failed_gens=d.get("failed_gens", []),
            generations=d.get("generations", []),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
        )


__all__ = ["STEPS", "STOP_REASONS", "GenerationState", "LoopResult"]
