"""VASP parallelisation and scaling models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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


__all__ = ["VaspParallelConfig", "ScalingPolicy"]
