"""Data models for the validate step."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class EosResult:
    """Outcome of an equation-of-state calculation for one structure."""
    structure_id: str           # e.g. "bcc", "fcc"
    volumes: list[float]        # Å³/atom
    energies: list[float]       # eV/atom
    # Birch-Murnaghan fit parameters (NaN if fit failed)
    V0: float = float("nan")    # equilibrium volume  Å³/atom
    E0: float = float("nan")    # equilibrium energy  eV/atom
    B0: float = float("nan")    # bulk modulus        GPa
    B0p: float = float("nan")   # pressure derivative (dimensionless)
    fit_ok: bool = False
    fit_error: Optional[str] = None


@dataclass
class ElasticResult:
    """Elastic constants for one structure (Voigt notation, GPa)."""
    structure_id: str
    C: dict = field(default_factory=dict)   # e.g. {"C11": 230.1, "C12": 134.5, ...}
    B_voigt: float = float("nan")           # Voigt bulk modulus
    G_voigt: float = float("nan")           # Voigt shear modulus
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Top-level result object written to validate/validate_manifest.json."""
    model_path: Path
    validate_dir: Path
    eos_results: list[EosResult] = field(default_factory=list)
    elastic_results: list[ElasticResult] = field(default_factory=list)
    plot_paths: dict = field(default_factory=dict)
    completed_at: str = field(default_factory=_now)

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def _clean(self, v):
        """Replace non-finite floats with None for JSON serialisation."""
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: self._clean(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [self._clean(x) for x in v]
        return v

    def save_manifest(self) -> Path:
        from mlip_pipeline.utils.fs import write_json

        eos_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "V0": r.V0, "E0": r.E0, "B0": r.B0, "B0p": r.B0p,
                "fit_ok": r.fit_ok,
                "fit_error": r.fit_error,
                "n_points": len(r.volumes),
            })
            for r in self.eos_results
        ]
        elastic_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "C": r.C,
                "B_voigt": r.B_voigt,
                "G_voigt": r.G_voigt,
                "compute_ok": r.compute_ok,
                "error": r.error,
            })
            for r in self.elastic_results
        ]
        path = self.validate_dir / "validate_manifest.json"
        write_json(path, self._clean({
            "step":             "validate",
            "model_path":       str(self.model_path),
            "eos_results":      eos_rows,
            "elastic_results":  elastic_rows,
            "plot_paths":       {k: str(v) for k, v in self.plot_paths.items()},
            "completed_at":     self.completed_at,
        }))
        return path

    @classmethod
    def load_manifest(cls, validate_dir: Path) -> "ValidationResult":
        d = json.loads((validate_dir / "validate_manifest.json").read_text())

        def _f(v):
            return float("nan") if v is None else float(v)

        eos = [
            EosResult(
                structure_id=r["structure_id"],
                volumes=[],
                energies=[],
                V0=_f(r.get("V0")), E0=_f(r.get("E0")),
                B0=_f(r.get("B0")), B0p=_f(r.get("B0p")),
                fit_ok=r.get("fit_ok", False),
                fit_error=r.get("fit_error"),
            )
            for r in d.get("eos_results", [])
        ]
        elastic = [
            ElasticResult(
                structure_id=r["structure_id"],
                C=r.get("C", {}),
                B_voigt=_f(r.get("B_voigt")),
                G_voigt=_f(r.get("G_voigt")),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in d.get("elastic_results", [])
        ]
        return cls(
            model_path=Path(d["model_path"]),
            validate_dir=validate_dir,
            eos_results=eos,
            elastic_results=elastic,
            plot_paths={k: Path(v) for k, v in d.get("plot_paths", {}).items()},
            completed_at=d.get("completed_at", ""),
        )
