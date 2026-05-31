"""Result dataclasses for the validate step."""
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
    structure_id: str
    volumes: list[float]
    energies: list[float]
    V0: float = float("nan")
    E0: float = float("nan")
    B0: float = float("nan")
    B0p: float = float("nan")
    fit_ok: bool = False
    fit_error: Optional[str] = None


@dataclass
class ElasticResult:
    """Elastic constants for one structure (Voigt notation, GPa)."""
    structure_id: str
    C: dict = field(default_factory=dict)
    B_voigt: float = float("nan")
    G_voigt: float = float("nan")
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class MeltingResult:
    """Melting temperature estimate from two-phase coexistence."""
    structure_id: str
    T_melt: float = float("nan")
    T_bracket_lo: float = float("nan")
    T_bracket_hi: float = float("nan")
    method: str = "two_phase"
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class ThermalExpansionResult:
    """Linear thermal expansion coefficient from NPT MD."""
    structure_id: str
    temperatures: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    alpha: float = float("nan")
    V_ref: float = float("nan")
    T_ref: float = float("nan")
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class VacancyResult:
    """Vacancy formation energy."""
    structure_id: str
    E_vac: float = float("nan")
    n_atoms_perfect: int = 0
    n_atoms_vacancy: int = 0
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class RdfResult:
    """Radial distribution function from NVT MD."""
    structure_id: str
    r: list[float] = field(default_factory=list)
    g_r: list[float] = field(default_factory=list)
    temperature: float = float("nan")
    first_peak_r: float = float("nan")
    first_peak_g: float = float("nan")
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Top-level result object written to validate/validate_manifest.json."""
    model_path: Path
    validate_dir: Path
    eos_results: list[EosResult] = field(default_factory=list)
    elastic_results: list[ElasticResult] = field(default_factory=list)
    melting_results: list[MeltingResult] = field(default_factory=list)
    thermal_expansion_results: list[ThermalExpansionResult] = field(default_factory=list)
    vacancy_results: list[VacancyResult] = field(default_factory=list)
    rdf_results: list[RdfResult] = field(default_factory=list)
    plot_paths: dict = field(default_factory=dict)
    completed_at: str = field(default_factory=_now)

    def _clean(self, v):
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
                "fit_ok": r.fit_ok, "fit_error": r.fit_error,
                "n_points": len(r.volumes),
            })
            for r in self.eos_results
        ]
        elastic_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "C": r.C, "B_voigt": r.B_voigt, "G_voigt": r.G_voigt,
                "compute_ok": r.compute_ok, "error": r.error,
            })
            for r in self.elastic_results
        ]
        melting_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "T_melt": r.T_melt,
                "T_bracket_lo": r.T_bracket_lo,
                "T_bracket_hi": r.T_bracket_hi,
                "method": r.method,
                "compute_ok": r.compute_ok, "error": r.error,
            })
            for r in self.melting_results
        ]
        thexp_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "alpha": r.alpha, "V_ref": r.V_ref, "T_ref": r.T_ref,
                "temperatures": r.temperatures, "volumes": r.volumes,
                "compute_ok": r.compute_ok, "error": r.error,
            })
            for r in self.thermal_expansion_results
        ]
        vacancy_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "E_vac": r.E_vac,
                "n_atoms_perfect": r.n_atoms_perfect,
                "n_atoms_vacancy": r.n_atoms_vacancy,
                "compute_ok": r.compute_ok, "error": r.error,
            })
            for r in self.vacancy_results
        ]
        rdf_rows = [
            self._clean({
                "structure_id": r.structure_id,
                "temperature": r.temperature,
                "first_peak_r": r.first_peak_r,
                "first_peak_g": r.first_peak_g,
                "n_points": len(r.r),
                "compute_ok": r.compute_ok, "error": r.error,
            })
            for r in self.rdf_results
        ]
        path = self.validate_dir / "validate_manifest.json"
        write_json(path, self._clean({
            "step":                      "validate",
            "model_path":                str(self.model_path),
            "eos_results":               eos_rows,
            "elastic_results":           elastic_rows,
            "melting_results":           melting_rows,
            "thermal_expansion_results": thexp_rows,
            "vacancy_results":           vacancy_rows,
            "rdf_results":               rdf_rows,
            "plot_paths":                {k: str(v) for k, v in self.plot_paths.items()},
            "completed_at":              self.completed_at,
        }))
        return path

    @classmethod
    def load_manifest(cls, validate_dir: Path) -> "ValidationResult":
        d = json.loads((validate_dir / "validate_manifest.json").read_text())

        def _f(v):
            return float("nan") if v is None else float(v)

        eos = [
            EosResult(
                structure_id=r["structure_id"], volumes=[], energies=[],
                V0=_f(r.get("V0")), E0=_f(r.get("E0")),
                B0=_f(r.get("B0")), B0p=_f(r.get("B0p")),
                fit_ok=r.get("fit_ok", False), fit_error=r.get("fit_error"),
            )
            for r in d.get("eos_results", [])
        ]
        elastic = [
            ElasticResult(
                structure_id=r["structure_id"],
                C=r.get("C", {}),
                B_voigt=_f(r.get("B_voigt")), G_voigt=_f(r.get("G_voigt")),
                compute_ok=r.get("compute_ok", False), error=r.get("error"),
            )
            for r in d.get("elastic_results", [])
        ]
        melting = [
            MeltingResult(
                structure_id=r["structure_id"],
                T_melt=_f(r.get("T_melt")),
                T_bracket_lo=_f(r.get("T_bracket_lo")),
                T_bracket_hi=_f(r.get("T_bracket_hi")),
                method=r.get("method", "two_phase"),
                compute_ok=r.get("compute_ok", False), error=r.get("error"),
            )
            for r in d.get("melting_results", [])
        ]
        thexp = [
            ThermalExpansionResult(
                structure_id=r["structure_id"],
                temperatures=r.get("temperatures", []),
                volumes=r.get("volumes", []),
                alpha=_f(r.get("alpha")), V_ref=_f(r.get("V_ref")),
                T_ref=_f(r.get("T_ref")),
                compute_ok=r.get("compute_ok", False), error=r.get("error"),
            )
            for r in d.get("thermal_expansion_results", [])
        ]
        vacancy = [
            VacancyResult(
                structure_id=r["structure_id"],
                E_vac=_f(r.get("E_vac")),
                n_atoms_perfect=r.get("n_atoms_perfect", 0),
                n_atoms_vacancy=r.get("n_atoms_vacancy", 0),
                compute_ok=r.get("compute_ok", False), error=r.get("error"),
            )
            for r in d.get("vacancy_results", [])
        ]
        rdf = [
            RdfResult(
                structure_id=r["structure_id"],
                temperature=_f(r.get("temperature")),
                first_peak_r=_f(r.get("first_peak_r")),
                first_peak_g=_f(r.get("first_peak_g")),
                compute_ok=r.get("compute_ok", False), error=r.get("error"),
            )
            for r in d.get("rdf_results", [])
        ]
        return cls(
            model_path=Path(d["model_path"]),
            validate_dir=validate_dir,
            eos_results=eos, elastic_results=elastic,
            melting_results=melting, thermal_expansion_results=thexp,
            vacancy_results=vacancy, rdf_results=rdf,
            plot_paths={k: Path(v) for k, v in d.get("plot_paths", {}).items()},
            completed_at=d.get("completed_at", ""),
        )


__all__ = [
    "EosResult",
    "ElasticResult",
    "MeltingResult",
    "ThermalExpansionResult",
    "VacancyResult",
    "RdfResult",
    "ValidationResult",
]
