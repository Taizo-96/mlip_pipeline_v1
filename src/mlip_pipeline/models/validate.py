"""Dataclasses for the validation step.

All public symbols are re-exported via ``mlip_pipeline.models``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Per-step result types ─────────────────────────────────────────────────────

@dataclass
class EosResult:
    structure_id: str
    volumes: list[float]
    energies: list[float]
    V0: float = 0.0
    E0: float = 0.0
    B0: float = 0.0
    B0p: float = 0.0
    fit_ok: bool = False
    fit_error: Optional[str] = None


@dataclass
class ElasticResult:
    structure_id: str
    C: dict = field(default_factory=dict)
    B_voigt: float = 0.0
    G_voigt: float = 0.0
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class MeltingResult:
    structure_id: str
    T_melt: float = 0.0
    T_bracket_lo: float = field(default_factory=lambda: float("nan"))
    T_bracket_hi: float = field(default_factory=lambda: float("nan"))
    compute_ok: bool = False
    error: Optional[str] = None

    @property
    def T_bracket(self) -> tuple:
        lo = None if math.isnan(self.T_bracket_lo) else self.T_bracket_lo
        hi = None if math.isnan(self.T_bracket_hi) else self.T_bracket_hi
        return (lo, hi)


@dataclass
class ThermalExpansionResult:
    structure_id: str
    temperatures: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    alpha: float = 0.0
    T_ref: float = 300.0
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class VacancyResult:
    structure_id: str
    E_vac: float = 0.0
    n_atoms_perfect: int = 0
    n_atoms_vacancy: int = 0
    compute_ok: bool = False
    error: Optional[str] = None


@dataclass
class RdfResult:
    structure_id: str
    r: list[float] = field(default_factory=list)
    g_r: list[float] = field(default_factory=list)
    first_peak_r: float = 0.0
    first_peak_g: float = 0.0
    temperature: float = 300.0
    compute_ok: bool = False
    error: Optional[str] = None


# ── Aggregated result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    model_path: Path
    validate_dir: Path
    eos_results: list[EosResult] = field(default_factory=list)
    elastic_results: list[ElasticResult] = field(default_factory=list)
    melting_results: list[MeltingResult] = field(default_factory=list)
    thermal_expansion_results: list[ThermalExpansionResult] = field(default_factory=list)
    vacancy_results: list[VacancyResult] = field(default_factory=list)
    rdf_results: list[RdfResult] = field(default_factory=list)
    plot_paths: dict[str, Path] = field(default_factory=dict)

    def save_manifest(self) -> Path:
        manifest = {
            "model_path": str(self.model_path),
            "validate_dir": str(self.validate_dir),
            "eos": [
                {
                    "structure_id": r.structure_id,
                    "V0": r.V0, "E0": r.E0, "B0": r.B0, "B0p": r.B0p,
                    "volumes": r.volumes, "energies": r.energies,
                    "fit_ok": r.fit_ok, "fit_error": r.fit_error,
                }
                for r in self.eos_results
            ],
            "elastic": [
                {
                    "structure_id": r.structure_id,
                    "C": r.C, "B_voigt": r.B_voigt, "G_voigt": r.G_voigt,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.elastic_results
            ],
            "melting": [
                {
                    "structure_id": r.structure_id,
                    "T_melt": r.T_melt,
                    "T_bracket": list(r.T_bracket),
                    "T_bracket_lo": r.T_bracket_lo,
                    "T_bracket_hi": r.T_bracket_hi,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.melting_results
            ],
            "thermal_expansion": [
                {
                    "structure_id": r.structure_id,
                    "temperatures": r.temperatures, "volumes": r.volumes,
                    "alpha": r.alpha, "T_ref": r.T_ref,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.thermal_expansion_results
            ],
            "vacancy": [
                {
                    "structure_id": r.structure_id,
                    "E_vac": r.E_vac,
                    "n_atoms_perfect": r.n_atoms_perfect,
                    "n_atoms_vacancy": r.n_atoms_vacancy,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.vacancy_results
            ],
            "rdf": [
                {
                    "structure_id": r.structure_id,
                    "r": r.r, "g_r": r.g_r,
                    "first_peak_r": r.first_peak_r, "first_peak_g": r.first_peak_g,
                    "temperature": r.temperature,
                    "compute_ok": r.compute_ok, "error": r.error,
                }
                for r in self.rdf_results
            ],
            "plots": {k: str(v) for k, v in self.plot_paths.items()},
        }
        path = self.validate_dir / "validate_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load_manifest(cls, manifest_path: Path) -> "ValidationResult":
        """Reconstruct a ValidationResult from a saved validate_manifest.json.

        No LAMMPS calls are made — all data is read from the JSON file.
        """
        data = json.loads(manifest_path.read_text())
        validate_dir = Path(data["validate_dir"])
        model_path   = Path(data["model_path"])

        eos_results = [
            EosResult(
                structure_id=r["structure_id"],
                volumes=r.get("volumes", []),
                energies=r.get("energies", []),
                V0=r.get("V0", 0.0),
                E0=r.get("E0", 0.0),
                B0=r.get("B0", 0.0),
                B0p=r.get("B0p", 0.0),
                fit_ok=r.get("fit_ok", False),
                fit_error=r.get("fit_error"),
            )
            for r in data.get("eos", [])
        ]
        elastic_results = [
            ElasticResult(
                structure_id=r["structure_id"],
                C=r.get("C") or {},
                B_voigt=r.get("B_voigt", 0.0),
                G_voigt=r.get("G_voigt", 0.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("elastic", [])
        ]
        melting_results = [
            MeltingResult(
                structure_id=r["structure_id"],
                T_melt=r.get("T_melt", 0.0),
                T_bracket_lo=r.get("T_bracket_lo", float("nan")),
                T_bracket_hi=r.get("T_bracket_hi", float("nan")),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("melting", [])
        ]
        thexp_results = [
            ThermalExpansionResult(
                structure_id=r["structure_id"],
                temperatures=r.get("temperatures", []),
                volumes=r.get("volumes", []),
                alpha=r.get("alpha", 0.0),
                T_ref=r.get("T_ref", 300.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("thermal_expansion", [])
        ]
        vacancy_results = [
            VacancyResult(
                structure_id=r["structure_id"],
                E_vac=r.get("E_vac", 0.0),
                n_atoms_perfect=r.get("n_atoms_perfect", 0),
                n_atoms_vacancy=r.get("n_atoms_vacancy", 0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("vacancy", [])
        ]
        rdf_results = [
            RdfResult(
                structure_id=r["structure_id"],
                r=r.get("r", []),
                g_r=r.get("g_r", []),
                first_peak_r=r.get("first_peak_r", 0.0),
                first_peak_g=r.get("first_peak_g", 0.0),
                temperature=r.get("temperature", 300.0),
                compute_ok=r.get("compute_ok", False),
                error=r.get("error"),
            )
            for r in data.get("rdf", [])
        ]
        plot_paths = {k: Path(v) for k, v in data.get("plots", {}).items()}

        return cls(
            model_path=model_path,
            validate_dir=validate_dir,
            eos_results=eos_results,
            elastic_results=elastic_results,
            melting_results=melting_results,
            thermal_expansion_results=thexp_results,
            vacancy_results=vacancy_results,
            rdf_results=rdf_results,
            plot_paths=plot_paths,
        )


# ── Execution plan ────────────────────────────────────────────────────────────

@dataclass
class RunPlan:
    """Immutable description of what ``run_validation`` will actually execute.

    Built once by :func:`build_run_plan` before any LAMMPS call is made, so
    the logic that interprets CLI flags lives in exactly one place.

    Attributes
    ----------
    mtp_steps:
        Ordered list of MTP steps to run.  Empty list means *skip all MTP*
        computation (useful when you only want reference comparisons or when
        a cached manifest already exists).
    ref_configs:
        Filtered, ordered list of reference-potential config dicts to run.
        Each entry is guaranteed to have ``pair_style`` and ``pair_coeff``.
    skip_mtp:
        True when the caller asked to bypass MTP computation entirely.
    """
    mtp_steps: list[str]
    ref_configs: list[dict]
    skip_mtp: bool = False

    def will_run_step(self, name: str) -> bool:
        return name in self.mtp_steps

    def will_run_any_refs(self) -> bool:
        return bool(self.ref_configs)


def build_run_plan(
    val_cfg: dict,
    all_mtp_steps: tuple[str, ...],
    only_steps: Optional[list[str]] = None,
    skip_steps: Optional[list[str]] = None,
    skip_mtp: bool = False,
    skip_refs: bool = False,
    only_refs: Optional[list[str]] = None,
    skip_ref: Optional[list[str]] = None,
) -> RunPlan:
    """Compute a :class:`RunPlan` from flags + config."""
    skip_steps = list(skip_steps or [])
    skip_ref = list(skip_ref or [])

    if skip_mtp:
        mtp_steps: list[str] = []
    elif only_steps is not None:
        mtp_steps = [s for s in all_mtp_steps if s in only_steps]
    else:
        mtp_steps = [
            s for s in all_mtp_steps
            if s not in skip_steps
            and val_cfg.get(s, {}).get("enabled", True)
        ]

    raw_refs = _resolve_reference_configs(val_cfg)
    refs = [r for r in raw_refs if r.get("enabled", False)]

    if skip_refs:
        refs = []
    elif only_refs is not None:
        _lower = [o.lower() for o in only_refs]
        refs = [r for r in refs if r.get("label", "").lower() in _lower]
    elif skip_ref:
        _lower = [s.lower() for s in skip_ref]
        refs = [r for r in refs if r.get("label", "").lower() not in _lower]

    return RunPlan(mtp_steps=mtp_steps, ref_configs=refs, skip_mtp=skip_mtp)


def _resolve_reference_configs(val_cfg: dict) -> list[dict]:
    """Normalise classical_references (list) or legacy classical_reference (dict)."""
    refs = val_cfg.get("classical_references")
    if isinstance(refs, list):
        return refs
    legacy = val_cfg.get("classical_reference")
    if isinstance(legacy, dict) and legacy:
        return [legacy]
    return []
