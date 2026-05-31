"""Orchestrator for physics validation.

Entry point:  run_validation(config, model_path, out_dir)

Design principle
----------------
Computation and reporting are kept strictly separate:
  1. A :class:`~mlip_pipeline.models.RunPlan` is built first from CLI flags
     + config, encoding exactly what will and will not run.
  2. ALL computation tasks (every MTP step × structure AND every classical
     reference) are submitted to a ThreadPoolExecutor and run concurrently.
     Plotting and reporting only start once every future has resolved.
  3. For each configured classical reference, ``plot_comparison()`` produces
     side-by-side MTP vs reference PNG files and ``write_report()`` writes
     a Markdown report and CSV summary.
  4. When two or more references are run, ``plot_combined()`` is called once
     to overlay all potentials on a single figure per property type.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.fs import ensure_dir
from mlip_pipeline.models import (
    ValidationResult, EosResult, ElasticResult,
    MeltingResult, ThermalExpansionResult, VacancyResult, RdfResult,
)
from mlip_pipeline.models.validate import RunPlan, build_run_plan
from mlip_pipeline.validate.eos import run_eos
from mlip_pipeline.validate.elastic import run_elastic
from mlip_pipeline.validate.melting import run_melting
from mlip_pipeline.validate.thermal_expansion import run_thermal_expansion
from mlip_pipeline.validate.vacancy import run_vacancy
from mlip_pipeline.validate.rdf import run_rdf
from mlip_pipeline.validate import plots
from mlip_pipeline.validate.report import write_report
from mlip_pipeline.validate._cli import banner, step, ok, warn
from mlip_pipeline.validate.classical_reference import run_classical_reference


# ── Internal helpers ───────────────────────────────────────────────────────

def _resolve_val_config(config: dict) -> dict:
    return config.get("validate", config)


def _resolve_structures(step_cfg: dict, val_cfg: dict) -> list[dict]:
    if "structures" in step_cfg:
        return step_cfg["structures"]
    top = val_cfg.get("structures", {})
    if isinstance(top, dict):
        result = []
        for sid, sdata in top.items():
            entry = dict(sdata)
            entry.setdefault("id", sid)
            if "data_file" in entry and "lammps_data" not in entry:
                entry["lammps_data"] = entry["data_file"]
            result.append(entry)
        return result
    if isinstance(top, list):
        return top
    return []


def _resolve_data_path(s: dict, config: dict) -> Path:
    key = "lammps_data" if "lammps_data" in s else "data_file"
    data_path = Path(s[key])
    if not data_path.is_absolute():
        root = Path(config.get("project_root", "."))
        data_path = (root / data_path).resolve()
    return data_path


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return slug or "reference"


# ── Per-task wrappers (called inside worker threads) ───────────────────────

def _run_mtp_step(
    step_name: str,
    s: dict,
    config: dict,
    model_path: Path,
    val_dir: Path,
    val_cfg: dict,
    element: str,
    lammps_cmd: str,
    mpi_command: Optional[str],
    mpi_np: Optional[int],
    cutoff: Optional[float],
):
    """Execute one (step, structure) MTP computation and return the result object."""
    sid = s.get("id", s.get("structure_id", "unknown"))
    data_path = _resolve_data_path(s, config)

    if step_name == "eos":
        if not data_path.exists():
            warn("eos", f"{sid}: data not found — {data_path}")
            return EosResult(structure_id=sid, volumes=[], energies=[], fit_error=f"data file not found: {data_path}")
        eos_cfg = val_cfg.get("eos", {})
        res = run_eos(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            scale_min=eos_cfg.get("scale_min", 0.85),
            scale_max=eos_cfg.get("scale_max", 1.15),
            n_points=eos_cfg.get("n_points", 21),
            cutoff=cutoff,
        )
        ok("eos", f"{sid}: B\u2080={res.B0:.1f} GPa  V\u2080={res.V0:.3f} \u00c5\u00b3") if res.fit_ok else warn("eos", f"{sid}: fit failed — {res.fit_error}")
        return res

    if step_name == "elastic":
        if not data_path.exists():
            warn("elastic", f"{sid}: data not found — {data_path}")
            return ElasticResult(structure_id=sid, error=f"data file not found: {data_path}")
        el_cfg = val_cfg.get("elastic", {})
        res = run_elastic(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            delta=el_cfg.get("delta", 0.01),
            cutoff=cutoff,
        )
        ok("elastic", f"{sid}: B={res.B_voigt:.1f}  G={res.G_voigt:.1f} GPa") if res.compute_ok else warn("elastic", f"{sid}: failed — {res.error}")
        return res

    if step_name == "melting":
        if not data_path.exists():
            warn("melting", f"{sid}: data not found — {data_path}")
            return MeltingResult(structure_id=sid, error=f"data file not found: {data_path}")
        melt_cfg = val_cfg.get("melting", {})
        res = run_melting(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            cutoff=cutoff,
            T_start=melt_cfg.get("T_start", 400.0),
            T_end=melt_cfg.get("T_end", 900.0),
            T_step=melt_cfg.get("T_step", 50.0),
            supercell_repeat=melt_cfg.get("supercell_repeat", 4),
            n_equil=melt_cfg.get("n_equil", 5000),
            n_prod=melt_cfg.get("n_prod", 20000),
            dt=melt_cfg.get("dt", 0.002),
        )
        ok("melting", f"{sid}: T_melt = {res.T_melt:.0f} K") if res.compute_ok else warn("melting", f"{sid}: failed — {res.error}")
        return res

    if step_name in ("thexp", "thermal_expansion"):
        if not data_path.exists():
            warn("thexp", f"{sid}: data not found — {data_path}")
            return ThermalExpansionResult(structure_id=sid, error=f"data file not found: {data_path}")
        thexp_cfg = val_cfg.get("thexp", val_cfg.get("thermal_expansion", {}))
        res = run_thermal_expansion(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            cutoff=cutoff,
            temperatures=thexp_cfg.get("temperatures"),
            T_ref=thexp_cfg.get("T_ref", 300.0),
            n_equil=thexp_cfg.get("n_equil", 5000),
            n_prod=thexp_cfg.get("n_prod", 10000),
            dt=thexp_cfg.get("dt", 0.002),
        )
        ok("thexp", f"{sid}: \u03b1 = {res.alpha*1e6:.2f}\u00d710\u207b\u2076 K\u207b\u00b9") if res.compute_ok else warn("thexp", f"{sid}: failed — {res.error}")
        return res

    if step_name == "vacancy":
        if not data_path.exists():
            warn("vacancy", f"{sid}: data not found — {data_path}")
            return VacancyResult(structure_id=sid, error=f"data file not found: {data_path}")
        vac_cfg = val_cfg.get("vacancy", {})
        res = run_vacancy(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            cutoff=cutoff,
            supercell_repeat=vac_cfg.get("supercell_repeat", 3),
        )
        ok("vacancy", f"{sid}: E_vac = {res.E_vac:.4f} eV") if res.compute_ok else warn("vacancy", f"{sid}: failed — {res.error}")
        return res

    if step_name == "rdf":
        if not data_path.exists():
            warn("rdf", f"{sid}: data not found — {data_path}")
            return RdfResult(structure_id=sid, error=f"data file not found: {data_path}")
        rdf_cfg = val_cfg.get("rdf", {})
        res = run_rdf(
            sid, data_path, model_path, val_dir, element=element,
            lammps_cmd=lammps_cmd, mpi_command=mpi_command, mpi_np=mpi_np,
            cutoff=cutoff,
            temperature=rdf_cfg.get("temperature", 300.0),
            r_max=rdf_cfg.get("r_max", 8.0),
            n_bins=rdf_cfg.get("n_bins", 200),
            n_equil=rdf_cfg.get("n_equil", 5000),
            n_prod=rdf_cfg.get("n_prod", 20000),
            dt=rdf_cfg.get("dt", 0.002),
            supercell_repeat=rdf_cfg.get("supercell_repeat", 3),
        )
        ok("rdf", f"{sid}: first peak at {res.first_peak_r:.3f} \u00c5") if res.compute_ok else warn("rdf", f"{sid}: failed — {res.error}")
        return res

    raise ValueError(f"Unknown step: {step_name}")


# ── Public runner ─────────────────────────────────────────────────────────────

def run_validation(
    config: dict,
    model_path: Path,
    out_dir: Path,
    # ── MTP step filters ───────────────────────────────────────────────
    only_steps: Optional[list[str]] = None,
    skip_steps: Optional[list[str]] = None,
    skip_mtp: bool = False,
    # ── Reference filters ──────────────────────────────────────────────
    skip_refs: bool = False,
    only_refs: Optional[list[str]] = None,
    skip_ref: Optional[list[str]] = None,
) -> ValidationResult:
    """Run physics validation according to the flags provided.

    Parameters
    ----------
    only_steps / skip_steps:
        Filter *which MTP steps* run.  Mutually exclusive.
    skip_mtp:
        Skip all MTP computation (handy when you only want fresh reference
        comparison plots against a cached manifest).
    skip_refs:
        Skip all classical-reference computations.
    only_refs:
        Run *only* the reference(s) whose label matches one of these strings
        (case-insensitive exact match).  Mutually exclusive with ``skip_ref``.
    skip_ref:
        Skip the reference(s) whose label matches one of these strings
        (case-insensitive exact match).  Mutually exclusive with ``only_refs``.
    """
    from mlip_pipeline.cli import VALIDATE_STEPS  # avoid circular at import time

    val_cfg = _resolve_val_config(config)
    val_dir = ensure_dir(out_dir)
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    lammps_cmd = val_cfg.get("lammps_cmd") or val_cfg.get("lammps_command") or "lmp_mpi"
    mpi_command = val_cfg.get("mpi_command") or val_cfg.get("mpi_prefix")
    mpi_np = val_cfg.get("mpi_np")
    element = val_cfg.get("element", "Fe")
    cutoff: Optional[float] = val_cfg.get("cutoff") or val_cfg.get("lammps_cutoff")
    max_workers: Optional[int] = val_cfg.get("max_workers")  # None = auto

    # Build execution plan — single source of truth for what runs
    plan: RunPlan = build_run_plan(
        val_cfg=val_cfg,
        all_mtp_steps=VALIDATE_STEPS,
        only_steps=only_steps,
        skip_steps=skip_steps,
        skip_mtp=skip_mtp,
        skip_refs=skip_refs,
        only_refs=only_refs,
        skip_ref=skip_ref,
    )

    # ── Setup banner ─────────────────────────────────────────────────────────
    banner("Validation")
    step("setup", f"model    : {model_path}")
    step("setup", f"out_dir  : {val_dir}")
    step("setup", f"lammps   : {lammps_cmd}")
    step("setup", f"element  : {element}")
    if plan.skip_mtp:
        step("setup", "mtp      : SKIPPED (--skip-mtp)")
    elif only_steps:
        step("setup", f"only     : {plan.mtp_steps}")
    elif skip_steps:
        step("setup", f"skip     : {skip_steps}")
    if plan.skip_mtp is False and plan.mtp_steps != list(VALIDATE_STEPS):
        pass  # already printed above
    if plan.will_run_any_refs():
        labels = [r.get("label", "?") for r in plan.ref_configs]
        step("setup", f"refs     : {labels}")
    elif skip_refs:
        step("setup", "refs     : SKIPPED (--skip-refs)")
    elif only_refs:
        step("setup", f"only-ref : {only_refs}")
    elif skip_ref:
        step("setup", f"skip-ref : {skip_ref}")

    # ── Collect all tasks ────────────────────────────────────────────────────
    # Each task is a (step_name, structure_dict) pair for MTP steps, or a
    # ("_ref", ref_cfg, idx) tuple for classical references.
    # All tasks are submitted concurrently; results are collected afterwards.

    # MTP task list: [(step_name, structure_dict), ...]
    mtp_tasks: list[tuple[str, dict]] = []
    if plan.mtp_steps:
        for step_name in plan.mtp_steps:
            cfg_key = "thexp" if step_name in ("thexp", "thermal_expansion") else step_name
            step_cfg = val_cfg.get(cfg_key, {})
            for s in _resolve_structures(step_cfg, val_cfg):
                mtp_tasks.append((step_name, s))

    # Reference task list: [(idx, ref_cfg), ...]
    ref_tasks: list[tuple[int, dict]] = list(enumerate(plan.ref_configs, start=1))

    n_total = len(mtp_tasks) + len(ref_tasks)
    _workers = max_workers or n_total or 1
    step("setup", f"workers  : {_workers} concurrent task(s)")

    if plan.mtp_steps:
        banner("MTP + Reference Computation  [parallel]")
    elif ref_tasks:
        banner("Reference Computation  [parallel]")

    # ── Submit and collect ───────────────────────────────────────────────────
    # future -> (kind, key) where kind in ('mtp', 'ref')
    future_meta: dict[Future, tuple[str, object]] = {}

    # Accumulate results keyed by (step_name, structure_id) / ref_idx
    mtp_raw: dict[tuple[str, str], object] = {}
    ref_raw: dict[int, object] = {}  # idx -> ref_result

    common_kw = dict(
        config=config,
        model_path=model_path,
        val_dir=val_dir,
        val_cfg=val_cfg,
        element=element,
        lammps_cmd=lammps_cmd,
        mpi_command=mpi_command,
        mpi_np=mpi_np,
        cutoff=cutoff,
    )

    with ThreadPoolExecutor(max_workers=_workers) as pool:
        # Submit MTP tasks
        for step_name, s in mtp_tasks:
            sid = s.get("id", s.get("structure_id", "unknown"))
            f = pool.submit(_run_mtp_step, step_name, s, **common_kw)
            future_meta[f] = ("mtp", (step_name, sid))

        # Submit reference tasks
        for idx, ref_cfg in ref_tasks:
            pair_style = ref_cfg.get("pair_style")
            pair_coeff = ref_cfg.get("pair_coeff")
            ref_label = ref_cfg.get("label", f"classical_{idx}")
            ref_slug = _slugify(ref_label)
            ref_dir = val_dir / f"classical_ref_{ref_slug}"

            if not (pair_style and pair_coeff):
                warn("compare", f"{ref_label}: missing pair_style/pair_coeff — skipping")
                continue

            f = pool.submit(
                run_classical_reference,
                config=config,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                out_dir=ref_dir,
                mtp_result=None,   # will be injected after MTP completes
                label=ref_label,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                cutoff=cutoff,
            )
            future_meta[f] = ("ref", idx)

        # Collect as they finish
        for f in as_completed(future_meta):
            kind, key = future_meta[f]
            try:
                result_obj = f.result()
            except Exception as exc:  # noqa: BLE001
                if kind == "mtp":
                    step_name, sid = key  # type: ignore[misc]
                    warn(step_name, f"{sid}: exception — {exc}")
                else:
                    warn("compare", f"reference {key}: exception — {exc}")
                result_obj = None

            if kind == "mtp":
                mtp_raw[key] = result_obj  # type: ignore[index]
            else:
                ref_raw[key] = result_obj  # type: ignore[index]

    # ── Assemble typed result lists ──────────────────────────────────────────
    eos_results: list[EosResult] = []
    elastic_results: list[ElasticResult] = []
    melting_results: list[MeltingResult] = []
    thexp_results: list[ThermalExpansionResult] = []
    vacancy_results: list[VacancyResult] = []
    rdf_results: list[RdfResult] = []

    for (step_name, _sid), res in mtp_raw.items():
        if res is None:
            continue
        if step_name == "eos":
            eos_results.append(res)  # type: ignore[arg-type]
        elif step_name == "elastic":
            elastic_results.append(res)  # type: ignore[arg-type]
        elif step_name == "melting":
            melting_results.append(res)  # type: ignore[arg-type]
        elif step_name in ("thexp", "thermal_expansion"):
            thexp_results.append(res)  # type: ignore[arg-type]
        elif step_name == "vacancy":
            vacancy_results.append(res)  # type: ignore[arg-type]
        elif step_name == "rdf":
            rdf_results.append(res)  # type: ignore[arg-type]

    result = ValidationResult(
        model_path=model_path,
        validate_dir=val_dir,
        eos_results=eos_results,
        elastic_results=elastic_results,
        melting_results=melting_results,
        thermal_expansion_results=thexp_results,
        vacancy_results=vacancy_results,
        rdf_results=rdf_results,
        plot_paths={},
    )
    result.save_manifest()

    # ── Classical reference comparisons (sequential — needs MTP result) ───────
    plot_paths: dict[str, Path] = {}
    all_ref_results = []

    for idx, ref_cfg in ref_tasks:
        ref_result = ref_raw.get(idx)
        if ref_result is None:
            continue

        ref_label = ref_cfg.get("label", f"classical_{idx}")
        ref_slug = _slugify(ref_label)
        cmp_dir = ensure_dir(val_dir / f"comparison_{ref_slug}")

        # Inject the now-complete MTP result into the reference result
        ref_result.mtp_result = result
        all_ref_results.append(ref_result)

        banner(f"Comparison: MTP vs {ref_label}")
        ref_plot_paths = plots.plot_comparison(result, ref_result, cmp_dir)
        for k, p in ref_plot_paths.items():
            ok("compare", f"{k} \u2192 {p.relative_to(val_dir)}")
        plot_paths.update({f"{ref_slug}:{k}": p for k, p in ref_plot_paths.items()})

        banner(f"Report: {ref_label}")
        md_path, csv_path = write_report(result, ref_result, val_dir)
        ok("report", f"Markdown \u2192 {md_path.name}")
        ok("report", f"CSV      \u2192 {csv_path.name}")

    # ── Combined plots (all refs on one figure) ────────────────────────────────
    if len(all_ref_results) >= 1:
        combined_dir = ensure_dir(val_dir / "comparison_combined")
        banner("Combined: MTP vs all references")
        combined_paths = plots.plot_combined(result, all_ref_results, combined_dir)
        for k, p in combined_paths.items():
            ok("combined", f"{k} \u2192 {p.relative_to(val_dir)}")
        plot_paths.update({f"combined:{k}": p for k, p in combined_paths.items()})

    banner("Done")
    step("output", f"manifest \u2192 {val_dir / 'validate_manifest.json'}")
    if plot_paths:
        n_refs = len({k.split(':', 1)[0] for k in plot_paths})
        step("output", f"plots    \u2192 {len(plot_paths)} file(s) across {n_refs} output group(s)")

    return result
