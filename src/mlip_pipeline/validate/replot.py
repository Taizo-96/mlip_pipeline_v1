"""Replot validation results from cached manifests without re-running LAMMPS.

Public API
----------
replot_validation(validate_dir, out_dir)
    Load validate_manifest.json + all ref_manifest.json files found under
    validate_dir, then regenerate every comparison and combined plot.
"""
from __future__ import annotations

import re
from pathlib import Path

from mlip_pipeline.models.validate import ValidationResult
from mlip_pipeline.validate.classical_reference import ClassicalReferenceResult
from mlip_pipeline.validate import plots
from mlip_pipeline.validate.report import write_report
from mlip_pipeline.validate._cli import banner, step, ok, warn
from mlip_pipeline.utils.fs import ensure_dir


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return slug or "reference"


def replot_validation(
    validate_dir: Path,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Regenerate all validation plots from cached manifests.

    Parameters
    ----------
    validate_dir:
        Directory that contains ``validate_manifest.json`` and
        ``classical_ref_*/ref_manifest.json`` sub-directories produced by
        a previous ``validate`` run.
    out_dir:
        Where to write the new plot files.  Defaults to ``validate_dir``
        (i.e. plots are regenerated in-place, overwriting previous ones).

    Returns
    -------
    dict mapping plot key -> Path for every PNG produced.
    """
    validate_dir = validate_dir.resolve()
    out_dir = (out_dir or validate_dir).resolve()

    # ── Load MTP manifest ────────────────────────────────────────────────────
    mtp_manifest = validate_dir / "validate_manifest.json"
    if not mtp_manifest.exists():
        raise FileNotFoundError(
            f"validate_manifest.json not found in {validate_dir}\n"
            "Run 'validate' first to generate the cache."
        )

    banner("Replot Validation")
    step("load", f"MTP manifest  → {mtp_manifest.relative_to(validate_dir)}")
    mtp_result = ValidationResult.load_manifest(mtp_manifest)

    # ── Discover reference manifests ─────────────────────────────────────────
    ref_manifest_paths = sorted(
        validate_dir.glob("classical_ref_*/ref_manifest.json")
    )
    if not ref_manifest_paths:
        warn("load", "No ref_manifest.json files found — only MTP data is available.")
        warn("load", "Nothing to plot (no reference potentials to compare against).")
        return {}

    ref_results: list[ClassicalReferenceResult] = []
    for rmp in ref_manifest_paths:
        step("load", f"ref manifest  → {rmp.relative_to(validate_dir)}")
        ref_results.append(ClassicalReferenceResult.load_manifest(rmp))

    # ── Pairwise comparison plots ────────────────────────────────────────────
    all_plot_paths: dict[str, Path] = {}

    for ref_result in ref_results:
        ref_slug = _slugify(ref_result.label)
        cmp_dir  = ensure_dir(out_dir / f"comparison_{ref_slug}")

        banner(f"Comparison: MTP vs {ref_result.label}")
        ref_plot_paths = plots.plot_comparison(mtp_result, ref_result, cmp_dir)
        for k, p in ref_plot_paths.items():
            ok("plot", f"{k} → {p.relative_to(out_dir)}")
        all_plot_paths.update(
            {f"{ref_slug}:{k}": p for k, p in ref_plot_paths.items()}
        )

        # Re-write the Markdown/CSV report alongside the fresh plots
        banner(f"Report: {ref_result.label}")
        md_path, csv_path = write_report(mtp_result, ref_result, out_dir)
        ok("report", f"Markdown → {md_path.name}")
        ok("report", f"CSV      → {csv_path.name}")

    # ── Combined plots ───────────────────────────────────────────────────────
    combined_dir = ensure_dir(out_dir / "comparison_combined")
    banner("Combined: MTP vs all references")
    combined_paths = plots.plot_combined(mtp_result, ref_results, combined_dir)
    for k, p in combined_paths.items():
        ok("combined", f"{k} → {p.relative_to(out_dir)}")
    all_plot_paths.update({f"combined:{k}": p for k, p in combined_paths.items()})

    banner("Done")
    step("output", f"{len(all_plot_paths)} plot file(s) written to {out_dir}")
    return all_plot_paths
