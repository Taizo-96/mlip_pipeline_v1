"""Write validation summary tables to Markdown and CSV.

Public API
----------
write_report(mtp_result, ref_result, out_dir, ref_label) -> (md_path, csv_path)

The Markdown file is the human-readable artefact; it contains one table
per property family with MTP value, reference value, and relative deviation.
The CSV mirrors the same rows for downstream scripting.

Files are written to ``out_dir`` with a label-based suffix when needed, e.g.:
  validation_report.md
  validation_summary.csv
  validation_report_Lee2003_MEAM.md
  validation_summary_Lee2003_MEAM.csv
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_pipeline.models import ValidationResult

if TYPE_CHECKING:
    from mlip_pipeline.validate.classical_reference import ClassicalReferenceResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _pct(mtp: float, ref: float) -> str:
    """Return percentage deviation string, or 'n/a' if either value is invalid."""
    if not math.isfinite(mtp) or not math.isfinite(ref) or ref == 0:
        return "n/a"
    return f"{(mtp - ref) / abs(ref) * 100:+.1f}%"


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{decimals}f}"


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return slug or "reference"


# ── row collectors ────────────────────────────────────────────────────────────

Row = dict[str, str]


def _collect_eos(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.eos_results}
    rows: list[Row] = []
    for mr in mtp.eos_results:
        rr = ref_map.get(mr.structure_id)
        rows.append({"property": "EOS", "structure": mr.structure_id, "quantity": "V0 (Å³/atom)", "MTP": _fmt(mr.V0, 4) if mr.fit_ok else "—", "reference": _fmt(rr.V0, 4) if (rr and rr.fit_ok) else "—", "delta_pct": _pct(mr.V0, rr.V0) if (mr.fit_ok and rr and rr.fit_ok) else "n/a"})
        rows.append({"property": "EOS", "structure": mr.structure_id, "quantity": "B0 (GPa)", "MTP": _fmt(mr.B0, 2) if mr.fit_ok else "—", "reference": _fmt(rr.B0, 2) if (rr and rr.fit_ok) else "—", "delta_pct": _pct(mr.B0, rr.B0) if (mr.fit_ok and rr and rr.fit_ok) else "n/a"})
        rows.append({"property": "EOS", "structure": mr.structure_id, "quantity": "B0' (—)", "MTP": _fmt(mr.B0p, 2) if mr.fit_ok else "—", "reference": _fmt(rr.B0p, 2) if (rr and rr.fit_ok) else "—", "delta_pct": _pct(mr.B0p, rr.B0p) if (mr.fit_ok and rr and rr.fit_ok) else "n/a"})
    return rows


def _collect_elastic(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.elastic_results}
    rows: list[Row] = []
    for mr in mtp.elastic_results:
        rr = ref_map.get(mr.structure_id)
        if not mr.compute_ok:
            continue
        all_keys = sorted(set((mr.C or {}).keys()) | set(((rr.C if rr else None) or {}).keys()))
        for k in all_keys:
            mv = (mr.C or {}).get(k)
            rv = ((rr.C if rr else None) or {}).get(k) if rr else None
            rows.append({"property": "Elastic", "structure": mr.structure_id, "quantity": f"{k} (GPa)", "MTP": _fmt(mv, 2) if mv is not None else "—", "reference": _fmt(rv, 2) if rv is not None else "—", "delta_pct": _pct(mv, rv) if (mv is not None and rv is not None) else "n/a"})
        for attr, label in (("B_voigt", "B_V (GPa)"), ("G_voigt", "G_V (GPa)")):
            mv = getattr(mr, attr, None)
            rv = getattr(rr, attr, None) if rr else None
            if mv is None and rv is None:
                continue
            rows.append({"property": "Elastic", "structure": mr.structure_id, "quantity": label, "MTP": _fmt(mv, 2) if mv is not None else "—", "reference": _fmt(rv, 2) if rv is not None else "—", "delta_pct": _pct(mv, rv) if (mv is not None and rv is not None) else "n/a"})
    return rows


def _collect_melting(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.melting_results}
    rows: list[Row] = []
    for mr in mtp.melting_results:
        rr = ref_map.get(mr.structure_id)
        rows.append({"property": "Melting", "structure": mr.structure_id, "quantity": "T_melt (K)", "MTP": _fmt(mr.T_melt, 0) if mr.compute_ok else "—", "reference": _fmt(rr.T_melt, 0) if (rr and rr.compute_ok) else "—", "delta_pct": _pct(mr.T_melt, rr.T_melt) if (mr.compute_ok and rr and rr.compute_ok) else "n/a"})
    return rows


def _collect_thexp(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.thermal_expansion_results}
    rows: list[Row] = []
    for mr in mtp.thermal_expansion_results:
        rr = ref_map.get(mr.structure_id)
        mv = mr.alpha * 1e6 if (mr.compute_ok and math.isfinite(mr.alpha)) else None
        rv = rr.alpha * 1e6 if (rr and rr.compute_ok and math.isfinite(rr.alpha)) else None
        rows.append({"property": "Therm. exp.", "structure": mr.structure_id, "quantity": "α (×10⁻⁶ K⁻¹)", "MTP": _fmt(mv, 2) if mv is not None else "—", "reference": _fmt(rv, 2) if rv is not None else "—", "delta_pct": _pct(mv, rv) if (mv is not None and rv is not None) else "n/a"})
    return rows


def _collect_vacancy(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.vacancy_results}
    rows: list[Row] = []
    for mr in mtp.vacancy_results:
        rr = ref_map.get(mr.structure_id)
        mv = mr.E_vac if (mr.compute_ok and math.isfinite(mr.E_vac)) else None
        rv = rr.E_vac if (rr and rr.compute_ok and math.isfinite(rr.E_vac)) else None
        rows.append({"property": "Vacancy", "structure": mr.structure_id, "quantity": "E_vac (eV)", "MTP": _fmt(mv, 4) if mv is not None else "—", "reference": _fmt(rv, 4) if rv is not None else "—", "delta_pct": _pct(mv, rv) if (mv is not None and rv is not None) else "n/a"})
    return rows


def _collect_rdf(mtp: ValidationResult, ref: "ClassicalReferenceResult") -> list[Row]:
    ref_map = {r.structure_id: r for r in ref.rdf_results}
    rows: list[Row] = []
    for mr in mtp.rdf_results:
        rr = ref_map.get(mr.structure_id)
        mv = mr.first_peak_r if (mr.compute_ok and math.isfinite(mr.first_peak_r)) else None
        rv = rr.first_peak_r if (rr and rr.compute_ok and math.isfinite(rr.first_peak_r)) else None
        rows.append({"property": "RDF", "structure": mr.structure_id, "quantity": "r_1 (Å)", "MTP": _fmt(mv, 3) if mv is not None else "—", "reference": _fmt(rv, 3) if rv is not None else "—", "delta_pct": _pct(mv, rv) if (mv is not None and rv is not None) else "n/a"})
    return rows


# ── Markdown helpers ──────────────────────────────────────────────────────────

_COLS = ["structure", "quantity", "MTP", "reference", "delta_pct"]
_HEADERS = ["Structure", "Quantity", "MTP", "Reference", "Δ%"]


def _md_table(rows: list[Row]) -> list[str]:
    widths = {c: len(h) for c, h in zip(_COLS, _HEADERS)}
    for row in rows:
        for c in _COLS:
            widths[c] = max(widths[c], len(row.get(c, "")))

    def _cell(val: str, w: int) -> str:
        return val.ljust(w)

    sep = "| " + " | ".join("-" * widths[c] for c in _COLS) + " |"
    header = "| " + " | ".join(_cell(h, widths[c]) for c, h in zip(_COLS, _HEADERS)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c, ""), widths[c]) for c in _COLS) + " |")
    return lines


# ── Public API ────────────────────────────────────────────────────────────────

def write_report(
    mtp_result: ValidationResult,
    ref_result: "ClassicalReferenceResult",
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    ref_label = ref_result.label
    slug = _slugify(ref_label)

    collectors = [
        ("EOS", _collect_eos),
        ("Elastic constants", _collect_elastic),
        ("Melting temperature", _collect_melting),
        ("Thermal expansion", _collect_thexp),
        ("Vacancy formation energy", _collect_vacancy),
        ("RDF first peak", _collect_rdf),
    ]

    section_rows: list[tuple[str, list[Row]]] = [(title, fn(mtp_result, ref_result)) for title, fn in collectors]
    all_rows: list[Row] = [{"property": title, **row} for title, rows in section_rows for row in rows]

    model_name = mtp_result.model_path.name if mtp_result.model_path else "MTP"
    md_lines: list[str] = [
        f"# Validation report — {model_name}",
        "",
        f"Reference potential: **{ref_label}**",
        "",
        "> Δ% = (MTP − reference) / |reference| × 100",
        "",
    ]
    for title, rows in section_rows:
        if not rows:
            continue
        md_lines += [f"## {title}", ""]
        md_lines += _md_table(rows)
        md_lines.append("")

    md_name = "validation_report.md" if slug == "classical" else f"validation_report_{slug}.md"
    csv_name = "validation_summary.csv" if slug == "classical" else f"validation_summary_{slug}.csv"

    md_path = out_dir / md_name
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = out_dir / csv_name
    fieldnames = ["property", "structure", "quantity", "MTP", "reference", "delta_pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return md_path, csv_path
