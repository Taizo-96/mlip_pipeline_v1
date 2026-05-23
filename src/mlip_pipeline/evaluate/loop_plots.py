"""loop_plots.py — per-generation record collection and loop-summary plots.

Public API
----------
collect_loop_records(runs_root, generations) -> list[dict]
plot_loop_summary(records, output_dir) -> list[Path]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match the style dict used in evaluate/plots.py
_STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "font.family": "sans-serif",
}

_TEAL   = "#01696f"
_BROWN  = "#964219"
_PURPLE = "#7a39bb"
_GREEN  = "#437a22"
_BLUE   = "#006494"


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON from *path*, returning None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def collect_loop_records(runs_root: Path, generations: list[int]) -> list[dict]:
    """Walk each generation directory and collect a summary record.

    Fields returned per generation
    --------------------------------
    generation            int
    rmse_energy           float  (from eval_manifest.json; NaN if absent)
    rmse_forces           float
    rmse_stress           float
    n_train_cfgs          int    (from fit_manifest.json;  -1 if absent)
    selected_count        int    (from selection_manifest.json; -1 if absent)
    replicate_tier        int    (from state.json; 0 if absent)
    converged_replicate_tier int
    status                str
    completed_at          str
    """
    records: list[dict] = []
    for gen in generations:
        gen_tag = f"gen_{gen:02d}"
        gen_dir = runs_root / gen_tag

        if not gen_dir.exists():
            continue

        # ── state.json ───────────────────────────────────────────────────
        state = _load_json(gen_dir / "state.json") or {}

        # ── eval_manifest.json ───────────────────────────────────────────
        eval_manifest_path = None
        eval_data: dict = {}

        # Try the path recorded in state first, then fall back to default location.
        recorded = state.get("evaluation_manifest")
        if recorded and Path(recorded).exists():
            eval_data = _load_json(Path(recorded)) or {}
            eval_manifest_path = recorded
        else:
            default_eval = gen_dir / "fit" / "eval" / "eval_manifest.json"
            if default_eval.exists():
                eval_data = _load_json(default_eval) or {}
                eval_manifest_path = str(default_eval)

        # ── fit_manifest.json ─────────────────────────────────────────────
        fit_data: dict = {}
        fit_manifest = gen_dir / "fit" / "fit_manifest.json"
        if fit_manifest.exists():
            fit_data = _load_json(fit_manifest) or {}

        # ── selection_manifest.json ───────────────────────────────────────
        sel_data: dict = {}
        sel_manifest = gen_dir / "select" / "selection_manifest.json"
        if sel_manifest.exists():
            sel_data = _load_json(sel_manifest) or {}

        records.append({
            "generation":                gen,
            "rmse_energy":               eval_data.get("rmse_energy", float("nan")),
            "rmse_forces":               eval_data.get("rmse_forces", float("nan")),
            "rmse_stress":               eval_data.get("rmse_stress", float("nan")),
            "n_train_cfgs":              fit_data.get("n_train_cfgs", -1),
            "selected_count":            sel_data.get("selected_count", -1),
            "replicate_tier":            state.get("replicate_tier", 0),
            "converged_replicate_tier":  state.get("converged_replicate_tier", 0),
            "status":                    state.get("status", "unknown"),
            "completed_at":              state.get("completed_at") or "",
        })

    return records


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def _gens(records: list[dict]) -> list[int]:
    return [r["generation"] for r in records]


def _valid(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Return (generations, values) for records where *key* is a finite float."""
    gens, vals = [], []
    for r in records:
        v = r.get(key, float("nan"))
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            gens.append(r["generation"])
            vals.append(float(v))
    return gens, vals


def _plot_rmse(records: list[dict], dest: Path) -> Path:
    """RMSE energy / forces / stress vs generation (log-y, line + scatter)."""
    series = [
        ("rmse_energy", "Energy (eV/atom)", _TEAL),
        ("rmse_forces", "Forces (eV/Å)",   _BROWN),
        ("rmse_stress", "Stress (GPa)",     _PURPLE),
    ]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        any_plotted = False
        for key, label, color in series:
            gens, vals = _valid(records, key)
            if gens:
                ax.plot(gens, vals, "-o", color=color, label=label,
                        linewidth=1.6, markersize=5)
                any_plotted = True
        if any_plotted:
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE")
        ax.set_title("Training RMSE vs Generation")
        ax.set_xticks(_gens(records))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_training_size(records: list[dict], dest: Path) -> Path:
    """Training set size (n_train_cfgs) vs generation."""
    gens, vals = [], []
    for r in records:
        v = r.get("n_train_cfgs", -1)
        if isinstance(v, int) and v >= 0:
            gens.append(r["generation"])
            vals.append(v)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        if gens:
            ax.bar(gens, vals, color=_TEAL, alpha=0.8, width=0.6)
            ax.plot(gens, vals, "-o", color=_TEAL, linewidth=1.4, markersize=5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Training configurations")
        ax.set_title("Training Set Size vs Generation")
        ax.set_xticks(_gens(records))
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_selected(records: list[dict], dest: Path) -> Path:
    """Selected structure count vs generation."""
    gens, vals = [], []
    for r in records:
        v = r.get("selected_count", -1)
        if isinstance(v, int) and v >= 0:
            gens.append(r["generation"])
            vals.append(v)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        if gens:
            # Only the bars are plotted now
            ax.bar(gens, vals, color=_BROWN, alpha=0.8, width=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Selected structures")
        ax.set_title("New Structures Selected vs Generation")
        ax.set_xticks(_gens(records))
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_replicate_tier(records: list[dict], dest: Path) -> Path:
    """Replicate tier (attempted vs converged) vs generation."""
    gens   = _gens(records)
    tiers  = [r.get("replicate_tier", 0) for r in records]
    ctiers = [r.get("converged_replicate_tier", 0) for r in records]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(gens, tiers,  where="mid", color=_PURPLE,  linewidth=1.6, label="attempted tier")
        ax.step(gens, ctiers, where="mid", color=_GREEN,   linewidth=1.6,
                linestyle="--", label="converged tier")
        ax.scatter(gens, tiers,  color=_PURPLE, s=30, zorder=3)
        ax.scatter(gens, ctiers, color=_GREEN,  s=30, zorder=3)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Replicate tier")
        ax.set_title("Replicate Tier vs Generation")
        ax.set_xticks(gens)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_summary_panel(records: list[dict], dest: Path) -> Path:
    """2×2 combined summary panel."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Active Learning Loop Summary", fontsize=13, fontweight="bold")

        ax = axes[0, 0]
        for key, label, color in [
            ("rmse_energy", "Energy (eV/atom)", _TEAL),
            ("rmse_forces", "Forces (eV/Å)",   _BROWN),
            ("rmse_stress", "Stress (GPa)",     _PURPLE),
        ]:
            gg, vv = _valid(records, key)
            if gg:
                ax.plot(gg, vv, "-o", color=color, label=label, linewidth=1.4, markersize=4)
        if any(_valid(records, k)[0] for k in ("rmse_energy", "rmse_forces", "rmse_stress")):
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE vs Generation")
        ax.set_xticks(_gens(records))
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        gens_t = [r["generation"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        vals_t = [r["n_train_cfgs"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        if gens_t:
            ax.bar(gens_t, vals_t, color=_TEAL, alpha=0.8, width=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Training configs")
        ax.set_title("Training Set Size")
        ax.set_xticks(_gens(records))

        ax = axes[1, 0]
        gens_s = [r["generation"] for r in records if r.get("selected_count", -1) >= 0]
        vals_s = [r["selected_count"] for r in records if r.get("selected_count", -1) >= 0]
        if gens_s:
            ax.bar(gens_s, vals_s, color=_BROWN, alpha=0.8, width=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Selected structures")
        ax.set_title("New Structures Selected")
        ax.set_xticks(_gens(records))

        ax = axes[1, 1]
        gens_r  = _gens(records)
        tiers   = [r.get("replicate_tier", 0) for r in records]
        ctiers  = [r.get("converged_replicate_tier", 0) for r in records]
        ax.step(gens_r, tiers,  where="mid", color=_PURPLE, linewidth=1.4, label="attempted")
        ax.step(gens_r, ctiers, where="mid", color=_GREEN,  linewidth=1.4,
                linestyle="--", label="converged")
        ax.scatter(gens_r, tiers,  color=_PURPLE, s=20, zorder=3)
        ax.scatter(gens_r, ctiers, color=_GREEN,  s=20, zorder=3)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Replicate tier")
        ax.set_title("Replicate Tier")
        ax.set_xticks(gens_r)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_loop_summary(records: list[dict], output_dir: Path) -> list[Path]:
    """Produce all loop-summary plots from *records* into *output_dir*.

    Returns
    -------
    list[Path]
        Paths of all PNG files written (always 5 files when records is non-empty).
    """
    if not records:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_rmse(records,           output_dir / "rmse_vs_generation.png"),
        _plot_training_size(records,  output_dir / "training_size_vs_generation.png"),
        _plot_selected(records,       output_dir / "selected_vs_generation.png"),
        _plot_replicate_tier(records, output_dir / "replicate_tier_vs_generation.png"),
        _plot_summary_panel(records,  output_dir / "loop_summary_panel.png"),
    ]
    return paths
