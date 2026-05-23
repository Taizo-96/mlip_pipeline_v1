"""loop_plots.py — per-generation record collection and loop-summary plots.

Public API
----------
collect_loop_records(runs_root, generations) -> list[dict]
plot_loop_summary(records, output_dir) -> list[Path]

Data sourcing strategy
----------------------
Every field in a record is read from its canonical typed manifest class
(``FitResult.load_manifest``, ``EvaluationResult.load_manifest``,
``SelectionResult.load_manifest``, ``ConvertResult.load_manifest``).
This keeps ``loop_plots.py`` decoupled from the raw JSON schema — adding a
field to a manifest class is all that is needed to make it available here.

Fallback chain for ``n_train_cfgs`` (the field most likely to be missing in
older runs):
  1. ``fit_manifest.json``   → ``FitResult.n_train_cfgs``
  2. ``convert_manifest.json`` → ``ConvertResult.n_total_cfgs``  (most
     reliable: written after accumulation, reflects every gen)
  3. Live ``BEGIN_CFG`` line-count from ``train.cfg`` on disk  (slow but
     accurate; only executed when both manifests are absent)
  4. ``-1``  (unknown — bar is omitted from the training-size plot)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
_RED    = "#c0392b"


# ---------------------------------------------------------------------------
# Manifest helpers — typed, not raw-dict
# ---------------------------------------------------------------------------

def _load_fit(gen_dir: Path) -> Optional[object]:
    from mlip_pipeline.models import FitResult
    fit_dir = gen_dir / "fit"
    try:
        return FitResult.load_manifest(fit_dir)
    except Exception:
        return None


def _load_eval(gen_dir: Path, state_eval_manifest: Optional[str]) -> Optional[object]:
    """Load EvaluationResult, preferring the path recorded in state.json."""
    from mlip_pipeline.models import EvaluationResult
    candidates: list[Path] = []
    if state_eval_manifest:
        p = Path(state_eval_manifest)
        if p.exists():
            candidates.append(p.parent)   # eval_dir
    candidates.append(gen_dir / "fit" / "eval")
    for eval_dir in candidates:
        try:
            return EvaluationResult.load_manifest(eval_dir)
        except Exception:
            continue
    return None


def _load_selection(gen_dir: Path) -> Optional[object]:
    from mlip_pipeline.models import SelectionResult
    select_dir = gen_dir / "select"
    try:
        return SelectionResult.load_manifest(select_dir)
    except Exception:
        return None


def _load_convert(gen_dir: Path) -> Optional[object]:
    from mlip_pipeline.models import ConvertResult
    # convert artefacts live under gen_dir directly (not a subdir)
    for candidate in [gen_dir / "convert", gen_dir]:
        try:
            return ConvertResult.load_manifest(candidate)
        except Exception:
            continue
    return None


def _count_train_cfgs_from_disk(fit_result) -> int:
    """Last-resort fallback: count BEGIN_CFG lines from the train.cfg on disk."""
    if fit_result is None or fit_result.train_cfg is None:
        return -1
    cfg_path = Path(fit_result.train_cfg)
    if not cfg_path.exists():
        return -1
    try:
        return sum(1 for line in cfg_path.open(encoding="utf-8") if line.strip() == "BEGIN_CFG")
    except Exception:
        return -1


def _load_state(gen_dir: Path) -> dict:
    import json
    try:
        return json.loads((gen_dir / "state.json").read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Record collection
# ---------------------------------------------------------------------------

def collect_loop_records(runs_root: Path, generations: list[int]) -> list[dict]:
    """Walk each generation directory and collect a summary record.

    Fields returned per generation
    --------------------------------
    generation                int
    rmse_energy               float  (NaN if absent)
    rmse_forces               float
    rmse_stress               float
    mean_gamma                float
    max_gamma                 float
    frac_above_save           float
    frac_above_break          float
    n_train_cfgs              int    (-1 if unknown after all fallbacks)
    selected_count            int    (-1 if absent)
    replicate_tier            int
    converged_replicate_tier  int
    status                    str
    completed_at              str
    """
    records: list[dict] = []
    for gen in generations:
        gen_tag = f"gen_{gen:02d}"
        gen_dir = runs_root / gen_tag
        if not gen_dir.exists():
            continue

        state = _load_state(gen_dir)

        # ── typed manifest loads ──────────────────────────────────────────────
        fit_result   = _load_fit(gen_dir)
        eval_result  = _load_eval(gen_dir, state.get("evaluation_manifest"))
        sel_result   = _load_selection(gen_dir)
        conv_result  = _load_convert(gen_dir)

        # ── n_train_cfgs: 4-level fallback chain ──────────────────────────────
        n_train_cfgs: int = -1
        if fit_result is not None and fit_result.n_train_cfgs > 0:
            n_train_cfgs = fit_result.n_train_cfgs                  # (1) fit manifest
        elif conv_result is not None and conv_result.n_total_cfgs > 0:
            n_train_cfgs = conv_result.n_total_cfgs                 # (2) convert manifest
        else:
            n_train_cfgs = _count_train_cfgs_from_disk(fit_result)  # (3) live disk count
        # (4) remains -1 — bar omitted from plot

        records.append({
            "generation":                gen,
            # ─ eval ─
            "rmse_energy":              eval_result.rmse_energy if eval_result else float("nan"),
            "rmse_forces":              eval_result.rmse_forces if eval_result else float("nan"),
            "rmse_stress":              eval_result.rmse_stress if eval_result else float("nan"),
            "mean_gamma":               eval_result.mean_gamma  if eval_result else float("nan"),
            "max_gamma":                eval_result.max_gamma   if eval_result else float("nan"),
            "frac_above_save":          eval_result.frac_above_save  if eval_result else float("nan"),
            "frac_above_break":         eval_result.frac_above_break if eval_result else float("nan"),
            # ─ fit / convert ─
            "n_train_cfgs":             n_train_cfgs,
            # ─ selection ─
            "selected_count":           sel_result.selected_count if sel_result else -1,
            # ─ state ─
            "replicate_tier":           state.get("replicate_tier", 0),
            "converged_replicate_tier": state.get("converged_replicate_tier", 0),
            "status":                   state.get("status", "unknown"),
            "completed_at":             state.get("completed_at") or "",
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


def _mark_failed(ax: plt.Axes, records: list[dict]) -> None:
    """Add a red × marker at the x-axis for any failed generation."""
    failed_gens = [r["generation"] for r in records if r.get("status") == "failed"]
    if failed_gens:
        ylim = ax.get_ylim()
        y_mark = ylim[0]
        ax.scatter(
            failed_gens,
            [y_mark] * len(failed_gens),
            marker="x",
            color=_RED,
            s=60,
            linewidths=2,
            zorder=5,
            label="failed gen",
        )


def _plot_rmse_combined(records: list[dict], dest: Path) -> Path:
    """
    Energy + forces on dual y-axes (left/right).  Stress is NOT shown here;
    see ``_plot_rmse_stress`` for the dedicated stress plot.

    Both y-axes use log scale so the vastly different magnitudes
    (energy ~meV/atom, forces ~10× larger) are each legible on their own
    scale without one series dominating the other.
    """
    e_gens, e_vals = _valid(records, "rmse_energy")
    f_gens, f_vals = _valid(records, "rmse_forces")

    with plt.rc_context(_STYLE):
        fig, ax1 = plt.subplots(figsize=(7, 4))

        if e_gens:
            ax1.plot(e_gens, e_vals, "-o", color=_TEAL, linewidth=1.6,
                     markersize=5, label="Energy RMSE (eV/atom)")
            ax1.set_yscale("log")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Energy RMSE  (eV/atom)", color=_TEAL)
        ax1.tick_params(axis="y", labelcolor=_TEAL)
        ax1.set_xticks(_gens(records))
        _mark_failed(ax1, records)

        if f_gens:
            ax2 = ax1.twinx()
            ax2.plot(f_gens, f_vals, "--s", color=_BROWN, linewidth=1.6,
                     markersize=5, label="Forces RMSE (eV/Å)")
            ax2.set_yscale("log")
            ax2.set_ylabel("Forces RMSE  (eV/Å)", color=_BROWN)
            ax2.tick_params(axis="y", labelcolor=_BROWN)
            ax2.spines["top"].set_visible(False)
            # Unified legend from both axes
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=9)
        else:
            ax1.legend(fontsize=9)

        ax1.set_title("Training RMSE vs Generation  (energy │ forces)")
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_rmse_energy(records: list[dict], dest: Path) -> Path:
    """Dedicated energy-RMSE plot on its own y-scale."""
    gens, vals = _valid(records, "rmse_energy")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        if gens:
            ax.plot(gens, vals, "-o", color=_TEAL, linewidth=1.6, markersize=5)
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE  (eV/atom)")
        ax.set_title("Energy RMSE vs Generation")
        ax.set_xticks(_gens(records))
        _mark_failed(ax, records)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_rmse_forces(records: list[dict], dest: Path) -> Path:
    """Dedicated forces-RMSE plot on its own y-scale."""
    gens, vals = _valid(records, "rmse_forces")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        if gens:
            ax.plot(gens, vals, "-o", color=_BROWN, linewidth=1.6, markersize=5)
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE  (eV/Å)")
        ax.set_title("Forces RMSE vs Generation")
        ax.set_xticks(_gens(records))
        _mark_failed(ax, records)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_rmse_stress(records: list[dict], dest: Path) -> Path:
    """Dedicated stress-RMSE plot on its own y-scale."""
    gens, vals = _valid(records, "rmse_stress")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        if gens:
            ax.plot(gens, vals, "-o", color=_PURPLE, linewidth=1.6, markersize=5)
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE  (GPa)")
        ax.set_title("Stress RMSE vs Generation")
        ax.set_xticks(_gens(records))
        _mark_failed(ax, records)
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


def _plot_gamma_evolution(records: list[dict], dest: Path) -> Path:
    """Mean and max extrapolation grade (gamma) vs generation."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))

        mean_gens, mean_vals = _valid(records, "mean_gamma")
        max_gens,  max_vals  = _valid(records, "max_gamma")

        if mean_gens:
            ax.plot(mean_gens, mean_vals, "-o", color=_TEAL,  linewidth=1.6,
                    markersize=5, label="mean \u03b3")
        if max_gens:
            ax.plot(max_gens, max_vals, "--s", color=_BROWN, linewidth=1.6,
                    markersize=5, label="max \u03b3")

        has_save  = any(math.isfinite(r.get("frac_above_save",  float("nan"))) for r in records)

        ax2 = None
        if has_save:
            fs_gens, fs_vals = _valid(records, "frac_above_save")
            if fs_gens:
                ax2 = ax.twinx()
                ax2.bar(fs_gens, fs_vals, color=_GREEN, alpha=0.25, width=0.5,
                        label="frac > \u03b3_save")
                ax2.set_ylabel("Fraction above save threshold", fontsize=9)
                ax2.set_ylim(0, max(fs_vals) * 2 + 0.01)
                ax2.spines["top"].set_visible(False)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Extrapolation grade \u03b3")
        ax.set_title("Gamma Evolution vs Generation")
        ax.set_xticks(_gens(records))
        _mark_failed(ax, records)

        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels  += l2
        if handles:
            ax.legend(handles, labels, fontsize=9)

        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_summary_panel(records: list[dict], dest: Path) -> Path:
    """
    2×2 combined summary panel.

    Top-left:  energy + forces RMSE on dual y-axes (stress excluded).
    Top-right: training set size.
    Bottom-left: selected structures.
    Bottom-right: gamma evolution.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Active Learning Loop Summary", fontsize=13, fontweight="bold")

        # ── top-left: energy + forces, dual y-axes ──────────────────────────
        ax1 = axes[0, 0]
        e_gens, e_vals = _valid(records, "rmse_energy")
        f_gens, f_vals = _valid(records, "rmse_forces")

        if e_gens:
            ax1.plot(e_gens, e_vals, "-o", color=_TEAL, linewidth=1.4,
                     markersize=4, label="Energy (eV/atom)")
            ax1.set_yscale("log")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Energy RMSE (eV/atom)", color=_TEAL)
        ax1.tick_params(axis="y", labelcolor=_TEAL)
        ax1.set_title("RMSE: Energy | Forces")
        ax1.set_xticks(_gens(records))
        _mark_failed(ax1, records)

        if f_gens:
            ax1r = ax1.twinx()
            ax1r.plot(f_gens, f_vals, "--s", color=_BROWN, linewidth=1.4,
                      markersize=4, label="Forces (eV/Å)")
            ax1r.set_yscale("log")
            ax1r.set_ylabel("Forces RMSE (eV/Å)", color=_BROWN)
            ax1r.tick_params(axis="y", labelcolor=_BROWN)
            ax1r.spines["top"].set_visible(False)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax1r.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=8)
        else:
            ax1.legend(fontsize=8)

        # ── top-right: training set size ──────────────────────────────────
        ax = axes[0, 1]
        gens_t = [r["generation"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        vals_t = [r["n_train_cfgs"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        if gens_t:
            ax.bar(gens_t, vals_t, color=_TEAL, alpha=0.8, width=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Training configs")
        ax.set_title("Training Set Size")
        ax.set_xticks(_gens(records))

        # ── bottom-left: selected structures ──────────────────────────────
        ax = axes[1, 0]
        gens_s = [r["generation"] for r in records if r.get("selected_count", -1) >= 0]
        vals_s = [r["selected_count"] for r in records if r.get("selected_count", -1) >= 0]
        if gens_s:
            ax.bar(gens_s, vals_s, color=_BROWN, alpha=0.8, width=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Selected structures")
        ax.set_title("New Structures Selected")
        ax.set_xticks(_gens(records))

        # ── bottom-right: gamma evolution ─────────────────────────────────
        ax = axes[1, 1]
        mean_gg, mean_vv = _valid(records, "mean_gamma")
        max_gg,  max_vv  = _valid(records, "max_gamma")
        if mean_gg:
            ax.plot(mean_gg, mean_vv, "-o", color=_TEAL,  linewidth=1.4,
                    markersize=4, label="mean \u03b3")
        if max_gg:
            ax.plot(max_gg, max_vv, "--s", color=_BROWN, linewidth=1.4,
                    markersize=4, label="max \u03b3")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Extrapolation grade \u03b3")
        ax.set_title("Gamma Evolution")
        ax.set_xticks(_gens(records))
        _mark_failed(ax, records)
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

    Returns a list of Paths of all PNG files written.
    Files produced:
      rmse_energy_vs_generation.png     — energy only, own scale
      rmse_forces_vs_generation.png     — forces only, own scale
      rmse_stress_vs_generation.png     — stress only, own scale
      rmse_combined_vs_generation.png   — energy + forces, dual y-axes (no stress)
      training_size_vs_generation.png
      selected_vs_generation.png
      replicate_tier_vs_generation.png
      gamma_vs_generation.png
      loop_summary_panel.png            — 2×2 panel
    """
    if not records:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_rmse_energy(records,    output_dir / "rmse_energy_vs_generation.png"),
        _plot_rmse_forces(records,    output_dir / "rmse_forces_vs_generation.png"),
        _plot_rmse_stress(records,    output_dir / "rmse_stress_vs_generation.png"),
        _plot_rmse_combined(records,  output_dir / "rmse_combined_vs_generation.png"),
        _plot_training_size(records,  output_dir / "training_size_vs_generation.png"),
        _plot_selected(records,       output_dir / "selected_vs_generation.png"),
        _plot_replicate_tier(records, output_dir / "replicate_tier_vs_generation.png"),
        _plot_gamma_evolution(records, output_dir / "gamma_vs_generation.png"),
        _plot_summary_panel(records,  output_dir / "loop_summary_panel.png"),
    ]
    return paths
