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

        fit_result   = _load_fit(gen_dir)
        eval_result  = _load_eval(gen_dir, state.get("evaluation_manifest"))
        sel_result   = _load_selection(gen_dir)
        conv_result  = _load_convert(gen_dir)

        n_train_cfgs: int = -1
        if fit_result is not None and fit_result.n_train_cfgs > 0:
            n_train_cfgs = fit_result.n_train_cfgs
        elif conv_result is not None and conv_result.n_total_cfgs > 0:
            n_train_cfgs = conv_result.n_total_cfgs
        else:
            n_train_cfgs = _count_train_cfgs_from_disk(fit_result)

        records.append({
            "generation":                gen,
            "rmse_energy":              eval_result.rmse_energy if eval_result else float("nan"),
            "rmse_forces":              eval_result.rmse_forces if eval_result else float("nan"),
            "rmse_stress":              eval_result.rmse_stress if eval_result else float("nan"),
            "mean_gamma":               eval_result.mean_gamma  if eval_result else float("nan"),
            "max_gamma":                eval_result.max_gamma   if eval_result else float("nan"),
            "frac_above_save":          eval_result.frac_above_save  if eval_result else float("nan"),
            "frac_above_break":         eval_result.frac_above_break if eval_result else float("nan"),
            "n_train_cfgs":             n_train_cfgs,
            "selected_count":           sel_result.selected_count if sel_result else -1,
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
    """Training set growth: cumulative line (left y) + per-gen delta bars (right y).

    When the total dataset is ~10 K configs and only changes by tens per gen,
    a plain bar chart gives the illusion of a flat line.  This dual-axis view
    makes *both* the absolute scale and the incremental additions readable at
    the same time.
    """
    gens, vals = [], []
    for r in records:
        v = r.get("n_train_cfgs", -1)
        if isinstance(v, int) and v >= 0:
            gens.append(r["generation"])
            vals.append(v)

    if not gens:
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title("Training Set Growth vs Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Training configurations")
            fig.tight_layout()
            fig.savefig(dest)
            plt.close(fig)
        return dest

    # per-generation deltas (first gen delta = its full size)
    deltas = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    with plt.rc_context(_STYLE):
        fig, ax1 = plt.subplots(figsize=(7, 4))

        # right axis — deltas as bars (drawn first so line sits on top)
        ax2 = ax1.twinx()
        bar_colors = [_GREEN if d >= 0 else _RED for d in deltas]
        ax2.bar(gens, deltas, color=bar_colors, alpha=0.4, width=0.55,
                label="Added configs (Δ)")
        ax2.set_ylabel("Configs added per generation  (Δ)", color=_GREEN, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=_GREEN)
        ax2.spines["top"].set_visible(False)
        # keep delta axis from being dwarfed by the cumulative scale
        if max(abs(d) for d in deltas) > 0:
            delta_range = max(abs(d) for d in deltas)
            ax2.set_ylim(-delta_range * 0.5, delta_range * 3.5)

        # left axis — cumulative total as a line
        ax1.plot(gens, vals, "-o", color=_TEAL, linewidth=2.0, markersize=6,
                 label="Cumulative total", zorder=3)
        # tight y-range so small movements are visible
        margin = max((max(vals) - min(vals)) * 0.5, 1)
        ax1.set_ylim(min(vals) - margin, max(vals) + margin)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Total training configurations", color=_TEAL)
        ax1.tick_params(axis="y", labelcolor=_TEAL)
        ax1.set_title("Training Set Growth vs Generation")
        ax1.set_xticks(gens)

        # unified legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9)

        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_rmse_pct_change(records: list[dict], dest: Path) -> Path:
    """Percentage change in energy and forces RMSE, generation-over-generation.

    Negative = improvement (RMSE fell).  Horizontal dashed line at 0.
    Makes convergence trend immediately visible without caring about units.
    """
    e_gens, e_vals = _valid(records, "rmse_energy")
    f_gens, f_vals = _valid(records, "rmse_forces")

    def _pct_change(gens: list[int], vals: list[float]):
        if len(vals) < 2:
            return [], []
        pct = [(vals[i] - vals[i - 1]) / abs(vals[i - 1]) * 100
               for i in range(1, len(vals))]
        return gens[1:], pct

    e_pg, e_pv = _pct_change(e_gens, e_vals)
    f_pg, f_pv = _pct_change(f_gens, f_vals)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.axhline(0, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=1)

        if e_pg:
            colors_e = [_GREEN if v < 0 else _RED for v in e_pv]
            ax.bar(
                [g - 0.18 for g in e_pg], e_pv,
                width=0.34, color=colors_e, alpha=0.75,
                label="Energy RMSE Δ%",
            )
            ax.plot(e_pg, e_pv, "-o", color=_TEAL, linewidth=1.4,
                    markersize=5, zorder=3)

        if f_pg:
            colors_f = [_GREEN if v < 0 else _RED for v in f_pv]
            ax.bar(
                [g + 0.18 for g in f_pg], f_pv,
                width=0.34, color=colors_f, alpha=0.45,
                label="Forces RMSE Δ%",
            )
            ax.plot(f_pg, f_pv, "--s", color=_BROWN, linewidth=1.4,
                    markersize=5, zorder=3)

        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE change  (%)")
        ax.set_title("RMSE % Change vs Previous Generation")
        all_gens = sorted(set(e_pg + f_pg))
        if all_gens:
            ax.set_xticks(all_gens)
        _mark_failed(ax, records)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(dest)
        plt.close(fig)
    return dest


def _plot_selected(records: list[dict], dest: Path) -> Path:
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

    Top-left:  energy + forces RMSE on dual y-axes.
    Top-right: training set growth (cumulative + delta).
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

        # ── top-right: training set growth (cumulative + delta) ──────────────
        ax_t = axes[0, 1]
        gens_t = [r["generation"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        vals_t = [r["n_train_cfgs"] for r in records if r.get("n_train_cfgs", -1) >= 0]
        if gens_t:
            deltas_t = [vals_t[0]] + [vals_t[i] - vals_t[i-1] for i in range(1, len(vals_t))]
            ax_t2 = ax_t.twinx()
            bar_colors = [_GREEN if d >= 0 else _RED for d in deltas_t]
            ax_t2.bar(gens_t, deltas_t, color=bar_colors, alpha=0.35, width=0.55)
            ax_t2.set_ylabel("Configs added (Δ)", color=_GREEN, fontsize=9)
            ax_t2.tick_params(axis="y", labelcolor=_GREEN)
            ax_t2.spines["top"].set_visible(False)
            if max(abs(d) for d in deltas_t) > 0:
                dr = max(abs(d) for d in deltas_t)
                ax_t2.set_ylim(-dr * 0.5, dr * 3.5)
            margin_t = max((max(vals_t) - min(vals_t)) * 0.5, 1)
            ax_t.plot(gens_t, vals_t, "-o", color=_TEAL, linewidth=1.6,
                      markersize=5, zorder=3)
            ax_t.set_ylim(min(vals_t) - margin_t, max(vals_t) + margin_t)
        ax_t.set_xlabel("Generation")
        ax_t.set_ylabel("Total configs", color=_TEAL)
        ax_t.tick_params(axis="y", labelcolor=_TEAL)
        ax_t.set_title("Training Set Growth")
        ax_t.set_xticks(_gens(records))

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
      rmse_energy_vs_generation.png
      rmse_forces_vs_generation.png
      rmse_stress_vs_generation.png
      rmse_combined_vs_generation.png
      rmse_pct_change_vs_generation.png   ← NEW
      training_size_vs_generation.png     (now: cumulative line + delta bars)
      selected_vs_generation.png
      replicate_tier_vs_generation.png
      gamma_vs_generation.png
      loop_summary_panel.png
    """
    if not records:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_rmse_energy(records,       output_dir / "rmse_energy_vs_generation.png"),
        _plot_rmse_forces(records,       output_dir / "rmse_forces_vs_generation.png"),
        _plot_rmse_stress(records,       output_dir / "rmse_stress_vs_generation.png"),
        _plot_rmse_combined(records,     output_dir / "rmse_combined_vs_generation.png"),
        _plot_rmse_pct_change(records,   output_dir / "rmse_pct_change_vs_generation.png"),
        _plot_training_size(records,     output_dir / "training_size_vs_generation.png"),
        _plot_selected(records,          output_dir / "selected_vs_generation.png"),
        _plot_replicate_tier(records,    output_dir / "replicate_tier_vs_generation.png"),
        _plot_gamma_evolution(records,   output_dir / "gamma_vs_generation.png"),
        _plot_summary_panel(records,     output_dir / "loop_summary_panel.png"),
    ]
    return paths
