"""Plotting helpers for the validate step."""
from __future__ import annotations

import math
from pathlib import Path

from mlip_pipeline.validate.models import EosResult, ElasticResult


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for validation plots") from exc


def plot_eos(result: EosResult, out_dir: Path) -> Path | None:
    """Plot E vs V with BM fit curve.  Returns the PNG path or None."""
    if not result.volumes or not result.energies:
        return None

    plt = _require_matplotlib()
    try:
        import numpy as np
        from mlip_pipeline.validate.eos import _birch_murnaghan
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    vs = result.volumes
    es = result.energies

    ax.scatter(vs, es, s=30, color="#2c7bb6", zorder=3, label="MTP")

    if result.fit_ok:
        v_fit = np.linspace(min(vs), max(vs), 200)
        # Convert B0 from GPa to eV/Å³
        B0_ev = result.B0 / 160.2176634
        e_fit = [_birch_murnaghan(v, result.V0, result.E0, B0_ev, result.B0p)
                 for v in v_fit]
        ax.plot(v_fit, e_fit, color="#d7191c", lw=1.5,
                label=f"BM fit  B₀={result.B0:.1f} GPa")
        ax.axvline(result.V0, color="grey", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel("Volume per atom (Å³)")
    ax.set_ylabel("Energy per atom (eV)")
    ax.set_title(f"EOS — {result.structure_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = out_dir / f"eos_{result.structure_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_elastic_bar(results: list[ElasticResult], out_dir: Path) -> Path | None:
    """Bar chart comparing Cij across structures.  Returns the PNG path or None."""
    valid = [r for r in results if r.compute_ok and r.C]
    if not valid:
        return None

    plt = _require_matplotlib()
    import numpy as np

    # Collect all unique Cij labels
    keys = sorted({k for r in valid for k in r.C})
    x = np.arange(len(keys))
    width = 0.8 / len(valid)

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 4))
    for i, r in enumerate(valid):
        vals = [r.C.get(k, float("nan")) for k in keys]
        ax.bar(x + i * width, vals, width, label=r.structure_id)

    ax.set_xticks(x + width * (len(valid) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Elastic constant (GPa)")
    ax.set_title("Elastic constants")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "elastic_constants.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
