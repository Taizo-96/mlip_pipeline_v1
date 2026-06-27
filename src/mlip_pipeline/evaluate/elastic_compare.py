"""elastic_compare.py -- Compare MTP (LAMMPS) vs DFT (VASP) elastic constants.

Produces:
  - elastic_comparison.json   -- structured comparison with % errors
  - elastic_comparison.png    -- grouped bar chart (MTP vs DFT per constant)
  - printed table to stdout
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Constants reported for cubic symmetry check + derived moduli
_CUBIC_KEYS = ["C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66"]
_DERIVED_KEYS = ["B_voigt", "G_voigt"]
_REPORT_KEYS = _CUBIC_KEYS + _DERIVED_KEYS


def compare_elastic(
    mtp_results: dict,
    dft_results: dict,
    output_dir: str | Path,
    label_mtp: str = "MTP",
    label_dft: str = "DFT",
) -> dict:
    """
    Compare MTP and DFT elastic constants, write JSON + bar plot.

    Parameters
    ----------
    mtp_results : dict returned by elastic_lammps.parse_elastic_from_logs()
    dft_results : dict returned by elastic_vasp.parse_elastic_vasp()
    output_dir  : directory to write elastic_comparison.json and .png
    label_mtp   : label for MTP column
    label_dft   : label for DFT column

    Returns
    -------
    comparison dict with per-key {mtp, dft, abs_err, rel_err_pct}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, dict] = {}
    for key in _REPORT_KEYS:
        mtp_val = mtp_results.get(key)
        dft_val = dft_results.get(key)
        if mtp_val is None or dft_val is None:
            continue
        abs_err = float(mtp_val) - float(dft_val)
        rel_err = abs_err / float(dft_val) * 100 if abs(float(dft_val)) > 1e-6 else float("nan")
        comparison[key] = {
            label_mtp: round(float(mtp_val), 2),
            label_dft: round(float(dft_val), 2),
            "abs_err_GPa": round(abs_err, 2),
            "rel_err_pct": round(rel_err, 2),
        }

    # Cubic symmetry check (should all be ~equal for BCC)
    _add_cubic_check(comparison, mtp_results, dft_results, label_mtp, label_dft)

    # Write JSON
    json_path = output_dir / "elastic_comparison.json"
    with json_path.open("w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Elastic comparison written to %s", json_path)

    # Print table
    _print_table(comparison, label_mtp, label_dft)

    # Bar chart
    try:
        plot_path = _plot_comparison(comparison, output_dir, label_mtp, label_dft)
        logger.info("Elastic comparison plot saved to %s", plot_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not generate elastic comparison plot: %s", exc)
        plot_path = None

    return {"comparison": comparison, "json_path": json_path, "plot_path": plot_path}


def _add_cubic_check(comparison: dict, mtp: dict, dft: dict,
                     label_mtp: str, label_dft: str) -> None:
    """Append cubic symmetry deviation metrics."""
    for label, src in [(label_mtp, mtp), (label_dft, dft)]:
        c11 = src.get("C11"); c22 = src.get("C22"); c33 = src.get("C33")
        c12 = src.get("C12"); c13 = src.get("C13"); c23 = src.get("C23")
        c44 = src.get("C44"); c55 = src.get("C55"); c66 = src.get("C66")
        if None in (c11, c22, c33, c12, c13, c23, c44, c55, c66):
            continue
        diag_spread = float(np.std([c11, c22, c33]))
        off_spread  = float(np.std([c12, c13, c23]))
        shear_spread = float(np.std([c44, c55, c66]))
        comparison.setdefault("_cubic_symmetry_check", {})[label] = {
            "C11/C22/C33_std_GPa": round(diag_spread, 3),
            "C12/C13/C23_std_GPa": round(off_spread, 3),
            "C44/C55/C66_std_GPa": round(shear_spread, 3),
        }


def _print_table(comparison: dict, label_mtp: str, label_dft: str) -> None:
    rows = {k: v for k, v in comparison.items() if not k.startswith("_")}
    if not rows:
        return
    print("\n" + "=" * 66)
    print(f"  ELASTIC CONSTANTS: {label_mtp} vs {label_dft} (GPa)")
    print("=" * 66)
    print(f"  {'Const':8s}  {label_mtp:>10s}  {label_dft:>10s}  {'|err|':>8s}  {'err%':>7s}")
    print("-" * 66)
    for key, v in rows.items():
        print(
            f"  {key:8s}  {v[label_mtp]:>10.2f}  {v[label_dft]:>10.2f}  "
            f"{v['abs_err_GPa']:>8.2f}  {v['rel_err_pct']:>6.1f}%"
        )
    print("=" * 66 + "\n")


def _plot_comparison(
    comparison: dict,
    output_dir: Path,
    label_mtp: str,
    label_dft: str,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {k: v for k, v in comparison.items() if not k.startswith("_")}
    keys = list(rows.keys())
    mtp_vals = [rows[k][label_mtp] for k in keys]
    dft_vals = [rows[k][label_dft] for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.9), 5))
    bars_mtp = ax.bar(x - width / 2, mtp_vals, width, label=label_mtp, color="#2a6a9e")
    bars_dft = ax.bar(x + width / 2, dft_vals, width, label=label_dft, color="#c0392b",
                      alpha=0.85)
    ax.set_xlabel("Elastic constant")
    ax.set_ylabel("Value (GPa)")
    ax.set_title(f"Elastic constants: {label_mtp} vs {label_dft}")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = output_dir / "elastic_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience loader for runner.py
# ---------------------------------------------------------------------------

def run_elastic_validation(
    config: dict,
    resolved_paths: dict,
    fit_result,
    eval_dir: Path,
) -> dict | None:
    """
    Run LAMMPS elastic constant calculation and compare to VASP DFT reference.

    Called from runner.run_evaluation() after parity/gamma steps.

    Config keys consumed (all under config["evaluate"]["elastic"]):
        lammps_cmd    : str   -- LAMMPS executable (default: "lmp")
        potential_file: str   -- path to .mtp potential (resolved relative to project_root)
        poscar        : str   -- path to relaxed POSCAR
        vasp_outcar   : str   -- path to VASP IBRION=6 OUTCAR (optional; skip DFT compare if absent)
        delta         : float -- strain magnitude (default: 0.005)
        reuse_existing: bool  -- skip re-running LAMMPS if logs exist (default: False)

    Returns None if elastic block is absent from config or disabled.
    """
    elastic_cfg = config.get("evaluate", {}).get("elastic")
    if not elastic_cfg:
        logger.debug("No evaluate.elastic config block found -- skipping elastic validation")
        return None

    if not elastic_cfg.get("enabled", True):
        logger.info("Elastic validation disabled in config")
        return None

    from mlip_pipeline.evaluate.elastic_lammps import compute_elastic_lammps
    from mlip_pipeline.evaluate.elastic_vasp import parse_elastic_vasp
    from mlip_pipeline.evaluate.elastic_compare import compare_elastic

    project_root = resolved_paths["project_root"]

    lammps_cmd = elastic_cfg.get("lammps_cmd", "lmp")
    delta      = float(elastic_cfg.get("delta", 0.005))
    reuse      = bool(elastic_cfg.get("reuse_existing", False))

    # Resolve potential file
    pot_file = elastic_cfg.get("potential_file")
    if pot_file is None:
        # Fall back to the fitted model path
        pot_file = fit_result.resolve_model_path()
    else:
        pot_file = (project_root / pot_file).resolve()

    poscar = elastic_cfg.get("poscar", "POSCAR")
    poscar = (project_root / poscar).resolve()

    elastic_dir = eval_dir / "elastic"

    print("\n  [elastic] Running LAMMPS finite-difference elastic constants...")
    try:
        mtp_results = compute_elastic_lammps(
            lammps_cmd=lammps_cmd,
            potential_file=pot_file,
            poscar=poscar,
            output_dir=elastic_dir / "lammps_runs",
            delta=delta,
            reuse_existing=reuse,
        )
        print(f"  [elastic] MTP: C11={mtp_results['C11']:.1f}  C12={mtp_results['C12']:.1f}  "
              f"C44={mtp_results['C44']:.1f}  B={mtp_results['B_voigt']:.1f} GPa")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LAMMPS elastic calculation failed: %s", exc)
        return None

    # DFT reference (optional)
    vasp_outcar = elastic_cfg.get("vasp_outcar")
    if vasp_outcar:
        vasp_outcar = (project_root / vasp_outcar).resolve()
        if vasp_outcar.exists():
            print("  [elastic] Parsing VASP IBRION=6 OUTCAR...")
            try:
                dft_results = parse_elastic_vasp(vasp_outcar)
                print(f"  [elastic] DFT: C11={dft_results['C11']:.1f}  C12={dft_results['C12']:.1f}  "
                      f"C44={dft_results['C44']:.1f}  B={dft_results['B_voigt']:.1f} GPa")
                return compare_elastic(
                    mtp_results, dft_results,
                    output_dir=elastic_dir,
                    label_mtp="MTP",
                    label_dft="DFT",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("VASP OUTCAR parsing failed: %s", exc)
        else:
            logger.warning("vasp_outcar path not found: %s", vasp_outcar)

    # No DFT reference -- just save MTP results
    import json
    out = elastic_dir / "elastic_mtp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: (v.tolist() if hasattr(v, 'tolist') else v)
              for k, v in mtp_results.items() if k != "raw_pressures"}
    out.write_text(json.dumps(serial, indent=2))
    print(f"  [elastic] MTP results saved to {out} (no DFT reference provided)")
    return {"mtp_results": mtp_results, "json_path": out, "plot_path": None}
