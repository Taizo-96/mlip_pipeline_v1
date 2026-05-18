from __future__ import annotations

import csv
import re
from pathlib import Path

# MLIP-3 writes a final summary block at the end of train.log, e.g.:
#
#   Energy (eV/atom):
#           RMS     absolute difference = 0.00312
#   Forces (eV/A):
#           RMS     absolute difference = 0.04521
#   Virial stresses (in pressure units):
#           RMS     absolute difference = 12.2741
#
# There are no per-iteration RMSE lines in this binary version.

_SECTION_RE = re.compile(
    r"(Energy|Forces|Virial stresses)[^\n]*\n"
    r"(?:[^\n]*\n){0,6}?"
    r"[^\n]*RMS\s+absolute difference\s*=\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)


def parse_train_log(log_path: Path) -> dict:
    """
    Parse the final summary block from MLIP-3 train.log.
    Returns dict with keys: rmse_e, rmse_f, rmse_s (None if absent).
    Returns empty dict if nothing is found.
    """
    text = log_path.read_text(encoding="utf-8")

    # Take only the last occurrence of each section (final summary)
    results: dict[str, float] = {}
    for m in _SECTION_RE.finditer(text):
        section = m.group(1).lower()
        value = float(m.group(2))
        if "energy" in section:
            results["rmse_e"] = value
        elif "force" in section:
            results["rmse_f"] = value
        elif "stress" in section or "virial" in section:
            results["rmse_s"] = value

    return results


def write_metrics_csv(metrics: dict, dest: Path) -> Path:
    with open(dest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        label_map = {
            "rmse_e": "final_rmse_energy_eV_per_atom",
            "rmse_f": "final_rmse_forces_eV_per_ang",
            "rmse_s": "final_rmse_stress_pressure_units",
        }
        for key, label in label_map.items():
            if key in metrics:
                w.writerow([label, metrics[key]])
    return dest
