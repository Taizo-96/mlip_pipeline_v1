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

# Canonical short keys used everywhere in the codebase.
# These are the keys stored in metrics.csv and read back by replot_evaluation.
_KEY_MAP = {
    "energy":  "rmse_e",
    "force":   "rmse_f",
    "stress":  "rmse_s",
    "virial":  "rmse_s",
}


def parse_train_log(log_path: Path) -> dict:
    """
    Parse the final summary block from MLIP-3 train.log.
    Returns dict with keys: rmse_e, rmse_f, rmse_s (None if absent).
    Returns empty dict if nothing is found.
    """
    text = log_path.read_text(encoding="utf-8")

    results: dict[str, float] = {}
    for m in _SECTION_RE.finditer(text):
        section = m.group(1).lower()
        value   = float(m.group(2))
        for fragment, key in _KEY_MAP.items():
            if fragment in section:
                results[key] = value
                break

    return results


def write_metrics_csv(metrics: dict, dest: Path) -> Path:
    """
    Write metrics to CSV using the canonical short keys (rmse_e, rmse_f, rmse_s)
    so that ``replot_evaluation`` can read them back without any translation.
    """
    with open(dest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in ("rmse_e", "rmse_f", "rmse_s"):
            if key in metrics:
                w.writerow([key, metrics[key]])
    return dest
