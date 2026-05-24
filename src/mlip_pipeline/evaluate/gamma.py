from __future__ import annotations

import json
import re
from pathlib import Path

# Grade is on the same line as the Feature keyword:
#   Feature   MV_grade 1.501604
_GRADE_RE = re.compile(
    r"Feature\s+(?:MV_grade|grade|EFT_grade)\s+([\d.eE+\-]+)",
    re.IGNORECASE,
)


def parse_grades_from_cfg(cfg_path: Path) -> list[float]:
    """
    Extract extrapolation grade (gamma) values from BEGIN_CFG blocks.
    Works with preselected.cfg or selected.cfg from explore/select step.
    """
    text = cfg_path.read_text(encoding="utf-8")
    return [float(m.group(1)) for m in _GRADE_RE.finditer(text)]


def parse_grades_with_provenance(
    cfg_paths: list[Path],
) -> list[dict]:
    """
    Return a list of records, one per grade value, with source provenance.

    Each record::

        {
            "grade":       float,
            "cfg_index":   int,   # 0-based index within the source file
            "source_file": str,   # relative or absolute path as string
        }

    This is the canonical representation written to ``gamma_grades.json``.
    """
    records: list[dict] = []
    for cfg_path in cfg_paths:
        text = cfg_path.read_text(encoding="utf-8")
        for idx, m in enumerate(_GRADE_RE.finditer(text)):
            records.append({
                "grade":       float(m.group(1)),
                "cfg_index":   idx,
                "source_file": str(cfg_path),
            })
    return records


# ── Persistence helpers ────────────────────────────────────────────────────────

GAMMA_GRADES_FILENAME = "gamma_grades.json"


def write_grades_json(grades: list[dict], dest: Path) -> Path:
    """
    Persist grade records to *dest* (a file path, e.g. ``eval/gamma_grades.json``).

    The file is a plain JSON object::

        {
            "n_grades": 1234,
            "grades":   [ {"grade": 1.5, "cfg_index": 0, "source_file": "..."}, ... ]
        }

    Returns the path written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"n_grades": len(grades), "grades": grades}, indent=2),
        encoding="utf-8",
    )
    return dest


def load_grades_json(path: Path) -> list[dict]:
    """
    Load grade records from a ``gamma_grades.json`` file.
    Returns an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("grades", [])


def grades_to_floats(records: list[dict]) -> list[float]:
    """Extract plain float list from provenance records for plotting / stats."""
    return [r["grade"] for r in records]
