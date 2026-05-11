from __future__ import annotations

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
