"""elastic_vasp.py -- Parse elastic constants from a VASP IBRION=6 OUTCAR.

VASP with IBRION=6 and ISIF=3 computes the full elastic stiffness tensor
using its internal finite-differences scheme and writes the result to OUTCAR.
This module extracts that tensor.

Usage
-----
    from mlip_pipeline.evaluate.elastic_vasp import parse_elastic_vasp
    results = parse_elastic_vasp("path/to/OUTCAR")
    # results["C11"], results["C12"], results["C44"], results["B_voigt"], ...
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

VOIGT_LABELS = ["xx", "yy", "zz", "xy", "xz", "yz"]


def parse_elastic_vasp(outcar_path: str | Path) -> dict:
    """
    Parse the elastic stiffness tensor from a VASP OUTCAR (IBRION=6, ISIF=3).

    VASP writes a block like::

        ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)
        Direction    XX          YY          ZZ          XY          YZ          ZX
        ------------------------------------------------------------------------
        XX         ...         ...         ...
        ...

    followed by::

        TOTAL ELASTIC MODULI (kBar)

    This function reads the TOTAL block (kBar) and converts to GPa.

    Returns
    -------
    dict with keys: C_matrix (6x6 GPa), C11..C66, B_voigt, G_voigt,
    voigt_labels, source='vasp'.
    Raises RuntimeError if the block is not found.
    """
    text = Path(outcar_path).read_text()

    # Locate the TOTAL elastic moduli block
    match = re.search(
        r"TOTAL ELASTIC MODULI \(kBar\).*?Direction.*?-{40,}(.*?)-{40,}",
        text,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(
            f"TOTAL ELASTIC MODULI block not found in {outcar_path}.\n"
            "Ensure the VASP run used IBRION = 6 and ISIF = 3."
        )

    block = match.group(1).strip()
    rows = []
    for line in block.splitlines():
        parts = line.split()
        # Each line starts with a label (XX, YY, ...) followed by 6 values
        if len(parts) == 7 and parts[0].upper() in ("XX", "YY", "ZZ", "XY", "YZ", "ZX"):
            rows.append([float(x) for x in parts[1:]])

    if len(rows) != 6:
        raise RuntimeError(
            f"Expected 6 rows in elastic block, got {len(rows)} in {outcar_path}"
        )

    # VASP order: XX YY ZZ XY YZ ZX -> remap to Voigt [xx yy zz xy xz yz]
    # VASP writes ZX in position 6; Voigt convention has XZ=ZX
    C_kbar = np.array(rows)  # 6x6, kBar
    # Reorder columns from VASP (XX YY ZZ XY YZ ZX) to Voigt (XX YY ZZ XY XZ YZ)
    vasp_order = [0, 1, 2, 3, 5, 4]  # swap YZ(4) and ZX(5)
    C_kbar = C_kbar[:, vasp_order][vasp_order, :]
    C = C_kbar * 0.1  # kBar -> GPa

    results = _build_results(C)
    results["source"] = "vasp"
    results["outcar"] = str(outcar_path)
    return results


def _build_results(C: np.ndarray) -> dict:
    results: dict = {"C_matrix": C, "voigt_labels": VOIGT_LABELS}
    results["C11"] = float(C[0, 0])
    results["C22"] = float(C[1, 1])
    results["C33"] = float(C[2, 2])
    results["C12"] = float(C[0, 1])
    results["C13"] = float(C[0, 2])
    results["C23"] = float(C[1, 2])
    results["C44"] = float(C[3, 3])
    results["C55"] = float(C[4, 4])
    results["C66"] = float(C[5, 5])
    results["B_voigt"] = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9
    results["G_voigt"] = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[0,2]+C[1,2])
                          + 3*(C[3,3]+C[4,4]+C[5,5])) / 15
    return results
