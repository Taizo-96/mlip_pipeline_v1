# This script has been superseded by the CLI command:
#
#   mlip-pipeline validate --config <yaml> --model <almtp> [--outdir <dir>]
#
# It is kept as a thin shim for backward compatibility only.
# Please use the CLI going forward.

from __future__ import annotations
import sys
from pathlib import Path

if __name__ == "__main__":
    print(
        "run_validation.py is deprecated.\n"
        "Use:  mlip-pipeline validate --config <yaml> --model <almtp> [--outdir <dir>]\n",
        file=sys.stderr,
    )
    sys.exit(1)
