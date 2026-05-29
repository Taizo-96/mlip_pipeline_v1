#!/usr/bin/env python
"""Standalone script to run physics validation on an already-trained MTP model.

Examples
--------
# Validate the model produced by generation 05:
python scripts/run_validation.py \\
    --model  runs/gen_05/FePb16.almtp \\
    --config configs/Fe_validate.yaml \\
    --outdir runs/gen_05/validate

# Use the validate section embedded in the main loop config:
python scripts/run_validation.py \\
    --model  runs/gen_05/FePb16.almtp \\
    --config configs/Fe_loop.yaml \\
    --outdir runs/gen_05/validate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package:
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    raw = yaml.safe_load(path.read_text())

    # Template expansion: replace {{ key }} and {{ section.key }} tokens.
    # This mirrors the minimal expansion done by config.py in the main pipeline.
    import re

    def _expand(text: str, ctx: dict) -> str:
        def _repl(m):
            expr = m.group(1).strip()
            if "." in expr:
                sec, key = expr.split(".", 1)
                return str(ctx.get(sec, {}).get(key, m.group(0)))
            return str(ctx.get(expr, m.group(0)))
        return re.sub(r"\{\{\s*([\w\.]+)\s*\}\}", _repl, text)

    # Two-pass expansion so nested references resolve
    text = path.read_text()
    for _ in range(3):
        text = _expand(text, yaml.safe_load(text) or {})
    return yaml.safe_load(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run physics validation on a trained MTP model."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the trained .almtp file",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a validation config YAML (Fe_validate.yaml or Fe_loop.yaml)",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Directory where validation output will be written",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    config_path = Path(args.config).resolve()
    out_dir     = Path(args.outdir).resolve()

    if not model_path.exists():
        raise SystemExit(f"ERROR: model not found: {model_path}")
    if not config_path.exists():
        raise SystemExit(f"ERROR: config not found: {config_path}")

    print(f"Loading config: {config_path}")
    config = _load_config(config_path)

    from mlip_pipeline.validate import run_validation
    result = run_validation(config, model_path, out_dir)

    # Print a short summary
    print("\n── EOS summary ──")
    for r in result.eos_results:
        status = f"B0={r.B0:.1f} GPa" if r.fit_ok else f"FAILED ({r.fit_error})"
        print(f"  {r.structure_id:12s}  {status}")

    print("\n── Elastic summary ──")
    for r in result.elastic_results:
        if r.compute_ok:
            print(f"  {r.structure_id:12s}  B_V={r.B_voigt:.1f} GPa  G_V={r.G_voigt:.1f} GPa")
        else:
            print(f"  {r.structure_id:12s}  FAILED ({r.error})")

    print(f"\nManifest: {result.validate_dir / 'validate_manifest.json'}")


if __name__ == "__main__":
    main()
