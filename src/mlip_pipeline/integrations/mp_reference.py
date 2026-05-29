"""Fetch reference EOS and elastic data from the Materials Project API.

Results are cached to a JSON file in the validate output directory so the
API is only hit once per mp_id.  Set the MP API key via the environment
variable named in the config (default: MP_API_KEY).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


_CACHE_FILENAME = "mp_reference.json"


def fetch_mp_reference(
    mp_id: str,
    out_dir: Path,
    api_key_env: str = "MP_API_KEY",
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """Return a dict with reference values for *mp_id*, or None on failure.

    Cached to ``out_dir/mp_reference.json`` after the first successful fetch.

    Returned keys (all optional — only present when MP has data):
        V0        float  Å³/atom
        E0        float  eV/atom  (formation energy per atom, GGA/PBE)
        B0        float  GPa      (Voigt bulk modulus)
        G0        float  GPa      (Voigt shear modulus)
        C11       float  GPa
        C12       float  GPa
        C44       float  GPa
        mp_id     str
        formula   str
    """
    cache_path = Path(out_dir) / _CACHE_FILENAME
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if data.get("mp_id") == mp_id:
                print(f"  [mp_ref]  loaded cached reference: {cache_path}")
                return data
        except Exception:
            pass

    key = api_key or os.environ.get(api_key_env)
    if not key:
        print(f"  [mp_ref]  WARNING: MP API key not found in env var '{api_key_env}' — skipping reference")
        return None

    try:
        from mp_api.client import MPRester  # type: ignore
    except ImportError:
        print("  [mp_ref]  WARNING: mp-api not installed (pip install mp-api) — skipping reference")
        return None

    ref: dict = {"mp_id": mp_id}

    try:
        with MPRester(key) as mpr:
            # ── Thermo: volume and energy ──────────────────────────────────
            thermo = mpr.materials.thermo.get_data_by_id(mp_id, fields=["volume", "energy_per_atom", "formula_pretty", "nsites"])
            if thermo:
                ref["formula"] = getattr(thermo, "formula_pretty", "?")
                nsites = getattr(thermo, "nsites", None)
                volume = getattr(thermo, "volume", None)  # Å³ total cell
                if volume is not None and nsites:
                    ref["V0"] = round(volume / nsites, 4)
                epa = getattr(thermo, "energy_per_atom", None)
                if epa is not None:
                    ref["E0"] = round(float(epa), 6)

            # ── Elasticity: bulk/shear moduli and Cij tensor ───────────────
            elast = mpr.materials.elasticity.get_data_by_id(mp_id, fields=["bulk_modulus", "shear_modulus", "elastic_tensor"])
            if elast:
                bm = getattr(elast, "bulk_modulus", None)
                gm = getattr(elast, "shear_modulus", None)
                if bm is not None:
                    voigt_b = getattr(bm, "voigt", None) or (bm if isinstance(bm, (int, float)) else None)
                    if voigt_b is not None:
                        ref["B0"] = round(float(voigt_b), 1)
                if gm is not None:
                    voigt_g = getattr(gm, "voigt", None) or (gm if isinstance(gm, (int, float)) else None)
                    if voigt_g is not None:
                        ref["G0"] = round(float(voigt_g), 1)
                tensor = getattr(elast, "elastic_tensor", None)
                if tensor is not None:
                    # elastic_tensor is a pymatgen ElasticTensor (9×9 or 6×6 Voigt)
                    try:
                        import numpy as np  # type: ignore
                        t = np.array(tensor.voigt) if hasattr(tensor, "voigt") else np.array(tensor)
                        if t.shape == (6, 6):
                            ref["C11"] = round(float(t[0, 0]), 1)
                            ref["C12"] = round(float(t[0, 1]), 1)
                            ref["C44"] = round(float(t[3, 3]), 1)
                    except Exception:
                        pass

    except Exception as exc:
        print(f"  [mp_ref]  WARNING: MP API call failed: {exc}")
        return None

    if len(ref) <= 1:  # only mp_id, nothing useful
        return None

    cache_path.write_text(json.dumps(ref, indent=2))
    print(f"  [mp_ref]  fetched reference for {mp_id} ({ref.get('formula', '?')}): {cache_path}")
    return ref


def print_deviation_table(mtp_vals: dict, ref: dict) -> None:
    """Print a side-by-side comparison table of MTP vs MP reference values."""
    keys = [
        ("V0",  "Å³/atom"),
        ("E0",  "eV/atom"),
        ("B0",  "GPa"),
        ("G0",  "GPa"),
        ("C11", "GPa"),
        ("C12", "GPa"),
        ("C44", "GPa"),
    ]
    label = ref.get("mp_id", "MP")
    formula = ref.get("formula", "")
    header = f"  {'Property':<8}  {'MTP':>10}  {'Ref ({})'.format(label):>14}  {'Δ%':>8}  Unit"
    print(f"\n  ── vs. {label} ({formula}) ──────────────────────────────")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, unit in keys:
        mtp_v = mtp_vals.get(key)
        ref_v = ref.get(key)
        if mtp_v is None or ref_v is None:
            continue
        try:
            delta = (float(mtp_v) - float(ref_v)) / abs(float(ref_v)) * 100
            flag = "  ✓" if abs(delta) < 10 else ("  !" if abs(delta) < 25 else "  ✗")
            print(f"  {key:<8}  {float(mtp_v):>10.3f}  {float(ref_v):>14.3f}  {delta:>+7.1f}%  {unit}{flag}")
        except (TypeError, ZeroDivisionError):
            continue
    print()
