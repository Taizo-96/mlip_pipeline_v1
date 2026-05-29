"""EOS runner and Birch-Murnaghan fitter."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import EosResult
from mlip_pipeline.validate.lammps import write_eos_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


# ------------------------------------------------------------------ #
# Birch-Murnaghan equation of state (3rd order)                        #
# ------------------------------------------------------------------ #

def _birch_murnaghan(V: float, V0: float, E0: float, B0: float, B0p: float) -> float:
    """3rd-order Birch-Murnaghan EOS — energy in eV, volume in Å³."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (
        E0
        + (9.0 * V0 * B0 / 16.0)
        * (
            (eta - 1.0) ** 3 * B0p
            + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
        )
    )


def _fit_bm(volumes: list[float], energies: list[float]) -> tuple[float, float, float, float]:
    """Fit a 3rd-order Birch-Murnaghan EOS using scipy.optimize.curve_fit.

    Returns (V0, E0, B0_GPa, B0p).
    Raises RuntimeError if the fit fails.
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy and numpy are required for EOS fitting") from exc

    vs = np.array(volumes)
    es = np.array(energies)

    i_min = int(np.argmin(es))
    p0 = [vs[i_min], es[i_min], 100.0, 4.0]  # V0, E0, B0 (GPa), B0p

    def model(V, V0, E0, B0_gpa, B0p):
        # Convert GPa → eV/Å³  (1 GPa = 1 / 160.2176634 eV/Å³)
        B0_ev = B0_gpa / 160.2176634
        return np.array([_birch_murnaghan(v, V0, E0, B0_ev, B0p) for v in V])

    popt, _ = curve_fit(model, vs, es, p0=p0, maxfev=10000)
    V0, E0, B0_gpa, B0p = popt
    return float(V0), float(E0), float(B0_gpa), float(B0p)


# ------------------------------------------------------------------ #
# Output parser                                                         #
# ------------------------------------------------------------------ #

def _parse_eos_output(out_file: Path) -> tuple[list[float], list[float]]:
    """Parse the LAMMPS EOS output file.

    Expected format (one data line per volume point)::

        # scale vol_per_atom energy_per_atom
        0.850  12.345  -4.321
        ...

    Returns (volumes, energies).
    """
    volumes: list[float] = []
    energies: list[float] = []
    for line in out_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            volumes.append(float(parts[1]))
            energies.append(float(parts[2]))
        except ValueError:
            continue
    return volumes, energies


# ------------------------------------------------------------------ #
# Public runner                                                         #
# ------------------------------------------------------------------ #

def run_eos(
    structure_id: str,
    lammps_data: Path,
    model_path: Path,
    validate_dir: Path,
    *,
    element: str,
    lammps_cmd: str = "lmp_mpi",
    mpi_command: Optional[str] = None,
    mpi_np: Optional[int] = None,
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    n_points: int = 21,
) -> EosResult:
    """Run an EOS calculation and return an EosResult.

    The work directory is <validate_dir>/eos/<structure_id>/.
    """
    work_dir = ensure_dir(validate_dir / "eos" / structure_id)
    out_file = work_dir / "eos_data.txt"

    print(f"  [eos]     {structure_id}: writing LAMMPS input ...")
    script = write_eos_input(
        lammps_data, model_path, out_file,
        element=element,
        scale_min=scale_min, scale_max=scale_max, n_points=n_points,
    )

    try:
        print(f"  [eos]     {structure_id}: running LAMMPS ({n_points} points) ...")
        run_lammps(
            script, work_dir,
            lammps_cmd=lammps_cmd,
            mpi_command=mpi_command,
            mpi_np=mpi_np,
            log_file=work_dir / "lammps.log",
        )
    except RuntimeError as exc:
        print(f"  [eos]     WARNING: LAMMPS failed for {structure_id}: {exc}")
        return EosResult(structure_id=structure_id, volumes=[], energies=[])

    volumes, energies = _parse_eos_output(out_file)
    if len(volumes) < 5:
        msg = f"too few EOS points parsed ({len(volumes)})"
        print(f"  [eos]     WARNING: {structure_id}: {msg}")
        return EosResult(
            structure_id=structure_id,
            volumes=volumes, energies=energies,
            fit_error=msg,
        )

    # Birch-Murnaghan fit
    try:
        V0, E0, B0, B0p = _fit_bm(volumes, energies)
        print(
            f"  [eos]     {structure_id}: V0={V0:.4f} Å³  "
            f"E0={E0:.6f} eV  B0={B0:.1f} GPa  B0'={B0p:.2f}"
        )
        return EosResult(
            structure_id=structure_id,
            volumes=volumes, energies=energies,
            V0=V0, E0=E0, B0=B0, B0p=B0p,
            fit_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [eos]     WARNING: BM fit failed for {structure_id}: {exc}")
        return EosResult(
            structure_id=structure_id,
            volumes=volumes, energies=energies,
            fit_error=str(exc),
        )
