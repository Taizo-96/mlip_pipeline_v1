"""EOS runner and Birch-Murnaghan fitter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import EosResult
from mlip_pipeline.validate.lammps import write_eos_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


# ------------------------------------------------------------------ #
# Birch-Murnaghan equation of state (3rd order)                        #
# ------------------------------------------------------------------ #

# Conversion: 1 GPa = 1 / 160.2176634 eV/Å³
_GPA_TO_EV_ANG3 = 1.0 / 160.2176634

# Points more than this many eV/atom above the minimum are trimmed before
# fitting.  MTP potentials extrapolate badly under high compression and can
# produce catastrophically negative energies that corrupt the BM fit.
_OUTLIER_THRESHOLD_EV = 2.0


def _birch_murnaghan(
    V: float, V0: float, E0: float, B0_ev: float, B0p: float
) -> float:
    """3rd-order BM EOS.  V and V0 in Å³/atom, E0 in eV/atom, B0_ev in eV/Å³."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (
        E0
        + (9.0 * V0 * B0_ev / 16.0)
        * (
            (eta - 1.0) ** 3 * B0p
            + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
        )
    )


def _fit_bm(
    volumes: list[float], energies: list[float]
) -> tuple[float, float, float, float]:
    """Fit a 3rd-order BM EOS.

    Before fitting, trims points that are more than _OUTLIER_THRESHOLD_EV
    above the minimum energy.  This prevents catastrophic MTP extrapolation
    at extreme compression from dominating the fit.

    Returns: (V0 [Å³/atom], E0 [eV/atom], B0 [GPa], B0p [dimensionless]).
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy and numpy are required for EOS fitting") from exc

    vs = np.array(volumes, dtype=float)
    es = np.array(energies, dtype=float)

    # Trim outliers: keep only points within _OUTLIER_THRESHOLD_EV of minimum
    e_min = es.min()
    mask = es <= e_min + _OUTLIER_THRESHOLD_EV
    vs_fit = vs[mask]
    es_fit = es[mask]

    n_trimmed = len(vs) - mask.sum()
    if n_trimmed > 0:
        print(f"  [eos]     BM fit: trimmed {n_trimmed} outlier points "
              f"(>{_OUTLIER_THRESHOLD_EV} eV/atom above minimum)")

    if len(vs_fit) < 5:
        raise RuntimeError(
            f"Only {len(vs_fit)} points remain after outlier trimming — "
            "scan range may be too narrow or model is unphysical"
        )

    i_min = int(np.argmin(es_fit))
    V0_guess  = float(vs_fit[i_min])
    E0_guess  = float(es_fit[i_min])
    B0_guess  = 40.0 * _GPA_TO_EV_ANG3   # 40 GPa in eV/Å³
    B0p_guess = 4.0

    p0 = [V0_guess, E0_guess, B0_guess, B0p_guess]
    lower = [V0_guess * 0.5, E0_guess - 10.0, 1e-5,  1.0]
    upper = [V0_guess * 2.0, E0_guess + 10.0, 1.0,  20.0]

    def model(V, V0, E0, B0_ev, B0p):
        return np.array([_birch_murnaghan(v, V0, E0, B0_ev, B0p) for v in V])

    popt, _ = curve_fit(
        model, vs_fit, es_fit,
        p0=p0,
        bounds=(lower, upper),
        maxfev=100000,
        method="trf",
    )
    V0, E0, B0_ev, B0p = popt
    B0_gpa = B0_ev / _GPA_TO_EV_ANG3
    return float(V0), float(E0), float(B0_gpa), float(B0p)


# ------------------------------------------------------------------ #
# Output parser                                                         #
# ------------------------------------------------------------------ #

def _parse_eos_output(out_file: Path) -> tuple[list[float], list[float]]:
    """Parse LAMMPS EOS output: returns (volumes_per_atom, energies_per_atom)."""
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
    cutoff: Optional[float] = None,
) -> EosResult:
    """Run EOS calculation and return an EosResult."""
    work_dir = ensure_dir(validate_dir / "eos" / structure_id)
    out_file = work_dir / "eos_data.txt"

    if out_file.exists():
        out_file.unlink()

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
            lammps_data=lammps_data,
            cutoff=cutoff,
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

    try:
        V0, E0, B0, B0p = _fit_bm(volumes, energies)
        print(
            f"  [eos]     {structure_id}: "
            f"V0={V0:.4f} Å³/atom  E0={E0:.6f} eV/atom  "
            f"B0={B0:.1f} GPa  B0\'={B0p:.2f}"
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
