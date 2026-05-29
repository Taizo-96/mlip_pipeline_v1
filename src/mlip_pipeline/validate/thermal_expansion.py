"""Linear thermal expansion coefficient from NPT MD.

Physics
-------
Run NPT MD at a series of temperatures T_i.  After equilibration, read the
time-averaged volume per atom V(T).  Fit a linear regression:

    V(T) = V_ref * (1 + 3*alpha*(T - T_ref))

so the volumetric expansion coefficient beta = 3*alpha, and the *linear*
thermal expansion coefficient alpha = (1/V_ref) * dV/dT / 3.

For Pb the experimental value is alpha ≈ 29e-6 K^-1 at 300 K.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import ThermalExpansionResult
from mlip_pipeline.validate.lammps import write_thermal_expansion_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


def _parse_thexp_output(out_file: Path) -> tuple[list[float], list[float]]:
    """Parse thexp_output.txt -> (temperatures, volumes_per_atom)."""
    temps, vols = [], []
    for line in out_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                temps.append(float(parts[0]))
                vols.append(float(parts[1]))
            except ValueError:
                continue
    return temps, vols


def _fit_alpha(
    temps: list[float], vols: list[float], T_ref: float
) -> tuple[float, float]:
    """Linear least-squares fit of V vs T; return (alpha, V_ref).

    alpha = (dV/dT) / (3 * V_ref)   [K^-1]
    """
    n = len(temps)
    if n < 2:
        return float("nan"), float("nan")

    mean_T = sum(temps) / n
    mean_V = sum(vols) / n
    num = sum((t - mean_T) * (v - mean_V) for t, v in zip(temps, vols))
    den = sum((t - mean_T) ** 2 for t in temps)
    if den == 0:
        return float("nan"), float("nan")

    dV_dT = num / den
    # V_ref = V at T_ref from the linear fit
    V_ref = mean_V + dV_dT * (T_ref - mean_T)
    if V_ref <= 0:
        V_ref = mean_V
    alpha = dV_dT / (3.0 * V_ref)
    return alpha, V_ref


def run_thermal_expansion(
    structure_id: str,
    lammps_data: Path,
    model_path: Path,
    validate_dir: Path,
    *,
    element: str,
    lammps_cmd: str = "lmp_mpi",
    mpi_command: Optional[str] = None,
    mpi_np: Optional[int] = None,
    cutoff: Optional[float] = None,
    temperatures: Optional[list[float]] = None,
    T_ref: float = 300.0,
    n_equil: int = 5000,
    n_prod: int = 10000,
    dt: float = 0.002,
) -> ThermalExpansionResult:
    if temperatures is None:
        temperatures = [100.0, 200.0, 300.0, 400.0, 500.0]

    work_dir = ensure_dir(validate_dir / "thexp" / structure_id)
    out_file = work_dir / "thexp_output.txt"
    if out_file.exists():
        out_file.unlink()

    print(f"  [thexp] {structure_id}: NPT MD at T = "
          f"{[int(t) for t in temperatures]} K ...")

    script = write_thermal_expansion_input(
        lammps_data, model_path, work_dir,
        temperatures=temperatures,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
    )
    try:
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
        print(f"  [thexp] WARNING: LAMMPS failed for {structure_id}: {exc}")
        return ThermalExpansionResult(structure_id=structure_id, error=str(exc))

    temps, vols = _parse_thexp_output(out_file)
    if len(temps) < 2:
        msg = f"only {len(temps)} data points parsed"
        print(f"  [thexp] WARNING: {structure_id}: {msg}")
        return ThermalExpansionResult(structure_id=structure_id, error=msg)

    alpha, V_ref = _fit_alpha(temps, vols, T_ref)
    print(f"  [thexp] {structure_id}: alpha = {alpha*1e6:.2f} x10^-6 K^-1  "
          f"V_ref({T_ref:.0f}K) = {V_ref:.4f} Å³/atom")
    return ThermalExpansionResult(
        structure_id=structure_id,
        temperatures=temps,
        volumes=vols,
        alpha=alpha,
        V_ref=V_ref,
        T_ref=T_ref,
        compute_ok=True,
    )
