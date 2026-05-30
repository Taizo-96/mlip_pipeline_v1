"""Thermal expansion coefficient runner."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import ThermalExpansionResult
from mlip_pipeline.validate.lammps import write_thermal_expansion_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> ThermalExpansionResult:
    """Run NPT MD at multiple temperatures and fit linear thermal expansion coefficient."""
    if temperatures is None:
        temperatures = [100.0, 200.0, 300.0, 400.0, 500.0]

    work_dir = ensure_dir(validate_dir / "thexp" / structure_id)
    print(f"  [thexp] {structure_id}: NPT MD at T = {temperatures} K ...")

    script = write_thermal_expansion_input(
        lammps_data, model_path, work_dir,
        temperatures=temperatures,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
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

    out_file = work_dir / "thexp_output.txt"
    Ts: list[float] = []
    Vs: list[float] = []
    for line in out_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                Ts.append(float(parts[0]))
                Vs.append(float(parts[1]))
            except ValueError:
                continue

    if len(Ts) < 3:
        msg = f"too few T-V points parsed ({len(Ts)})"
        print(f"  [thexp] WARNING: {structure_id}: {msg}")
        return ThermalExpansionResult(structure_id=structure_id, error=msg)

    # Linear fit: V(T) = V_ref * (1 + alpha*(T - T_ref))
    # => alpha = slope / V_ref
    import numpy as np  # type: ignore
    Ts_arr = np.array(Ts)
    Vs_arr = np.array(Vs)
    coeffs = np.polyfit(Ts_arr, Vs_arr, 1)
    slope = float(coeffs[0])
    V_ref_fit = float(np.polyval(coeffs, T_ref))
    alpha = slope / V_ref_fit if V_ref_fit != 0 else 0.0

    print(f"  [thexp] {structure_id}: alpha = {alpha*1e6:.2f}e-6 K\u207b\u00b9")
    return ThermalExpansionResult(
        structure_id=structure_id,
        temperatures=Ts,
        volumes=Vs,
        alpha=alpha,
        T_ref=T_ref,
        compute_ok=True,
    )
