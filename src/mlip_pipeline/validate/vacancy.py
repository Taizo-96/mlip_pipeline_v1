"""Vacancy formation energy.

Physics
-------
Create a relaxed supercell of N atoms ("perfect") and a supercell with one
atom removed ("vacancy").  The vacancy formation energy is:

    E_vac = E(N-1) - (N-1)/N * E(N)

where E(N) is the total DFT/MTP energy of the N-atom perfect supercell and
E(N-1) is the total energy of the (N-1)-atom vacancy supercell.

For Pb fcc, the experimental value is ~0.57 eV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import VacancyResult
from mlip_pipeline.validate.lammps import write_vacancy_inputs, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


def _parse_energy_file(path: Path) -> tuple[float, int]:
    """Read '<etot> <natoms>' from a one-line LAMMPS print output."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), int(parts[1])
            except ValueError:
                continue
    raise ValueError(f"Could not parse energy file: {path}")


def run_vacancy(
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
    supercell_repeat: int = 3,
) -> VacancyResult:
    work_dir = ensure_dir(validate_dir / "vacancy" / structure_id)

    print(f"  [vacancy] {structure_id}: minimising perfect and vacancy supercells "
          f"({supercell_repeat}x{supercell_repeat}x{supercell_repeat}) ...")

    perfect_script, vacancy_script = write_vacancy_inputs(
        lammps_data, model_path, work_dir,
        supercell_repeat=supercell_repeat,
    )

    for name, script in [("perfect", perfect_script), ("vacancy", vacancy_script)]:
        try:
            run_lammps(
                script, work_dir,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                log_file=work_dir / f"{name}_lammps.log",
                lammps_data=lammps_data,
                cutoff=cutoff,
            )
        except RuntimeError as exc:
            msg = f"LAMMPS failed ({name}): {exc}"
            print(f"  [vacancy] WARNING: {structure_id}: {msg}")
            return VacancyResult(structure_id=structure_id, error=msg)

    try:
        E_perf, N_perf = _parse_energy_file(work_dir / "perfect_energy.txt")
        E_vac_cell, N_vac = _parse_energy_file(work_dir / "vacancy_energy.txt")
    except (ValueError, FileNotFoundError) as exc:
        msg = str(exc)
        print(f"  [vacancy] WARNING: {structure_id}: {msg}")
        return VacancyResult(structure_id=structure_id, error=msg)

    # E_vac = E(N-1) - (N-1)/N * E(N)
    E_vac = E_vac_cell - (N_vac / N_perf) * E_perf
    print(f"  [vacancy] {structure_id}: E_vac = {E_vac:.4f} eV  "
          f"(N_perfect={N_perf}, N_vacancy={N_vac})")
    return VacancyResult(
        structure_id=structure_id,
        E_vac=E_vac,
        n_atoms_perfect=N_perf,
        n_atoms_vacancy=N_vac,
        compute_ok=True,
    )
