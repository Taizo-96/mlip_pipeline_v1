"""Vacancy formation energy runner."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import VacancyResult
from mlip_pipeline.validate.lammps import write_vacancy_inputs, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> VacancyResult:
    """Compute vacancy formation energy: E_vac = E(N-1) - (N-1)/N * E(N)."""
    work_dir = ensure_dir(validate_dir / "vacancy" / structure_id)

    print(
        f"  [vacancy] {structure_id}: minimising perfect and vacancy supercells "
        f"({supercell_repeat}x{supercell_repeat}x{supercell_repeat}) ..."
    )

    perfect_script, vacancy_script = write_vacancy_inputs(
        lammps_data, model_path, work_dir,
        supercell_repeat=supercell_repeat,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )

    def _run(script: Path, label: str) -> Optional[tuple[float, int]]:
        try:
            run_lammps(
                script, work_dir,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                log_file=work_dir / f"{label}_lammps.log",
                lammps_data=lammps_data,
                cutoff=cutoff,
                supercell_repeat=supercell_repeat,
            )
        except RuntimeError as exc:
            print(f"  [vacancy] WARNING: LAMMPS failed ({label}): {exc}")
            return None
        out_file = work_dir / f"{label}_energy.txt"
        for line in out_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[0]), int(parts[1])
                except ValueError:
                    continue
        return None

    perfect = _run(perfect_script, "perfect")
    vacancy = _run(vacancy_script, "vacancy")

    if perfect is None or vacancy is None:
        msg = "LAMMPS run failed for perfect or vacancy supercell"
        return VacancyResult(structure_id=structure_id, error=msg)

    E_perf, N_perf = perfect
    E_vac_cell, N_vac = vacancy

    if N_perf == 0:
        return VacancyResult(structure_id=structure_id, error="zero atoms in perfect cell")

    E_vac = E_vac_cell - (N_vac / N_perf) * E_perf

    print(
        f"  [vacancy] {structure_id}: E_vac = {E_vac:.4f} eV "
        f" (N_perfect={N_perf}, N_vacancy={N_vac})"
    )
    return VacancyResult(
        structure_id=structure_id,
        E_vac=E_vac,
        n_atoms_perfect=N_perf,
        n_atoms_vacancy=N_vac,
        compute_ok=True,
    )
