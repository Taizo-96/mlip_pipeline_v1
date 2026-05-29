"""Radial distribution function (RDF) from NVT MD.

Physics
-------
Run NVT MD at temperature T (default 300 K) after equilibration, then
accumulate g(r) via LAMMPS compute rdf + fix ave/time.

The first peak position r_1 is the nearest-neighbour distance.
For Pb fcc at 300 K:  r_1 ≈ 3.49 Å  (a/sqrt(2), a ≈ 4.95 Å)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import RdfResult
from mlip_pipeline.validate.lammps import write_rdf_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


def _parse_rdf_output(out_file: Path) -> tuple[list[float], list[float]]:
    """Parse LAMMPS ave/time RDF file -> (r_list, g_r_list).

    The ave/time file format has a header block followed by rows:
      TimeStep  Number-of-rows
      row  r  g(r)  coord_number  ...
    We average over all timestep blocks.
    """
    r_acc: dict[int, list[float]] = {}   # bin_idx -> list of g(r) values
    r_vals: dict[int, float] = {}        # bin_idx -> r

    lines = out_file.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for timestep header lines: two numbers
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    _ts = int(parts[0])
                    n_rows = int(parts[1])
                    i += 1
                    for row_idx in range(n_rows):
                        if i >= len(lines):
                            break
                        row = lines[i].strip().split()
                        i += 1
                        if len(row) < 3:
                            continue
                        try:
                            bin_id = int(row[0])
                            r = float(row[1])
                            g = float(row[2])
                            r_vals[bin_id] = r
                            r_acc.setdefault(bin_id, []).append(g)
                        except ValueError:
                            continue
                    continue
                except ValueError:
                    pass
        i += 1

    if not r_vals:
        return [], []

    bins = sorted(r_vals.keys())
    r_list = [r_vals[b] for b in bins]
    g_list = [sum(r_acc[b]) / len(r_acc[b]) for b in bins]
    return r_list, g_list


def _first_peak(r_list: list[float], g_list: list[float]) -> tuple[float, float]:
    """Return (r_peak, g_peak) of the first peak in g(r).

    Skip the near-zero bins (r < 1.5 Å) where g(r) is always 0.
    """
    best_g = -1.0
    best_r = float("nan")
    for r, g in zip(r_list, g_list):
        if r < 1.5:
            continue
        if g > best_g:
            best_g = g
            best_r = r
        # Stop after the first peak has been passed (g starts decreasing)
        elif best_g > 1.0 and g < best_g * 0.5:
            break
    return best_r, best_g


def run_rdf(
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
    temperature: float = 300.0,
    r_max: float = 8.0,
    n_bins: int = 200,
    n_equil: int = 5000,
    n_prod: int = 20000,
    dt: float = 0.002,
    supercell_repeat: int = 3,
) -> RdfResult:
    work_dir = ensure_dir(validate_dir / "rdf" / structure_id)
    out_file = work_dir / "rdf_output.txt"
    if out_file.exists():
        out_file.unlink()

    print(f"  [rdf] {structure_id}: NVT MD at T={temperature:.0f} K, "
          f"accumulating g(r) ...")

    script = write_rdf_input(
        lammps_data, model_path, work_dir,
        temperature=temperature,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        r_max=r_max,
        n_bins=n_bins,
        supercell_repeat=supercell_repeat,
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
        print(f"  [rdf] WARNING: LAMMPS failed for {structure_id}: {exc}")
        return RdfResult(structure_id=structure_id, error=str(exc))

    r_list, g_list = _parse_rdf_output(out_file)
    if len(r_list) < 5:
        msg = f"only {len(r_list)} RDF bins parsed"
        print(f"  [rdf] WARNING: {structure_id}: {msg}")
        return RdfResult(structure_id=structure_id, error=msg)

    r_peak, g_peak = _first_peak(r_list, g_list)
    print(f"  [rdf] {structure_id}: first peak r={r_peak:.3f} Å  g(r)={g_peak:.3f}  "
          f"(T={temperature:.0f} K)")
    return RdfResult(
        structure_id=structure_id,
        r=r_list,
        g_r=g_list,
        temperature=temperature,
        first_peak_r=r_peak,
        first_peak_g=g_peak,
        compute_ok=True,
    )
