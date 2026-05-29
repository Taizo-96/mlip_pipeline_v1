"""Melting temperature estimation via two-phase coexistence.

Physics
-------
A supercell is split into solid (bottom half) and liquid (top half) by z.
The liquid seed is created by briefly heating that half to ~T+50 K in NVT.
The full cell is then run in NVT at T_target.

The potential energy time series reveals which phase is stable at T_target
under NVT (constant volume):
  - PE increasing  ->  liquid growing  ->  T > T_melt
  - PE decreasing  ->  solid growing   ->  T < T_melt
  - PE flat        ->  ambiguous; scan continues

A bracket search across candidate temperatures is performed to locate T_melt
to within `T_step` K.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import MeltingResult
from mlip_pipeline.validate.lammps import write_melting_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


# ------------------------------------------------------------------ #
# PE trend classifier                                                  #
# ------------------------------------------------------------------ #

def _classify_pe_trend(thermo_file: Path) -> str:
    """Return 'growing_liquid', 'growing_solid', or 'ambiguous'.

    Reads the NVT production thermo file (coex_thermo.txt) and fits a
    linear slope to the POTENTIAL ENERGY (col 3, 0-indexed) time series.
    Under NVT the volume is fixed, so the PE signal is the correct
    indicator of which phase is growing:
      PE slope > 0  ->  liquid growing  (higher PE per atom)
      PE slope < 0  ->  solid growing   (lower  PE per atom)
      |slope| small ->  ambiguous; neither phase clearly winning
    """
    pes: list[float] = []
    try:
        for line in thermo_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    pes.append(float(parts[3]))  # col 3 = pe
                except ValueError:
                    continue
    except OSError:
        return "ambiguous"

    if len(pes) < 4:
        return "ambiguous"

    # Use last 50% of points for the slope (skip transient)
    mid = len(pes) // 2
    tail = pes[mid:]
    n = len(tail)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_p = sum(tail) / n
    num = sum((x - mean_x) * (p - mean_p) for x, p in zip(xs, tail))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return "ambiguous"
    slope = num / den

    # Normalise slope by |mean PE| to get fractional change per step.
    # PE is negative for metals; use abs() to avoid sign flip.
    if mean_p == 0:
        return "ambiguous"
    rel_slope = slope / abs(mean_p)
    THRESHOLD = 5e-8   # fractional PE change per timestep
    if rel_slope > THRESHOLD:
        return "growing_liquid"
    elif rel_slope < -THRESHOLD:
        return "growing_solid"
    return "ambiguous"


# ------------------------------------------------------------------ #
# Public runner                                                        #
# ------------------------------------------------------------------ #

def run_melting(
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
    T_start: float = 500.0,
    T_end: float = 750.0,
    T_step: float = 25.0,
    supercell_repeat: int = 4,
    n_equil: int = 5000,
    n_prod: int = 20000,
    dt: float = 0.002,
) -> MeltingResult:
    """Bracket T_melt by scanning temperatures and classifying PE trend."""
    work_dir = ensure_dir(validate_dir / "melting" / structure_id)
    print(f"  [melting] {structure_id}: scanning T = {T_start:.0f}..{T_end:.0f} K "
          f"in steps of {T_step:.0f} K ...")

    candidates = []
    T = T_start
    while T <= T_end + 1e-3:
        candidates.append(round(T, 1))
        T += T_step

    T_lo: Optional[float] = None   # highest T where solid grew (PE decreasing)
    T_hi: Optional[float] = None   # lowest  T where liquid grew (PE increasing)
    trend_map: dict[float, str] = {}

    for T_cand in candidates:
        T_dir = ensure_dir(work_dir / f"T{T_cand:.0f}")
        script = write_melting_input(
            lammps_data, model_path, T_dir,
            temperature=T_cand,
            n_solid=supercell_repeat,
            n_liquid=supercell_repeat,
            supercell_repeat=supercell_repeat,
            dt=dt,
            n_equil=n_equil,
            n_prod=n_prod,
        )
        try:
            run_lammps(
                script, T_dir,
                lammps_cmd=lammps_cmd,
                mpi_command=mpi_command,
                mpi_np=mpi_np,
                log_file=T_dir / "lammps.log",
                lammps_data=lammps_data,
                cutoff=cutoff,
            )
        except RuntimeError as exc:
            print(f"  [melting] WARNING: LAMMPS failed at T={T_cand}: {exc}")
            trend_map[T_cand] = "error"
            continue

        thermo_file = T_dir / "coex_thermo.txt"
        trend = _classify_pe_trend(thermo_file)
        trend_map[T_cand] = trend
        print(f"  [melting]   T={T_cand:.0f} K -> {trend}")

        if trend == "growing_solid":
            T_lo = T_cand          # update: keep highest solid-growing T
        elif trend == "growing_liquid":
            if T_hi is None:
                T_hi = T_cand      # record: lowest liquid-growing T
            # Early exit only once we have a proper bracket (both sides seen)
            if T_lo is not None and T_hi is not None and T_lo < T_hi:
                break
        # "ambiguous": continue scanning — do NOT update bracket or break

    if T_lo is None and T_hi is None:
        msg = "could not bracket T_melt in the given range"
        print(f"  [melting] WARNING: {structure_id}: {msg}")
        return MeltingResult(structure_id=structure_id, error=msg)

    # Estimate T_melt as midpoint of bracket
    if T_lo is not None and T_hi is not None and T_lo < T_hi:
        T_melt = (T_lo + T_hi) / 2.0
    elif T_lo is not None and T_hi is not None and T_lo == T_hi:
        T_melt = T_lo
    elif T_lo is None:
        T_melt = T_hi   # only upper bound known
    else:
        T_melt = T_lo   # only lower bound known

    print(f"  [melting] {structure_id}: T_melt ≈ {T_melt:.0f} K "
          f"(bracket [{T_lo}, {T_hi}] K)")
    return MeltingResult(
        structure_id=structure_id,
        T_melt=T_melt,
        T_bracket_lo=T_lo if T_lo is not None else float("nan"),
        T_bracket_hi=T_hi if T_hi is not None else float("nan"),
        compute_ok=True,
    )
