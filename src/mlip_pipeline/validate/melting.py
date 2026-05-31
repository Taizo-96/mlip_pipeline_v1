"""Melting temperature estimation via two-phase coexistence.

Physics
-------
A supercell is split into solid (bottom half) and liquid (top half) by z.
The liquid seed is created by briefly heating that half to min(1.3*T, T+300) K
in NVT at a reduced timestep, while the solid half is held at T_target.
The full cell is then run in NPH (constant enthalpy/pressure) at P=0.

The potential energy time series reveals which phase is stable at T_target
under NPH (constant pressure — the physically correct ensemble for TPC):
  - PE increasing  ->  liquid growing  ->  T > T_melt
  - PE decreasing  ->  solid growing   ->  T < T_melt
  - PE flat        ->  ambiguous; scan continues

Trend classification uses a two-sample Welch t-test comparing the first
and last quartile of the PE series, which is more robust than a simple
linear slope threshold.

A linear bracket scan across candidate temperatures locates T_melt
to within `T_step` K; the reported uncertainty is ±T_step/2.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import MeltingResult
from mlip_pipeline.validate.lammps import write_melting_input, run_lammps
from mlip_pipeline.utils.fs import ensure_dir


# ------------------------------------------------------------------ #
# PE trend classifier                                                  #
# ------------------------------------------------------------------ #

def _welch_t_statistic(a: list[float], b: list[float]) -> float:
    """Return Welch t-statistic for two independent samples (b - a direction)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    se2 = var_a / n_a + var_b / n_b
    if se2 <= 0:
        return 0.0
    return (mean_b - mean_a) / math.sqrt(se2)


def _classify_pe_trend(thermo_file: Path) -> str:
    """Return 'growing_liquid', 'growing_solid', or 'ambiguous'.

    Uses a two-sample Welch t-test comparing the first quartile (early
    production) vs the last quartile (late production) of the PE series.
    A |t| >= 2.0 threshold (roughly p < 0.05 for n >= 30) avoids the
    arbitrary relative-slope magic number and gives a statistically
    grounded classification.
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
                    pes.append(float(parts[3]))
                except ValueError:
                    continue
    except OSError:
        return "ambiguous"

    if len(pes) < 8:
        return "ambiguous"

    n = len(pes)
    q = max(2, n // 4)
    first_q = pes[:q]
    last_q  = pes[n - q:]

    # t > 0 means PE increased => liquid growing => T > T_melt
    t_stat = _welch_t_statistic(first_q, last_q)

    T_THRESHOLD = 2.0  # |t| >= 2 ~ p < 0.05
    if t_stat > T_THRESHOLD:
        return "growing_liquid"
    elif t_stat < -T_THRESHOLD:
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
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> MeltingResult:
    """Bracket T_melt by scanning temperatures and classifying PE trend.

    The reported T_melt is the midpoint of the bracket [T_lo, T_hi].
    The inherent resolution is ±T_step/2 K, stored as T_melt_uncertainty
    in the result.  One-sided brackets (scan range too narrow) produce
    a warning and still return the best available bound, but with a
    NaN on the missing side so callers can detect the issue.
    """
    work_dir = ensure_dir(validate_dir / "melting" / structure_id)
    print(f"  [melting] {structure_id}: scanning T = {T_start:.0f}..{T_end:.0f} K "
          f"in steps of {T_step:.0f} K ...")

    candidates = []
    T = T_start
    while T <= T_end + 1e-3:
        candidates.append(round(T, 1))
        T += T_step

    T_lo: Optional[float] = None
    T_hi: Optional[float] = None
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
            pair_style=pair_style,
            pair_coeff=pair_coeff,
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
            T_lo = T_cand
        elif trend == "growing_liquid":
            if T_hi is None:
                T_hi = T_cand
            if T_lo is not None and T_hi is not None and T_lo < T_hi:
                break

    if T_lo is None and T_hi is None:
        msg = "could not bracket T_melt in the given range"
        print(f"  [melting] WARNING: {structure_id}: {msg}")
        return MeltingResult(structure_id=structure_id, error=msg)

    # Warn if only one side of the bracket was found — scan range is likely
    # too narrow and T_melt may lie outside the tested interval.
    if T_lo is None:
        print(
            f"  [melting] WARNING: {structure_id}: only upper bound found "
            f"(T_hi={T_hi:.0f} K). T_melt may be below T_start={T_start:.0f} K. "
            f"Consider re-running with a lower T_start."
        )
    if T_hi is None:
        print(
            f"  [melting] WARNING: {structure_id}: only lower bound found "
            f"(T_lo={T_lo:.0f} K). T_melt may be above T_end={T_end:.0f} K. "
            f"Consider re-running with a higher T_end."
        )

    if T_lo is not None and T_hi is not None and T_lo < T_hi:
        T_melt = (T_lo + T_hi) / 2.0
        # Actual bracket width may be larger than T_step if ambiguous temps
        # fell in between; use half the actual bracket as the uncertainty.
        uncertainty = (T_hi - T_lo) / 2.0
    elif T_lo is not None and T_hi is not None and T_lo == T_hi:
        T_melt = T_lo
        uncertainty = T_step / 2.0
    elif T_lo is None:
        T_melt = T_hi
        uncertainty = T_step / 2.0
    else:
        T_melt = T_lo
        uncertainty = T_step / 2.0

    print(
        f"  [melting] {structure_id}: T_melt \u2248 {T_melt:.0f} \u00b1 {uncertainty:.0f} K "
        f"(bracket [{T_lo}, {T_hi}] K)"
    )
    return MeltingResult(
        structure_id=structure_id,
        T_melt=T_melt,
        T_melt_uncertainty=uncertainty,
        T_bracket_lo=T_lo if T_lo is not None else float("nan"),
        T_bracket_hi=T_hi if T_hi is not None else float("nan"),
        compute_ok=True,
    )
