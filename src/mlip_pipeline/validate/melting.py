"""Melting temperature estimation via two-phase coexistence (TPC).

Physics
-------
A supercell is split into solid (bottom half, z < zmid) and liquid
(top half, z > zmid) by assembling two independently equilibrated
single-phase snapshots.  The combined cell is then run in NPH
(constant enthalpy / pressure) at P = 0.

The potential energy time series reveals which phase is stable at
T_target under NPH (the physically correct ensemble for TPC):
  - PE increasing  ->  liquid growing  ->  T > T_melt
  - PE decreasing  ->  solid growing   ->  T < T_melt
  - PE flat        ->  ambiguous; scan continues

Trend classification uses a two-sample Welch t-test comparing the
first and last quartile of the PE series.

All candidate temperatures are run concurrently (ThreadPoolExecutor).
Bracket detection is applied after all futures resolve.

Preparation protocol (splice approach)
---------------------------------------
All three LAMMPS scripts run only ONE thermostat on ALL atoms at a
time — there are no region/group commands in any of them.  This
entirely avoids the MPI ghost-atom double-integration failure that
affects all dual-thermostat approaches with 4+ MPI ranks.

  Step 1 — solid_eq.in:
    NPT ramp 1 K → T_target, then NPT hold at T_target.
    Final snapshot → solid_final.dump.

  Step 2 — liquid_eq.in:
    NPT ramp 1 K → T_dis = min(1.3*T, T+300), then NPT hold.
    Final snapshot → liquid_final.dump.

  Step 3 — Python splice (splice_tpc_cell):
    Read both dumps.  lo-z atoms from solid dump, hi-z atoms from
    liquid dump → tpc_start.lammps.

  Step 4 — coex.in:
    Read tpc_start.lammps.  NVT interface relax (Stage A), then NPH
    production (Stage B).  No groups.  Velocities inherited.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from mlip_pipeline.validate.models import MeltingResult
from mlip_pipeline.validate.lammps import (
    write_solid_eq_input,
    write_liquid_eq_input,
    splice_tpc_cell,
    write_coex_input,
    run_lammps,
)
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
    |t| >= 2.0 threshold (roughly p < 0.05 for n >= 30 per group).
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

    t_stat = _welch_t_statistic(first_q, last_q)

    T_THRESHOLD = 2.0
    if t_stat > T_THRESHOLD:
        return "growing_liquid"
    elif t_stat < -T_THRESHOLD:
        return "growing_solid"
    return "ambiguous"


# ------------------------------------------------------------------ #
# Single-temperature TPC run                                           #
# ------------------------------------------------------------------ #

def _run_tpc_temperature(
    T_cand: float,
    T_dir: Path,
    lammps_data: Path,
    model_path: Path,
    *,
    supercell_repeat: int,
    dt: float,
    n_equil: int,
    n_prod: int,
    lammps_cmd: str,
    mpi_command: Optional[str],
    mpi_np: Optional[int],
    cutoff: Optional[float],
    pair_style: Optional[str],
    pair_coeff: Optional[str],
) -> str:
    """Run the full splice TPC protocol for one temperature.

    Returns the PE trend string: 'growing_liquid', 'growing_solid',
    'ambiguous', or 'error'.
    """
    run_kw = dict(
        lammps_cmd=lammps_cmd,
        mpi_command=mpi_command,
        mpi_np=mpi_np,
        lammps_data=lammps_data,
        cutoff=cutoff,
        supercell_repeat=supercell_repeat,
    )

    solid_script = write_solid_eq_input(
        lammps_data, model_path, T_dir,
        temperature=T_cand,
        supercell_repeat=supercell_repeat,
        dt=dt,
        n_equil=n_equil,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )
    try:
        run_lammps(solid_script, T_dir, log_file=T_dir / "solid_eq.log", **run_kw)
    except RuntimeError as exc:
        print(f"  [melting] WARNING: solid_eq failed at T={T_cand}: {exc}")
        return "error"

    liquid_script = write_liquid_eq_input(
        lammps_data, model_path, T_dir,
        temperature=T_cand,
        supercell_repeat=supercell_repeat,
        dt=dt,
        n_equil=n_equil,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )
    try:
        run_lammps(liquid_script, T_dir, log_file=T_dir / "liquid_eq.log", **run_kw)
    except RuntimeError as exc:
        print(f"  [melting] WARNING: liquid_eq failed at T={T_cand}: {exc}")
        return "error"

    solid_dump  = T_dir / "solid_final.dump"
    liquid_dump = T_dir / "liquid_final.dump"
    tpc_data    = T_dir / "tpc_start.lammps"
    try:
        splice_tpc_cell(solid_dump, liquid_dump, tpc_data)
    except Exception as exc:
        print(f"  [melting] WARNING: splice failed at T={T_cand}: {exc}")
        return "error"

    coex_script = write_coex_input(
        tpc_data, model_path, T_dir,
        temperature=T_cand,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )
    try:
        run_lammps(coex_script, T_dir, log_file=T_dir / "coex.log", **run_kw)
    except RuntimeError as exc:
        print(f"  [melting] WARNING: coex failed at T={T_cand}: {exc}")
        return "error"

    return _classify_pe_trend(T_dir / "coex_thermo.txt")


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
    max_workers: Optional[int] = None,
    pair_style: Optional[str] = None,
    pair_coeff: Optional[str] = None,
) -> MeltingResult:
    """Bracket T_melt by scanning temperatures concurrently.

    All candidate temperatures are submitted to a ThreadPoolExecutor
    and run in parallel.  Each temperature writes to its own T<N>/
    subdirectory, so there are no file-system races.

    Bracket detection (T_lo / T_hi) is applied after all futures
    resolve.  The reported T_melt is the midpoint of [T_lo, T_hi];
    uncertainty = (T_hi - T_lo) / 2, or T_step / 2 for one-sided
    brackets.

    Parameters
    ----------
    max_workers:
        Maximum number of concurrent temperature runs.  Defaults to
        len(candidates) (all at once).  Set in config as
        ``melting.max_workers`` to cap concurrent MPI jobs.
    """
    work_dir = ensure_dir(validate_dir / "melting" / structure_id)

    candidates: list[float] = []
    T = T_start
    while T <= T_end + 1e-3:
        candidates.append(round(T, 1))
        T += T_step

    _workers = max_workers or len(candidates)
    print(
        f"  [melting] {structure_id}: scanning {len(candidates)} temperatures "
        f"({T_start:.0f}–{T_end:.0f} K, step {T_step:.0f} K) "
        f"with {_workers} concurrent worker(s) ..."
    )

    common_kw = dict(
        lammps_data=lammps_data,
        model_path=model_path,
        supercell_repeat=supercell_repeat,
        dt=dt,
        n_equil=n_equil,
        n_prod=n_prod,
        lammps_cmd=lammps_cmd,
        mpi_command=mpi_command,
        mpi_np=mpi_np,
        cutoff=cutoff,
        pair_style=pair_style,
        pair_coeff=pair_coeff,
    )

    trend_map: dict[float, str] = {}

    with ThreadPoolExecutor(max_workers=_workers) as pool:
        future_to_T = {
            pool.submit(
                _run_tpc_temperature,
                T_cand,
                ensure_dir(work_dir / f"T{T_cand:.0f}"),
                **common_kw,
            ): T_cand
            for T_cand in candidates
        }
        for f in as_completed(future_to_T):
            T_cand = future_to_T[f]
            try:
                trend = f.result()
            except Exception as exc:  # noqa: BLE001
                trend = "error"
                print(f"  [melting] WARNING: T={T_cand:.0f} K raised exception: {exc}")
            trend_map[T_cand] = trend
            print(f"  [melting]   T={T_cand:.0f} K -> {trend}")

    # ------------------------------------------------------------------ #
    # Bracket detection — applied to the full sorted result set           #
    # ------------------------------------------------------------------ #
    T_lo: Optional[float] = None
    T_hi: Optional[float] = None

    for T_cand in sorted(trend_map):
        trend = trend_map[T_cand]
        if trend == "growing_solid":
            T_lo = T_cand
        elif trend == "growing_liquid":
            if T_hi is None:
                T_hi = T_cand

    if T_lo is None and T_hi is None:
        msg = "could not bracket T_melt in the given range"
        print(f"  [melting] WARNING: {structure_id}: {msg}")
        return MeltingResult(structure_id=structure_id, error=msg)

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
