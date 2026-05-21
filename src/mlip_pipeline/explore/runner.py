from __future__ import annotations

from pathlib import Path

from mlip_pipeline.models import ExploreResult
from mlip_pipeline.utils.shell import run_command
from mlip_pipeline.utils.logging import info, warn


def run_exploration_runs(
    config: dict, resolved_paths: dict, explore_root: Path
) -> ExploreResult:
    """Run all LAMMPS exploration jobs found under explore_root.

    Non-zero exit codes are warned about but do not abort the loop —
    some runs may diverge (e.g. near the melt) and that is expected.
    The select step downstream will simply find fewer preselected.cfg
    candidates from those runs.

    Returns an :class:`ExploreResult` and writes ``explore_manifest.json``
    to *explore_root* on completion.
    """
    explore_cfg    = config["explore"]
    lammps_command = explore_cfg.get("lammps_command", "lmp")
    mpi_prefix     = explore_cfg.get("mpi_prefix", "")
    replicate      = list(explore_cfg.get("replicate", [1, 1, 1]))
    candidate_filename = explore_cfg.get("candidate_filename", "preselected.cfg")

    input_files = sorted(explore_root.rglob("in.mlip.pb"))
    if not input_files:
        raise FileNotFoundError(f"No exploration inputs found in {explore_root}")

    run_records: list[dict] = []
    failed_run_dirs: list[Path] = []
    preselected_cfgs: list[Path] = []
    n_ok   = 0
    n_fail = 0

    for input_file in input_files:
        command: list[str] = []
        if mpi_prefix:
            command.extend(mpi_prefix.split())
        command.extend([lammps_command, "-in", input_file.name])

        log_path    = input_file.with_suffix(input_file.suffix + ".log")
        return_code = run_command(command, cwd=input_file.parent, log_file=log_path)

        status = "ok" if return_code == 0 else "failed"

        # Collect the candidate cfg path for this run (may not exist if run failed
        # or if potential did not extrapolate — that is normal).
        candidate = input_file.parent / candidate_filename
        if candidate.exists():
            preselected_cfgs.append(candidate)

        run_records.append({
            "run_dir":        str(input_file.parent.relative_to(explore_root)),
            "status":         status,
            "exit_code":      return_code,
            "log":            str(log_path.relative_to(explore_root)),
            "has_candidates": candidate.exists(),
        })

        if return_code != 0:
            warn(
                f"LAMMPS run {input_file.parent.name} exited {return_code} — "
                f"continuing (see {log_path.name})"
            )
            failed_run_dirs.append(input_file.parent)
            n_fail += 1
        else:
            n_ok += 1

    if n_fail:
        warn(f"Exploration complete: {n_ok} ok, {n_fail} failed (non-fatal).")

    result = ExploreResult(
        explore_root=explore_root,
        run_dirs=[f.parent for f in input_files],
        n_runs=len(input_files),
        n_ok=n_ok,
        n_failed=n_fail,
        replicate=replicate,
        run_records=run_records,
        preselected_cfgs=preselected_cfgs,
        failed_runs=failed_run_dirs,
    )
    result.save_manifest()
    info(
        f"Explore manifest written → {explore_root / 'explore_manifest.json'} "
        f"({len(preselected_cfgs)} candidate file(s) from {n_ok}/{len(input_files)} runs)"
    )
    return result
