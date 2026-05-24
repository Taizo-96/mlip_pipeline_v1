from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mlip_pipeline.utils.shell import run_command

logger = logging.getLogger(__name__)

# Only these files are synced to Dardel — outputs stay local until sync-back
VASP_INPUT_FILES = {"POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh"}

# rsync exit codes that are non-fatal (partial transfer warnings)
_RSYNC_OK = {0, 23, 24}

# Slurm states that mean a job will never run again
_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}

# ---------------------------------------------------------------------------
# Default retry schedule for sync_outputs_from_dardel.
# These values are used when the config does not override them via
#   label.dardel.sync_retry_delays      (list[int], seconds)
#   label.dardel.sync_retry_interval    (int, seconds)
#   label.dardel.sync_retry_deadline    (int, seconds)
# ---------------------------------------------------------------------------
_DEFAULT_RETRY_DELAYS   = [60, 300, 900, 1800]   # 1 min, 5 min, 15 min, 30 min
_DEFAULT_RETRY_INTERVAL = 3_600                   # 1 h between subsequent attempts
_DEFAULT_RETRY_DEADLINE = 24 * 3_600             # stop retrying after 24 h of total wait


def _retry_settings(dardel_cfg: dict) -> tuple[list[int], int, int]:
    """
    Read retry settings from the dardel sub-config, falling back to defaults.

    Returns
    -------
    (delays, interval, deadline)
        delays   – explicit per-attempt wait list (seconds)
        interval – wait between attempts once the list is exhausted (seconds)
        deadline – give up after this many cumulative seconds of waiting
    """
    delays   = list(dardel_cfg.get("sync_retry_delays",   _DEFAULT_RETRY_DELAYS))
    interval = int(dardel_cfg.get("sync_retry_interval", _DEFAULT_RETRY_INTERVAL))
    deadline = int(dardel_cfg.get("sync_retry_deadline", _DEFAULT_RETRY_DEADLINE))
    return delays, interval, deadline


class SyncRetryExhausted(RuntimeError):
    """Raised when rsync sync-back keeps failing for the configured deadline."""


# ---------------------------------------------------------------------------
# HpcJobState — persisted job-ID record written right after submission
# ---------------------------------------------------------------------------

class HpcJobState:
    """
    Thin persistence wrapper for Slurm job IDs produced by submit_jobs_on_dardel.

    Written to  <local_label_dir>/slurm_job_ids.json  immediately after
    sbatch so the watcher can be killed and restarted without losing track
    of which jobs belong to this pipeline run.

    Schema
    ------
    {
        "job_ids":   ["12345", "12346", ...],
        "user":      "username",
        "host":      "dardel.pdc.kth.se",
        "remote_dir": "/cfs/...",
        "submitted_at": "2026-05-22T08:00:00Z"
    }
    """
    _FILENAME = "slurm_job_ids.json"

    def __init__(
        self,
        job_ids: list[str],
        user: str,
        host: str,
        remote_dir: str,
        label_dir: Path,
    ) -> None:
        self.job_ids    = job_ids
        self.user       = user
        self.host       = host
        self.remote_dir = remote_dir
        self.label_dir  = Path(label_dir)

    @property
    def path(self) -> Path:
        return self.label_dir / self._FILENAME

    def save(self) -> Path:
        from mlip_pipeline.utils.fs import write_json
        from datetime import datetime, timezone
        write_json(self.path, {
            "job_ids":      self.job_ids,
            "user":         self.user,
            "host":         self.host,
            "remote_dir":   self.remote_dir,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return self.path

    @classmethod
    def load(cls, label_dir: Path) -> "HpcJobState":
        p = Path(label_dir) / cls._FILENAME
        if not p.exists():
            raise FileNotFoundError(
                f"{cls._FILENAME} not found in {label_dir}. "
                "Run 'submit-remote' first."
            )
        d = json.loads(p.read_text())
        return cls(
            job_ids=d["job_ids"],
            user=d["user"],
            host=d["host"],
            remote_dir=d["remote_dir"],
            label_dir=label_dir,
        )

    @classmethod
    def exists(cls, label_dir: Path) -> bool:
        return (Path(label_dir) / cls._FILENAME).exists()


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _socket_path(user: str, host: str) -> str:
    return f"/tmp/ssh_ctrl_{user}@{host}"


def _open_master(user: str, host: str, socket: str) -> None:
    """
    Open a persistent SSH ControlMaster socket.

    If a stale socket file already exists at *socket* it is removed first so
    that a dropped connection from a previous attempt does not silently block
    the new master from starting.
    """
    # Remove any leftover socket from a previous (dead) master.
    if os.path.exists(socket):
        try:
            os.remove(socket)
            logger.debug("_open_master: removed stale socket %s", socket)
        except OSError as exc:
            logger.warning("_open_master: could not remove stale socket %s: %s", socket, exc)

    master_cmd = [
        "ssh", "-MNf",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={socket}",
        "-o", "ControlPersist=yes",
        f"{user}@{host}",
    ]
    subprocess.Popen(master_cmd)
    time.sleep(2)


def _close_master(user: str, host: str, socket: str) -> None:
    """Close the persistent SSH ControlMaster socket."""
    exit_cmd = [
        "ssh", "-O", "exit",
        "-o", f"ControlPath={socket}",
        f"{user}@{host}",
    ]
    run_command(exit_cmd)
    print("SSH master connection closed.")


def _ssh_via_socket(user: str, host: str, socket: str, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-o", "ControlMaster=no",
        "-o", f"ControlPath={socket}",
        f"{user}@{host}",
        remote_cmd,
    ]


def _rsync_via_socket(socket: str, src: str, dest: str, extra_flags: list[str]) -> list[str]:
    return [
        "rsync", "-avz",
        "-e", f"ssh -o ControlMaster=no -o ControlPath={socket}",
        *extra_flags,
        src,
        dest,
    ]


# ---------------------------------------------------------------------------
# sacct-based terminal-state detection
# ---------------------------------------------------------------------------

def _check_jobs_via_sacct(
    user: str,
    host: str,
    socket: str,
    job_ids: list[str],
) -> tuple[bool, dict[str, str]]:
    """
    Query sacct for the terminal state of the given job IDs.

    Returns
    -------
    (all_done, state_map)
        all_done  - True when every job_id is in a terminal Slurm state.
        state_map - {job_id: state_string} for all returned rows.

    Notes
    -----
    sacct is used instead of squeue because it includes jobs that have
    already left the queue, eliminating the race condition where a job
    finishes between two squeue polls and the queue appears empty for the
    wrong reason.

    Only the base job ID (no array suffix) is requested so that both plain
    jobs and job-array elements are covered uniformly.
    """
    id_arg = ",".join(job_ids)
    remote_cmd = (
        f"sacct -j {id_arg} --format=JobID,State --noheader --parsable2"
    )
    result = subprocess.run(
        _ssh_via_socket(user, host, socket, remote_cmd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        logger.warning("sacct returned exit %d — falling back to squeue", result.returncode)
        return False, {}

    state_map: dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        raw_jid, state = parts[0], parts[1].strip()
        base_jid = re.split(r"[._]", raw_jid)[0]
        existing = state_map.get(base_jid, "")
        if state in _TERMINAL_STATES or existing not in _TERMINAL_STATES:
            state_map[base_jid] = state

    if not state_map:
        return False, {}

    all_done = all(
        state_map.get(jid, "") in _TERMINAL_STATES
        for jid in job_ids
    )
    return all_done, state_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sync_inputs_to_dardel(
    local_label_dir: Path,
    user: str,
    host: str,
    remote_label_dir: str,
    socket: str | None = None,
) -> None:
    """Rsync only VASP input files to Dardel."""
    s = socket or _socket_path(user, host)

    run_command(_ssh_via_socket(user, host, s, f"mkdir -p {remote_label_dir}"))

    include_flags = [f"--include=task.*/{f}" for f in sorted(VASP_INPUT_FILES)]
    cmd = _rsync_via_socket(
        s,
        f"{local_label_dir}/",
        f"{user}@{host}:{remote_label_dir}/",
        ["--include=task.*/", *include_flags, "--exclude=*"],
    )
    run_command(cmd)


def submit_jobs_on_dardel(
    user: str,
    host: str,
    remote_label_dir: str,
    local_label_dir: Optional[Path] = None,
    socket: str | None = None,
) -> list[str]:
    """
    Submit VASP jobs via sbatch on Dardel and return the captured job IDs.

    If *local_label_dir* is provided the job IDs are also persisted to
    ``<local_label_dir>/slurm_job_ids.json`` via :class:`HpcJobState` so
    that the watcher can be restarted independently.
    """
    s = socket or _socket_path(user, host)
    remote_cmd = (
        f"find {remote_label_dir} -name job.sh | sort | while read job; do "
        f"sbatch --chdir=$(dirname $(realpath $job)) $job; done"
    )
    result = subprocess.run(
        _ssh_via_socket(user, host, s, remote_cmd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Job submission failed on {host} (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    job_ids = re.findall(r"Submitted batch job (\d+)", result.stdout)
    if not job_ids:
        logger.warning(
            "submit_jobs_on_dardel: no job IDs parsed from sbatch output. "
            "Queue watching will fall back to user-scoped squeue."
        )
    else:
        print(f"Submitted {len(job_ids)} job(s): {', '.join(job_ids)}")

    if local_label_dir is not None and job_ids:
        state = HpcJobState(
            job_ids=job_ids,
            user=user,
            host=host,
            remote_dir=remote_label_dir,
            label_dir=local_label_dir,
        )
        state.save()
        print(f"Job IDs persisted to {state.path}")

    return job_ids


def watch_queue(
    user: str,
    host: str,
    poll_interval: int = 300,
    socket: str | None = None,
    job_ids: Optional[list[str]] = None,
) -> tuple[bool, dict[str, str]]:
    """
    Poll until all submitted Slurm jobs reach a terminal state.

    Uses ``sacct`` for terminal-state detection when job IDs are known
    (avoids the race condition where squeue returns empty between polls).
    Falls back to ``squeue -u <user>`` when no job IDs are available.

    Parameters
    ----------
    user, host        : Dardel credentials.
    poll_interval     : Seconds between polls (minimum 60).
    socket            : Caller-owned SSH ControlMaster socket path.  When
                        None this function opens and closes its own socket.
    job_ids           : Slurm job IDs to watch.  When None the function
                        falls back to watching the entire user queue via
                        squeue (legacy behaviour).

    Returns
    -------
    (finished_cleanly, state_map)
        finished_cleanly - True when all jobs reached a terminal state.
        state_map        - {job_id: final_state} dict (empty on KeyboardInterrupt
                           or when no job_ids were provided).
    """
    s = socket or _socket_path(user, host)
    poll_interval = max(60, poll_interval)
    owns_socket = socket is None

    if owns_socket:
        _open_master(user, host, s)

    scoped = bool(job_ids)
    print(
        f"Watching {'jobs ' + ', '.join(job_ids) if scoped else 'user queue'} "
        f"every {poll_interval}s via {'sacct' if scoped else 'squeue'}  —  Ctrl+C to stop"
    )
    try:
        while True:
            if scoped:
                all_done, state_map = _check_jobs_via_sacct(user, host, s, job_ids)  # type: ignore[arg-type]
                if all_done:
                    failed = [
                        jid for jid, st in state_map.items()
                        if st not in {"COMPLETED"}
                    ]
                    if failed:
                        logger.warning(
                            "Some jobs finished in non-COMPLETED state: %s",
                            {jid: state_map[jid] for jid in failed},
                        )
                        print(f"WARNING  {len(failed)} job(s) did not COMPLETE: "
                              + ", ".join(f"{j}={state_map[j]}" for j in failed))
                    else:
                        print(f"All {len(job_ids)} job(s) COMPLETED.")
                    return True, state_map
                from collections import Counter
                counts = Counter(state_map.values())
                print("  jobs: " + "  ".join(f"{st}={n}" for st, n in sorted(counts.items())))
            else:
                check_cmd = _ssh_via_socket(user, host, s, f"squeue -u {user} -h")
                result = subprocess.run(check_cmd, text=True, capture_output=True)
                if not result.stdout.strip():
                    print("Queue is empty.")
                    return True, {}
                print(result.stdout.strip())

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")
        return False, {}
    finally:
        if owns_socket:
            _close_master(user, host, s)


def sync_outputs_from_dardel(
    local_label_dir: Path,
    user: str,
    host: str,
    remote_label_dir: str,
    socket: str | None = None,
    dardel_cfg: dict | None = None,
) -> None:
    """
    Rsync OUTCAR/vasprun.xml/OSZICAR back from Dardel, with automatic retry.

    Retry schedule is read from *dardel_cfg* (label.dardel sub-dict) with
    these optional keys (all in seconds):

        sync_retry_delays   : list[int]  – per-attempt waits before going hourly
                                           default: [60, 300, 900, 1800]
        sync_retry_interval : int        – interval once the list is exhausted
                                           default: 3600  (1 h)
        sync_retry_deadline : int        – give up after this total wait
                                           default: 86400 (24 h)

    Example YAML::

        label:
          dardel:
            sync_retry_delays: [60, 300, 900, 1800]
            sync_retry_interval: 3600
            sync_retry_deadline: 86400

    Raises
    ------
    SyncRetryExhausted
        If rsync keeps failing until the deadline is reached.
    """
    retry_delays, retry_interval, retry_deadline = _retry_settings(dardel_cfg or {})

    s = socket or _socket_path(user, host)

    cmd = _rsync_via_socket(
        s,
        f"{user}@{host}:{remote_label_dir}/",
        f"{local_label_dir}/",
        [
            "--update",
            "--include=*/",
            "--include=OUTCAR",
            "--include=vasprun.xml",
            "--include=OSZICAR",
            "--exclude=*",
        ],
    )

    print(f"=== Syncing outputs from {host} ===")

    total_waited = 0
    attempt = 0

    while True:
        # Re-open a fresh master before every attempt; _open_master() removes
        # any stale socket first so a dropped connection never silently blocks.
        _open_master(user, host, s)
        try:
            exit_code = run_command(cmd)
        finally:
            _close_master(user, host, s)

        if exit_code in _RSYNC_OK:
            return

        attempt += 1

        if attempt <= len(retry_delays):
            wait = retry_delays[attempt - 1]
        else:
            wait = retry_interval

        if total_waited + wait >= retry_deadline:
            raise SyncRetryExhausted(
                f"rsync sync-back failed (exit {exit_code}) — SSH connection lost or "
                f"remote path missing. Gave up after "
                f"{total_waited // 3600:.1f} h of retries ({attempt} attempt(s))."
            )

        wait_min = wait / 60
        now = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[{now}] rsync sync-back failed (exit {exit_code}), attempt {attempt}. "
            f"Retrying in {wait_min:.0f} min "
            f"(total waited so far: {total_waited // 60:.0f} min)."
        )
        logger.warning(msg)
        print(f"WARNING  {msg}")
        time.sleep(wait)
        total_waited += wait


def submit_label_jobs(
    label_result,
    config: dict,
    on_submitted=None,
) -> tuple[bool, dict[str, str]]:
    """
    Full HPC labelling workflow:
      open master -> sync inputs -> submit (persist job IDs) -> watch queue -> sync back

    Parameters
    ----------
    label_result  : LabelResult with .label_root set.
    config        : Pipeline config dict.
    on_submitted  : Optional zero-arg callback invoked *after* sbatch returns
                    and job IDs are persisted but *before* the watcher blocks.
                    Use this hook to start background work (e.g. evaluation)
                    that should run concurrently with the HPC jobs.

    Returns
    -------
    (finished_cleanly, state_map) — forwarded from watch_queue.
    """
    label_cfg        = config["label"]
    dardel_cfg       = label_cfg["dardel"]
    user             = dardel_cfg["user"]
    host             = dardel_cfg.get("host", "dardel.pdc.kth.se")
    remote_root      = dardel_cfg["remote_root"]
    poll             = dardel_cfg.get("poll_interval", 60)

    from pathlib import Path as _Path
    runs_root        = _Path(config.get("runs_root", "runs"))
    label_subdir     = label_cfg["output_subdir"]
    local_label_dir  = label_result.label_root
    remote_label_dir = f"{remote_root}/{runs_root.name}/{label_subdir}"

    socket = _socket_path(user, host)
    _open_master(user, host, socket)

    job_ids: list[str] = []
    try:
        print("\n=== Syncing inputs to Dardel ===")
        sync_inputs_to_dardel(local_label_dir, user, host, remote_label_dir, socket=socket)

        print("\n=== Submitting jobs on Dardel ===")
        job_ids = submit_jobs_on_dardel(
            user, host, remote_label_dir,
            local_label_dir=local_label_dir,
            socket=socket,
        )

        if on_submitted is not None:
            try:
                on_submitted()
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_submitted callback raised: %s", exc)

        print("\n=== Watching queue ===")
        finished, state_map = watch_queue(
            user, host,
            poll_interval=poll,
            socket=socket,
            job_ids=job_ids or None,
        )

    finally:
        _close_master(user, host, socket)

    print("\n=== Syncing outputs back ===")
    sync_outputs_from_dardel(
        local_label_dir, user, host, remote_label_dir,
        dardel_cfg=dardel_cfg,
    )
    print("Done.")
    return finished, state_map
