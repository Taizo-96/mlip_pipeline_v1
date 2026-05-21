from __future__ import annotations

import logging
import time
import subprocess
from pathlib import Path
from mlip_pipeline.utils.shell import run_command

logger = logging.getLogger(__name__)

# Only these files are synced to Dardel — outputs stay local until sync-back
VASP_INPUT_FILES = {"POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh"}

# rsync exit codes that are non-fatal (partial transfer warnings)
_RSYNC_OK = {0, 23, 24}

# ---------------------------------------------------------------------------
# Retry schedule for sync_outputs_from_dardel
# ---------------------------------------------------------------------------
# First four explicit waits (seconds), then hourly until the 24-h deadline.
_RETRY_DELAYS   = [60, 300, 900, 1800]   # 1 min, 5 min, 15 min, 30 min
_RETRY_INTERVAL = 3_600                   # 1 h between subsequent attempts
_RETRY_DEADLINE = 24 * 3_600             # stop retrying after 24 h of total wait


class SyncRetryExhausted(RuntimeError):
    """Raised when rsync sync-back keeps failing for 24 h straight."""


def _socket_path(user: str, host: str) -> str:
    return f"/tmp/ssh_ctrl_{user}@{host}"


def _open_master(user: str, host: str, socket: str) -> None:
    """Open a persistent SSH ControlMaster socket."""
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
    socket: str | None = None,
) -> None:
    """Submit VASP jobs via sbatch on Dardel."""
    s = socket or _socket_path(user, host)
    remote_cmd = (
        f"find {remote_label_dir} -name job.sh | sort | while read job; do "
        f"sbatch --chdir=$(dirname $(realpath $job)) $job; done"
    )
    exit_code = run_command(_ssh_via_socket(user, host, s, remote_cmd))
    if exit_code != 0:
        raise RuntimeError(f"Job submission failed on {host} (exit {exit_code})")


def watch_queue(
    user: str,
    host: str,
    poll_interval: int = 300,
    socket: str | None = None,
) -> bool:
    """
    Poll squeue until all jobs finish.

    If *socket* is provided the caller owns the master connection lifecycle
    and this function will NOT open or close it.
    If *socket* is None this function manages its own master connection
    (legacy / standalone use).
    """
    s = socket or _socket_path(user, host)
    poll_interval = max(60, poll_interval)
    owns_socket = socket is None

    if owns_socket:
        _open_master(user, host, s)

    print(f"Watching queue every {poll_interval}s  —  Ctrl+C to stop")
    try:
        while True:
            check_cmd = _ssh_via_socket(user, host, s, f"squeue -u {user} -h")
            result = subprocess.run(check_cmd, text=True, capture_output=True)
            if not result.stdout.strip():
                print("Queue is empty.")
                return True
            print(result.stdout.strip())
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return False
    finally:
        if owns_socket:
            _close_master(user, host, s)


def sync_outputs_from_dardel(
    local_label_dir: Path,
    user: str,
    host: str,
    remote_label_dir: str,
    socket: str | None = None,
) -> None:
    """
    Rsync OUTCAR/vasprun.xml/OSZICAR back from Dardel, with automatic retry.

    Retry schedule (cumulative wait before giving up at 24 h):
        attempt 1 → wait  1 min
        attempt 2 → wait  5 min
        attempt 3 → wait 15 min
        attempt 4 → wait 30 min
        attempt 5+ → wait  1 h  (repeated until 24 h total elapsed)

    Raises
    ------
    SyncRetryExhausted
        If rsync keeps failing for 24 h of cumulative waiting.
    """
    # When the socket is externally managed (e.g. from submit_label_jobs) the
    # master connection may have dropped. Re-open it fresh for the sync-back so
    # rsync always has a live multiplexed channel.
    owned_socket = socket is None
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
        # Re-open a fresh master before every attempt so a dropped connection
        # from the watch_queue phase does not permanently block sync-back.
        _open_master(user, host, s)
        try:
            exit_code = run_command(cmd)
        finally:
            _close_master(user, host, s)

        if exit_code in _RSYNC_OK:
            return  # success

        attempt += 1

        # Determine wait for this attempt (follow the schedule, then hourly)
        if attempt <= len(_RETRY_DELAYS):
            wait = _RETRY_DELAYS[attempt - 1]
        else:
            wait = _RETRY_INTERVAL

        if total_waited + wait >= _RETRY_DEADLINE:
            raise SyncRetryExhausted(
                f"rsync sync-back failed (exit {exit_code}) — SSH connection lost or "
                f"remote path missing. Gave up after "
                f"{total_waited // 3600:.1f} h of retries ({attempt} attempt(s))."
            )

        wait_min = wait / 60
        msg = (
            f"rsync sync-back failed (exit {exit_code}), attempt {attempt}. "
            f"Retrying in {wait_min:.0f} min "
            f"(total waited so far: {total_waited // 60:.0f} min)."
        )
        logger.warning(msg)
        print(f"WARNING  {msg}")
        time.sleep(wait)
        total_waited += wait


def submit_label_jobs(label_result, config: dict) -> None:
    """
    Full HPC labelling workflow:
      open master → sync inputs → submit → watch queue → sync back → close master

    sync_outputs_from_dardel manages its own socket lifecycle internally so
    that a dropped master from watch_queue never blocks the sync-back.
    """
    label_cfg        = config["label"]
    dardel_cfg       = label_cfg["dardel"]
    user             = dardel_cfg["user"]
    host             = dardel_cfg.get("host", "dardel.pdc.kth.se")
    remote_root      = dardel_cfg["remote_root"]
    poll             = dardel_cfg.get("poll_interval", 60)

    runs_root        = Path(config.get("runs_root", "runs"))
    label_subdir     = label_cfg["output_subdir"]
    local_label_dir  = label_result.label_root
    remote_label_dir = f"{remote_root}/{runs_root}/{label_subdir}"

    socket = _socket_path(user, host)
    _open_master(user, host, socket)

    try:
        print("\n=== Syncing inputs to Dardel ===")
        sync_inputs_to_dardel(local_label_dir, user, host, remote_label_dir, socket=socket)

        print("\n=== Submitting jobs on Dardel ===")
        submit_jobs_on_dardel(user, host, remote_label_dir, socket=socket)

        print("\n=== Watching queue ===")
        watch_queue(user, host, poll_interval=poll, socket=socket)

    finally:
        # Always close the watch-phase master before sync-back,
        # which opens its own fresh connection per attempt.
        _close_master(user, host, socket)

    # sync_outputs_from_dardel opens/closes its own master on every attempt
    print("\n=== Syncing outputs back ===")
    sync_outputs_from_dardel(local_label_dir, user, host, remote_label_dir)
    print("Done.")
