from __future__ import annotations

import time
import subprocess
from pathlib import Path
from mlip_pipeline.utils.shell import run_command  # Use your utility

# Only these files are synced to Dardel — outputs stay local until sync-back
VASP_INPUT_FILES = {"POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh"}


def _ssh_cmd(user: str, host: str, remote_cmd: str) -> list[str]:
    """Helper to format SSH commands for run_command."""
    return ["ssh", f"{user}@{host}", remote_cmd]

def sync_inputs_to_dardel(local_label_dir: Path, user: str, host: str, remote_label_dir: str) -> None:
    """Rsync only VASP input files to Dardel."""
    # Ensure remote directory exists
    run_command(["ssh", f"{user}@{host}", f"mkdir -p {remote_label_dir}"]) #

    include_flags = [f"--include='task.*/{f}'" for f in sorted(VASP_INPUT_FILES)]
    cmd = [
        "rsync", "-avz",
        "--include='task.*/'",
        *include_flags,
        "--exclude='*'",
        f"{local_label_dir}/",
        f"{user}@{host}:{remote_label_dir}/"
    ]
    run_command(cmd) #


def submit_jobs_on_dardel(
        user: str,
        host: str,
        remote_label_dir: str,
) -> None:
    """Submits VASP jobs using the remote scheduler."""
    # Build a single robust remote command string
    remote_cmd = (
        f"find {remote_label_dir} -name job.sh | sort | while read job; do "
        f"sbatch --chdir=$(dirname $(realpath $job)) $job; done"
    )

    # Use the helper to wrap it in an SSH list and execute via your utility
    full_cmd = _ssh_cmd(user, host, remote_cmd)
    exit_code = run_command(full_cmd)

    if exit_code != 0:
        print(f"Error: Submission failed on {host}")


def watch_queue(user: str, host: str, poll_interval: int = 300) -> bool:
    """Monitors the Slurm queue using SSH multiplexing."""
    socket_path = f"/tmp/ssh_ctrl_{user}@{host}"
    poll_interval = max(60, poll_interval)

    # 1. Establish Master Connection
    # We still use Popen here because this process must stay alive in the background
    master_cmd = [
        "ssh", "-MNf",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={socket_path}",
        "-o", "ControlPersist=yes",
        f"{user}@{host}",
    ]
    subprocess.Popen(master_cmd)
    time.sleep(2)

    print(f"Watching queue every {poll_interval}s  —  Ctrl+C to stop")
    try:
        while True:
            # 2. Check Queue using utility
            # Use capture_output logic or redirect to a temp log if needed
            check_cmd = [
                "ssh", "-o", "ControlMaster=no", "-o", f"ControlPath={socket_path}",
                f"{user}@{host}", f"squeue -u {user} -h"
            ]

            # Since run_command doesn't currently return stdout text,
            # we use a temporary subprocess call or enhance run_command.
            # For "Senior" consistency, let's assume we want to see the output:
            result = subprocess.run(check_cmd, text=True, capture_output=True)

            if not result.stdout.strip():
                print("Queue is empty.")
                return True

            print(result.stdout.strip())
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        # 3. Cleanup using utility
        exit_cmd = [
            "ssh", "-O", "exit",
            "-o", f"ControlPath={socket_path}",
            f"{user}@{host}",
        ]
        run_command(exit_cmd)
        print("SSH master connection closed.")


def sync_outputs_from_dardel(
        local_label_dir: Path,
        user: str,
        host: str,
        remote_label_dir: str,
) -> None:
    """
    Syncs results back.
    Uses --update to prevent overwriting newer local files and
    --exclude to avoid pulling back massive unnecessary system files.
    """
    # Using your existing shell utility instead of raw subprocess
    cmd = [
        "rsync", "-avzu",
        "--include='*/'",
        "--include='OUTCAR'", "--include='vasprun.xml'", "--include='OSZICAR'",
        "--exclude='*'",
        f"{user}@{host}:{remote_label_dir}/",
        f"{local_label_dir}/"
    ]
    print(f"=== Syncing outputs from {host} ===")
    # run_command returns exit code
    exit_code = run_command(cmd)
    if exit_code != 0:
        print(f"Error: Sync failed with exit code {exit_code}")


def submit_label_jobs(label_result, config: dict) -> None:
    """Top-level entry point called from CLI (label-submit-dardel command)."""
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

    print("\n=== Syncing inputs to Dardel ===")
    sync_inputs_to_dardel(local_label_dir, user, host, remote_label_dir)

    print("\n=== Submitting jobs on Dardel ===")
    submit_jobs_on_dardel(user, host, remote_label_dir)

    print("\n=== Watching queue ===")
    watch_queue(user, host, poll_interval=poll)

    print("\n=== Syncing outputs back ===")
    sync_outputs_from_dardel(local_label_dir, user, host, remote_label_dir)
    print("Done.")




