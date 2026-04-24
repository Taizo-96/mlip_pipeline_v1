from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mlip_pipeline.models import LabelResult

# Only these files are synced to Dardel — outputs stay local until sync-back
VASP_INPUT_FILES = {"POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh"}


def _run(cmd: str) -> None:
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _ssh(user: str, host: str, remote_cmd: str) -> None:
    subprocess.run(["ssh", f"{user}@{host}", remote_cmd], check=True)


def sync_inputs_to_dardel(
    local_label_dir: Path,
    user: str,
    host: str,
    remote_label_dir: str,
) -> None:
    """Rsync only VASP input files + job.sh to Dardel. Excludes outputs."""
    # Ensure remote directory exists before rsync
    _ssh(user, host, f"mkdir -p {remote_label_dir}")

    include_flags = " ".join(
        f"--include='task.*/{f}'" for f in sorted(VASP_INPUT_FILES)
    )
    _run(
        f"rsync -avz "
        f"--include='task.*/' "
        f"{include_flags} "
        f"--exclude='*' "
        f"{local_label_dir}/ {user}@{host}:{remote_label_dir}/"
    )


def submit_jobs_on_dardel(
    user: str,
    host: str,
    remote_label_dir: str,
) -> None:
    _ssh(user, host,
        f"find {remote_label_dir} -name job.sh | sort | while read job; do "
        f"sbatch --chdir=$(dirname $(realpath $job)) $job; done"
    )


def watch_queue(
    user: str,
    host: str,
    poll_interval: int = 300,
) -> None:
    socket_path = f"/tmp/ssh_ctrl_{user}@{host}"
    poll_interval = max(60, poll_interval)

    # Open one persistent SSH connection
    subprocess.Popen([
        "ssh", "-MNf",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={socket_path}",
        "-o", "ControlPersist=yes",
        f"{user}@{host}",
    ])
    time.sleep(2)  # let the master establish

    print(f"Watching queue every {poll_interval}s  —  Ctrl+C to stop")
    try:
        while True:
            result = subprocess.run(
                ["ssh",
                 "-o", "ControlMaster=no",
                 "-o", f"ControlPath={socket_path}",
                 f"{user}@{host}",
                 f"squeue -u {user} --format='%.10i %.20j %.8T %.10M %.6D %R'"],
                text=True, capture_output=True,
            )
            os.system("clear")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}]  polling every {poll_interval}s  —  Ctrl+C to stop\n")
            if result.returncode != 0:
                print(f"  SSH/squeue error:\n{result.stderr.strip()}")
            else:
                print(result.stdout.strip() or "  No jobs in queue.")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        # Close the master connection on exit
        subprocess.run([
            "ssh", "-O", "exit",
            "-o", f"ControlPath={socket_path}",
            f"{user}@{host}",
        ])
        print("SSH master connection closed.")


def sync_outputs_from_dardel(
    local_label_dir: Path,
    user: str,
    host: str,
    remote_label_dir: str,
) -> None:
    _run(
        f"rsync -avz {user}@{host}:{remote_label_dir}/ {local_label_dir}/"
    )


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



