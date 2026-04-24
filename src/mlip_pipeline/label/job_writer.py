from __future__ import annotations
from pathlib import Path

from mlip_pipeline.models import VaspParallelConfig


def write_job_sh(
        task_dir: Path,
        n_atoms: int,
        slurm_cfg: dict | None = None,
        sys_override: VaspParallelConfig | None = None
) -> Path:
    cfg = slurm_cfg or {}

    # Cascading logic: Override > YAML Config > Default
    partition = sys_override.partition if sys_override else cfg.get("partition", "shared")
    nodes = sys_override.nodes if sys_override else cfg.get("nodes", 1)
    ntasks = sys_override.ntasks if sys_override else cfg.get("ntasks", 32)

    # Other values not handled by the scaling logic
    account = cfg.get("account", "naiss2025-3-54")
    time = cfg.get("time", "01:00:00")
    vasp_cmd = cfg.get("vasp_cmd", "vasp_std")
    env_block = cfg.get("env_block", "# no env_block configured")
    omp = sys_override.omp_num_threads if sys_override else cfg.get("omp", 1)

    job_name = f"vasp_{task_dir.name}"

    script = f"""\
#!/bin/bash
#SBATCH -A {account}
#SBATCH -J {job_name}
#SBATCH -t {time}
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -p {partition}

{env_block}

mkdir -p output
export OMP_NUM_THREADS={omp}
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

LOGFILE="output/simulation_log.txt"
echo "Starting simulation at $(date)" >> $LOGFILE
start_time=$(date +%s)

{{ time srun --hint=nomultithread {vasp_cmd} > output/vasp.log 2>&1; }} 2>> $LOGFILE
vasp_exit=$?

end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

if [ $vasp_exit -eq 0 ]; then
    echo "Simulation completed successfully." >> $LOGFILE
else
    echo "Error in simulation. Check output/vasp.log for details." >> $LOGFILE
fi

echo "VASP run ended at:   $(date)" >> $LOGFILE
echo "Total runtime: ${{hours}}h ${{minutes}}m ${{seconds}}s" >> $LOGFILE
echo "------------------------------------------------" >> $LOGFILE
"""
    job_path = task_dir / "job.sh"
    job_path.write_text(script)
    return job_path