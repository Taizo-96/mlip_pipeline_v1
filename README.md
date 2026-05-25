# mlip_pipeline_v1

An automated active-learning pipeline for fitting **Moment Tensor Potentials (MTP)**
using [MLIP-3](https://gitlab.com/ashapeev/mlip-3), driven by LAMMPS molecular-dynamics
exploration and VASP DFT labelling.

Designed for the **Pb–Fe / liquid-lead** system, but straightforwardly adaptable to any
mono- or multi-element system by editing the YAML configuration file.

---

## Architecture overview

```
mlip_pipeline_v1/
├── configs/
│   └── Pb_loop.yaml          # Single source of truth for all settings
├── MTP_templates/            # MLIP-3 initial .mtp template files
├── scripts/
│   └── prepare_initial_training.py   # One-time DeePMD→CFG conversion (pre-loop)
└── src/mlip_pipeline/
    ├── cli.py                # Typer CLI entry-point
    ├── config.py             # YAML loading and path resolution
    ├── models.py             # Dataclasses + manifest I/O for every step result
    ├── checks.py             # Sanity checks (training-set size, etc.)
    ├── data/
    │   └── outcar_to_cfg.py  # VASP OUTCAR → MLIP-3 CFG conversion
    ├── fit/                  # mlp train wrapper
    ├── explore/              # LAMMPS input generation + runner
    ├── select/               # mlp select_add wrapper
    ├── label/                # VASP task preparation and local runner
    ├── io/
    │   └── dardel.py         # SSH/SFTP helpers for Dardel HPC
    ├── integrations/
    │   ├── materials_project.py   # mp-api structure download
    │   └── lammps_export.py       # Structure → LAMMPS data file
    ├── evaluate/             # MTP error metrics + parity plots
    ├── loop/
    │   └── runner.py         # Multi-generation loop orchestrator
    └── utils/
        ├── logging.py        # Rich-aware logging + step_timed() helper
        ├── fs.py             # File-system helpers (ensure_dir, write_json)
        └── shell.py          # Subprocess helpers
```

### Active-learning loop steps

| Step | What it does |
|---|---|
| **fit** | Calls `mlp train` to produce a new `.almtp` potential |
| **explore** | Runs LAMMPS NPT/NVT MD with on-the-fly γ grading |
| **select** | Calls `mlp select_add` to choose maximally-diverse candidates |
| **label** | Prepares VASP POSCAR/INCAR/KPOINTS task directories |
| **label_hpc** | Syncs tasks to Dardel, submits SLURM jobs, watches queue |
| **label_local** | Runs VASP locally via `mpirun` (no scheduler) |
| **convert** | Parses OUTCARs and appends new CFGs to the training set |
| **evaluate** | Runs `mlp calculate_efs` and generates parity/RMSE plots |

Each completed step writes a `*_manifest.json` inside the generation directory;
these manifests are the crash-recovery mechanism — the loop resumes from the
last incomplete step.

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Language runtime |
| MLIP-3 (`mlp` binary) | MTP training and selection |
| LAMMPS (with MLIP-3 interface) | MD exploration |
| VASP | DFT single-point labelling |
| numpy ≥ 1.24 | CFG array I/O |
| PyYAML ≥ 6.0 | Config parsing |
| typer ≥ 0.9 | CLI framework |
| rich ≥ 13.0 | Terminal output (optional but recommended) |
| mp-api ≥ 0.41 | Materials Project structure download |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Taizo-96/mlip_pipeline_v1.git
cd mlip_pipeline_v1

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install the package and all Python dependencies
pip install -e .

# Alternatively, install from requirements.txt only (no editable install):
# pip install -r requirements.txt
```

> **Note:** MLIP-3, LAMMPS, and VASP are external binaries compiled separately.
> Set their paths in `configs/Pb_loop.yaml` under the `bins:` section.

### Transferring to a new machine

The Python environment is fully described by `pyproject.toml` (minimal bounds)
and `requirements.txt` (pinned versions for reproducibility):

```bash
# On the new machine:
pip install -r requirements.txt
pip install -e .
```

For a fully-locked reproducible environment (recommended for HPC):

```bash
pip install pip-tools
pip-compile pyproject.toml -o requirements-lock.txt
# Commit requirements-lock.txt and use it on the target machine:
pip install -r requirements-lock.txt
```

---

## Configuration

All settings live in a single YAML file (`configs/Pb_loop.yaml`).
The file supports `{{ key }}` template interpolation.

Key sections:

```yaml
project_root: "/path/to/project"      # absolute path on your machine
paths:
  data_root:       "/path/to/data"    # DeePMD system dirs (for initial CFG prep)
  runs_root:       "{{ project_root }}/runs"
  datasets_root:   "{{ project_root }}/datasets"
bins:
  mlp:    "mlp"                       # path to mlp binary
  lammps: "/path/to/lmp_mpi"          # path to LAMMPS binary
  vasp:   "vasp_std"                  # VASP binary name
```

Set `MP_API_KEY` as an environment variable for Materials Project downloads:

```bash
export MP_API_KEY="your_key_here"
```

---

## Workflow

### 1. Prepare initial training set (one time)

If your initial data is in DeePMD `.npy` format:

```bash
python scripts/prepare_initial_training.py \
    --data-root /path/to/deepmd/systems \
    --glob "Pb*" \
    --output-dir /path/to/datasets/pb_cfg \
    --merge-name train.cfg
```

This produces `pb_cfg/train.cfg` which you reference in `fit.train_cfg` of the YAML.

### 2. Run the active-learning loop

```bash
# Run generations 0 through 10 (auto-resume if interrupted):
mlip-pipeline run-loop configs/Pb_loop.yaml --start-gen 0 --end-gen 10

# Run a single generation:
mlip-pipeline run-gen configs/Pb_loop.yaml 3

# Force-redo specific steps within a generation:
mlip-pipeline run-gen configs/Pb_loop.yaml 3 --force --only fit,evaluate
```

### 3. Run individual steps manually

```bash
mlip-pipeline fit       --config configs/Pb_loop.yaml
mlip-pipeline explore   --config configs/Pb_loop.yaml
mlip-pipeline select    --config configs/Pb_loop.yaml
mlip-pipeline label     --config configs/Pb_loop.yaml
mlip-pipeline convert-cfg --config configs/Pb_loop.yaml
mlip-pipeline evaluate  --config configs/Pb_loop.yaml --gen 5
```

### 4. HPC (Dardel) labelling workflow

```bash
# Prepare VASP task directories and sync to Dardel:
mlip-pipeline label           --config configs/Pb_loop.yaml
mlip-pipeline sync-to-remote  --config configs/Pb_loop.yaml
mlip-pipeline submit-remote   --config configs/Pb_loop.yaml

# Watch the queue and sync back when done:
mlip-pipeline watch-remote    --config configs/Pb_loop.yaml
mlip-pipeline sync-from-remote --config configs/Pb_loop.yaml

# Convert OUTCARs to CFG:
mlip-pipeline convert-cfg     --config configs/Pb_loop.yaml
```

### 5. Evaluation and plotting

```bash
# Evaluate a single generation:
mlip-pipeline evaluate --config configs/Pb_loop.yaml --gen 7

# Evaluate all generations:
mlip-pipeline evaluate --config configs/Pb_loop.yaml --gens all

# Re-generate plots from cached data (no MTP inference):
mlip-pipeline plot-eval --config configs/Pb_loop.yaml

# Side-by-side parity comparison across generations:
mlip-pipeline compare-parity --config configs/Pb_loop.yaml --gens 0,5,10

# Loop-summary convergence plots:
mlip-pipeline plot-loop --config configs/Pb_loop.yaml
```

---

## State management

Each generation stores its progress in `runs/gen_NN/state.json`.  If a run
crashes, re-invoking the same command resumes from the last incomplete step.

```bash
# Inspect current state of generation 5:
mlip-pipeline show-state --config configs/Pb_loop.yaml --gen 5

# Reset specific steps (e.g. to re-run fit and evaluate):
mlip-pipeline reset-steps --config configs/Pb_loop.yaml --gen 5 --steps fit,evaluate
```

---

## Output artefacts

After a full loop, `runs/` has the following structure:

```
runs/
├── gen_00/
│   ├── state.json             # generation state (crash-recovery)
│   ├── config_snapshot.yaml   # config used for this generation
│   ├── fit/
│   │   ├── Pb16.almtp         # trained MTP
│   │   ├── fit_manifest.json  # provenance: command, paths, timestamp
│   │   └── eval/
│   │       ├── eval_manifest.json
│   │       └── *.png          # parity and RMSE plots
│   ├── explore/
│   │   └── explore_manifest.json
│   ├── select/
│   │   ├── selected.cfg
│   │   └── selection_manifest.json
│   ├── label/
│   │   ├── task.*/            # VASP input directories
│   │   └── label_manifest.json
│   └── convergence.json
├── gen_01/ ...
├── loop_manifest.json         # whole-loop summary
└── loop_summary/
    └── *.png                  # loop-level convergence plots
```

`datasets/converted_cfg/` accumulates the growing training set across all
generations.

---

## Design notes

### Manifest JSON (no duplication)

Every step result dataclass (`FitResult`, `ExploreResult`, `SelectionResult`, etc.)
has a unified `save_manifest()` / `load_manifest()` interface that writes a single
`*_manifest.json` per step.  There is no separate `metadata.json` — all provenance
(command, paths, timestamps) lives in the manifest.

### Step timing

Every step logs its start time, end time, and elapsed duration via the
`step_timed()` context manager in `utils/logging.py`:

```
[09:14:02]  ▶ fit  started
[09:16:55]  ✓ fit  finished  (elapsed: 2m 52.8s)
```

### Replicate schedule

If `select` finds 0 candidates in the current supercell, the pipeline
automatically scales the simulation box to the next tier in
`explore.replicate_schedule` and re-runs explore + select.  Convergence is
declared when 0 candidates are found at all tiers.
