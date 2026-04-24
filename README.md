# mlip-pipeline

Minimal Python package skeleton for:
- fitting a potential with MLIP-3
- generating LAMMPS exploration runs across temperatures

## Commands

```bash
python -m mlip_pipeline.cli download-structures --config fit/config.yaml
python -m mlip_pipeline.cli prepare-train --config fit/config.yaml
python -m mlip_pipeline.cli fit --config fit/config.yaml
python -m mlip_pipeline.cli explore --config configs/config.yaml
python -m mlip_pipeline.cli select --config configs/config.yaml
```
