# mlip-pipeline

Minimal Python package skeleton for:
- fitting a potential with MLIP-3
- generating LAMMPS exploration runs across temperatures

## Commands

```bash
mlip-pipeline fit --config configs/fit.yaml
mlip-pipeline explore --config configs/explore.yaml
mlip-pipeline run --fit-config configs/fit.yaml --explore-config configs/explore.yaml
```
