from __future__ import annotations

from pathlib import Path
import glob
import os
import numpy as np

from mlip_pipeline.models import PrepareTrainResult
from mlip_pipeline.utils.fs import ensure_dir


def _write_cfg(system_dir: str | Path, out_path: str | Path) -> None:
    system_dir = Path(system_dir)
    out_path = Path(out_path)
    base = system_dir / "set.000"

    coord = np.load(base / "coord.npy")
    box = np.load(base / "box.npy")
    energy = np.load(base / "energy.npy")
    force = np.load(base / "force.npy")
    types = np.loadtxt(system_dir / "type.raw", dtype=int)

    nframes = coord.shape[0]
    natoms = len(types)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(nframes):
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"   {natoms}\n")
            f.write(" Supercell\n")
            cell = box[i].reshape(3, 3)
            for row in cell:
                f.write(
                    "  "
                    + "  ".join(f"{float(x):20.12f}" for x in row)
                    + "\n"
                )
            f.write(
                " AtomData:  id type       cartes_x      cartes_y      cartes_z"
                "           fx          fy          fz\n"
            )
            xyz = coord[i].reshape(natoms, 3)
            frc = force[i].reshape(natoms, 3)
            for j in range(natoms):
                atype = int(types[j])
                f.write(
                    f"{j+1:13d}{atype:5d}"
                    + "".join(f"{float(x):14.6f}" for x in xyz[j])
                    + "".join(f"{float(x):12.6f}" for x in frc[j])
                    + "\n"
                )
            e = float(
                energy[i, 0] if getattr(energy[i], "shape", ()) else energy[i]
            )
            f.write(" Energy\n")
            f.write(f"{e:20.12f}\n")
            f.write(" PlusStress:  xx yy zz yz xz xy\n")
            f.write("  0.0 0.0 0.0 0.0 0.0 0.0\n")
            f.write(" Feature   conf_id\n")
            f.write(f"   {i}\n")
            f.write("END_CFG\n\n")


def prepare_training_cfgs(config: dict, resolved_paths: dict) -> PrepareTrainResult:
    train_cfg = config["training"]
    input_glob = train_cfg.get("input_glob", "Pb*")
    absolute_glob = str(resolved_paths["data_root"] / input_glob)
    systems = sorted([p for p in glob.glob(absolute_glob) if os.path.isdir(p)])
    if not systems:
        raise FileNotFoundError(
            f"No system directories matched: {absolute_glob}"
        )

    output_dir = ensure_dir(
        resolved_paths["datasets_root"] / train_cfg.get("output_subdir", "pb_cfg")
    )
    generated: list[Path] = []
    for system_dir in systems:
        name = Path(system_dir).name
        out_path = output_dir / f"{name}.cfg"
        _write_cfg(system_dir, out_path)
        generated.append(out_path)

    merge_name = train_cfg.get("merge_name", "train.cfg")
    merged_cfg: Path | None = None
    if merge_name:
        merged_cfg = output_dir / merge_name
        with open(merged_cfg, "w", encoding="utf-8") as merged:
            for cfg in generated:
                text = Path(cfg).read_text(encoding="utf-8")
                merged.write(text)
                if not text.endswith("\n"):
                    merged.write("\n")

    return PrepareTrainResult(
        output_dir=output_dir, merged_cfg=merged_cfg, generated_cfgs=generated
    )