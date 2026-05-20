from pathlib import Path

train_path = Path("/home/saitto/mlip_pipeline_v1/datasets/converted_cfg/train.cfg")
sel_path = Path("/home/saitto/mlip_pipeline_v1/runs/gen_13/select/selected.cfg")

def read_blocks(path):
    txt = path.read_text(errors="ignore")
    parts = [p.strip() for p in txt.split("BEGIN_CFG") if p.strip()]
    return ["BEGIN_CFG\n" + p if not p.startswith("BEGIN_CFG") else p for p in parts]

train_blocks = read_blocks(train_path)
sel_blocks = read_blocks(sel_path)

train_set = set(train_blocks)
new_blocks = [b for b in sel_blocks if b not in train_set]
dup_blocks = [b for b in sel_blocks if b in train_set]

print("train blocks:", len(train_blocks))
print("selected blocks:", len(sel_blocks))
print("new selected blocks:", len(new_blocks))
print("duplicate selected blocks:", len(dup_blocks))

for i, b in enumerate(new_blocks[:3], 1):
    print(f"\nNEW BLOCK {i}\n{b[:800]}")