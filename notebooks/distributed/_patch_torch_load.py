"""
Fixes two issues in all distributed notebooks:

1. torch.load() calls that load checkpoints (dicts with mixed Python objects +
   tensors) need weights_only=False explicitly. PyTorch 2.6 changed the default
   to True, which rejects anything that isn't pure tensors.

2. In hospital notebooks the download cell created the model AFTER torch.load,
   so if torch.load failed, model was never defined and the next cell crashed
   with NameError. Fixed by creating the model first, then loading weights.
"""
import json, os, re

NOTEBOOK_DIR = os.path.dirname(__file__)

NOTEBOOKS = [
    "fl_initializer.ipynb",
    "hospital_0_training.ipynb",
    "hospital_1_training.ipynb",
    "hospital_2_training.ipynb",
    "fl_aggregator.ipynb",
]

# ── Fix 1: add weights_only=False to checkpoint torch.load calls ──────────────
# Matches:  torch.load(<anything>, map_location='cpu')
# but NOT lines that already have weights_only
LOAD_RE = re.compile(
    r"torch\.load\(([^)]+map_location=['\"]cpu['\"][^)]*)\)",
)

def add_weights_only(m):
    args = m.group(1)
    if "weights_only" in args:
        return m.group(0)   # already set, leave alone
    return f"torch.load({args.rstrip()}, weights_only=False)"


# ── Fix 2: in hospital notebooks, create model before loading checkpoint ──────
# Old order (breaks if torch.load raises):
#   ckpt  = torch.load(...)
#   model = create_model()
#   model.load_state_dict(ckpt['model_state_dict'])
#   model = model.to(device)
#
# New order:
#   model = create_model().to(device)
#   ckpt  = torch.load(...)
#   model.load_state_dict(ckpt['model_state_dict'])

OLD_LOAD_BLOCK = (
    "ckpt     = torch.load(os.path.join(art_dir, 'global_model.pth'), map_location='cpu')\n"
    "model    = create_model()\n"
    "model.load_state_dict(ckpt['model_state_dict'])\n"
    "model    = model.to(device)"
)

NEW_LOAD_BLOCK = (
    "model    = create_model().to(device)\n"
    "ckpt     = torch.load(os.path.join(art_dir, 'global_model.pth'), map_location='cpu', weights_only=False)\n"
    "model.load_state_dict(ckpt['model_state_dict'])"
)


def patch(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        # Fix 2 first (more specific, before the generic regex)
        if OLD_LOAD_BLOCK in new_src:
            new_src = new_src.replace(OLD_LOAD_BLOCK, NEW_LOAD_BLOCK)

        # Fix 1: add weights_only=False to remaining torch.load calls
        new_src = LOAD_RE.sub(add_weights_only, new_src)

        if new_src != src:
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched: {os.path.basename(path)}")
    else:
        print(f"No changes: {os.path.basename(path)}")


if __name__ == "__main__":
    for fname in NOTEBOOKS:
        patch(os.path.join(NOTEBOOK_DIR, fname))
