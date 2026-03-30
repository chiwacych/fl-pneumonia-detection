"""
Patches create_model() in all 5 distributed notebooks to work with
PyTorch 2.6+ where torch.load defaults to weights_only=True.

Opacus ModuleValidator.fix() uses torch.load internally to deep-copy
the module (clone_module). With weights_only=True this raises
UnpicklingError. We temporarily set weights_only=False for that call only.
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

PATCH_FIX = (
    "    # PyTorch 2.6+ fix: ModuleValidator.fix() uses torch.load internally\n"
    "    # to clone the module. The new weights_only=True default breaks this.\n"
    "    # We restore the old behaviour just for this call.\n"
    "    _orig_load = torch.load\n"
    "    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})\n"
)

# Two patterns depending on how the notebook calls ModuleValidator.fix()
PATTERNS = [
    # initializer: model = ModuleValidator.fix(model)
    (
        "    model = ModuleValidator.fix(model)",
        PATCH_FIX +
        "    model = ModuleValidator.fix(model)\n"
        "    torch.load = _orig_load"
    ),
    # hospital / aggregator: return ModuleValidator.fix(model)
    (
        "    return ModuleValidator.fix(model)",
        PATCH_FIX +
        "    model = ModuleValidator.fix(model)\n"
        "    torch.load = _orig_load\n"
        "    return model"
    ),
]

for fname in NOTEBOOKS:
    path = os.path.join(NOTEBOOK_DIR, fname)
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in PATTERNS:
            if old in src:
                src = src.replace(old, new)
                patched = True
                break
        if patched:
            cell["source"] = [line + "\n" for line in src.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

    if patched:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched: {fname}")
    else:
        print(f"WARNING: pattern not found in {fname}")
