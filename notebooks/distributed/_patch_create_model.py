"""
Replaces create_model() in all 5 distributed notebooks with a clean
implementation that avoids ModuleValidator.fix() entirely.

Root cause: Opacus ModuleValidator.fix() calls clone_module() which uses
torch.load internally. PyTorch 2.6+ changed the default to weights_only=True,
breaking the clone. The monkey-patch workaround also failed because Python
lambdas capture variables by reference, causing infinite recursion when the
patch was applied more than once.

Solution: implement BatchNorm→GroupNorm replacement directly (recursive walk)
without any torch.load involvement. This is exactly what ModuleValidator.fix()
does internally, minus the broken clone step.
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

# ── Replacement code ──────────────────────────────────────────────────────────
# fix_for_opacus replaces BatchNorm2d with GroupNorm in-place (no clone needed)
# and copies the learned affine parameters so no information is lost.
FIX_HELPER = """\
def fix_for_opacus(module):
    \"\"\"Replace BatchNorm2d with GroupNorm recursively (Opacus compatibility).
    Avoids ModuleValidator.fix() which internally uses torch.load and breaks
    with PyTorch 2.6+ (weights_only=True default change).\"\"\"
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = min(32, num_channels)
            while num_channels % num_groups != 0:
                num_groups -= 1
            gn = nn.GroupNorm(num_groups, num_channels,
                              eps=child.eps, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            fix_for_opacus(child)
    return module\
"""

CLEAN_CREATE_MODEL = """\
def create_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 2),
    )
    return fix_for_opacus(model)\
"""

# Regex that matches the create_model function from 'def create_model():' through
# its closing line, including any previously applied torch.load patches.
CREATE_MODEL_RE = re.compile(
    r"def create_model\(\):.*?(?=\ndef |\Z)",
    re.DOTALL,
)


def patch_notebook(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def create_model():" not in src:
            continue

        # Replace the whole create_model function
        new_src, n = CREATE_MODEL_RE.subn(
            FIX_HELPER + "\n\n" + CLEAN_CREATE_MODEL,
            src,
        )
        if n == 0:
            print(f"  WARNING: regex found no match in {os.path.basename(path)}")
            continue

        # Also remove any leftover ModuleValidator import (no longer needed)
        # Keep it if present — it's harmless and validate() is still useful
        cell["source"] = [line + "\n" for line in new_src.splitlines()]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        patched = True
        break   # only one create_model per notebook

    if patched:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched: {os.path.basename(path)}")
    else:
        print(f"SKIPPED (no create_model found): {os.path.basename(path)}")


if __name__ == "__main__":
    for fname in NOTEBOOKS:
        patch_notebook(os.path.join(NOTEBOOK_DIR, fname))
