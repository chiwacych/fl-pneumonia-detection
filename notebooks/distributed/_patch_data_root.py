"""
Replaces the hardcoded DATA_ROOT line in all notebooks with auto-detection
code that finds the dataset regardless of how Kaggle mounted it.
"""
import json, os, re

NOTEBOOK_DIR = os.path.dirname(__file__)
NOTEBOOKS = [
    os.path.join(NOTEBOOK_DIR, "hospital_0_training.ipynb"),
    os.path.join(NOTEBOOK_DIR, "hospital_1_training.ipynb"),
    os.path.join(NOTEBOOK_DIR, "hospital_2_training.ipynb"),
    os.path.join(NOTEBOOK_DIR, "fl_aggregator.ipynb"),
    os.path.join(NOTEBOOK_DIR, "..", "05_FL_FedAvg_DP_Kaggle.ipynb"),
]

# Matches any line like:  DATA_ROOT  = '/kaggle/input/.../chest_xray'
OLD_RE = re.compile(r"^DATA_ROOT\s*=\s*'[^']*'$", re.MULTILINE)

NEW = """\
# Auto-detect dataset path (handles different Kaggle mounting conventions)
def _find_data_root():
    search_bases = [
        '/kaggle/input/chest-xray-pneumonia',
        '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia',
        '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray',
    ]
    for base in search_bases:
        if not os.path.exists(base):
            continue
        for root, dirs, _ in os.walk(base):
            if 'train' in dirs:
                train_path = os.path.join(root, 'train')
                if any(c in os.listdir(train_path) for c in ['NORMAL', 'PNEUMONIA']):
                    return root
    raise RuntimeError(
        'Chest X-ray dataset not found.\\n'
        'Add it via Add Data and search paultimothymooney/chest-xray-pneumonia'
    )
DATA_ROOT = _find_data_root()
print(f'Data root: {DATA_ROOT}')\
"""


def patch(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not OLD_RE.search(src):
            continue
        new_src = OLD_RE.sub(NEW, src)
        cell["source"] = [line + "\n" for line in new_src.splitlines()]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        patched = True

    if patched:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Patched: {os.path.basename(path)}")
    else:
        print(f"Skipped (DATA_ROOT line not found): {os.path.basename(path)}")


if __name__ == "__main__":
    for path in NOTEBOOKS:
        patch(path)
