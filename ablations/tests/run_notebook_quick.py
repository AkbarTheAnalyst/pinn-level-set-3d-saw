"""Run a generated notebook locally at tiny size (tests only).

google.colab is replaced by a stub, PROJECT_DIR by a temporary folder holding the
core file, saw_ablation.py and any replay files, and saw_ablation is shrunk right
after import. The notebook is run twice to check that finished runs are skipped.
"""
import json
import os
import shutil
import sys
import tempfile
import types

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.join(EXP, "..")
sys.path.insert(0, HERE)

colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda path: None)
sys.modules["google"] = types.ModuleType("google")
sys.modules["google.colab"] = colab


def run(nb_path, project):
    nb = json.load(open(nb_path, encoding="utf8"))
    ns = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = src.replace("'/content/drive/MyDrive/SAW_ablation'", repr(project))
        src = src.replace("import saw_ablation as sa", "import saw_ablation as sa\nfrom small import shrink; shrink(sa)")
        src = src.replace("device = 'cuda'", "device = 'cpu'")
        src = src.replace("print(torch.cuda.get_device_name(0))", "")
        exec(compile(src, nb_path, "exec"), ns)
    for m in [m for m in sys.modules if m.endswith("_core") or m == "saw_ablation"]:
        del sys.modules[m]


for name in sys.argv[1:]:
    path = os.path.join(EXP, "notebooks", name + ".ipynb")
    bench = name.split("_")[-1] if name.split("_")[-1].endswith("3D") else name.split("_")[-2]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(ROOT, bench.lower(), bench.lower() + "_core.py"), tmp)
        shutil.copy(os.path.join(EXP, "saw_ablation.py"), tmp)
        for s in (42, 43, 44):
            f = os.path.join(ROOT, "results", "weight_history", f"{bench}_saw_whist_q0.95_seed{s}.npz")
            if os.path.exists(f):
                shutil.copy(f, tmp)
        print(f"\n######## {name}: first pass")
        run(path, tmp)
        print(f"######## {name}: second pass (should skip everything)")
        run(path, tmp)
