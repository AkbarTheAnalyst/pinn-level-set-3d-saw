"""Generate one Colab notebook per experiment into notebooks/.

Each notebook imports <bench>_core.py and saw_ablation.py from
MyDrive/SAW_ablation/ and saves its results to MyDrive/SAW_ablation/<experiment>/.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebooks")
DRIVE = "/content/drive/MyDrive/SAW_ablation"

EXPERIMENTS = [
    ("replay_ZD3D", "ZD3D", "SAW's own weight schedule replayed, no gate",
     [dict(mode="replay", seed=s) for s in (42, 43, 44)], 90),
    ("replay_RV3D", "RV3D", "SAW's own weight schedule replayed, no gate",
     [dict(mode="replay", seed=s) for s in (42, 43, 44)], 120),
    ("gate_diag_ZD3D", "ZD3D", "SAW with diagnostics: what the gate removes during training",
     [dict(mode="saw", seed=s, q=0.95, diag=True) for s in (42, 43, 44)], 120),
    ("gate_diag_RV3D", "RV3D", "SAW with diagnostics: what the gate removes during training",
     [dict(mode="saw", seed=s, q=0.95, diag=True) for s in (42, 43, 44)], 150),
    ("gate_diag_TR3D", "TR3D", "SAW with diagnostics on a control (exact solution is an SDF)",
     [dict(mode="saw", seed=42, q=0.95, diag=True)], 20),
    ("gate_diag_RO3D", "RO3D", "SAW with diagnostics on a control (exact solution is an SDF)",
     [dict(mode="saw", seed=42, q=0.95, diag=True)], 20),
    ("random_drop_ZD3D", "ZD3D", "Drop a random 5% of points instead of the top 5%",
     [dict(mode="random", seed=42, q=0.95)], 40),
    ("random_drop_RV3D", "RV3D", "Drop a random 5% of points instead of the top 5%",
     [dict(mode="random", seed=42, q=0.95)], 50),
    ("seeds_ZD3D_SAW", "ZD3D", "Two more seeds for SAW",
     [dict(mode="saw", seed=s, q=0.95) for s in (45, 46)], 75),
    ("seeds_ZD3D_fixed", "ZD3D", "Two more seeds for the fixed weight 1e-3",
     [dict(mode="fixed", seed=s, w_eik=1e-3) for s in (45, 46)], 60),
    ("q099_ZD3D", "ZD3D", "Gate at q = 0.99 instead of 0.95",
     [dict(mode="saw", seed=42, q=0.99)], 40),
    ("q099_RV3D", "RV3D", "Gate at q = 0.99 instead of 0.95",
     [dict(mode="saw", seed=42, q=0.99)], 50),
    ("q090_ZD3D", "ZD3D", "Gate at q = 0.90 instead of 0.95",
     [dict(mode="saw", seed=42, q=0.90)], 40),
    ("q090_RV3D", "RV3D", "Gate at q = 0.90 instead of 0.95",
     [dict(mode="saw", seed=42, q=0.90)], 50),
    ("q097_ZD3D", "ZD3D", "Gate at q = 0.97, matching the ~3% kink fraction",
     [dict(mode="saw", seed=42, q=0.97)], 40),
    ("q071_RV3D", "RV3D", "Gate at q = 0.71, matching the ~29% departure fraction",
     [dict(mode="saw", seed=42, q=0.71)], 50),
    ("q090_ZD3D_s43", "ZD3D", "Gate at q = 0.90, seed 43",
     [dict(mode="saw", seed=43, q=0.90)], 40),
    ("q090_ZD3D_s44", "ZD3D", "Gate at q = 0.90, seed 44",
     [dict(mode="saw", seed=44, q=0.90)], 40),
    ("q090_RV3D_s43", "RV3D", "Gate at q = 0.90, seed 43",
     [dict(mode="saw", seed=43, q=0.90)], 50),
    ("q090_RV3D_s44", "RV3D", "Gate at q = 0.90, seed 44",
     [dict(mode="saw", seed=44, q=0.90)], 50),
    ("gateoff_diag_ZD3D", "ZD3D", "Gate off (q = 1) with diagnostics: gradient norms without the gate",
     [dict(mode="saw", seed=42, q=1.0, diag=True)], 50),
    ("gateoff_diag_RV3D", "RV3D", "Gate off (q = 1) with diagnostics: gradient norms without the gate",
     [dict(mode="saw", seed=42, q=1.0, diag=True)], 65),
]


def lines(src):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(src)}


def notebook(name, bench, purpose, runs, minutes):
    core = bench.lower() + "_core"
    replay = any(r["mode"] == "replay" for r in runs)
    files = [f"`{core}.py`", "`saw_ablation.py`"]
    if replay:
        files += [f"`{bench}_saw_whist_q0.95_seed{s}.npz`" for s in (42, 43, 44)]
    if bench == "RV3D":
        files.append("`RV3D_reference_n201_T2.npz` (optional; computed and saved there if missing)")

    config = f"RUNS = {runs!r}\nCSV_PATH = f'{{OUT}}/{{EXP}}.csv'\ngrid = core.Grid(n=sa.GRID_N)\nPHI_E = None\n"
    if replay:
        config += (f"REPLAY = {{s: np.load(f'{{PROJECT_DIR}}/{bench}_saw_whist_q0.95_seed{{s}}.npz')['w_hist']\n"
                   f"          for s in (42, 43, 44)}}\n")
    if bench == "RV3D":
        config += ("times, _ = core.snapshot_times(sa.BENCH[BENCH]['T'], sa.N_SNAP)\n"
                   "PHI_E = core.reference_fields(grid, times, sa.BENCH[BENCH]['T'], device, nsteps=sa.RK4_STEPS,\n"
                   "                              cache_path=f'{PROJECT_DIR}/RV3D_reference_n201_T2.npz')\n")

    replay_line = "    if run['mode'] == 'replay':\n        run['replay'] = REPLAY[run['seed']]\n" if replay else ""
    cells = [
        md(f"# {name}\n\n{purpose}. Benchmark {bench}, {len(runs)} run(s), about {minutes} min on a T4 GPU.\n\n"
           f"Files needed in `MyDrive/SAW_ablation/`: " + ", ".join(files) + ".\n\n"
           f"Results are saved to `MyDrive/SAW_ablation/{name}/` after every run; "
           "re-running the notebook skips finished runs."),
        code(f"""import os, sys, csv, time
import numpy as np, torch
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '{DRIVE}'
sys.path.insert(0, PROJECT_DIR)
import {core} as core
import saw_ablation as sa

EXP, BENCH = {name!r}, {bench!r}
OUT = f'{{PROJECT_DIR}}/{{EXP}}'
os.makedirs(OUT, exist_ok=True)
device = 'cuda'
print(torch.cuda.get_device_name(0))"""),
        code(config.rstrip("\n")),
        code(f"""def key(r):
    return (r['mode'], str(r['seed']), str(r.get('q', '')), str(r.get('w_eik', '')))

done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        done = {{(r['mode'], r['seed'], r['q'], r['w_eik']) for r in csv.DictReader(f)}}
todo = [r for r in RUNS if key(r) not in done]
print(f'{{len(RUNS) - len(todo)}} done, {{len(todo)}} to go')

for run in todo:
    run = dict(run)
{replay_line}    print(run['mode'], 'seed', run['seed'], flush=True)
    row, tag = sa.run_one(core, BENCH, run, device, OUT, grid, PHI_E)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  rel L2 {{row['avg_rel_L2']:.4f}}%  MAPE {{row['avg_vol_MAPE']:.4f}}%  "
          f"final w {{row['w_eik_final']}}  {{row['minutes']:.1f}} min", flush=True)"""),
        code("""for r in csv.DictReader(open(CSV_PATH)):
    extra = f"  F_fill {float(r['avg_slot_filled']):.2f}%" if 'avg_slot_filled' in r else ''
    print(f"{r['mode']:7s} seed {r['seed']}  rel L2 {float(r['avg_rel_L2']):.4f}%  "
          f"MAPE {float(r['avg_vol_MAPE']):.4f}%{extra}  final w {r['w_eik_final']}")"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


os.makedirs(OUT, exist_ok=True)
for name, bench, purpose, runs, minutes in EXPERIMENTS:
    with open(os.path.join(OUT, f"{name}.ipynb"), "w", encoding="utf8") as f:
        json.dump(notebook(name, bench, purpose, runs, minutes), f, indent=1)
    print("wrote", name)
