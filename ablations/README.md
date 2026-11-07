# Controls, diagnostics and sensitivity runs for SAW

This folder holds the experiments that test *why* SAW works: the replay and
random-drop controls, the gate diagnostics against the exact solution, the
gate-off gradient-norm logging, the sensitivity to the quantile `q`, and two
extra seeds on ZD3D. Results are in [`../results/ablations/`](../results/ablations/).

| File | Purpose |
|---|---|
| `saw_ablation.py` | One module for every run. `SAWLoss` reproduces the original SAW loss exactly and adds the `replay`, `random` and `fixed` modes and the diagnostics. `run_one` trains, evaluates and saves one run. |
| `make_notebooks.py` | Writes one Colab notebook per experiment into `notebooks/`. |
| `notebooks/` | The generated notebooks (one experiment each). |
| `tests/test_local.py` | CPU check: `SAWLoss` gives the same weight trajectory and final loss as the original SAW of each `*_core.py` (TR3D, ZD3D, RV3D), then a smoke test of every mode at tiny size. |
| `tests/run_notebook_quick.py` | Runs a generated notebook on CPU at tiny size, twice, to check that finished runs are skipped. |
| `tests/small.py` | The tiny sizes used by the two tests. |

## Experiments

All runs use the protocol of the main study ($\beta = 0.999$, $2\times10^4$ Adam
iterations, then L-BFGS, $201^3$ evaluation grid, six snapshots).

| Notebook | What it tests | Runs |
|---|---|---|
| `replay_ZD3D`, `replay_RV3D` | SAW's own weight trajectory (q = 0.95) replayed with the gate off | seeds 42–44 |
| `random_drop_ZD3D`, `random_drop_RV3D` | a random 5% of points excluded instead of the top 5% | seed 42 |
| `gate_diag_ZD3D`, `gate_diag_RV3D` | which points the gate removes, compared with labels from the exact solution; gradient norms | seeds 42–44 |
| `gate_diag_TR3D`, `gate_diag_RO3D` | the same on the two smooth rigid benchmarks | seed 42 |
| `gateoff_diag_ZD3D`, `gateoff_diag_RV3D` | gradient norms with the gate off (q = 1) | seed 42 |
| `q090_*`, `q097_ZD3D`, `q099_*`, `q071_RV3D` | sensitivity to the quantile q | seed 42; q = 0.90 also seeds 43, 44 |
| `seeds_ZD3D_SAW`, `seeds_ZD3D_fixed` | two more seeds (45, 46) for SAW at q = 0.95 and for the fixed weight 1e-3 | seeds 45, 46 |

Every diagnostic run reproduces the matching run without diagnostics to every
printed digit, so recording the diagnostics does not change training.

## Running a notebook on Colab

1. Make the folder `MyDrive/SAW_ablation/` and upload into it:
   - `saw_ablation.py`
   - the core file of the benchmark, from `../tr3d/`, `../ro3d/`, `../zd3d/` or `../rv3d/`
   - for the replay notebooks: `<BENCH>_saw_whist_q0.95_seed42.npz`, `..._seed43.npz`,
     `..._seed44.npz` from `../results/weight_history/`
   - for the RV3D notebooks (optional): `RV3D_reference_n201_T2.npz`; if it is
     missing, the reference is computed and saved there
2. Open the notebook in Colab, choose a T4 GPU runtime, and run all cells.

Results are written to `MyDrive/SAW_ablation/<notebook name>/` after every run, and
running the notebook again skips finished runs. A ZD3D run takes 30–50 minutes on a
T4, an RV3D run 30–60 minutes.

## Output files

For each run, in `../results/ablations/<notebook name>/`:

- `<notebook name>.csv`: one row per run, with the same columns as the main
  sweeps (per-snapshot and averaged relative $L_2$ error, volume error, eikonal
  deviation, and for ZD3D the feature-restricted errors `slot_filled` and
  `solid_eroded`), plus the final weight and the run time
- `<tag>_whist.npz`: the eikonal weight at every loss evaluation
- `<tag>_diag.json` (diagnostic runs): per checkpoint, the weight, the three
  gradient norms, the gate threshold and fraction, the full and gated eikonal
  loss, and the precision, recall and enrichment of the gate against the
  exact-solution labels
- `<tag>_diag_arrays.npz` (diagnostic runs): the 8000 subsampled points and their
  residuals, labels and gate flags at each checkpoint (`k-1` is the end of training)
- `<notebook name>.ipynb`: the executed notebook

`../figures/fig_gate.py` makes the gate figure of the paper from these files.

## Local checks (CPU)

```bash
cd ablations
python tests/test_local.py
python tests/run_notebook_quick.py replay_ZD3D gate_diag_RV3D
```

On Windows, set `PYTHONIOENCODING=utf-8` for the RV3D notebooks.
