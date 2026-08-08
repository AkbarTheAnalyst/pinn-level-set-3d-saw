# TR3D — Level-Set PINN for 3D Linear Transport

A physics-informed neural network solving the level-set transport equation

    phi_t + u . grad(phi) = 0

for a sphere of radius 0.15 translating through the unit cube with velocity
(0, 0.1, 0). The initial condition is the exact signed-distance function; the
sphere centre moves from (0.5, 0.2, 0.5) at t=0 to (0.5, 0.7, 0.5) at t=T=5.

Because the motion is a rigid translation, the exact solution remains an exact
signed-distance function for all time, which makes the analytic solution
available at every snapshot for error measurement.

## Files

| file | what it does | runtime |
|---|---|---|
| `tr3d_core.py` | model, loss, training, evaluation, figures | — |
| `01_eikonal_ablation.ipynb` | 6 weights x 3 seeds → results table, selects `W_EIK` | ~3.3 h (T4) |
| `02_train_and_evaluate.ipynb` | one configuration → metrics, figures, checkpoint | ~11 min (T4) |

Notebooks are numbered in execution order: `01` selects the eikonal weight,
`02` trains and evaluates at that value. **Most readers want `02`** — `01` only
needs re-running to reproduce the ablation table.

Both notebooks import from `tr3d_core.py`, so there is one definition of the
method and no risk of the two drifting apart.

## Quick start

1. Put all four files in the same folder (on Colab: `MyDrive/TR3D/`).
2. Open **`02_train_and_evaluate.ipynb`** and Run All. This is the short path
   (~11 min, or under a minute with the pretrained checkpoint).

To reproduce the eikonal-weight selection as well, run
`01_eikonal_ablation.ipynb` first (~3.3 h), copy the `selected w_eik` it prints
into the `W_EIK` setting of `02`, then run `02`.

If `TR3D_pretrained.pth` is in the folder, training is skipped and the notebook
reproduces every metric and figure in under a minute, CPU only. Delete or
rename it to train from scratch.

## Configuration

All tunable settings are in the second code cell of each notebook. The main one:

    W_EIK = 0.1     # eikonal loss weight, selected by 01_eikonal_ablation

## Method summary

- **Architecture** — 8 hidden layers, 256 units, tanh, Xavier init
- **Input** — (x, y, z, tau) with tau = t/T normalised to [0,1]; the residual
  therefore carries an explicit factor of T in the advection term
- **Loss** — `w_pde * L_pde + w_ic * L_ic + w_eik * L_eik`, with
  (1.0, 10.0, W_EIK). L_eik penalises deviation of |grad phi| from 1
- **Sampling** — 10,000 uniform collocation points in the 4-D domain,
  5,000 initial-condition points
- **Optimisation** — Adam (1e-3) under a full cosine cycle to eta_min = 1e-5
  over 20,000 iterations, then L-BFGS with strong-Wolfe line search
- **Component losses are logged unweighted**, so runs at different `w_eik`
  remain comparable term by term

## Metrics

Evaluated on a 201^3 grid (dx = 0.005, so the planes z=0.5 and y=0.7 fall on
grid points) at six snapshots t = 0, T/5, ..., T.

| metric | definition | note |
|---|---|---|
| `l2_abs` | sqrt(mean(e^2)) | equals the discrete L2 norm, since the domain has unit measure |
| `rel_L2` | ‖e‖ / ‖phi_exact‖ x 100% | standard relative-L2 PINN convention |
| `vol_MAPE` | \|V_pred − V_exact\| / V_exact | V by cell counting; ~0.1% estimator floor at 201^3 |
| `eik_dev` | mean \|\|grad phi\|−1\| | restricted to a 3-cell band, excluding the SDF kink at the sphere centre |

Both per-snapshot values and their averages are written to CSV.

## Figures

- **Fig. A** — shaded zero-isosurface at six snapshots, analytic sphere overlaid
  as a wireframe
- **Fig. B** — three orthogonal sections (three circular cross-sections is what
  distinguishes a sphere from a cylinder), pointwise error, eikonal deviation,
  and error growth in time
- **Fig. C** — training loss history with the Adam → L-BFGS transition marked

Saved as both PDF (vector, for the paper) and PNG at 300 dpi, sized 7.2 in wide
for a two-column layout.

## Eikonal weight selection

`W_EIK` was chosen with `01_eikonal_ablation.ipynb`: a sweep over
{0, 1e-4, 1e-3, 1e-2, 1e-1, 1} with three seeds each, at the same 20,000-Adam
budget as the reported run so the ranking transfers without extrapolation.

Decision rule, fixed before running:

1. lowest average relative L2
2. tiebreak: lowest average volume MAPE
3. tiebreak: lowest eikonal deviation
4. tiebreak: the smaller weight

Two weights count as tied when the gap between their means is smaller than the
larger of their two seed standard deviations. Eikonal deviation is deliberately
only a second-order tiebreak: it is essentially the quantity the swept term
minimises, so ranking by it would recover the sweep by construction.

**Caveat on transfer.** Under rigid translation the exact solution stays an
exact SDF, so the eikonal constraint is consistent with the PDE and this
benchmark is favourable to it. In deformation cases (vortex, Zalesak) the
advected level set does not remain an SDF and the term competes with the
transport residual. A weight selected here should be re-checked before being
adopted as a shared constant.

## Reproducing

Fixed seeds control both network initialisation (`seed`) and collocation
sampling (`seed + 58`); `seed = 42` reproduces the reference run. Results are
deterministic on a given device, but may differ slightly between GPU models
because of floating-point non-associativity.
