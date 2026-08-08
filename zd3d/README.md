# ZD3D — Level-Set PINN for the 3D Zalesak Slotted Sphere

A physics-informed neural network solving the level-set transport equation

    phi_t + u . grad(phi) = 0

for a **Zalesak slotted sphere** carried by solid-body rotation about the
vertical axis through (0.5, 0.5):

    u = -(y - 0.5),   v = (x - 0.5),   w = 0

The shape is a sphere of radius 0.15 centred at (0.5, 0.75, 0.5) with a
rectangular slot of half-extents (0.025, 0.125, 0.10) centred at
(0.5, 0.725, 0.5). The slot has **finite depth in z**, so it is a blind pocket:
a section near the pole shows an unbroken sphere while a section through the
middle shows the notch. This is what makes the test genuinely three
dimensional rather than an extruded 2D disk. Fig. D (cutaway) shows it.

The centre orbits at radius 0.25 in the plane z = 0.5, completing exactly one
revolution by t = T = 2*pi and returning to its starting position — so the
error at t=T measures accumulated drift over a closed orbit. Zalesak is a
*rigid* motion, not a deformation test: it probes preservation of corners and
thin features under rotation.

## The exact solution is not an exact SDF

The initial field is the CSG difference `max(phi_sphere, -phi_slot)`. Near the
slot's reentrant edges this returns the distance to the nearer face rather than
to the edge, so `|grad phi| != 1` there **in the exact solution itself**.

Two consequences:

- the eikonal term competes with the initial-condition term near the slot
  instead of reinforcing it, so this benchmark wants a much smaller eikonal
  weight than the plain-sphere cases;
- a nonzero `eik_dev` near the slot is expected and is not purely network
  error.

## Files

| file | what it does | runtime |
|---|---|---|
| `zd3d_core.py` | model, loss, training, evaluation, figures | — |
| `01_eikonal_ablation.ipynb` | 6 weights x 3 seeds → results table, selects `W_EIK` | ~4.5 h (T4, estimate) |
| `02_train_and_evaluate.ipynb` | one configuration → metrics, figures, checkpoint | ~15 min (T4, estimate) |

Notebooks are numbered in execution order: `01` selects the eikonal weight,
`02` trains and evaluates at that value. **Most readers want `02`** — `01` only
needs re-running to reproduce the ablation table.

Both notebooks import from `zd3d_core.py`, so there is one definition of the
method and no risk of the two drifting apart.

## Quick start

1. Put all four files in the same folder (on Colab: `MyDrive/ZD3D/`).
2. Open **`02_train_and_evaluate.ipynb`** and Run All. This is the short path
   (~11 min, or under a minute with the pretrained checkpoint).

To reproduce the eikonal-weight selection as well, run
`01_eikonal_ablation.ipynb` first (~3.3 h), copy the `selected w_eik` it prints
into the `W_EIK` setting of `02`, then run `02`.

If `ZD3D_pretrained.pth` is in the folder, training is skipped and the notebook
reproduces every metric and figure in under a minute, CPU only. Delete or
rename it to train from scratch.

## Configuration

All tunable settings are in the second code cell of each notebook. The main one:

    W_EIK = 0.001   # eikonal loss weight, selected by 01_eikonal_ablation

## Method summary

- **Architecture** — random Fourier feature encoding (128 features, B ~ N(0, 5))
  of (x, y, z, tau), then 8 hidden layers, 256 units, tanh, Xavier init
- **Input** — (x, y, z, tau) with tau = t/T normalised to [0,1], T = 2*pi;
  the residual therefore carries an explicit factor of T in the advection term
- **Loss** — `w_pde * L_pde + w_ic * L_ic + w_eik * L_eik`, with
  (1.0, 10.0, W_EIK). L_eik penalises deviation of |grad phi| from 1
- **Sampling** — 10,000 collocation points, resampled every 1,000 iterations by
  residual-based adaptive distribution (RAD), plus residual-based refinement
  (RAR-G) adding 500 points every 1,800 iterations; 5,000 initial-condition points
- **Residual weighting** — causal weighting of Wang et al. (2022), 32 chunks
  sorted by tau, eps = 1.0
- **Optimisation** — Adam (1e-3) under a full cosine cycle to eta_min = 1e-5
  over 20,000 iterations, then L-BFGS with strong-Wolfe line search
- **Component losses are logged unweighted**, so runs at different `w_eik`
  remain comparable term by term

## Metrics

Evaluated on a 201^3 grid (dx = 0.005, so the planes x=0.5, y=0.75 and z=0.5
fall on grid points) at six snapshots t = 0, T/5, ..., T.

| metric | definition | note |
|---|---|---|
| `l2_abs` | sqrt(mean(e^2)) | equals the discrete L2 norm, since the domain has unit measure |
| `rel_L2` | ‖e‖ / ‖phi_exact‖ x 100% | standard relative-L2 PINN convention |
| `vol_MAPE` | \|V_pred − V_ref\| / V_ref | V by cell counting; the slotted sphere has no convenient closed form, so V_ref is the counted volume of the **exact** field at t=0, which shares the estimator and cancels its bias to leading order |
| `vol_floor` | drift of the counted exact volume across snapshots | the noise floor of the volume metric; printed by `print_metrics` |
| `eik_dev` | mean \|\|grad phi\|−1\| | restricted to a 3-cell band, excluding the SDF kink at the sphere centre |

Both per-snapshot values and their averages are written to CSV.

## Figures

- **Fig. A** — shaded zero-isosurface at six snapshots, with the exact zero
  level set overlaid as a silhouette outline (the shape is non-convex, so the
  analytic wireframe used for the sphere benchmarks does not apply)
- **Fig. D** — half-space cutaway at x = 0.5, predicted and exact side by side.
  The slot mouth is a narrow opening near z = 0.5, so from outside the shape
  reads as an almost intact sphere; this is the panel that shows the pocket
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

The grid runs lower here — {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} — because the
corner artefact described above makes a large eikonal weight actively harmful
on this benchmark, unlike the plain-sphere cases.

**Caveat on transfer.** The eikonal weight selected here should not be assumed
to carry over to the translating- or rotating-sphere benchmarks, or vice versa:
for a smooth sphere the exact solution is an exact SDF everywhere and the
eikonal term is free supervision, whereas here it conflicts with the exact
solution near the slot. Report the selected value per benchmark rather than as
a shared constant.

## Reproducing

Fixed seeds control both network initialisation (`seed`) and collocation
sampling (`seed + 58`); `seed = 42` reproduces the reference run. Results are
deterministic on a given device, but may differ slightly between GPU models
because of floating-point non-associativity.
