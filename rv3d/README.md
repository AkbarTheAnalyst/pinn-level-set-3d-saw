# RV3D — Level-Set PINN for the 3D Reversed Single Vortex

A physics-informed neural network solving the level-set transport equation

    phi_t + u . grad(phi) = 0

for a sphere of radius 0.15 centred at (0.35, 0.35, 0.35), carried by the
Enright et al. (2002) three-dimensional deformation field, written in
normalised time tau = t/T:

    u =  2 sin^2(pi x) sin(2 pi y) sin(2 pi z) cos(pi tau)
    v =   -sin(2 pi x) sin^2(pi y) sin(2 pi z) cos(pi tau)
    w =   -sin(2 pi x) sin(2 pi y) sin^2(pi z) cos(pi tau)

The sphere stretches into a thin folded sheet and returns exactly to its
initial shape at t = T = 2, because the cos(pi tau) factor integrates to zero
over [0, 1] and the flow reverses at tau = 1/2.

The field is exactly divergence free (2 pi - pi - pi = 0), so the enclosed
volume is constant in time and the volume metric stays meaningful at every
snapshot.

## Why this benchmark exists

Translation, rotation and Zalesak are all **rigid** motions: the exact solution
remains an exact signed distance function for all time, so the eikonal term is
free supervision that never conflicts with the transport residual. Under the
reversed vortex the interface stretches, the advected field stops being an SDF,
and enforcing |grad phi| = 1 competes directly with the PDE. This is the only
one of the four benchmarks where the eikonal weight is genuinely contested.

**Note on T.** T = 2 gives milder deformation than the T = 3 of Enright et al.
The interface departs less from a signed distance function, so the eikonal
conflict is weaker than it would be at larger T. If the ablation shows little
sensitivity, that is partly a consequence of T and should be reported as such
rather than as evidence that the term is harmless under deformation. T is a
one-line change and the reference tracer handles any value.

## No closed-form solution

Since phi is advected as a passive scalar, phi(x, tau) = phi_0(X(0; x, tau))
along backward characteristics. The velocity is analytic, so the reference is
obtained by integrating dX/dtau = T u backward with RK4. Measured
self-consistency at t = T, where the reference must reproduce phi_0 exactly:

| RK4 steps | max error | rms error |
|---|---|---|
| 50 | 1.5e-05 | 1.6e-06 |
| 100 | 4.7e-07 | 6.2e-08 |
| **200 (default)** | **1.7e-07** | **3.4e-08** |

Orders of magnitude below the network error. The reference does not depend on
the network, so it is computed once, cached as `.npz`, and reused by all 18
ablation runs. The self-consistency figure is printed whenever it is computed.

## Files

| file | what it does | runtime |
|---|---|---|
| `rv3d_core.py` | model, loss, training, evaluation, figures | — |
| `01_eikonal_ablation.ipynb` | 6 weights x 3 seeds → results table, selects `W_EIK` | ~4.5 h (T4, estimate) |
| `02_train_and_evaluate.ipynb` | one configuration → metrics, figures, checkpoint | ~15 min (T4, estimate) |

Notebooks are numbered in execution order: `01` selects the eikonal weight,
`02` trains and evaluates at that value. **Most readers want `02`** — `01` only
needs re-running to reproduce the ablation table.

Both notebooks import from `rv3d_core.py`, so there is one definition of the
method and no risk of the two drifting apart.

## Quick start

1. Put all four files in the same folder (on Colab: `MyDrive/RV3D/`).
2. Open **`02_train_and_evaluate.ipynb`** and Run All. This is the short path
   (~11 min, or under a minute with the pretrained checkpoint).

To reproduce the eikonal-weight selection as well, run
`01_eikonal_ablation.ipynb` first (~3.3 h), copy the `selected w_eik` it prints
into the `W_EIK` setting of `02`, then run `02`.

If `RV3D_pretrained.pth` is in the folder, training is skipped and the notebook
reproduces every metric and figure in under a minute, CPU only. Delete or
rename it to train from scratch.

## Configuration

All tunable settings are in the second code cell of each notebook. The main one:

    W_EIK = 0.0     # eikonal loss weight, selected by 01_eikonal_ablation

## Method summary

- **Architecture** — 8 hidden layers, 256 units, tanh, Xavier init
- **Input** — (x, y, z, tau) with tau = t/T normalised to [0,1], T = 2*pi;
  the residual therefore carries an explicit factor of T in the advection term
- **Loss** — `w_pde * L_pde + w_ic * L_ic + w_eik * L_eik`, with
  (1.0, 10.0, W_EIK). L_eik penalises deviation of |grad phi| from 1
- **Sampling** — 10,000 collocation points, resampled every 1,000 iterations by
  residual-based adaptive distribution (RAD), plus residual-based refinement
  (RAR-G) adding 500 points every 1,800 iterations; 5,000 initial-condition points
- **Residual weighting** — causal weighting of Wang et al. (2022), 32 chunks
  sorted by tau (10 chunks), eps = 1.0
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
| `vol_MAPE` | \|V_pred − V_exact\| / V_exact | the field is divergence free, so V_exact is the analytic sphere volume at every snapshot |
| `vol_floor` | \|counted reference − V_exact\| / V_exact | noise floor of the volume metric; grows where the sheet is thinnest. At 65³ the worst snapshot loses 0.3% at T=2 (vs 1.9% at T=8), so at 201³ it is negligible |
| `eik_dev` | mean \|\|grad phi\|−1\| | restricted to a 3-cell band, excluding the SDF kink at the sphere centre |

Both per-snapshot values and their averages are written to CSV.

## Figures

- **Fig. A** — shaded zero-isosurface at six snapshots, with the exact zero
  level set overlaid as a silhouette outline (the shape is non-convex, so the
  analytic wireframe used for the sphere benchmarks does not apply)
On the folded sheet the exact silhouette has many branches and can read as
speckle over the predicted surface. `figure_interface(..., show_exact=False)`
gives a clean qualitative panel; Fig. B carries the quantitative comparison.
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

The grid is {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}. The 2D counterpart of this test
used w_eik = 0, so a value at or near zero is the expected outcome; confirming
or contradicting that in 3D is the point of the sweep.

**Caveat on transfer.** Do not assume the weight selected here carries over to
the rigid benchmarks or vice versa. For translation and rotation the exact
solution is an exact SDF everywhere and the eikonal term is free supervision;
for Zalesak it conflicts locally near the slot corners; here it conflicts
globally wherever the interface stretches. Report the selected value per
benchmark.

## Reproducing

Fixed seeds control both network initialisation (`seed`) and collocation
sampling (`seed + 58`); `seed = 42` reproduces the reference run. Results are
deterministic on a given device, but may differ slightly between GPU models
because of floating-point non-associativity.
