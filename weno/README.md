# WENO — Classical Baseline

Fifth-order WENO reconstruction of the convective derivatives with third-order
TVD Runge-Kutta time stepping, solving the same four benchmarks on the same
201^3 grid with the same error definitions as the PINN.

| file | purpose |
|---|---|
| `weno_core.py` | WENO5 + RK3 solver (torch, GPU), benchmark-agnostic |
| `weno_baseline.ipynb` | runs the four benchmarks, writes `weno_results.csv` |

## Setup

Place this folder alongside `TR3D/`, `RO3D/`, `ZD3D/` and `RV3D/` in Drive. The
notebook adds all of them to `sys.path` and imports each `*_core` module for the
velocity field, initial condition and exact solution, so nothing about the
problem definitions is duplicated here.

## Verification

The WENO derivative reproduces fifth-order convergence on a smooth function:

| grid | max error | observed order |
|---|---|---|
| 41 | 7.02e-05 | — |
| 81 | 2.19e-06 | 5.00 |
| 161 | 6.85e-08 | 5.00 |
| 321 | 2.14e-09 | 5.00 |

On the translation benchmark the full solver converges at second order in the
L2 norm, limited by the gradient singularity at the centre of the sphere rather
than by the scheme.

## Boundary conditions

Where the exact field is known it is imposed as Dirichlet data on a three-cell
boundary shell, evaluated only on that shell. This is not optional: the
level-set field grows towards the boundary, so at an inflow face zero-gradient
extrapolation injects incorrect data. Without it the translation benchmark
sits at a relative L2 error of about 36% that does not reduce under refinement,
even though the interface itself is transported correctly.

The reversed vortex needs no boundary data: that velocity field vanishes on the
faces of the unit cube, so no information enters the domain.

## Reinitialisation

None is applied. Standard level-set practice periodically re-solves the
reinitialisation equation to restore |grad phi| = 1, but the PINN maintains that
property through its eikonal loss term rather than by re-solving. Omitting it
keeps the comparison like for like, and the eikonal deviation is reported for
both methods.

## Cost

The step count is CFL-limited and differs by benchmark: roughly 250 steps for
translation, 3000 for rotation and Zalesak, 4000 for the vortex at 201^3.
Expect 5-20 minutes each on a T4. `weno_results.csv` records wall-clock time,
step count and peak GPU memory alongside the errors so that accuracy and cost
can be compared on the same footing.
