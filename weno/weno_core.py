"""
weno_core.py — classical baseline for the 3D level-set advection benchmarks.

Fifth-order WENO reconstruction of the convective derivatives (Jiang and Shu
1996, in the level-set form of Osher and Fedkiw 2003) with third-order TVD
Runge--Kutta time stepping (Shu and Osher 1988), solving

    phi_t + u . grad(phi) = 0

in non-conservative form on a uniform Cartesian grid. Written in torch so it
runs on the GPU: at 201^3 with a CFL-limited step the solver takes O(10^3)
steps of 3 stages over 8.1e6 cells, which is hours in NumPy and minutes on a
T4.

The solver is benchmark-agnostic. It takes the initial field and a velocity
callable, so the same implementation serves TR3D, RO3D, ZD3D and RV3D and the
comparison is not confounded by four separate implementations.

NO REINITIALISATION is applied. Standard level-set practice periodically
re-solves the reinitialisation equation to restore |grad phi| = 1, but the PINN
being compared against enforces that property through the eikonal loss term
rather than by re-solving. Omitting it here keeps the comparison like for like;
the eikonal deviation is reported for both methods so the reader can see what
each does to the signed-distance property.

Boundary treatment: where the exact field is known analytically it is imposed
as Dirichlet data on a three-cell boundary shell. This matters. The level-set
field grows towards the domain boundary, so at an inflow face zero-gradient
extrapolation feeds in incorrect data which then propagates inwards; without
Dirichlet data the translation benchmark shows a relative L2 error of ~36% that
does not reduce under grid refinement, even though the interface itself is
transported correctly. For the reversed vortex no boundary data is needed: that
velocity field vanishes on the faces of the unit cube, so there is no inflow.
"""

import time
import numpy as np
import torch

EPS_WENO = 1e-6
_C = (13.0 / 12.0, 0.25)


def _pad_axis(f, axis, n_ghost=3):
    """Replicate-pad `f` along one axis (zero-gradient boundary)."""
    n = f.shape[axis]
    idx = torch.arange(-n_ghost, n + n_ghost, device=f.device).clamp_(0, n - 1)
    return f.index_select(axis, idx)


def _slice(D, axis, start, length):
    return D.narrow(axis, start, length)


def _weno5(v1, v2, v3, v4, v5):
    """Fifth-order WENO combination of three third-order stencils."""
    a, b = _C
    s1 = a * (v1 - 2*v2 + v3)**2 + b * (v1 - 4*v2 + 3*v3)**2
    s2 = a * (v2 - 2*v3 + v4)**2 + b * (v2 - v4)**2
    s3 = a * (v3 - 2*v4 + v5)**2 + b * (3*v3 - 4*v4 + v5)**2

    a1 = 0.1 / (s1 + EPS_WENO)**2
    a2 = 0.6 / (s2 + EPS_WENO)**2
    a3 = 0.3 / (s3 + EPS_WENO)**2
    inv = 1.0 / (a1 + a2 + a3)

    p1 = (v1 / 3.0) - (7.0 * v2 / 6.0) + (11.0 * v3 / 6.0)
    p2 = (-v2 / 6.0) + (5.0 * v3 / 6.0) + (v4 / 3.0)
    p3 = (v3 / 3.0) + (5.0 * v4 / 6.0) - (v5 / 6.0)
    return (a1 * p1 + a2 * p2 + a3 * p3) * inv


def weno_derivative(f, axis, dx, vel):
    """Upwinded WENO5 approximation of df/dx along `axis`.

    Both one-sided reconstructions are formed and selected pointwise by the
    sign of the advecting velocity, which is what makes the scheme upwind.
    """
    n = f.shape[axis]
    fp = _pad_axis(f, axis, 3)
    D = (fp.narrow(axis, 1, n + 5) - fp.narrow(axis, 0, n + 5)) / dx

    minus = _weno5(*[_slice(D, axis, k, n) for k in (0, 1, 2, 3, 4)])
    plus = _weno5(*[_slice(D, axis, k, n) for k in (5, 4, 3, 2, 1)])
    return torch.where(vel > 0, minus, plus)


def rhs(phi, U, V, W, d):
    """-u . grad(phi) with WENO5 upwinding in each direction."""
    return -(U * weno_derivative(phi, 0, d, U)
             + V * weno_derivative(phi, 1, d, V)
             + W * weno_derivative(phi, 2, d, W))


def solve(phi0, vel_fn, T_final, d, times, cfl=0.4, device='cuda',
          verbose=True, max_steps=200000, bc_fn=None, bc_grid=None,
          bc_width=3):
    """Advance phi0 to each time in `times` with WENO5 + TVD-RK3.

    vel_fn(t) must return (U, V, W) as tensors on the same grid as phi0, with
    t the PHYSICAL time.

    bc_fn(t, X, Y, Z), if given, returns the exact field at physical time t at
    the supplied coordinates; it is evaluated only on a shell of `bc_width`
    cells at the domain boundary (about 9% of the points at 201^3) and imposed
    there after every Runge-Kutta stage. `bc_grid` is the (X, Y, Z) coordinate
    triple of the full grid. Omit both only when the velocity vanishes on the
    boundary, so that no information enters the domain.

    Returns {t: numpy field}, plus timing, step count and peak memory.
    """
    phi = torch.as_tensor(phi0, dtype=torch.float32, device=device).clone()

    shell = None
    if bc_fn is not None:
        w = bc_width
        m = torch.zeros_like(phi, dtype=torch.bool)
        m[:w] = m[-w:] = True
        m[:, :w] = m[:, -w:] = True
        m[:, :, :w] = m[:, :, -w:] = True
        shell = m
        mnp = m.cpu().numpy()
        Xs, Ys, Zs = (a[mnp] for a in bc_grid)   # shell coordinates only

    def apply_bc(f, t_phys):
        if shell is None:
            return f
        vals = torch.as_tensor(np.ascontiguousarray(bc_fn(t_phys, Xs, Ys, Zs)),
                               dtype=torch.float32, device=device)
        f = f.clone()
        f[shell] = vals
        return f

    out, targets = {}, sorted(times)
    t, nstep = 0.0, 0
    t0 = time.time()

    if targets and targets[0] <= 0.0:
        out[targets[0]] = phi.cpu().numpy().copy()
        targets = targets[1:]

    for t_target in targets:
        while t < t_target - 1e-12:
            U, V, W = vel_fn(t)
            vmax = (U.abs().max() + V.abs().max() + W.abs().max()).item()
            dt = cfl * d / max(vmax, 1e-12)
            dt = min(dt, t_target - t)

            # Shu-Osher third-order TVD Runge-Kutta
            p1 = apply_bc(phi + dt * rhs(phi, U, V, W, d), t + dt)

            U2, V2, W2 = vel_fn(t + dt)
            p2 = 0.75 * phi + 0.25 * (p1 + dt * rhs(p1, U2, V2, W2, d))
            p2 = apply_bc(p2, t + 0.5 * dt)

            U3, V3, W3 = vel_fn(t + 0.5 * dt)
            phi = (phi + 2.0 * (p2 + dt * rhs(p2, U3, V3, W3, d))) / 3.0
            phi = apply_bc(phi, t + dt)

            t += dt
            nstep += 1
            if nstep > max_steps:
                raise RuntimeError('step limit exceeded — check the CFL number')
        out[t_target] = phi.cpu().numpy().copy()
        if verbose:
            print(f'  t = {t:.4f} reached after {nstep} steps '
                  f'({time.time()-t0:.0f}s)', flush=True)

    mem = (torch.cuda.max_memory_allocated() / 2**20) if device == 'cuda' else float('nan')
    return {'fields': out, 'minutes': (time.time() - t0) / 60,
            'steps': nstep, 'peak_MB': mem}


# --------------------------------------------------------------------------
# Metrics. Definitions are identical to `evaluate` in the benchmark cores, so
# the two methods are scored on exactly the same quantities and grid.
# --------------------------------------------------------------------------
def metrics(grid, times, PHI_P, PHI_E, V_exact, band_cells=3):
  res = {k: [] for k in ('l2_abs', 'rel_L2', 'vol_abs', 'vol_MAPE', 'eik_dev',
                         'vol_floor')}
  for t_val in times:
    pp, pe = PHI_P[t_val], PHI_E[t_val]
    res['l2_abs'].append(float(np.sqrt(np.mean((pp - pe)**2))))
    res['rel_L2'].append(float(np.sqrt(np.sum((pp - pe)**2))
                               / np.sqrt(np.sum(pe**2)) * 100))
    V_pred = np.sum(pp < 0) * grid.d**3
    res['vol_abs'].append(float(abs(V_pred - V_exact)))
    res['vol_MAPE'].append(float(abs(V_pred - V_exact) / V_exact * 100))
    V_ex = np.sum(pe < 0) * grid.d**3
    res['vol_floor'].append(float(abs(V_ex - V_exact) / V_exact * 100))

    gx, gy, gz = np.gradient(pp, grid.d, grid.d, grid.d)
    band = np.abs(pe) < band_cells * grid.d
    res['eik_dev'].append(
        float(np.mean(np.abs(np.sqrt(gx**2 + gy**2 + gz**2)[band] - 1.0))))
    del gx, gy, gz, band

  for k in list(res):
    res['avg_' + k] = float(np.mean(res[k]))
  return res


def print_metrics(times, res, label=''):
  print(f"\n{label}")
  print(f"{'Snapshot':<12}{'abs L2':>12}{'rel L2 (%)':>13}"
        f"{'vol abs':>13}{'MAPE (%)':>12}{'eik dev':>13}")
  print('-' * 75)
  for i, t in enumerate(times):
    print(f"t={t:<10.3f}{res['l2_abs'][i]:>12.2e}{res['rel_L2'][i]:>12.3f}%"
          f"{res['vol_abs'][i]:>13.2e}{res['vol_MAPE'][i]:>11.3f}%"
          f"{res['eik_dev'][i]:>13.2e}")
  print('-' * 75)
  print(f"{'avg':<12}{res['avg_l2_abs']:>12.2e}{res['avg_rel_L2']:>12.3f}%"
        f"{res['avg_vol_abs']:>13.2e}{res['avg_vol_MAPE']:>11.3f}%"
        f"{res['avg_eik_dev']:>13.2e}")
