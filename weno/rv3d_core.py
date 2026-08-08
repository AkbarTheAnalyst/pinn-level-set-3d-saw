"""
rv3d_core.py — shared definitions for the 3D reversed single-vortex level-set PINN.

Both notebooks import from here:

    01_eikonal_ablation.ipynb     w_eik x seed sweep   -> results table
    02_train_and_evaluate.ipynb   single configuration -> metrics, figures, checkpoint

Geometry: unit cube, sphere of radius 0.15 centred at (0.35, 0.35, 0.35).

Velocity (Enright et al. 2002 three-dimensional deformation field), written in
normalised time tau = t/T so that the network sees tau in [0,1]:

    u =  2 sin^2(pi x) sin(2 pi y) sin(2 pi z) cos(pi tau)
    v =   -sin(2 pi x) sin^2(pi y) sin(2 pi z) cos(pi tau)
    w =   -sin(2 pi x) sin(2 pi y) sin^2(pi z) cos(pi tau)

The field is exactly divergence free (2 pi - pi - pi = 0), so the enclosed
volume is constant in time and the volume metric stays meaningful at every
snapshot. The cos(pi tau) factor integrates to zero over [0,1], so the flow
reverses at tau = 1/2 and the sphere returns to its initial shape at tau = 1.

WHY THIS BENCHMARK MATTERS FOR THE EIKONAL TERM. Translation, rotation and
Zalesak are all *rigid* motions: the exact solution stays an exact signed
distance function for all time, so the eikonal constraint is free supervision
that never conflicts with the transport residual. Here the interface stretches
into a thin sheet, the advected field stops being an SDF, and enforcing
|grad phi| = 1 actively competes with the PDE. This is the regime where the
eikonal weight actually matters.

NO CLOSED-FORM SOLUTION. Because phi is advected as a passive scalar,
phi(x, tau) = phi_0(X(0; x, tau)) where X is the backward characteristic. The
velocity is known analytically, so the reference is obtained by integrating
dX/dtau = T * u backward with RK4. With 400 steps at T = 8 the self-consistency
error at tau = 1 (where the reference must recover phi_0 exactly) is ~3e-6 in
max norm, several orders below the network error. `reference_fields` computes
these once and caches them, since they do not depend on the network.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------- geometry
XC, YC, ZC, R = 0.35, 0.35, 0.35, 0.15
V_EXACT = (4.0 / 3.0) * np.pi * R**3      # exact: the field is divergence free


def velocity_field(x, y, z, tau):
  """Enright deformation field in normalised time. `tau` may be a tensor
  column (training) or a broadcastable scalar tensor (reference tracing)."""
  f = torch.cos(np.pi * tau)
  u = 2 * torch.sin(np.pi*x)**2 * torch.sin(2*np.pi*y) * torch.sin(2*np.pi*z) * f
  v = -torch.sin(2*np.pi*x) * torch.sin(np.pi*y)**2 * torch.sin(2*np.pi*z) * f
  w = -torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y) * torch.sin(np.pi*z)**2 * f
  return u, v, w


def initial_condition(x, y, z):
  return torch.sqrt((x - XC)**2 + (y - YC)**2 + (z - ZC)**2) - R


def initial_condition_np(X, Y, Z):
  return np.sqrt((X - XC)**2 + (Y - YC)**2 + (Z - ZC)**2) - R


def snapshot_times(T_final, n=6):
  ts = [i * T_final / (n - 1) for i in range(n)]
  labs = ['0'] + [(f'{i}T/{n-1}' if i > 1 else f'T/{n-1}') for i in range(1, n-1)] + ['T']
  return ts, labs


# -------------------------------------------------------------------- model
class PINN(nn.Module):
  def __init__(self, layers=8, neurons=256):
    super().__init__()
    net = [nn.Linear(4, neurons), nn.Tanh()]
    for _ in range(layers - 1):
      net += [nn.Linear(neurons, neurons), nn.Tanh()]
    net += [nn.Linear(neurons, 1)]
    self.net = nn.Sequential(*net)
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, x):
    return self.net(x)


def build_model(device, seed=42, layers=8, neurons=256):
  torch.manual_seed(seed)
  if device == 'cuda':
    torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  return PINN(layers, neurons).to(device)


def sample_points(N_f, N_i, device, seed=42):
  torch.manual_seed(seed + 58)
  if device == 'cuda':
    torch.cuda.manual_seed(seed + 58)
  X_f = torch.rand(N_f, 4, dtype=torch.float32, device=device).requires_grad_(True)
  xi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  yi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  zi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  ti = torch.zeros(N_i, 1, dtype=torch.float32, device=device)
  return X_f, torch.cat([xi, yi, zi, ti], dim=1)


# ------------------------------------------------------- adaptive sampling
class AdaptiveSampler:
  """RAD resampling of the base set (Wu et al. 2023) plus RAR-G growth
  (Lu et al. 2021)."""

  def __init__(self, net, X_base, T_final, device,
               rad_freq=1000, rad_candidates=20000, rad_k=1.0, rad_c=1.0,
               rar_freq=1800, rar_candidates=50000, rar_add=500):
    self.net, self.T_final, self.device = net, T_final, device
    self.X_base, self.X_rar = X_base, None
    self.N_f = X_base.shape[0]
    self.rad_freq, self.rad_candidates = rad_freq, rad_candidates
    self.rad_k, self.rad_c = rad_k, rad_c
    self.rar_freq, self.rar_candidates, self.rar_add = rar_freq, rar_candidates, rar_add

  def _residuals(self, X_cand):
    Xc = X_cand.detach().requires_grad_(True)
    phi = self.net(Xc)
    g = torch.autograd.grad(phi, Xc, grad_outputs=torch.ones_like(phi),
                            create_graph=False)[0]
    u, v, w = velocity_field(Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3], Xc[:, 3:4])
    res = g[:, 3:4] + self.T_final * (u*g[:, 0:1] + v*g[:, 1:2] + w*g[:, 2:3])
    return res.detach().abs().squeeze()

  def step(self, epoch):
    if epoch > 0 and epoch % self.rad_freq == 0:
      Xc = torch.rand(self.rad_candidates, 4, dtype=torch.float32,
                      device=self.device)
      rk = self._residuals(Xc) ** self.rad_k
      wts = rk / (rk.mean() + 1e-8) + self.rad_c
      idx = torch.multinomial(wts / wts.sum(), self.N_f, replacement=False)
      self.X_base = Xc[idx].detach().requires_grad_(True)
    if epoch > 0 and epoch % self.rar_freq == 0:
      Xc = torch.rand(self.rar_candidates, 4, dtype=torch.float32,
                      device=self.device)
      _, idx = torch.topk(self._residuals(Xc), self.rar_add)
      new = Xc[idx].detach()
      self.X_rar = new if self.X_rar is None else torch.cat([self.X_rar, new], 0)

  def points(self):
    if self.X_rar is not None and self.X_rar.shape[0] > 0:
      return torch.cat([self.X_base, self.X_rar.detach().requires_grad_(True)], 0)
    return self.X_base

  def n_points(self):
    return self.N_f + (0 if self.X_rar is None else self.X_rar.shape[0])


# --------------------------------------------------------------------- loss
def make_loss(net, sampler, X_i, T_final, w_pde=1.0, w_ic=10.0, w_eik=0.0,
              m_chunks=10, eps_causal=1.0):
  """Causally weighted PDE residual + IC + eikonal. Component losses are
  unweighted so runs at different w_eik stay comparable term by term."""
  def loss_fn():
    X_f = sampler.points()
    phi = net(X_f)
    g = torch.autograd.grad(phi, X_f, grad_outputs=torch.ones_like(phi),
                            create_graph=True)[0]
    px, py, pz, pt = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    u, v, w = velocity_field(X_f[:, 0:1], X_f[:, 1:2], X_f[:, 2:3], X_f[:, 3:4])
    residuals = (pt + T_final * (u*px + v*py + w*pz))**2

    chunk = X_f.shape[0] // m_chunks
    res_sorted = residuals[torch.argsort(X_f[:, 3].detach())]
    chunk_means = [torch.mean(res_sorted[m*chunk:(m+1)*chunk])
                   for m in range(m_chunks)]
    cumsum = torch.zeros((), device=X_f.device)
    weights = []
    for m in range(m_chunks):
      weights.append(torch.exp(-eps_causal * cumsum))
      cumsum = cumsum + chunk_means[m].detach()
    loss_pde = sum(weights[m] * chunk_means[m] for m in range(m_chunks)) / m_chunks

    loss_ic = torch.mean((initial_condition(X_i[:, 0:1], X_i[:, 1:2],
                                            X_i[:, 2:3]) - net(X_i))**2)
    loss_eik = torch.mean((torch.sqrt(px**2 + py**2 + pz**2 + 1e-8) - 1)**2)
    return (loss_pde, loss_ic, loss_eik,
            w_pde*loss_pde + w_ic*loss_ic + w_eik*loss_eik)
  return loss_fn


# ----------------------------------------------------------------- training
def train(net, loss_fn, sampler=None, adam_epochs=20000, lbfgs_epochs=2000,
          lr=1e-3, eta_min=1e-5, clip=1.0, verbose=True, log_every=2000):
  hist = {'epoch': [], 'total': [], 'pde': [], 'ic': [], 'eik': [], 'n_pts': []}
  t0 = time.time()

  opt = optim.Adam(net.parameters(), lr=lr)
  sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_epochs,
                                             eta_min=eta_min)
  for ep in range(adam_epochs):
    if sampler is not None:
      sampler.step(ep)
    opt.zero_grad()
    l_pde, l_ic, l_eik, loss = loss_fn()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    sch.step()

    hist['epoch'].append(ep)
    hist['total'].append(loss.item())
    hist['pde'].append(l_pde.item())
    hist['ic'].append(l_ic.item())
    hist['eik'].append(l_eik.item())
    hist['n_pts'].append(sampler.n_points() if sampler else 0)

    if verbose and (ep % log_every == 0 or ep == adam_epochs - 1):
      print(f'Adam {ep:>6d} | total {loss.item():.3e} | pde {l_pde.item():.3e} '
            f'| ic {l_ic.item():.3e} | eik {l_eik.item():.3e} '
            f'| lr {sch.get_last_lr()[0]:.3e} | pts {hist["n_pts"][-1]}',
            flush=True)

  loss_adam = float(hist['total'][-1])
  if verbose:
    print('L-BFGS starting...', flush=True)

  opt2 = optim.LBFGS(net.parameters(), lr=1.0, history_size=200,
                     line_search_fn='strong_wolfe')
  store = {}

  def closure():
    opt2.zero_grad()
    a, b, c, l = loss_fn()
    l.backward()
    store.update(total=l.item(), pde=a.item(), ic=b.item(), eik=c.item())
    return l

  for ep in range(lbfgs_epochs):
    opt2.step(closure)
    hist['epoch'].append(adam_epochs + ep)
    for k in ('total', 'pde', 'ic', 'eik'):
      hist[k].append(store[k])
    hist['n_pts'].append(sampler.n_points() if sampler else 0)
    if verbose and (ep % 400 == 0 or ep == lbfgs_epochs - 1):
      print(f'LBFGS {ep:>5d} | total {store["total"]:.3e} '
            f'| pde {store["pde"]:.3e} | ic {store["ic"]:.3e} '
            f'| eik {store["eik"]:.3e}', flush=True)

  loss_final = float(hist['total'][-1])
  minutes = (time.time() - t0) / 60
  if verbose:
    print(f'training done in {minutes:.1f} min '
          f'(L-BFGS gain {loss_adam - loss_final:.2e})', flush=True)

  return {'history': hist, 'minutes': minutes, 'loss_adam': loss_adam,
          'loss_final': loss_final, 'lbfgs_gain': loss_adam - loss_final,
          'n_points': sampler.n_points() if sampler else 0}


# ------------------------------------------------- reference by backtracing
class Grid:
  """Uniform evaluation grid on the unit cube. n=201 gives dx=0.005."""

  def __init__(self, n=201):
    self.n = n
    self.x = np.linspace(0, 1, n, dtype=np.float32)
    self.d = float(self.x[1] - self.x[0])
    self.X, self.Y, self.Z = np.meshgrid(self.x, self.x, self.x, indexing='ij')
    self.pts = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1)

  def predict(self, net, t_norm, device, chunk=200000):
    out = []
    for s in range(0, self.pts.shape[0], chunk):
      blk = self.pts[s:s + chunk]
      tc = np.full((blk.shape[0], 1), t_norm, dtype=np.float32)
      inp = torch.tensor(np.concatenate([blk, tc], axis=1),
                         dtype=torch.float32, device=device)
      with torch.no_grad():
        out.append(net(inp).cpu().numpy())
    return np.concatenate(out, 0).reshape(self.n, self.n, self.n)


def _backtrace(x, y, z, tau, T_final, nsteps):
  """RK4 integration of dX/dtau = T*u backward from tau to 0."""
  h = -tau / nsteps
  t = tau
  for _ in range(nsteps):
    tt = torch.full_like(x, t)
    th = torch.full_like(x, t + 0.5*h)
    te = torch.full_like(x, t + h)
    k1 = velocity_field(x, y, z, tt)
    k2 = velocity_field(x + 0.5*h*T_final*k1[0], y + 0.5*h*T_final*k1[1],
                        z + 0.5*h*T_final*k1[2], th)
    k3 = velocity_field(x + 0.5*h*T_final*k2[0], y + 0.5*h*T_final*k2[1],
                        z + 0.5*h*T_final*k2[2], th)
    k4 = velocity_field(x + h*T_final*k3[0], y + h*T_final*k3[1],
                        z + h*T_final*k3[2], te)
    x = x + h*T_final*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
    y = y + h*T_final*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6
    z = z + h*T_final*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6
    t = t + h
  return x, y, z


def reference_fields(grid, times, T_final, device, nsteps=400, chunk=1000000,
                     cache_path=None, verbose=True):
  """Reference phi at each snapshot by backward characteristics.

  Independent of the network, so compute once and reuse across every run of
  the ablation. Set `cache_path` to persist as .npz.
  """
  import os
  if cache_path and os.path.exists(cache_path):
    z = np.load(cache_path)
    if int(z['n']) == grid.n and abs(float(z['T']) - T_final) < 1e-12:
      if verbose:
        print(f'loaded reference fields from {os.path.basename(cache_path)}')
      return {t: z[f'f{i}'] for i, t in enumerate(times)}
    if verbose:
      print('cached reference has different grid/T — recomputing')

  out = {}
  t0 = time.time()
  for t_val in times:
    tau = t_val / T_final
    parts = []
    for s in range(0, grid.pts.shape[0], chunk):
      p = torch.tensor(grid.pts[s:s + chunk], dtype=torch.float32, device=device)
      with torch.no_grad():
        bx, by, bz = _backtrace(p[:, 0], p[:, 1], p[:, 2], tau, T_final, nsteps)
        parts.append((torch.sqrt((bx - XC)**2 + (by - YC)**2 + (bz - ZC)**2)
                      - R).cpu().numpy())
    out[t_val] = np.concatenate(parts, 0).reshape(grid.n, grid.n, grid.n)
    if verbose:
      print(f'  reference at t={t_val:.3f} done ({time.time()-t0:.0f}s)', flush=True)

  # self-consistency: at tau=1 the flow has reversed exactly, so the reference
  # must reproduce the initial field
  phi0 = initial_condition_np(grid.X, grid.Y, grid.Z)
  err = np.abs(out[times[-1]] - phi0)
  if verbose:
    print(f'reference self-consistency at t=T: max {err.max():.2e}, '
          f'rms {np.sqrt((err**2).mean()):.2e}  '
          f'(should be far below the network error)')

  if cache_path:
    np.savez_compressed(cache_path, n=grid.n, T=T_final,
                        **{f'f{i}': out[t] for i, t in enumerate(times)})
    if verbose:
      print('cached reference →', cache_path)
  return out


# --------------------------------------------------------------- evaluation
def evaluate(net, grid, times, T_final, device, PHI_E, band_cells=3,
             cache=False):
  """Per-snapshot metrics against the backtraced reference.

  The velocity field is divergence free, so V_EXACT is the analytic sphere
  volume at every snapshot. `vol_floor` reports how far cell-counting the
  reference itself lands from that analytic value — the noise floor of the
  volume metric, which grows where the interface is most stretched.
  """
  net.eval()
  res = {k: [] for k in ('l2_abs', 'rel_L2', 'vol_abs', 'vol_MAPE', 'eik_dev',
                         'vol_floor')}
  PHI_P = {}

  for t_val in times:
    pp = grid.predict(net, t_val / T_final, device)
    pe = PHI_E[t_val]

    res['l2_abs'].append(float(np.sqrt(np.mean((pp - pe)**2))))
    res['rel_L2'].append(float(np.sqrt(np.sum((pp - pe)**2))
                               / np.sqrt(np.sum(pe**2)) * 100))
    V_pred = np.sum(pp < 0) * grid.d**3
    res['vol_abs'].append(float(abs(V_pred - V_EXACT)))
    res['vol_MAPE'].append(float(abs(V_pred - V_EXACT) / V_EXACT * 100))
    V_ex = np.sum(pe < 0) * grid.d**3
    res['vol_floor'].append(float(abs(V_ex - V_EXACT) / V_EXACT * 100))

    gx, gy, gz = np.gradient(pp, grid.d, grid.d, grid.d)
    band = np.abs(pe) < band_cells * grid.d
    res['eik_dev'].append(
        float(np.mean(np.abs(np.sqrt(gx**2 + gy**2 + gz**2)[band] - 1.0))))
    del gx, gy, gz, band

    if cache:
      PHI_P[t_val] = pp
    else:
      del pp

  for k in list(res):
    res['avg_' + k] = float(np.mean(res[k]))
  if cache:
    res['PHI_P'], res['PHI_E'] = PHI_P, PHI_E
  return res


def print_metrics(times, res):
  print(f"\n{'Snapshot':<12}{'abs L2':>12}{'rel L2 (%)':>13}"
        f"{'vol abs':>13}{'MAPE (%)':>12}{'eik dev':>13}")
  print('-' * 75)
  for i, t in enumerate(times):
    print(f"t={t:<10.2f}{res['l2_abs'][i]:>12.2e}{res['rel_L2'][i]:>12.3f}%"
          f"{res['vol_abs'][i]:>13.2e}{res['vol_MAPE'][i]:>11.3f}%"
          f"{res['eik_dev'][i]:>13.2e}")
  print('-' * 75)
  print(f"{'avg':<12}{res['avg_l2_abs']:>12.2e}{res['avg_rel_L2']:>12.3f}%"
        f"{res['avg_vol_abs']:>13.2e}{res['avg_vol_MAPE']:>11.3f}%"
        f"{res['avg_eik_dev']:>13.2e}")
  print(f"\nvolume-metric noise floor (counting the reference): "
        f"{min(res['vol_floor']):.3f}% – {max(res['vol_floor']):.3f}%")


# ------------------------------------------------------------------ figures
ELEV, AZIM = 22, -58
XLIM, YLIM, ZLIM = (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)

PAPER_RC = {
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300,
}


def _shaded_isosurface(ax, field, spacing, base_rgb=(0.26, 0.45, 0.71),
                       light=(0.4, 0.5, 0.75), ambient=0.42):
  """Marching-cubes zero level set with Lambertian shading."""
  from skimage import measure
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  if field.min() >= 0.0 or field.max() <= 0.0:
    # degenerate prediction: no zero level set to extract (can happen when the
    # eikonal term is switched off entirely)
    ax.text2D(0.5, 0.5, 'no interface', transform=ax.transAxes,
              ha='center', va='center', fontsize=7, color='0.5')
    return
  verts, faces, normals, _ = measure.marching_cubes(field, level=0.0,
                                                    spacing=spacing)
  tris = verts[faces]
  n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
  n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
  n[np.sum(n * normals[faces].mean(axis=1), axis=1) > 0] *= -1
  L = np.array(light, float)
  L /= np.linalg.norm(L)
  shade = ambient + (1 - ambient) * np.clip(n @ L, 0, 1)
  fc = np.clip(np.array(base_rgb)[None, :] * shade[:, None], 0, 1)
  coll = Poly3DCollection(tris, facecolors=fc, edgecolors='none')
  coll.set_zsort('average')
  ax.add_collection3d(coll)


def _mc(field, spacing):
  """Marching cubes with outward-oriented face normals, or None if degenerate."""
  from skimage import measure
  if field.min() >= 0.0 or field.max() <= 0.0:
    return None
  verts, faces, normals, _ = measure.marching_cubes(field, level=0.0,
                                                    spacing=spacing)
  tris = verts[faces]
  n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
  n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
  n[np.sum(n * normals[faces].mean(axis=1), axis=1) > 0] *= -1
  return verts, faces, n


def _view_vec():
  e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
  return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])


def _exact_silhouette(ax, field, spacing, color='0.1', lw=0.3, alpha=0.55):
  """Silhouette outline of the EXACT zero level set.

  The slotted sphere is non-convex, so the analytic front-hemisphere wireframe
  used for the sphere benchmarks does not apply, and a strided triangle
  wireframe reads as speckle. Instead we draw only the true silhouette: edges
  whose two adjacent faces disagree on front/back facing. For a closed surface
  this is a clean outline, including the rim of the slot when it is visible.
  """
  from mpl_toolkits.mplot3d.art3d import Line3DCollection
  mc = _mc(field, spacing)
  if mc is None:
    return
  verts, faces, n = mc
  front = (n @ _view_vec()) > 0.0

  e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
  e = np.sort(e, axis=1)
  fid = np.tile(np.arange(faces.shape[0]), 3)
  order = np.lexsort((e[:, 1], e[:, 0]))
  e, fid = e[order], fid[order]

  shared = np.all(e[:-1] == e[1:], axis=1)
  i = np.flatnonzero(shared)
  sil = front[fid[i]] != front[fid[i + 1]]
  segs = verts[e[i][sil]]
  # mplot3d has no depth buffer: nudge the outline toward the camera so it is
  # not lost inside the shaded surface it traces
  segs = segs + 2.5 * float(spacing[0]) * _view_vec()[None, None, :]
  if segs.shape[0]:
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw,
                                         alpha=alpha, zorder=12))


def _style_3d(ax):
  ax.view_init(elev=ELEV, azim=AZIM)
  ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_zlim(*ZLIM)
  ax.set_box_aspect([XLIM[1]-XLIM[0], YLIM[1]-YLIM[0], ZLIM[1]-ZLIM[0]])
  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor('white')
    axis.pane.set_edgecolor('0.75')
    axis.pane.set_alpha(1.0)
    axis._axinfo['grid'].update(color='0.90', linewidth=0.6)
  # Tick marks without labels: on a 6-panel row the numerals collide at the
  # front corner where the x and y sequences meet. The crop is given in the
  # caption instead.
  ax.set_xticks(np.linspace(XLIM[0], XLIM[1], 3)); ax.set_xticklabels([])
  ax.set_yticks(np.linspace(YLIM[0], YLIM[1], 3)); ax.set_yticklabels([])
  ax.set_zticks(np.linspace(ZLIM[0], ZLIM[1], 3)); ax.set_zticklabels([])
  ax.tick_params(length=2, pad=-2.5)


def figure_interface(grid, times, tlabels, PHI_P, PHI_E, savepath=None,
                     ncols=3, show_exact=True):
  """Fig. A — shaded predicted isosurface with the analytic sphere overlaid."""
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  from matplotlib.lines import Line2D
  plt.rcParams.update(PAPER_RC)

  nrows = int(np.ceil(len(times) / ncols))
  fig = plt.figure(figsize=(7.2, 2.15 * nrows))
  gs = GridSpec(nrows, ncols, figure=fig, wspace=-0.01, hspace=0.10,
                left=0.005, right=0.995, top=1.06, bottom=0.0)

  for i, (t_val, lab) in enumerate(zip(times, tlabels)):
    ax = fig.add_subplot(gs[divmod(i, ncols)], projection='3d')
    ax.set_zorder(10 - i)
    ax.patch.set_visible(False)
    _style_3d(ax)
    _shaded_isosurface(ax, PHI_P[t_val], (grid.d, grid.d, grid.d))
    if show_exact:
      # on the folded sheet the exact silhouette has many branches; set
      # show_exact=False for a cleaner qualitative panel and rely on Fig. B
      _exact_silhouette(ax, PHI_E[t_val], (grid.d, grid.d, grid.d))
    ax.set_title(rf'$t={lab}$', pad=-2)
    if i == 0:
      # axis letters in DATA coordinates: reliable, unlike mplot3d's labelpad
      sx = XLIM[1] - XLIM[0]; sy = YLIM[1] - YLIM[0]; sz = ZLIM[1] - ZLIM[0]
      ax.text(0.5*(XLIM[0]+XLIM[1]), YLIM[0] - 0.14*sy, ZLIM[0],
              r'$x$', ha='center', va='top')
      ax.text(XLIM[1] + 0.10*sx, 0.5*(YLIM[0]+YLIM[1]), ZLIM[0] - 0.06*sz,
              r'$y$', ha='left', va='top')
      ax.text(XLIM[1] + 0.06*sx, YLIM[1] + 0.05*sy, 0.5*(ZLIM[0]+ZLIM[1]),
              r'$z$', ha='left', va='center')
    else:
      ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

  fig.legend(handles=[
      Line2D([], [], color=(0.26, 0.45, 0.71), lw=6, label=r'PINN  $\hat{\phi}=0$'),
      Line2D([], [], color='k', lw=0.9, label=r'Exact  $\phi=0$')],
      loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06),
      handlelength=1.6, columnspacing=2.0)

  if savepath:
    fig.savefig(savepath + '.pdf', bbox_inches='tight')
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
  return fig


def figure_verification(grid, times, res, T_final, savepath=None,
                        include_growth=True):
  """Fig. B — three orthogonal planes, error maps, and error growth.

  Panels (a)-(c) are mutually perpendicular sections: three circular
  cross-sections is what distinguishes a sphere from a cylinder.
  """
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  plt.rcParams.update(PAPER_RC)

  t_end = times[-1]
  pp, pe = res['PHI_P'][t_end], res['PHI_E'][t_end]
  X, Y, Z, x = grid.X, grid.Y, grid.Z, grid.x
  i_x = int(np.argmin(abs(x - XC)))
  j_y = int(np.argmin(abs(x - YC)))
  k_z = int(np.argmin(abs(x - ZC)))

  gx, gy, gz = np.gradient(pp, grid.d, grid.d, grid.d)
  gradmag = np.sqrt(gx**2 + gy**2 + gz**2)

  # 3 rows: slices, error maps, then error growth full width. The growth curve
  # is a line plot and reads badly squeezed into a square slot beside two maps.
  nrow = 3 if include_growth else 2
  hr = [1.0, 1.0, 0.62] if include_growth else [1.0, 1.0]
  fig = plt.figure(figsize=(7.2, 6.2 if include_growth else 4.5))
  gs = GridSpec(nrow, 6, figure=fig, height_ratios=hr,
                hspace=0.45, wspace=1.45,
                left=0.075, right=0.955, top=0.94, bottom=0.10)

  slices = [
      (X[:, :, k_z], Y[:, :, k_z], pp[:, :, k_z], pe[:, :, k_z],
       r'$x$', r'$y$', rf'(a) $z={x[k_z]:.2f}$ plane'),
      (X[:, j_y, :], Z[:, j_y, :], pp[:, j_y, :], pe[:, j_y, :],
       r'$x$', r'$z$', rf'(b) $y={x[j_y]:.2f}$ plane'),
      (Y[i_x, :, :], Z[i_x, :, :], pp[i_x, :, :], pe[i_x, :, :],
       r'$y$', r'$z$', rf'(c) $x={x[i_x]:.2f}$ plane'),
  ]
  vmax = max(abs(pp).max(), abs(pe).max())
  for k, (A, B, Pp, Pe, xl, yl, ttl) in enumerate(slices):
    ax = fig.add_subplot(gs[0, 2*k:2*k+2])
    im = ax.contourf(A, B, Pp, levels=np.linspace(-vmax, vmax, 41),
                     cmap='RdBu_r', extend='both')
    ax.contour(A, B, Pp, levels=[0], colors='k', linewidths=1.6)
    ax.contour(A, B, Pe, levels=[0], colors='#e8000b', linewidths=1.1,
               linestyles='--')
    ax.set_xlabel(xl, labelpad=1); ax.set_ylabel(yl, labelpad=0)
    ax.set_title(ttl, pad=4); ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
  cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.012,
                    ticks=[-vmax, 0, vmax])
  cb.set_label(r'$\hat{\phi}$', labelpad=-6)
  cb.ax.set_yticklabels([f'{-vmax:.2f}', '0', f'{vmax:.2f}'])
  cb.ax.tick_params(labelsize=7)

  from matplotlib.colors import LogNorm
  axd = fig.add_subplot(gs[1, 1:3])
  err_d = np.abs(pp - pe)[:, :, k_z]
  imd = axd.pcolormesh(X[:, :, k_z], Y[:, :, k_z],
                       np.clip(err_d, err_d.max()/1e4, None),
                       cmap='magma', shading='auto', rasterized=True,
                       norm=LogNorm(vmin=err_d.max()/1e4, vmax=err_d.max()))
  axd.contour(X[:, :, k_z], Y[:, :, k_z], pe[:, :, k_z], levels=[0],
              colors='w', linewidths=1.0)
  axd.set_aspect('equal'); axd.set_xlabel(r'$x$', labelpad=1)
  axd.set_ylabel(r'$y$', labelpad=0)
  axd.set_title(r'(d) $|\hat{\phi}-\phi|$ at $t=T$', pad=4)
  axd.set_xticks([0, 0.5, 1]); axd.set_yticks([0, 0.5, 1])
  cbd = fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.03)
  cbd.ax.tick_params(labelsize=6, pad=1)
  cbd.ax.yaxis.set_major_locator(__import__('matplotlib').ticker.LogLocator(numticks=4))

  axe = fig.add_subplot(gs[1, 3:5])
  eik = np.abs(gradmag[:, :, k_z] - 1.0)
  band2d = np.abs(pe[:, :, k_z]) < 5 * grid.d
  vmax_e = float(np.percentile(eik[band2d], 99.0)) if band2d.any() else eik.max()
  ime = axe.pcolormesh(X[:, :, k_z], Y[:, :, k_z], eik, cmap='viridis',
                       shading='auto', rasterized=True, vmin=0, vmax=vmax_e)
  axe.contour(X[:, :, k_z], Y[:, :, k_z], pe[:, :, k_z], levels=[0],
              colors='w', linewidths=1.0)
  axe.set_aspect('equal'); axe.set_xlabel(r'$x$', labelpad=1)
  axe.set_ylabel(r'$y$', labelpad=0)
  axe.set_title(r'(e) $\left| \, |\nabla\hat{\phi}|-1 \right|$ at $t=T$', pad=4)
  axe.set_xticks([0, 0.5, 1]); axe.set_yticks([0, 0.5, 1])
  fig.colorbar(ime, ax=axe, fraction=0.046, pad=0.03,
               extend='max').ax.tick_params(labelsize=7)

  if include_growth:
    axf = fig.add_subplot(gs[2, 0:6])
    axf.plot(times, res['rel_L2'], 'o-', color='#1f4e79', lw=1.4, ms=4,
             label=r'rel. $L_2$')
    axf.set_xlabel(r'$t$', labelpad=1)
    axf.set_ylabel(r'rel. $L_2$ error (%)', color='#1f4e79')
    axf.tick_params(axis='y', labelcolor='#1f4e79')
    axf.set_title('(f) error growth', pad=4)
    axf.grid(alpha=0.25, lw=0.5)
    ax2 = axf.twinx()
    ax2.plot(times, res['vol_MAPE'], 's--', color='#c0392b', lw=1.4, ms=4,
             label='volume')
    ax2.set_ylabel('volume MAPE (%)', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    h1, l1 = axf.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axf.legend(h1+h2, l1+l2, loc='upper left', frameon=False, ncol=2)
    axf.set_xlim(times[0], times[-1])

  if savepath:
    fig.savefig(savepath + '.pdf', bbox_inches='tight')
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
  return fig


def figure_loss_history(history, adam_epochs, title='', savepath=None):
  """Raw traces plus log-space moving average, with the optimiser switch."""
  import matplotlib.pyplot as plt
  from scipy.ndimage import uniform_filter1d
  plt.rcParams.update(PAPER_RC)

  ep = np.array(history['epoch'])
  fig, ax = plt.subplots(figsize=(7.2, 3.4))
  series = [('total', 'Total', '#1f4e79'), ('pde', 'PDE', '#e67e22'),
            ('ic', 'IC', '#27ae60'), ('eik', 'Eikonal', '#c0392b')]
  for key, lab, col in series:
    d = np.clip(np.array(history[key]), 1e-14, None)
    ax.semilogy(ep, d, color=col, alpha=0.18, lw=0.5)
    ax.semilogy(ep, 10**uniform_filter1d(np.log10(d), size=200),
                color=col, lw=1.6, label=lab)
  ax.axvline(adam_epochs, color='k', ls='--', lw=1.0, label='Adam → L-BFGS')
  ax.set_xlabel('epoch'); ax.set_ylabel('loss')
  if title:
    ax.set_title(title, pad=4)
  ax.legend(frameon=False, ncol=5, loc='upper center',
            bbox_to_anchor=(0.5, -0.22))
  ax.grid(alpha=0.25, lw=0.5)
  if savepath:
    fig.savefig(savepath + '.pdf', bbox_inches='tight')
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
  return fig


def figure_growth(times, res, savepath=None):
  """Error growth over the integration window, as a standalone figure.

  Separated from the verification panels because it is the only temporal plot
  there: squeezed into a square slot beside two field maps it renders as a flat
  strip with no room for the twin axis labels.
  """
  import matplotlib.pyplot as plt
  plt.rcParams.update(PAPER_RC)

  fig, axf = plt.subplots(figsize=(7.2, 2.7))
  fig.subplots_adjust(left=0.085, right=0.912, top=0.94, bottom=0.19)

  axf.plot(times, res['rel_L2'], 'o-', color='#1f4e79', lw=1.5, ms=4.5,
           label=r'relative $L_2$')
  axf.set_xlabel(r'$t$', labelpad=2)
  axf.set_ylabel(r'rel. $L_2$ error (%)', color='#1f4e79')
  axf.tick_params(axis='y', labelcolor='#1f4e79')
  axf.grid(alpha=0.25, lw=0.5)
  axf.set_xlim(times[0], times[-1])

  ax2 = axf.twinx()
  ax2.plot(times, res['vol_MAPE'], 's--', color='#c0392b', lw=1.5, ms=4.5,
           label='volume')
  ax2.set_ylabel('volume MAPE (%)', color='#c0392b')
  ax2.tick_params(axis='y', labelcolor='#c0392b')

  # headroom so the legend cannot sit on the curves
  lo, hi = axf.get_ylim(); axf.set_ylim(lo, hi + 0.28*(hi - lo))
  lo, hi = ax2.get_ylim(); ax2.set_ylim(lo, hi + 0.28*(hi - lo))

  h1, l1 = axf.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  axf.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False, ncol=2)

  if savepath:
    fig.savefig(savepath + '.pdf', bbox_inches='tight')
    fig.savefig(savepath + '.png', dpi=300, bbox_inches='tight')
  return fig
