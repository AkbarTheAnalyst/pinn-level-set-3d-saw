"""
tr3d_core.py — shared definitions for the 3D translating-sphere level-set PINN.

Both notebooks import from here, so there is exactly one definition of the
method:

    01_train_and_evaluate.ipynb   single configuration -> metrics, figures, checkpoint
    02_eikonal_ablation.ipynb     w_eik x seed sweep   -> results table

Geometry: unit cube, sphere of radius 0.15 centred at (0.5, 0.2, 0.5) at t=0,
translating with velocity (0, 0.1, 0) so the centre reaches y=0.7 at t=T=5.

Time is normalised: the network sees tau = t/T in [0,1], which is why the
advection term in the residual carries an explicit factor of T_final.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------- geometry
XC, YC, ZC, R = 0.5, 0.2, 0.5, 0.15
UVEL, VVEL, WVEL = 0.0, 0.1, 0.0
V_EXACT = (4.0 / 3.0) * np.pi * R**3


def velocity_field(x, y, z):
  return (UVEL * torch.ones_like(x),
          VVEL * torch.ones_like(y),
          WVEL * torch.ones_like(z))


def initial_condition(x, y, z):
  return torch.sqrt((x - XC)**2 + (y - YC)**2 + (z - ZC)**2) - R


def exact_solution(x, y, z, t):
  return torch.sqrt((x - XC - UVEL*t)**2 + (y - YC - VVEL*t)**2
                    + (z - ZC - WVEL*t)**2) - R


def exact_np(X, Y, Z, t):
  return np.sqrt((X - XC - UVEL*t)**2 + (Y - YC - VVEL*t)**2
                 + (Z - ZC - WVEL*t)**2) - R


def snapshot_times(T_final, n=6):
  """n evenly spaced snapshots on [0, T], and their LaTeX labels."""
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
  """Collocation and initial-condition points. Offset mirrors the 42 -> 100
  pairing used in the original notebook, so seed=42 reproduces it exactly."""
  torch.manual_seed(seed + 58)
  if device == 'cuda':
    torch.cuda.manual_seed(seed + 58)
  X_f = torch.rand(N_f, 4, dtype=torch.float32, device=device)
  X_f.requires_grad = True
  xi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  yi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  zi = torch.rand(N_i, 1, dtype=torch.float32, device=device)
  ti = torch.zeros(N_i, 1, dtype=torch.float32, device=device)
  return X_f, torch.cat([xi, yi, zi, ti], dim=1)


# --------------------------------------------------------------------- loss
def make_loss(net, X_f, X_i, T_final, w_pde=1.0, w_ic=10.0, w_eik=0.1):
  """Returns fn() -> (loss_pde, loss_ic, loss_eik, total)."""
  def loss_fn():
    phi = net(X_f)
    g = torch.autograd.grad(phi, X_f, grad_outputs=torch.ones_like(phi),
                            create_graph=True)[0]
    px, py, pz, pt = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    u, v, w = velocity_field(X_f[:, 0:1], X_f[:, 1:2], X_f[:, 2:3])

    loss_pde = torch.mean((pt + T_final * (u*px + v*py + w*pz))**2)
    loss_ic = torch.mean((initial_condition(X_i[:, 0:1], X_i[:, 1:2],
                                            X_i[:, 2:3]) - net(X_i))**2)
    loss_eik = torch.mean((torch.sqrt(px**2 + py**2 + pz**2 + 1e-8) - 1)**2)
    return (loss_pde, loss_ic, loss_eik,
            w_pde*loss_pde + w_ic*loss_ic + w_eik*loss_eik)
  return loss_fn


# ----------------------------------------------------------------- training
def train(net, loss_fn, adam_epochs=20000, lbfgs_epochs=500, lr=1e-3,
          eta_min=1e-5, clip=1.0, verbose=True, log_every=2000):
  """Adam under a full cosine cycle, then L-BFGS. Component losses in the
  history are unweighted, so runs at different w_eik stay comparable."""
  hist = {'epoch': [], 'total': [], 'pde': [], 'ic': [], 'eik': []}
  t0 = time.time()

  opt = optim.Adam(net.parameters(), lr=lr)
  sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_epochs,
                                             eta_min=eta_min)
  for ep in range(adam_epochs):
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

    if verbose and (ep % log_every == 0 or ep == adam_epochs - 1):
      print(f'Adam {ep:>6d} | total {loss.item():.3e} | pde {l_pde.item():.3e} '
            f'| ic {l_ic.item():.3e} | eik {l_eik.item():.3e} '
            f'| lr {sch.get_last_lr()[0]:.3e}', flush=True)

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
    if verbose and (ep % 200 == 0 or ep == lbfgs_epochs - 1):
      print(f'LBFGS {ep:>5d} | total {store["total"]:.3e} '
            f'| pde {store["pde"]:.3e} | ic {store["ic"]:.3e} '
            f'| eik {store["eik"]:.3e}', flush=True)

  loss_final = float(hist['total'][-1])
  minutes = (time.time() - t0) / 60
  if verbose:
    print(f'training done in {minutes:.1f} min '
          f'(L-BFGS gain {loss_adam - loss_final:.2e})', flush=True)

  return {'history': hist, 'minutes': minutes, 'loss_adam': loss_adam,
          'loss_final': loss_final, 'lbfgs_gain': loss_adam - loss_final}


# --------------------------------------------------------------- evaluation
class Grid:
  """Uniform evaluation grid on the unit cube.

  n=201 gives dx=0.005 exactly, so the slice planes z=0.5 and y=0.7 land on
  grid points, and the cell-counting volume estimator sits ~0.1% from the
  analytic volume (vs ~0.5% at n=101).
  """

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


def evaluate(net, grid, times, T_final, device, band_cells=3, cache=False):
  """Per-snapshot metrics. Returns lists (one entry per snapshot) plus their
  means. Set cache=True to also keep the predicted/exact fields for plotting.

  l2_abs  : sqrt(mean(e^2)). On the unit cube this equals the discrete L2
            norm, since |Omega| = 1.
  rel_L2  : standard relative-L2 PINN convention, in percent.
  vol_MAPE: |V_pred - V_exact| / V_exact, V_pred by cell counting.
  eik_dev : mean ||grad phi| - 1| restricted to |phi_exact| < band_cells*dx,
            which excludes the SDF kink at the sphere centre.
  """
  net.eval()
  res = {k: [] for k in ('l2_abs', 'rel_L2', 'vol_abs', 'vol_MAPE', 'eik_dev')}
  PHI_P, PHI_E = {}, {}

  for t_val in times:
    pp = grid.predict(net, t_val / T_final, device)
    pe = exact_np(grid.X, grid.Y, grid.Z, t_val)

    res['l2_abs'].append(float(np.sqrt(np.mean((pp - pe)**2))))
    res['rel_L2'].append(float(np.sqrt(np.sum((pp - pe)**2))
                               / np.sqrt(np.sum(pe**2)) * 100))
    V_pred = np.sum(pp < 0) * grid.d**3
    res['vol_abs'].append(float(abs(V_pred - V_EXACT)))
    res['vol_MAPE'].append(float(abs(V_pred - V_EXACT) / V_EXACT * 100))

    gx, gy, gz = np.gradient(pp, grid.d, grid.d, grid.d)
    band = np.abs(pe) < band_cells * grid.d
    res['eik_dev'].append(
        float(np.mean(np.abs(np.sqrt(gx**2 + gy**2 + gz**2)[band] - 1.0))))
    del gx, gy, gz, band

    if cache:
      PHI_P[t_val], PHI_E[t_val] = pp, pe
    else:
      del pp, pe

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


# ------------------------------------------------------------------ figures
ELEV, AZIM = 22, -58
XLIM, YLIM, ZLIM = (0.18, 0.82), (0.0, 1.0), (0.18, 0.82)

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


def _exact_wireframe(ax, center, r, n_meri=6, n_para=3, color='k', lw=0.7):
  """Analytic sphere, front hemisphere only (mplot3d has no depth buffer)."""
  e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
  view = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
  cx, cy, cz = center

  def draw(px, py, pz):
    d = ((px-cx)*view[0] + (py-cy)*view[1] + (pz-cz)*view[2]) / r
    px, py, pz = px.copy(), py.copy(), pz.copy()
    m = d <= 0.02
    px[m] = np.nan; py[m] = np.nan; pz[m] = np.nan
    ax.plot(px, py, pz, color=color, lw=lw, alpha=0.9, zorder=10)

  u = np.linspace(0, 2*np.pi, 200)
  for th in np.linspace(0, np.pi, n_meri, endpoint=False):
    draw(cx + r*np.sin(u)*np.cos(th), cy + r*np.sin(u)*np.sin(th), cz + r*np.cos(u))
  for ph in np.linspace(0, np.pi, n_para+2)[1:-1]:
    draw(cx + r*np.sin(ph)*np.cos(u), cy + r*np.sin(ph)*np.sin(u),
         cz + r*np.cos(ph)*np.ones_like(u))


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
  ax.set_xticks([0.2, 0.5, 0.8]); ax.set_xticklabels([])
  ax.set_yticks([0.0, 0.5, 1.0]); ax.set_yticklabels([])
  ax.set_zticks([0.2, 0.5, 0.8]); ax.set_zticklabels([])
  ax.tick_params(length=2, pad=-2.5)


def figure_interface(grid, times, tlabels, PHI_P, savepath=None, ncols=3):
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
    ctr = (XC + UVEL*t_val, YC + VVEL*t_val, ZC + WVEL*t_val)
    _style_3d(ax)
    ax.plot([XC, XC], [YLIM[0], YLIM[1]], [ZLIM[0], ZLIM[0]],
            color='0.6', lw=0.7, ls=':')
    u = np.linspace(0, 2*np.pi, 100)
    ax.plot_trisurf(ctr[0] + R*np.cos(u), ctr[1] + R*np.sin(u),
                    np.full(100, ZLIM[0] + 1e-4), color='0.82',
                    shade=False, alpha=0.55, edgecolor='none')
    _shaded_isosurface(ax, PHI_P[t_val], (grid.d, grid.d, grid.d))
    _exact_wireframe(ax, ctr, R)
    ax.set_title(rf'$t={lab}$', pad=-2)
    if i == 0:
      # axis letters in DATA coordinates: reliable, unlike mplot3d's labelpad
      mx = 0.5*(XLIM[0] + XLIM[1]); my = 0.5*(YLIM[0] + YLIM[1])
      mz = 0.5*(ZLIM[0] + ZLIM[1])
      ax.text(mx, YLIM[0] - 0.14, ZLIM[0], r'$x$', ha='center', va='top')
      ax.text(XLIM[1] + 0.10, my, ZLIM[0] - 0.03, r'$y$', ha='left', va='top')
      ax.text(XLIM[1] + 0.06, YLIM[1] + 0.05, mz, r'$z$', ha='left', va='center')
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
  j_y = int(np.argmin(abs(x - (YC + VVEL*t_end))))
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
