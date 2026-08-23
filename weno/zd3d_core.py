"""
zd3d_core.py — shared definitions for the 3D Zalesak slotted-sphere level-set PINN.

Both notebooks import from here, so there is exactly one definition of the
method:

    01_eikonal_ablation.ipynb     w_eik x seed sweep   -> results table
    02_train_and_evaluate.ipynb   single configuration -> metrics, figures, checkpoint

Geometry: unit cube, sphere of radius 0.15 centred at (0.5, 0.75, 0.5) with a
rectangular slot of half-extents (0.025, 0.125, 0.10) centred at
(0.5, 0.725, 0.5). The slot has FINITE depth in z, so it is a blind pocket:
a section near the pole shows an unbroken sphere while a section through the
middle shows the notch. This is what makes the test genuinely three
dimensional rather than an extruded 2D disk.

Motion: solid-body rotation about the vertical axis through (0.5, 0.5), one
full revolution by t = T = 2*pi. Rotation is rigid, so the exact solution is
the initial field composed with the inverse rotation.

IMPORTANT — the initial field is not an exact SDF everywhere. It is built as
the CSG difference max(phi_sphere, -phi_slot), which returns the distance to
the nearer face rather than to the corner in the neighbourhood of the slot's
reentrant edges. So |grad phi| != 1 there IN THE EXACT SOLUTION, and the
eikonal term is in genuine conflict with the initial-condition term near those
edges. This is why this benchmark wants a much smaller eikonal weight than the
translating/rotating sphere cases, and it is worth stating explicitly.

Architecture follows the 2D ZD experiments: random Fourier feature encoding,
residual-based adaptive sampling (RAD), residual-based refinement (RAR-G), and
the causal weighting of Wang et al. (2022).

Time is normalised: the network sees tau = t/T in [0,1], which is why the
advection term in the residual carries an explicit factor of T_final.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------- geometry
CX, CY = 0.5, 0.5                     # rotation axis (vertical, through this point)
XC, YC, ZC, R = 0.5, 0.75, 0.5, 0.15  # sphere centre at t = 0
SX, SY, SZ = 0.5, 0.725, 0.5          # slot centre at t = 0
ORBIT = float(np.hypot(XC - CX, YC - CY))   # 0.25, orbital radius of the centre
HW, HH, HD = 0.025, 0.125, 0.10       # slot half-extents (HD finite -> blind pocket)


def center(t):
  """Exact sphere centre at time t (solid-body rotation about the z axis)."""
  th = t                              # omega = 2*pi / T_final = 1
  dx, dy = XC - CX, YC - CY
  return (CX + np.cos(th)*dx - np.sin(th)*dy,
          CY + np.sin(th)*dx + np.cos(th)*dy,
          ZC)


def velocity_field(x, y, z):
  return (-(y - CY), (x - CX), torch.zeros_like(z))


def _box_sdf_torch(x, y, z, cx, cy, cz, hw, hh, hd):
  dx = torch.abs(x - cx) - hw
  dy = torch.abs(y - cy) - hh
  dz = torch.abs(z - cz) - hd
  outside = torch.sqrt(torch.clamp(dx, min=0)**2 +
                       torch.clamp(dy, min=0)**2 +
                       torch.clamp(dz, min=0)**2)
  inside = torch.clamp(torch.maximum(torch.maximum(dx, dy), dz), max=0)
  return outside + inside


def _box_sdf_np(x, y, z, cx, cy, cz, hw, hh, hd):
  dx = np.abs(x - cx) - hw
  dy = np.abs(y - cy) - hh
  dz = np.abs(z - cz) - hd
  outside = np.sqrt(np.clip(dx, 0, None)**2 +
                    np.clip(dy, 0, None)**2 +
                    np.clip(dz, 0, None)**2)
  inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0.0)
  return outside + inside


def initial_condition(x, y, z):
  """Slotted sphere: sphere minus box, as the CSG difference."""
  phi_sphere = torch.sqrt((x - XC)**2 + (y - YC)**2 + (z - ZC)**2) - R
  phi_slot = _box_sdf_torch(x, y, z, SX, SY, SZ, HW, HH, HD)
  return torch.maximum(phi_sphere, -phi_slot)


def _rotate_back_torch(x, y, t):
  """Map a query point into the t=0 frame (inverse rotation about the z axis)."""
  c, s = float(np.cos(t)), float(np.sin(t))
  dx, dy = x - CX, y - CY
  return CX + c*dx + s*dy, CY - s*dx + c*dy


def exact_solution(x, y, z, t):
  xl, yl = _rotate_back_torch(x, y, t)
  return initial_condition(xl, yl, z)


def exact_np(X, Y, Z, t):
  c, s = np.cos(t), np.sin(t)
  dX, dY = X - CX, Y - CY
  Xl = CX + c*dX + s*dY
  Yl = CY - s*dX + c*dY
  phi_sphere = np.sqrt((Xl - XC)**2 + (Yl - YC)**2 + (Z - ZC)**2) - R
  phi_slot = _box_sdf_np(Xl, Yl, Z, SX, SY, SZ, HW, HH, HD)
  return np.maximum(phi_sphere, -phi_slot)


def sphere_np(X, Y, Z, t):
  """The sphere WITHOUT the slot, transported by the same rigid rotation.
  Used to isolate the region the slot removes."""
  c, s = np.cos(t), np.sin(t)
  dX, dY = X - CX, Y - CY
  Xl = CX + c*dX + s*dY
  Yl = CY - s*dX + c*dY
  return np.sqrt((Xl - XC)**2 + (Yl - YC)**2 + (Z - ZC)**2) - R


def snapshot_times(T_final, n=6):
  """n evenly spaced snapshots on [0, T], and their LaTeX labels."""
  ts = [i * T_final / (n - 1) for i in range(n)]
  labs = ['0'] + [(f'{i}T/{n-1}' if i > 1 else f'T/{n-1}') for i in range(1, n-1)] + ['T']
  return ts, labs


# -------------------------------------------------------------------- model
class PINN(nn.Module):
  """MLP on a random Fourier feature encoding of (x, y, z, tau)."""

  def __init__(self, layers=8, neurons=256, n_fourier=128, b_scale=5.0):
    super().__init__()
    self.register_buffer('B', torch.randn(4, n_fourier) * b_scale)
    net = [nn.Linear(2 * n_fourier, neurons), nn.Tanh()]
    for _ in range(layers - 1):
      net += [nn.Linear(neurons, neurons), nn.Tanh()]
    net += [nn.Linear(neurons, 1)]
    self.net = nn.Sequential(*net)
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, x):
    proj = x @ self.B
    return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))


def build_model(device, seed=42, layers=8, neurons=256, n_fourier=128,
                b_scale=5.0):
  torch.manual_seed(seed)
  if device == 'cuda':
    torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  return PINN(layers, neurons, n_fourier, b_scale).to(device)


def sample_points(N_f, N_i, device, seed=42):
  """Base collocation set and initial-condition points."""
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
  """RAD resampling of the base set plus RAR-G growth of a refinement set."""

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
    u, v, w = velocity_field(Xc[:, 0:1], Xc[:, 1:2], Xc[:, 2:3])
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
def make_loss(net, sampler, X_i, T_final, w_pde=1.0, w_ic=10.0, w_eik=0.001,
              m_chunks=32, eps_causal=1.0):
  """Causally weighted PDE residual (Wang et al. 2022) + IC + eikonal.

  Returns fn() -> (loss_pde, loss_ic, loss_eik, total). Component losses are
  unweighted so runs at different w_eik stay comparable term by term.
  """
  def loss_fn():
    X_f = sampler.points()
    phi = net(X_f)
    g = torch.autograd.grad(phi, X_f, grad_outputs=torch.ones_like(phi),
                            create_graph=True)[0]
    px, py, pz, pt = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    u, v, w = velocity_field(X_f[:, 0:1], X_f[:, 1:2], X_f[:, 2:3])
    residuals = (pt + T_final * (u*px + v*py + w*pz))**2

    # causal weighting: sort by tau, then down-weight later chunks by the
    # accumulated residual of everything before them
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

class SAW:
  """SDF-Aware Weighting — adaptive eikonal regularisation.

  Adaptive loss balancing normally assumes every residual should be driven to
  zero.  For level-set advection the eikonal residual of the EXACT solution is
  nonzero wherever the transported field departs from a signed-distance
  function, so that assumption fails and a scheme which enforces it moves the
  network away from the correct answer.  On this benchmark the initial field is
  the CSG difference max(phi_sphere, -phi_slot), which returns the distance to
  the nearer face rather than to the edge near the reentrant edges of the slot,
  so |grad phi| != 1 there in the exact solution.

  SAW has two parts:

  * Residual-quantile gate — points whose eikonal residual exceeds the q-th
                             percentile are dropped from the eikonal loss, on
                             the reasoning that a large eikonal residual is
                             evidence of genuine SDF violation rather than of
                             network error.  The threshold is EMA-smoothed.
  * Gradient-norm ratio    — the surviving eikonal term is scaled so its
                             parameter-gradient norm matches the combined
                             PDE + IC norms, clamped to [0, w_pde] and
                             EMA-smoothed.

  The gate is what allows the ratio to move: with the gate disabled the
  eikonal loss stays small, the ratio remains pinned at its clamp, and the
  weight never leaves its initial value.

  Buffers are warm-started to the population mean whenever RAD/RAR changes the
  collocation count, preventing cold-start spikes.

  Call signature matches make_loss: fn() -> (pde, ic, eik, total).
  The adaptive weights are exposed as fn.w_global after each call, and the
  full weight trajectory as fn.w_hist.
  """

  def __init__(self, net, sampler, X_i, T_final,
               beta=0.999, q=0.95,
               m_chunks=32, eps_causal=1.0):
    self.net        = net
    self._gn_params = [p for p in list(net.parameters())[-2:] if p.requires_grad]
    self.sampler    = sampler
    self.X_i        = X_i
    self.T_final    = T_final
    self.beta       = beta          # EMA smoothing
    self.q          = q             # quantile threshold for the eikonal gate
    self.m_chunks   = m_chunks
    self.eps_causal = eps_causal
    self.w_pde      = 1.0
    self.w_ic       = 10.0
    self.w_global   = None
    self.w_hist     = []      # trajectory of the adaptive weight, one per call

    self._ema_r_pde = None
    self._ema_r_eik = None
    self._ema_r_ic  = None
    self._ema_tau   = None    # smoothed quantile threshold
    self._ema_w_eik = None    # smoothed gradient-norm ratio
    self._n_f       = 0
    self._n_i       = 0

  def __call__(self):
    X_f = self.sampler.points()
    X_i = self.X_i

    # 1. Pointwise squared residuals for all three terms -------------------
    phi_f = self.net(X_f)
    g = torch.autograd.grad(phi_f, X_f, grad_outputs=torch.ones_like(phi_f),
                            create_graph=True)[0]
    px, py, pz, pt = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
    u, v, w_vel = velocity_field(X_f[:, 0:1], X_f[:, 1:2], X_f[:, 2:3])

    res_pde = (pt + self.T_final * (u*px + v*py + w_vel*pz))**2   # (N_f, 1)
    res_eik = (torch.sqrt(px**2 + py**2 + pz**2 + 1e-8) - 1.0)**2  # (N_f, 1)

    phi_i    = self.net(X_i)
    phi_ex_i = initial_condition(X_i[:, 0:1], X_i[:, 1:2], X_i[:, 2:3])
    res_ic   = (phi_i - phi_ex_i)**2                               # (N_i, 1)

    # 2. Causal PDE loss (Wang et al. 2022 — structure unchanged) ----------
    m_chunks = self.m_chunks
    chunk    = X_f.shape[0] // m_chunks
    sort_idx = torch.argsort(X_f[:, 3].detach())
    rs       = res_pde[sort_idx]
    cmeans   = [torch.mean(rs[m*chunk:(m+1)*chunk]) for m in range(m_chunks)]
    cumsum   = torch.zeros((), device=X_f.device)
    cweights = []
    for m in range(m_chunks):
      cweights.append(torch.exp(-self.eps_causal * cumsum))
      cumsum = cumsum + cmeans[m].detach()
    loss_pde = sum(cweights[m] * cmeans[m] for m in range(m_chunks)) / m_chunks

    # 3. Gradient-norm ratio ----------------------------------------------
    N_f, N_i = X_f.shape[0], X_i.shape[0]

    def _grad_norm(loss_scalar):
      # last-layer proxy: the ratio is preserved and the cost falls sharply
      grads = torch.autograd.grad(loss_scalar, self._gn_params,
                                  retain_graph=True, allow_unused=True)
      return sum(gr.norm()**2 for gr in grads if gr is not None).sqrt()

    gn_pde = _grad_norm(loss_pde).detach()
    gn_ic  = _grad_norm(self.w_ic * torch.mean(res_ic)).detach()
    gn_eik = _grad_norm(torch.mean(res_eik)).detach()

    with torch.no_grad():
      # warm-start when RAD/RAR changes the collocation count
      if N_f != self._n_f:
        m_p = (self._ema_r_pde.mean() if self._ema_r_pde is not None
               else res_pde.detach().mean())
        m_e = (self._ema_r_eik.mean() if self._ema_r_eik is not None
               else res_eik.detach().mean())
        self._ema_r_pde = torch.full((N_f, 1), float(m_p), device=X_f.device)
        self._ema_r_eik = torch.full((N_f, 1), float(m_e), device=X_f.device)
        self._n_f = N_f

      if N_i != self._n_i:
        m_i = (self._ema_r_ic.mean() if self._ema_r_ic is not None
               else res_ic.detach().mean())
        self._ema_r_ic = torch.full((N_i, 1), float(m_i), device=X_i.device)
        self._n_i = N_i

      b = self.beta
      self._ema_r_pde = b * self._ema_r_pde + (1 - b) * res_pde.detach()
      self._ema_r_eik = b * self._ema_r_eik + (1 - b) * res_eik.detach()
      self._ema_r_ic  = b * self._ema_r_ic  + (1 - b) * res_ic.detach()

      # w_eik clamped to [0, w_pde] BEFORE the EMA: gn_eik -> 0 near
      # convergence sends the raw ratio arbitrarily high, and an unclamped
      # value baked into the EMA state would take ~1/(1-beta) steps to unwind
      w_eik_raw = torch.clamp((gn_pde + gn_ic) / (gn_eik + 1e-8),
                              0.0, self.w_pde)
      if self._ema_w_eik is None:
        self._ema_w_eik = w_eik_raw
      self._ema_w_eik = self.beta * self._ema_w_eik + (1 - self.beta) * w_eik_raw
      w_eik_scalar = float(self._ema_w_eik)

      # Residual-quantile gate ------------------------------------------
      curr_q = torch.quantile(res_eik.detach().squeeze(), self.q)
      if self._ema_tau is None:
        self._ema_tau = curr_q
      self._ema_tau = 0.99 * self._ema_tau + 0.01 * curr_q
      w_pt_eik = torch.where(res_eik.detach() > self._ema_tau,
                             torch.zeros_like(res_eik),
                             torch.ones_like(res_eik))

      self.w_global = torch.tensor([self.w_pde, self.w_ic, w_eik_scalar])
      self.w_hist.append(w_eik_scalar)

    # 4. Loss assembly -----------------------------------------------------
    loss_ic  = torch.mean(res_ic)
    loss_eik = torch.mean(res_eik)
    total    = (self.w_pde * loss_pde
                + self.w_ic * loss_ic
                + w_eik_scalar * torch.mean(w_pt_eik * res_eik))

    return loss_pde, loss_ic, loss_eik, total


def make_loss_saw(net, sampler, X_i, T_final,
                  beta=0.999, q=0.95,
                  m_chunks=32, eps_causal=1.0):
  """Factory returning a callable SAW loss object.

  Drop-in replacement for make_loss.  make_loss is retained because
  01_eikonal_ablation.ipynb sweeps a fixed w_eik.
  """
  return SAW(net, sampler, X_i, T_final,
             beta=beta, q=q,
             m_chunks=m_chunks, eps_causal=eps_causal)
# ----------------------------------------------------------------- training
def train(net, loss_fn, sampler=None, adam_epochs=20000, lbfgs_epochs=500,
          lr=1e-3, eta_min=1e-5, clip=1.0, verbose=True, log_every=2000):
  """Adam under a full cosine cycle, then L-BFGS. RAD/RAR updates happen at
  the start of each Adam iteration; the collocation set is frozen for L-BFGS."""
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
          'loss_final': loss_final, 'lbfgs_gain': loss_adam - loss_final,
          'n_points': sampler.n_points() if sampler else 0}


# --------------------------------------------------------------- evaluation
class Grid:
  """Uniform evaluation grid on the unit cube.

  n=201 gives dx=0.005 exactly, so the slice planes x=0.5, y=0.75 and z=0.5
  land on grid points.
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
  """Per-snapshot metrics. Returns lists (one per snapshot) plus their means.

  The slotted sphere has no convenient closed-form volume, so the reference
  volume is obtained by cell-counting the EXACT field at t=0 on the same grid.
  Prediction and reference therefore share the same estimator, which cancels
  the staircase bias to leading order. `vol_floor` reports how much the counted
  volume of the exact field itself drifts across snapshots — the noise floor
  of the volume metric.

  eik_dev is restricted to |phi_exact| < band_cells*dx. Note that the exact
  field is not an exact SDF near the slot's reentrant edges, so a nonzero
  eik_dev there is expected and is not purely network error.
  """
  net.eval()
  res = {k: [] for k in ('l2_abs', 'rel_L2', 'vol_abs', 'vol_MAPE', 'eik_dev',
                         'vol_floor', 'slot_filled', 'solid_eroded')}
  PHI_P, PHI_E = {}, {}

  pe0 = exact_np(grid.X, grid.Y, grid.Z, 0.0)
  V_ref = float(np.sum(pe0 < 0) * grid.d**3)
  del pe0

  for t_val in times:
    pp = grid.predict(net, t_val / T_final, device)
    pe = exact_np(grid.X, grid.Y, grid.Z, t_val)

    res['l2_abs'].append(float(np.sqrt(np.mean((pp - pe)**2))))
    res['rel_L2'].append(float(np.sqrt(np.sum((pp - pe)**2))
                               / np.sqrt(np.sum(pe**2)) * 100))
    V_pred = np.sum(pp < 0) * grid.d**3
    res['vol_abs'].append(float(abs(V_pred - V_ref)))
    res['vol_MAPE'].append(float(abs(V_pred - V_ref) / V_ref * 100))
    V_ex_t = np.sum(pe < 0) * grid.d**3
    res['vol_floor'].append(float(abs(V_ex_t - V_ref) / V_ref * 100))

    # Feature-specific classification errors. The global norms cannot certify
    # the slot: replacing the slotted sphere by an unslotted one changes the
    # relative L2 error by only about one percentage point, because the slot
    # occupies a small fraction of the domain over which the norm is taken.
    # These two quantities are sign-based and therefore cannot cancel.
    psph = sphere_np(grid.X, grid.Y, grid.Z, t_val)
    removed = (psph < 0) & (pe > 0)          # cells the slot takes out
    res['slot_filled'].append(
        float(np.mean(pp[removed] < 0) * 100) if removed.any() else 0.0)
    solid = pe < 0
    res['solid_eroded'].append(float(np.mean(pp[solid] >= 0) * 100))
    del psph, removed, solid

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
  res['V_ref'] = V_ref
  if cache:
    res['PHI_P'], res['PHI_E'] = PHI_P, PHI_E
  return res


def print_metrics(times, res):
  print(f"\n{'Snapshot':<12}{'abs L2':>12}{'rel L2 (%)':>13}"
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
  print(f"\n{'Snapshot':<12}{'slot filled (%)':>18}{'solid eroded (%)':>19}")
  print('-' * 49)
  for i, t in enumerate(times):
    print(f"t={t:<10.3f}{res['slot_filled'][i]:>17.3f}%{res['solid_eroded'][i]:>18.3f}%")
  print('-' * 49)
  print(f"{'avg':<12}{res['avg_slot_filled']:>17.3f}%{res['avg_solid_eroded']:>18.3f}%")
  print(f"\nreference volume (counted, exact field at t=0): {res['V_ref']:.6e}")
  print(f"volume-metric noise floor across snapshots: "
        f"{min(res['vol_floor']):.3f}% – {max(res['vol_floor']):.3f}%")


# ------------------------------------------------------------------ figures
ELEV, AZIM = 22, -58
XLIM, YLIM, ZLIM = (0.05, 0.95), (0.05, 0.95), (0.25, 0.75)

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


def _exact_silhouette(ax, field, spacing, color='k', lw=0.9, alpha=0.95):
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
                     ncols=3):
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
    ctr = center(t_val)
    _style_3d(ax)
    _th = np.linspace(0, 2*np.pi, 200)
    ax.plot(CX - ORBIT*np.sin(_th), CY + ORBIT*np.cos(_th),
            np.full(200, ZLIM[0]), color='0.6', lw=0.7, ls=':')
    u = np.linspace(0, 2*np.pi, 100)
    ax.plot_trisurf(ctr[0] + R*np.cos(u), ctr[1] + R*np.sin(u),
                    np.full(100, ZLIM[0] + 1e-4), color='0.82',
                    shade=False, alpha=0.55, edgecolor='none')
    _shaded_isosurface(ax, PHI_P[t_val], (grid.d, grid.d, grid.d))
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
  cx_e, cy_e, cz_e = center(t_end)
  i_x = int(np.argmin(abs(x - cx_e)))
  j_y = int(np.argmin(abs(x - cy_e)))
  k_z = int(np.argmin(abs(x - cz_e)))

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


def figure_cutaway(grid, PHI_P, PHI_E, t_val, savepath=None):
  """Half-space cutaway at x = 0.5, which slices through the middle of the slot.

  The slot is a blind pocket whose mouth is a narrow opening near z = 0.5, so
  from the outside the shape reads as an almost intact sphere. This panel
  establishes the geometry for the reader: predicted on the left, exact on the
  right, both cut on the same plane.
  """
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  plt.rcParams.update(PAPER_RC)

  cut = grid.X > (CX + 1e-9)          # keep the near half
  fig = plt.figure(figsize=(7.2, 3.2))
  gs = GridSpec(1, 2, figure=fig, wspace=0.02, left=0.01, right=0.99,
                top=1.22, bottom=-0.14)

  for j, (field, name) in enumerate(((PHI_P[t_val], r'PINN  $\hat{\phi}$'),
                                     (PHI_E[t_val], r'Exact  $\phi$'))):
    ax = fig.add_subplot(gs[0, j], projection='3d')
    clipped = np.where(cut, np.abs(field) + grid.d, field)
    _style_3d(ax)
    cx, cy, cz = center(t_val)
    ax.set_xlim(cx - 1.6*R, cx + 1.6*R)
    ax.set_ylim(cy - 1.6*R, cy + 1.6*R)
    ax.set_zlim(cz - 1.6*R, cz + 1.6*R)
    ax.set_box_aspect([1, 1, 1])
    _shaded_isosurface(ax, clipped, (grid.d, grid.d, grid.d))
    ax.set_title(name + rf'   ($t={t_val:.2f}$, cut at $x={CX}$)', pad=-14)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

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
