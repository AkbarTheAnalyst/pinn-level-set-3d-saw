"""SAW ablations: one shared module for every notebook.

Uses the original benchmark code (tr3d_core, ro3d_core, zd3d_core, rv3d_core)
unchanged. Adds one loss class, SAWLoss, which reproduces the original SAW exactly
in its default mode and adds three controls:

  weight = 'ratio'   SAW's gradient-norm ratio (original behaviour)
           'replay'  a saved weight trajectory, one value per loss evaluation
  gate   = 'quantile' drop points above the smoothed q-quantile (original)
           'random'   drop a random 5% of points at every evaluation
           'none'     keep every point (full eikonal loss)

With diag=True it also records, at fixed checkpoints, the three gradient norms,
the weight, the threshold, the gated and full eikonal losses, and how well the
gate matches the points where the EXACT solution leaves the signed-distance
property (precision, recall and enrichment against an exact-solution label).
"""
import json
import math
import os
import time

import numpy as np
import torch

# ------------------------------------------------------------------ settings
BENCH = {
    'TR3D': dict(N_F=10000, N_I=5000, T=5.0, lbfgs=500, chunks=None, sampler=False, rff=False),
    'RO3D': dict(N_F=10000, N_I=5000, T=2 * math.pi, lbfgs=500, chunks=None, sampler=False, rff=False),
    'ZD3D': dict(N_F=20000, N_I=10000, T=2 * math.pi, lbfgs=500, chunks=32, sampler=True, rff=True),
    'RV3D': dict(N_F=10000, N_I=5000, T=2.0, lbfgs=2000, chunks=10, sampler=True, rff=False),
}
ADAM_EPOCHS, LAYERS, NEURONS = 20000, 8, 256
W_PDE, W_IC, BETA = 1.0, 10.0, 0.999
RAD = dict(rad_freq=1000, rad_candidates=20000, rad_k=1.0, rad_c=1.0,
           rar_freq=1800, rar_candidates=50000, rar_add=500)
GRID_N, N_SNAP, RK4_STEPS = 201, 6, 200


# ------------------------------------------------------------------ the loss
class SAWLoss:
    def __init__(self, core, bench, net, X_i, T_final, points_fn, q=0.95, beta=BETA,
                 weight='ratio', gate='quantile', replay=None, drop_frac=0.05,
                 m_chunks=None, eps_causal=1.0, diag=False, seed=0):
        self.core, self.bench, self.net = core, bench, net
        self.X_i, self.T, self.points_fn = X_i, T_final, points_fn
        self.q, self.beta = q, beta
        self.weight, self.gate = weight, gate
        self.replay = None if replay is None else np.asarray(replay, dtype=np.float64)
        self.drop_frac = drop_frac
        self.m_chunks, self.eps_causal = m_chunks, eps_causal
        self._gn_params = [p for p in list(net.parameters())[-2:] if p.requires_grad]
        self._ema_w = None
        self._thresh = None
        self.w_hist = []
        self.k = 0
        dev = next(net.parameters()).device
        self._rng = torch.Generator(device=dev).manual_seed(10_000 + seed)
        self.diag = diag
        self.diag_rows, self.diag_arrays = [], {}
        self.diag_label = make_label_fn(core, bench, T_final) if diag else None

    # -- velocity: RV3D's field depends on tau
    def _vel(self, X):
        if self.bench == 'RV3D':
            return self.core.velocity_field(X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4])
        return self.core.velocity_field(X[:, 0:1], X[:, 1:2], X[:, 2:3])

    def _pde_loss(self, X_f, res_pde):
        if not self.m_chunks:
            return torch.mean(res_pde)
        m = self.m_chunks
        chunk = X_f.shape[0] // m
        rs = res_pde[torch.argsort(X_f[:, 3].detach())]
        cmeans = [torch.mean(rs[i * chunk:(i + 1) * chunk]) for i in range(m)]
        cumsum = torch.zeros((), device=X_f.device)
        cw = []
        for i in range(m):
            cw.append(torch.exp(-self.eps_causal * cumsum))
            cumsum = cumsum + cmeans[i].detach()
        return sum(cw[i] * cmeans[i] for i in range(m)) / m

    def __call__(self):
        X_f, X_i = self.points_fn(), self.X_i
        phi_f = self.net(X_f)
        g = torch.autograd.grad(phi_f, X_f, grad_outputs=torch.ones_like(phi_f), create_graph=True)[0]
        px, py, pz, pt = g[:, 0:1], g[:, 1:2], g[:, 2:3], g[:, 3:4]
        u, v, w_vel = self._vel(X_f)
        res_pde = (pt + self.T * (u * px + v * py + w_vel * pz)) ** 2
        res_eik = (torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-8) - 1.0) ** 2
        phi_i = self.net(X_i)
        res_ic = (phi_i - self.core.initial_condition(X_i[:, 0:1], X_i[:, 1:2], X_i[:, 2:3])) ** 2
        loss_pde = self._pde_loss(X_f, res_pde)

        need_gn = self.weight == 'ratio' or self.diag
        if need_gn:
            def _gn(loss_scalar):
                grads = torch.autograd.grad(loss_scalar, self._gn_params, retain_graph=True, allow_unused=True)
                return sum(gr.norm() ** 2 for gr in grads if gr is not None).sqrt()
            gn_pde = _gn(loss_pde).detach()
            gn_ic = _gn(W_IC * torch.mean(res_ic)).detach()
            gn_eik = _gn(torch.mean(res_eik)).detach()

        with torch.no_grad():
            if self.weight == 'ratio':
                raw = torch.clamp((gn_pde + gn_ic) / (gn_eik + 1e-8), 0.0, W_PDE)
                if self._ema_w is None:
                    self._ema_w = raw
                self._ema_w = self.beta * self._ema_w + (1 - self.beta) * raw
                w = float(self._ema_w)
            else:  # replay
                w = float(self.replay[min(self.k, len(self.replay) - 1)])

            r = res_eik.detach()
            if self.gate == 'quantile' or self.diag:
                curr_q = torch.quantile(r.squeeze(), self.q)
                if self._thresh is None:
                    self._thresh = curr_q
                self._thresh = 0.99 * self._thresh + 0.01 * curr_q
            if self.gate == 'quantile':
                mask = torch.where(r > self._thresh, torch.zeros_like(r), torch.ones_like(r))
            elif self.gate == 'random':
                keep = torch.rand(r.shape, generator=self._rng, device=r.device) >= self.drop_frac
                mask = keep.to(r.dtype)
            else:
                mask = torch.ones_like(r)
            self.w_hist.append(w)

            if self.diag and is_checkpoint(self.k):
                gnorm = torch.sqrt(px ** 2 + py ** 2 + pz ** 2).detach()
                self._record(self.k, X_f.detach(), r, mask, gnorm, w, gn_pde, gn_ic, gn_eik)

        loss_ic = torch.mean(res_ic)
        loss_eik = torch.mean(res_eik)
        total = W_PDE * loss_pde + W_IC * loss_ic + w * torch.mean(mask * res_eik)
        self.k += 1
        return loss_pde, loss_ic, loss_eik, total

    # -- diagnostics
    def _record(self, k, X, r, mask, gnorm, w, gn_pde, gn_ic, gn_eik):
        gated = (mask.squeeze() == 0)
        row = dict(call=k, w=w, gn_pde=float(gn_pde), gn_ic=float(gn_ic), gn_eik=float(gn_eik),
                   thresh=float(self._thresh) if self._thresh is not None else float('nan'),
                   gate_frac=float(gated.float().mean()), n_points=int(X.shape[0]),
                   L_eik_full=float(r.mean()), L_eik_gated=float((mask * r).mean()))
        row['wL_eik_full'] = w * row['L_eik_full']
        row['wL_eik_gated'] = w * row['L_eik_gated']
        # two gates are scored: the smoothed-threshold gate SAW applies ('gate_'; for
        # 'none'/'random' runs, the one it would apply) and the plain top (1-q) of the
        # residual at this instant ('top_'), which never depends on the threshold history
        rs = r.squeeze()
        applied = (rs > self._thresh) if self._thresh is not None else gated
        instant = rs > torch.quantile(rs, self.q)
        stats, arrays = self.diag_label(X, rs, {'gate': applied, 'top': instant}, gnorm.squeeze())
        row.update(stats)
        self.diag_rows.append(row)
        self.diag_arrays[k] = arrays

    def final_diag(self):
        """One diagnostic at the end of training, without updating any state."""
        X_f = self.points_fn().detach().clone().requires_grad_(True)
        phi = self.net(X_f)
        g = torch.autograd.grad(phi.sum(), X_f)[0]
        gnorm = g[:, :3].norm(dim=1, keepdim=True).detach()
        r = (torch.sqrt(gnorm ** 2 + 1e-8) - 1.0) ** 2
        mask = torch.where(r > self._thresh, torch.zeros_like(r), torch.ones_like(r)) \
            if self._thresh is not None else torch.ones_like(r)
        w = self.w_hist[-1] if self.w_hist else float('nan')
        nan = torch.tensor(float('nan'))
        with torch.no_grad():
            self._record(-1, X_f.detach(), r, mask, gnorm, w, nan, nan, nan)


def is_checkpoint(k):
    if k < ADAM_EPOCHS:
        return k in (0, 500, 1000) or k % 2000 == 0 or k == ADAM_EPOCHS - 1
    return (k - ADAM_EPOCHS) % 1000 == 0


# ------------------------------------------------------------------ exact-solution labels
def _stats(label, gate, name):
    base = float(label.float().mean())
    ng = int(gate.sum())
    prec = float(label[gate].float().mean()) if ng else float('nan')
    rec = float(gate[label].float().mean()) if bool(label.any()) else float('nan')
    enr = prec / base if base > 0 and ng else float('nan')
    return {f'{name}_base': base, f'{name}_prec': prec, f'{name}_rec': rec, f'{name}_enrich': enr}


def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-30))


def _kink_jump(exact_fn, X, h, ndir=24):
    """Largest change of the exact gradient within distance h (24 random directions)."""
    def grad(Y):
        Y = Y.clone().requires_grad_(True)
        with torch.enable_grad():
            phi = exact_fn(Y)
            return torch.autograd.grad(phi.sum(), Y)[0][:, :3]
    Xd = X.double()
    g0 = grad(Xd)
    worst = torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
    gen = torch.Generator(device=X.device).manual_seed(11)
    for _ in range(ndir):
        v = torch.randn(X.shape[0], 3, generator=gen, dtype=torch.float64, device=X.device)
        v = v / v.norm(dim=1, keepdim=True) * h
        Y = Xd.clone()
        Y[:, :3] += v
        worst = torch.maximum(worst, (grad(Y) - g0).norm(dim=1))
    return worst


def make_label_fn(core, bench, T, n_sub=8000):
    gen_dev = None

    def subsample(X):
        nonlocal gen_dev
        n = X.shape[0]
        if n <= n_sub:
            return torch.arange(n, device=X.device)
        g = torch.Generator(device=X.device).manual_seed(5)
        return torch.randperm(n, generator=g, device=X.device)[:n_sub]

    if bench in ('TR3D', 'RO3D', 'ZD3D'):
        def exact_fn(Y):
            x, y, z, tau = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
            t = tau * T
            if bench == 'TR3D':
                return torch.sqrt((x - core.XC - core.UVEL * t) ** 2 + (y - core.YC - core.VVEL * t) ** 2
                                  + (z - core.ZC - core.WVEL * t) ** 2) - core.R
            if bench == 'RO3D':
                cx = core.CX - core.ORBIT * torch.sin(t)
                cy = core.CY + core.ORBIT * torch.cos(t)
                return torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - core.ZC) ** 2) - core.R
            c, s = torch.cos(t), torch.sin(t)
            dx, dy = x - core.CX, y - core.CY
            xl, yl = core.CX + c * dx + s * dy, core.CY - s * dx + c * dy
            pa = torch.sqrt((xl - core.XC) ** 2 + (yl - core.YC) ** 2 + (z - core.ZC) ** 2) - core.R
            pb = -core._box_sdf_torch(xl, yl, z, core.SX, core.SY, core.SZ, core.HW, core.HH, core.HD)
            return torch.maximum(pa, pb)

        def label(X, r, gates, gnorm):
            idx = subsample(X)
            Xs, rs = X[idx], r[idx]
            out = {}
            arr = {'X': Xs.cpu().numpy().astype(np.float32), 'r': rs.cpu().numpy()}
            jumps = {h: _kink_jump(exact_fn, Xs, h) for h in (0.02, 0.04)}
            with torch.no_grad():
                phi = exact_fn(Xs.double())
            for h, jump in jumps.items():
                out[f'kink{h:g}_spearman'] = _spearman(rs.double(), jump)
                arr[f'kinkjump{h:g}'] = jump.cpu().numpy().astype(np.float32)
            for gname, gate in gates.items():
                gs = gate[idx]
                arr[gname] = gs.cpu().numpy()
                for h, jump in jumps.items():
                    out.update(_stats(jump > 0.5, gs, f'{gname}_kink{h:g}'))
                far = gs & ~(jumps[0.04] > 0.5)
                out[f'{gname}_notkink_median_absphi'] = float(phi[far].abs().median()) if bool(far.any()) else float('nan')
                out[f'{gname}_notkink_share_near_interface'] = float((phi[far].abs() < 0.05).float().mean()) if bool(far.any()) else float('nan')
            return out, arr
        return label

    # RV3D: exact |grad phi_ref| from backward characteristics, central differences
    def phi_ref(x, y, z, tau):
        h = -tau / RK4_STEPS
        t = tau.clone()
        for _ in range(RK4_STEPS):
            k1 = core.velocity_field(x, y, z, t)
            k2 = core.velocity_field(x + .5 * h * T * k1[0], y + .5 * h * T * k1[1], z + .5 * h * T * k1[2], t + .5 * h)
            k3 = core.velocity_field(x + .5 * h * T * k2[0], y + .5 * h * T * k2[1], z + .5 * h * T * k2[2], t + .5 * h)
            k4 = core.velocity_field(x + h * T * k3[0], y + h * T * k3[1], z + h * T * k3[2], t + h)
            x = x + h * T * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            y = y + h * T * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            z = z + h * T * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            t = t + h
        return torch.sqrt((x - core.XC) ** 2 + (y - core.YC) ** 2 + (z - core.ZC) ** 2) - core.R

    def label(X, r, gates, gnorm):
        idx = subsample(X)
        Xs, rs, gn = X[idx].double(), r[idx].double(), gnorm[idx].double()
        x, y, z, tau = Xs[:, 0], Xs[:, 1], Xs[:, 2], Xs[:, 3]
        hfd, comps = 1e-4, []
        with torch.no_grad():
            for c in range(3):
                e = [torch.zeros_like(x) for _ in range(3)]
                e[c] = e[c] + hfd
                comps.append((phi_ref(x + e[0], y + e[1], z + e[2], tau)
                              - phi_ref(x - e[0], y - e[1], z - e[2], tau)) / (2 * hfd))
        gref = torch.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)
        d, e_net = gref - 1.0, gn - gref
        out = {'spearman_r_vs_d2': _spearman(rs, d ** 2),
               'all_share_exact_dominates': float((d.abs() > e_net.abs()).double().mean())}
        arr = {'X': Xs.cpu().numpy().astype(np.float32), 'r': rs.cpu().numpy(),
               'd': d.cpu().numpy().astype(np.float32), 'e': e_net.cpu().numpy().astype(np.float32)}
        nan = float('nan')
        for gname, gate in gates.items():
            gs = gate[idx]
            arr[gname] = gs.cpu().numpy()
            for eps in (0.1, 0.3, 0.5):
                out.update(_stats(d.abs() > eps, gs, f'{gname}_dep{eps:g}'))
            any_g = bool(gs.any())
            out[f'{gname}_median_abs_d'] = float(d[gs].abs().median()) if any_g else nan
            out[f'{gname}_median_abs_e'] = float(e_net[gs].abs().median()) if any_g else nan
            out[f'{gname}_share_exact_dominates'] = float((d[gs].abs() > e_net[gs].abs()).double().mean()) if any_g else nan
        return out, arr
    return label


# ------------------------------------------------------------------ one run
def run_one(core, bench, run, device, out_dir, grid, phi_e=None):
    """run: dict(mode in {'saw','replay','random','fixed'}, seed, q, w_eik, replay, diag)."""
    cfg = dict(BENCH[bench])
    adam = ADAM_EPOCHS
    seed = run['seed']
    if cfg['rff']:
        net = core.build_model(device, seed=seed, layers=LAYERS, neurons=NEURONS, n_fourier=128, b_scale=5.0)
    else:
        net = core.build_model(device, seed=seed, layers=LAYERS, neurons=NEURONS)
    X_f, X_i = core.sample_points(cfg['N_F'], cfg['N_I'], device, seed=seed)
    sampler = None
    if cfg['sampler']:
        sampler = core.AdaptiveSampler(net, X_f, cfg['T'], device, **RAD)
        points_fn = sampler.points
    else:
        points_fn = lambda: X_f

    mode = run['mode']
    if mode == 'fixed':
        if cfg['sampler']:
            loss_fn = core.make_loss(net, sampler, X_i, cfg['T'], W_PDE, W_IC, run['w_eik'],
                                     m_chunks=cfg['chunks'], eps_causal=1.0)
        else:
            loss_fn = core.make_loss(net, X_f, X_i, cfg['T'], W_PDE, W_IC, run['w_eik'])
    else:
        loss_fn = SAWLoss(core, bench, net, X_i, cfg['T'], points_fn, q=run.get('q', 0.95),
                         weight='replay' if mode == 'replay' else 'ratio',
                         gate={'saw': 'quantile', 'replay': 'none', 'random': 'random'}[mode],
                         replay=run.get('replay'), m_chunks=cfg['chunks'],
                         diag=run.get('diag', False), seed=seed)

    kw = dict(adam_epochs=adam, lbfgs_epochs=cfg['lbfgs'], verbose=False)
    if cfg['sampler']:
        kw['sampler'] = sampler
    tr = core.train(net, loss_fn, **kw)

    if getattr(loss_fn, 'diag', False):
        loss_fn.final_diag()

    times, _ = core.snapshot_times(cfg['T'], N_SNAP)
    if bench == 'RV3D':
        res = core.evaluate(net, grid, times, cfg['T'], device, phi_e, cache=False)
    else:
        res = core.evaluate(net, grid, times, cfg['T'], device, cache=False)

    row = dict(bench=bench, mode=mode, seed=seed, q=run.get('q', ''), w_eik=run.get('w_eik', ''))
    metrics = ['rel_L2', 'vol_MAPE', 'eik_dev', 'l2_abs', 'vol_abs']
    if bench == 'ZD3D':
        metrics += ['slot_filled', 'solid_eroded']
    for m in metrics:
        for i in range(N_SNAP):
            row[f'{m}_t{i}'] = res[m][i]
        row[f'avg_{m}'] = res['avg_' + m]
    wh = getattr(loss_fn, 'w_hist', [])
    row.update(w_eik_final=wh[-1] if wh else run.get('w_eik', ''),
               w_eik_geomean_adam=float(np.exp(np.mean(np.log(np.clip(wh[:adam], 1e-12, None))))) if wh else '',
               n_loss_calls=len(wh), loss_adam=tr['loss_adam'], loss_final=tr['loss_final'],
               lbfgs_gain=tr['lbfgs_gain'], minutes=tr['minutes'])

    tag = f"{bench}_{mode}_seed{seed}" + (f"_q{run['q']:g}" if mode in ('saw', 'random') and 'q' in run else '') \
          + (f"_w{run['w_eik']:g}" if mode == 'fixed' else '')
    if wh:
        np.savez_compressed(os.path.join(out_dir, f'{tag}_whist.npz'), w_hist=np.asarray(wh, dtype=np.float32))
    if getattr(loss_fn, 'diag', False):
        with open(os.path.join(out_dir, f'{tag}_diag.json'), 'w') as f:
            json.dump(loss_fn.diag_rows, f)
        flat = {f'k{k}_{n}': a for k, d in loss_fn.diag_arrays.items() for n, a in d.items()}
        np.savez_compressed(os.path.join(out_dir, f'{tag}_diag_arrays.npz'), **flat)
    del net, loss_fn, sampler
    if device == 'cuda':
        torch.cuda.empty_cache()
    return row, tag
