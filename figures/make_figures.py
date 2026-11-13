"""Paper figures made from the sweep CSVs and the saved seed-42 models.

    python figures/make_figures.py [sweep|growth|verification|rv|zd|all]

The CSVs are read from ../results/. The model checkpoints (*.pth, not in the
repository) are read from MODELS/<BENCH>/, by default the folder that holds this
repository; set the environment variable SAW_MODELS to change it. Runs on CPU.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
MODELS = os.environ.get("SAW_MODELS", os.path.dirname(ROOT))
torch.set_num_threads(max(1, os.cpu_count() - 1))

RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
      "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8, "font.family": "serif",
      "mathtext.fontset": "dejavuserif", "savefig.dpi": 300}
plt.rcParams.update(RC)
BENCH = ["TR3D", "RO3D", "ZD3D", "RV3D"]
NAME = {"TR3D": "TR3D, translating sphere", "RO3D": "RO3D, rotating sphere",
        "ZD3D": "ZD3D, slotted sphere", "RV3D": "RV3D, reversed vortex"}
COL = {"TR3D": "#0072B2", "RO3D": "#E69F00", "ZD3D": "#009E73", "RV3D": "#D55E00"}
MARK = {"TR3D": "o", "RO3D": "s", "ZD3D": "^", "RV3D": "D"}
T_END = {"TR3D": 5.0, "RO3D": 2 * np.pi, "ZD3D": 2 * np.pi, "RV3D": 2.0}
SELECTED = {"TR3D": 1e-1, "RO3D": 1e-1, "ZD3D": 1e-3, "RV3D": 1e-5}
INK, GREY, SAWC = "#1a1a1a", "#8c8c8c", "#1f4e79"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved", name)


def rows(path):
    return list(csv.DictReader(open(path)))


def sweep_file(b):
    return os.path.join(RES, f"{b}_eikonal_ablation_20K.csv" if b == "ZD3D" else f"{b}_eikonal_ablation.csv")


# ------------------------------------------------------------------ fixed-weight sweep
def fig_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x0 = 1e-6                                     # where w = 0 is drawn
    for b in BENCH:
        g = {}
        for r in rows(sweep_file(b)):
            g.setdefault(float(r["w_eik"]), []).append((float(r["avg_rel_L2"]), float(r["avg_vol_MAPE"])))
        ws = sorted(g)
        xs = np.array([w if w > 0 else x0 for w in ws])
        for ax, k in zip(axes, (0, 1)):
            v = [np.array([p[k] for p in g[w]]) for w in ws]
            m = np.array([a.mean() for a in v])
            lo, hi = m - np.array([a.min() for a in v]), np.array([a.max() for a in v]) - m
            ax.errorbar(xs[1:], m[1:], yerr=[lo[1:], hi[1:]], color=COL[b], marker=MARK[b], ms=4.5,
                        lw=1.4, capsize=2, elinewidth=0.8, label=NAME[b])
            ax.errorbar(xs[:1], m[:1], yerr=[lo[:1], hi[:1]], color=COL[b], marker=MARK[b], ms=4.5,
                        lw=0, capsize=2, elinewidth=0.8, mfc="white")
            s = ws.index(SELECTED[b])
            ax.plot(xs[s], m[s], marker="o", ms=11, mfc="none", mec=COL[b], mew=1.2)
    for ax, lab in zip(axes, (r"rel. $L_2$ error (%)", "volume MAPE (%)")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
        ax.minorticks_off()
        ax.axvline(3.2e-6, color=GREY, lw=0.6, ls=":")
        ax.set_xlabel(r"fixed eikonal weight $w_{\mathrm{eik}}$"); ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) field error", loc="left"); axes[1].set_title("(b) volume error", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_sweep")


# ------------------------------------------------------------------ error growth, all benchmarks
def per_snap(r, key):
    return np.array([float(r[f"{key}_t{i}"]) for i in range(6)])


def fig_growth():
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex="col")
    for j, b in enumerate(BENCH):
        fixed = [r for r in rows(sweep_file(b)) if float(r["w_eik"]) == SELECTED[b]]
        saw = [r for r in rows(os.path.join(RES, f"{b}_saw_sweep.csv")) if float(r["q"]) == 0.95]
        t = np.linspace(0, T_END[b], 6)
        for i, key in enumerate(("rel_L2", "vol_MAPE")):
            ax = axes[i, j]
            for runs, c, lab, mk in ((fixed, GREY, "selected fixed weight", "s"), (saw, SAWC, r"SAW, $q=0.95$", "o")):
                a = np.array([per_snap(r, key) for r in runs])
                ax.fill_between(t, a.min(0), a.max(0), color=c, alpha=0.18, lw=0)
                ax.plot(t, a.mean(0), color=c, marker=mk, ms=3.2, lw=1.3, label=lab)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(NAME[b].replace(", ", "\n"), fontsize=8.5)
            else:
                ax.set_xlabel(r"$t$")
        axes[0, j].set_xticks(np.linspace(0, T_END[b], 3))
        axes[1, j].set_xticklabels(["0", r"$T/2$", r"$T$"])
    axes[0, 0].set_ylabel(r"rel. $L_2$ error (%)"); axes[1, 0].set_ylabel("volume MAPE (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_growth")


# ------------------------------------------------------------------ models and fields
def load_core(b):
    sys.path.insert(0, os.path.join(MODELS, b))
    core = __import__(b.lower() + "_core")
    sys.path.pop(0)
    return core


def load_net(b, core):
    ckpt = {"TR3D": "TR3D_weik0.1_seed42.pth", "RO3D": "RO3D_weik0.1_seed42.pth",
            "ZD3D": "ZD3D_saw_b0.999_q0.95_seed42.pth", "RV3D": "RV3D_saw_b0.999_q0.95_seed42.pth"}[b]
    ck = torch.load(os.path.join(MODELS, b, ckpt), map_location="cpu", weights_only=False)
    net = core.build_model("cpu", seed=42, n_fourier=128, b_scale=5.0) if b == "ZD3D" else core.build_model("cpu", seed=42)
    net.load_state_dict(ck["model_state_dict"])
    return net.eval()


def predict(net, P, grad=False, chunk=100000):
    out, gn = [], []
    for s in range(0, P.shape[0], chunk):
        p = torch.tensor(P[s:s + chunk], dtype=torch.float32, requires_grad=grad)
        if grad:
            f = net(p)
            g = torch.autograd.grad(f.sum(), p)[0][:, :3]
            out.append(f.detach().numpy().ravel()); gn.append(g.norm(dim=1).detach().numpy())
        else:
            with torch.no_grad():
                out.append(net(p).numpy().ravel())
    return (np.concatenate(out), np.concatenate(gn)) if grad else np.concatenate(out)


def plane_points(axis, val, n=201, tn=1.0):
    u = np.linspace(0, 1, n, dtype=np.float32)
    A, B = np.meshgrid(u, u, indexing="ij")
    c = np.full(A.size, val, np.float32)
    cols = {"z": (A.ravel(), B.ravel(), c), "y": (A.ravel(), c, B.ravel()), "x": (c, A.ravel(), B.ravel())}[axis]
    return A, B, np.stack(list(cols) + [np.full(A.size, tn, np.float32)], 1)


def verification(name, planes, maps, time_label, exact_eik=None):
    """planes: list of (A, B, Pp, Pe, xlabel, ylabel, title); maps: (A, B, err, eik_net, Pe) on plane (a)."""
    ncol = 3 if exact_eik is not None else 2
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.45, wspace=1.3,
                          left=0.07, right=0.95, top=0.94, bottom=0.08)
    lo = min(min(p[2].min(), p[3].min()) for p in planes)
    hi = max(max(p[2].max(), p[3].max()) for p in planes)
    norm = TwoSlopeNorm(vmin=min(lo, -1e-3), vcenter=0.0, vmax=hi)
    levels = np.concatenate([np.linspace(norm.vmin, 0, 12)[:-1], np.linspace(0, norm.vmax, 30)])
    for k, (A, B, Pp, Pe, xl, yl, ttl) in enumerate(planes):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        im = ax.contourf(A, B, Pp, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        ax.contour(A, B, Pp, levels=[0], colors="k", linewidths=1.5)
        ax.contour(A, B, Pe, levels=[0], colors="#e8000b", linewidths=1.0, linestyles="--")
        ax.set(xlabel=xl, ylabel=yl, aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.set_title(ttl, pad=4)
    cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.012, ticks=[norm.vmin, 0, norm.vmax])
    cb.set_label(r"$\hat{\phi}$", labelpad=-4); cb.ax.set_yticklabels([f"{norm.vmin:.2f}", "0", f"{norm.vmax:.2f}"])
    cb.ax.tick_params(labelsize=7)
    A, B, err, eik_n, Pe, plane_lab = maps
    starts = [0, 2, 4] if ncol == 3 else [1, 3]
    axd = fig.add_subplot(gs[1, starts[0]:starts[0] + 2])
    e = np.clip(err, err.max() / 1e4, None)
    imd = axd.pcolormesh(A, B, e, cmap="magma", shading="auto", rasterized=True,
                         norm=LogNorm(vmin=err.max() / 1e4, vmax=err.max()))
    axd.contour(A, B, Pe, levels=[0], colors="w", linewidths=0.9)
    axd.set(aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel=planes[0][4], ylabel=planes[0][5])
    axd.set_title(rf"(d) $|\hat{{\phi}}-\phi|$, {plane_lab}", pad=4)
    fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    band = np.abs(Pe) < 5 * (A[1, 0] - A[0, 0])
    devs = [np.abs(eik_n - 1)] + ([np.abs(exact_eik - 1)] if exact_eik is not None else [])
    vmax_e = float(np.percentile(np.concatenate([d[band] for d in devs]), 99)) if band.any() else 1
    titles = [rf"(e) $\left|\,|\nabla\hat{{\phi}}|-1\right|$, {plane_lab}", rf"(f) $\left|\,|\nabla\phi|-1\right|$, exact"]
    for i, d in enumerate(devs):
        ax = fig.add_subplot(gs[1, starts[i + 1]:starts[i + 1] + 2])
        ime = ax.pcolormesh(A, B, d, cmap="viridis", shading="auto", rasterized=True, vmin=0, vmax=vmax_e)
        ax.contour(A, B, Pe, levels=[0], colors="w", linewidths=0.9)
        ax.set(aspect="equal", xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel=planes[0][4])
        ax.set_title(titles[i], pad=4)
        fig.colorbar(ime, ax=ax, fraction=0.046, pad=0.03, extend="max").ax.tick_params(labelsize=6)
    save(fig, name)


def fig_verification_rigid(b):
    core, n = load_core(b), 201
    net = load_net(b, core)
    T = T_END[b]
    centre = {"TR3D": (0.5, 0.7, 0.5), "RO3D": (0.5, 0.75, 0.5), "ZD3D": (0.5, 0.75, 0.5)}[b]
    planes, maps = [], None
    for axis, lab, (xl, yl) in (("z", "a", ("$x$", "$y$")), ("y", "b", ("$x$", "$z$")), ("x", "c", ("$y$", "$z$"))):
        val = centre["xyz".index(axis)]
        A, B, P = plane_points(axis, val, n, 1.0)
        pp, gn = predict(net, P, grad=True)
        X, Y, Z = P[:, 0].reshape(n, n), P[:, 1].reshape(n, n), P[:, 2].reshape(n, n)
        pe = core.exact_np(X, Y, Z, T)
        pp = pp.reshape(n, n)
        planes.append((A, B, pp, pe, xl, yl, rf"({lab}) ${axis}={val:.2f}$ plane"))
        if axis == "z":
            maps = (A, B, np.abs(pp - pe), gn.reshape(n, n), pe, rf"$z={val:.2f}$")
    verification(f"{b.lower()}_verification", planes, maps, rf"{NAME[b]}, $t=T$")


def rv_reference(core, P, nsteps=200):
    x, y, z, tau = (torch.tensor(P[:, i], dtype=torch.float64) for i in range(4))
    T = T_END["RV3D"]
    h = -tau / nsteps
    t = tau.clone()
    for _ in range(nsteps):
        k1 = core.velocity_field(x, y, z, t)
        k2 = core.velocity_field(x + .5 * h * T * k1[0], y + .5 * h * T * k1[1], z + .5 * h * T * k1[2], t + .5 * h)
        k3 = core.velocity_field(x + .5 * h * T * k2[0], y + .5 * h * T * k2[1], z + .5 * h * T * k2[2], t + .5 * h)
        k4 = core.velocity_field(x + h * T * k3[0], y + h * T * k3[1], z + h * T * k3[2], t + h)
        x = x + h * T * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y = y + h * T * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        z = z + h * T * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        t = t + h
    return (torch.sqrt((x - core.XC) ** 2 + (y - core.YC) ** 2 + (z - core.ZC) ** 2) - core.R).numpy()


def fig_verification_rv():
    b, n, tn = "RV3D", 201, 0.5
    core = load_core(b)
    net = load_net(b, core)
    # sections through the centroid of the deformed body
    u = np.linspace(0, 1, 101, dtype=np.float32)
    G = np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    inside = predict(net, np.c_[G, np.full(len(G), tn, np.float32)]) < 0
    c = G[inside].mean(0)
    c = np.round(c / 0.005) * 0.005
    print("RV3D centroid at T/2:", c)
    planes, maps = [], None
    for axis, lab, (xl, yl) in (("z", "a", ("$x$", "$y$")), ("y", "b", ("$x$", "$z$")), ("x", "c", ("$y$", "$z$"))):
        val = float(c["xyz".index(axis)])
        A, B, P = plane_points(axis, val, n, tn)
        pp, gn = predict(net, P, grad=True)
        pe = rv_reference(core, P)
        planes.append((A, B, pp.reshape(n, n), pe.reshape(n, n), xl, yl, rf"({lab}) ${axis}={val:.2f}$ plane"))
        if axis == "z":
            hd, comps = 1e-4, []
            for k in range(3):
                Pp_, Pm_ = P.copy(), P.copy()
                Pp_[:, k] += hd; Pm_[:, k] -= hd
                comps.append((rv_reference(core, Pp_) - rv_reference(core, Pm_)) / (2 * hd))
            ge = np.sqrt(sum(cc ** 2 for cc in comps)).reshape(n, n)
            maps = (A, B, np.abs(pp - pe).reshape(n, n), gn.reshape(n, n), pe.reshape(n, n), rf"$z={val:.2f}$")
    verification("rv3d_verification", planes, maps, r"RV3D, reversed vortex, $t=T/2$ (largest deformation)", exact_eik=ge)


# ------------------------------------------------------------------ slotted sphere in the body frame
def fig_zd_body():
    """Sections of the slotted sphere in the frame that rotates with it."""
    b = "ZD3D"
    core = load_core(b)
    net = load_net(b, core)
    n = 241
    cx, cy = 0.5, 0.5
    u1 = np.linspace(0.575, 0.925, n, dtype=np.float32)        # body y
    u2 = np.linspace(0.325, 0.675, n, dtype=np.float32)        # body z and body x
    secs = {"x": np.meshgrid(u1, u2, indexing="ij"),           # x_b = 0.5: (y_b, z_b)
            "z": np.meshgrid(u2, u1, indexing="ij")}           # z_b = 0.5: (x_b, y_b)
    times = np.linspace(0, T_END[b], 6)
    labels = ["0", "T/5", "2T/5", "3T/5", "4T/5", "T"]
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.15))
    for row, key in enumerate(("x", "z")):
        A, B = secs[key]
        if key == "x":
            XB, YB, ZB = np.full_like(A, 0.5), A, B
        else:
            XB, YB, ZB = A, B, np.full_like(A, 0.5)
        pe = core.exact_np(XB, YB, ZB, 0.0)
        for j, (t, lab) in enumerate(zip(times, labels)):
            c, s_ = np.cos(t), np.sin(t)
            XW = cx + c * (XB - cx) - s_ * (YB - cy)
            YW = cy + s_ * (XB - cx) + c * (YB - cy)
            P = np.stack([XW.ravel(), YW.ravel(), ZB.ravel(), np.full(XB.size, t / T_END[b], np.float32)], 1)
            pp = predict(net, P).reshape(A.shape)
            ax = axes[row, j]
            ax.contourf(A, B, (pp < 0).astype(float), levels=[0.5, 1.5], colors=["#4a74b4"])
            ax.contour(A, B, pe, levels=[0], colors="#e8000b", linewidths=0.9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.7")
            if row == 0:
                ax.set_title(rf"$t={lab}$", pad=3)
        axes[row, 0].set_ylabel(r"$x_b=0.5$ plane" if key == "x" else r"$z_b=0.5$ plane", fontsize=8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color="#4a74b4", label=r"PINN solid $\hat\phi<0$"),
                        Line2D([], [], color="#e8000b", lw=1.2, label=r"exact $\phi=0$")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.1, wspace=0.06, hspace=0.08)
    save(fig, "zd3d_bodyframe")


# ------------------------------------------------------------------ zoomed verification, rigid benchmarks
def fig_verification_zoom():
    """Signed error and eikonal deviation near the interface at t = T, common colour scales."""
    from matplotlib.colors import SymLogNorm
    cols = [("TR3D", "z", (0.5, 0.7, 0.5), "TR3D, $z=0.50$", ("$x$", "$y$")),
            ("RO3D", "z", (0.5, 0.75, 0.5), "RO3D, $z=0.50$", ("$x$", "$y$")),
            ("ZD3D", "z", (0.5, 0.75, 0.5), "ZD3D, $z=0.50$", ("$x$", "$y$")),
            ("ZD3D", "x", (0.5, 0.75, 0.5), "ZD3D, $x=0.50$", ("$y$", "$z$"))]
    half, n = 0.2, 241
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.25))
    enorm = SymLogNorm(linthresh=3e-4, vmin=-3e-2, vmax=3e-2, base=10)
    dnorm = LogNorm(vmin=1e-3, vmax=1.0)
    cores, nets = {}, {}
    for j, (b, axis, ctr, title, (xl, yl)) in enumerate(cols):
        if b not in cores:
            cores[b] = load_core(b); nets[b] = load_net(b, cores[b])
        core, net = cores[b], nets[b]
        k = "xyz".index(axis)
        ia, ib = [m for m in range(3) if m != k]
        ua = np.linspace(ctr[ia] - half, ctr[ia] + half, n, dtype=np.float32)
        ub = np.linspace(ctr[ib] - half, ctr[ib] + half, n, dtype=np.float32)
        A, Bm = np.meshgrid(ua, ub, indexing="ij")
        P = np.zeros((A.size, 4), np.float32)
        P[:, ia], P[:, ib], P[:, k], P[:, 3] = A.ravel(), Bm.ravel(), ctr[k], 1.0
        pp, gn = predict(net, P, grad=True)
        pp, gn = pp.reshape(n, n), gn.reshape(n, n)
        pe = core.exact_np(P[:, 0].reshape(n, n), P[:, 1].reshape(n, n), P[:, 2].reshape(n, n), T_END[b])
        ax = axes[0, j]
        ime = ax.pcolormesh(A, Bm, pp - pe, cmap="RdBu_r", norm=enorm, shading="auto", rasterized=True)
        ax.contour(A, Bm, pe, levels=[0], colors="k", linewidths=1.0, linestyles="--")
        ax.contour(A, Bm, pp, levels=[0], colors="k", linewidths=0.9)
        ax.set_title(title, pad=3)
        ax2 = axes[1, j]
        imd = ax2.pcolormesh(A, Bm, np.clip(np.abs(gn - 1), 1e-3, None), cmap="viridis", norm=dnorm,
                             shading="auto", rasterized=True)
        ax2.contour(A, Bm, pe, levels=[0], colors="w", linewidths=0.8, linestyles="--")
        for a in (ax, ax2):
            a.set_aspect("equal")
            a.set_xticks([round(float(ua[0]) + .05, 2), round(float(ua[-1]) - .05, 2)])
            a.set_yticks([round(float(ub[0]) + .05, 2), round(float(ub[-1]) - .05, 2)])
            a.tick_params(labelsize=7)
            a.set_xlabel(xl, labelpad=0)
        ax.set_ylabel(yl, labelpad=0); ax2.set_ylabel(yl, labelpad=0)
        print("panel", title, "max |err| %.2e" % np.abs(pp - pe).max(), flush=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.94, bottom=0.08, wspace=0.42, hspace=0.32)
    c1 = fig.add_axes([0.9, 0.55, 0.015, 0.37]); c2 = fig.add_axes([0.9, 0.09, 0.015, 0.37])
    cb1 = fig.colorbar(ime, cax=c1, ticks=[-1e-2, -1e-3, 0, 1e-3, 1e-2], extend="both")
    cb1.ax.set_yticklabels([r"$-10^{-2}$", r"$-10^{-3}$", "0", r"$10^{-3}$", r"$10^{-2}$"]); cb1.ax.tick_params(labelsize=7)
    cb1.set_label(r"$\hat\phi-\phi$", labelpad=2)
    cb2 = fig.colorbar(imd, cax=c2, extend="both"); cb2.ax.tick_params(labelsize=7)
    cb2.set_label(r"$\left|\,|\nabla\hat\phi|-1\right|$", labelpad=2)
    save(fig, "fig_verification_rigid")


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    if "sweep" in what or "all" in what:
        fig_sweep()
    if "growth" in what or "all" in what:
        fig_growth()
    if "verification" in what or "all" in what:
        for b in ("TR3D", "RO3D", "ZD3D"):
            fig_verification_rigid(b)
    if "rv" in what or "all" in what:
        fig_verification_rv()
    if "zd" in what or "all" in what:
        fig_zd_body()
    if "zoom" in what or "all" in what:
        fig_verification_zoom()
