"""ZD3D interface figure in matplotlib, with the camera turning with the body.

    python figures/zd3d_interface_follow.py          # full domain box, as the original figure
    python figures/zd3d_interface_follow.py zoom     # each panel cropped to the body, orbit in an inset

Same style as the original figure of zd3d_core.figure_interface, but in each panel
the camera azimuth (and the light) is turned by the rotation angle t, so the slot,
which opens towards the rotation axis, faces the viewer at every snapshot.
Needs the seed-42 SAW checkpoint (see make_figures.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import HERE, load_core, load_net  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

B, T = "ZD3D", 2 * np.pi
core = load_core(B)
CACHE = os.environ.get("ZD3D_FIELDS", os.path.join(HERE, "_zd3d_fields.npz"))
ZOOM = "zoom" in sys.argv[1:]
times = np.linspace(0, T, 6)
labels = ["0", "T/5", "2T/5", "3T/5", "4T/5", "T"]
grid = core.Grid(201)

if os.path.exists(CACHE):
    z = np.load(CACHE)
    PP = [z[f"p{i}"] for i in range(6)]
else:
    net = load_net(B, core)
    PP = []
    for i, t in enumerate(times):
        PP.append(grid.predict(net, t / T, "cpu"))
        print("field", labels[i], "done", flush=True)
    np.savez_compressed(CACHE, **{f"p{i}": p for i, p in enumerate(PP)})
PE = [core.exact_np(grid.X, grid.Y, grid.Z, t) for t in times]

plt.rcParams.update(core.PAPER_RC)
fig = plt.figure(figsize=(7.2, 2.15 * 2) if not ZOOM else (7.2, 5.0))
gs = (GridSpec(2, 3, figure=fig, wspace=-0.01, hspace=0.10, left=0.005, right=0.995, top=1.06, bottom=0.0) if not ZOOM
      else GridSpec(2, 3, figure=fig, wspace=0.0, hspace=0.06, left=0.0, right=1.0, top=0.95, bottom=0.06))
AZ0, L0 = core.AZIM, np.array([0.4, 0.5, 0.75])
for i, (t, lab) in enumerate(zip(times, labels)):
    az = AZ0 + np.degrees(t)
    core.AZIM = az                                   # read by _style_3d and _view_vec
    c, s = np.cos(t), np.sin(t)
    light = (c * L0[0] - s * L0[1], s * L0[0] + c * L0[1], L0[2])
    ax = fig.add_subplot(gs[divmod(i, 3)], projection="3d")
    ax.set_zorder(10 - i)
    ax.patch.set_visible(False)
    ctr = core.center(t)
    core._style_3d(ax)
    th = np.linspace(0, 2 * np.pi, 200)
    if ZOOM:
        h = 0.165
        ax.set_xlim(ctr[0] - h, ctr[0] + h); ax.set_ylim(ctr[1] - h, ctr[1] + h); ax.set_zlim(0.5 - h, 0.5 + h)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ins = ax.inset_axes([0.74, 0.72, 0.24, 0.24])
        ins.plot(core.CX + core.ORBIT * np.cos(th), core.CY + core.ORBIT * np.sin(th), color="0.6", lw=0.8)
        ins.plot(core.CX, core.CY, "+", color="k", ms=4)
        ins.plot(ctr[0], ctr[1], "o", color=(0.26, 0.45, 0.71), ms=4)
        cam = np.array([np.cos(np.radians(az)), np.sin(np.radians(az))])
        ins.annotate("", xy=ctr[:2], xytext=(ctr[0] + 0.24 * cam[0], ctr[1] + 0.24 * cam[1]),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
        ins.set_xlim(0.1, 0.9); ins.set_ylim(0.1, 0.9); ins.set_aspect("equal")
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color("0.75")
    else:
        ax.plot(core.CX - core.ORBIT * np.sin(th), core.CY + core.ORBIT * np.cos(th),
                np.full(200, core.ZLIM[0]), color="0.6", lw=0.7, ls=":")
        u = np.linspace(0, 2 * np.pi, 100)
        ax.plot_trisurf(ctr[0] + core.R * np.cos(u), ctr[1] + core.R * np.sin(u),
                        np.full(100, core.ZLIM[0] + 1e-4), color="0.82", shade=False, alpha=0.55, edgecolor="none")
    core._shaded_isosurface(ax, PP[i], (grid.d, grid.d, grid.d), light=light)
    core._exact_silhouette(ax, PE[i], (grid.d, grid.d, grid.d))
    ax.set_title(rf"$t={lab}$", pad=-2 if not ZOOM else 0)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
core.AZIM = AZ0
fig.legend(handles=[Line2D([], [], color=(0.26, 0.45, 0.71), lw=6, label=r"PINN  $\hat{\phi}=0$"),
                    Line2D([], [], color="k", lw=0.9, label=r"Exact  $\phi=0$")],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06 if not ZOOM else 0.0),
           handlelength=1.6, columnspacing=2.0)
for ext in ("pdf", "png"):
    name = "zd3d_interface_zoom" if ZOOM else "zd3d_interface_follow"
    fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
print("saved", name)
