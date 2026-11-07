"""Figure: what the gate removes (slotted sphere slice, vortex precision)."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "ablations")
OUT = os.path.join(HERE, "fig_gate.pdf")

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "mathtext.fontset": "dejavuserif"})
GREY, ACCENT, INK = "#b8b8b8", "#c0392b", "#1a1a1a"

fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw={"width_ratios": [1, 1.35]})

# (a) slotted sphere: points in the body frame, slab |z - 0.5| < 0.05, three seeds pooled
xs, crease, gated = [], [], []
for s in (42, 43, 44):
    z = np.load(os.path.join(RES, "gate_diag_ZD3D", f"ZD3D_saw_seed{s}_q0.95_diag_arrays.npz"))
    X, g, k = z["k-1_X"], z["k-1_gate"], z["k-1_kinkjump0.02"] > 0.5
    t = X[:, 3] * 2 * np.pi
    dx, dy = X[:, 0] - 0.5, X[:, 1] - 0.5
    bx = 0.5 + np.cos(t) * dx + np.sin(t) * dy          # rotate back by -t
    by = 0.5 - np.sin(t) * dx + np.cos(t) * dy
    m = np.abs(X[:, 2] - 0.5) < 0.05
    xs.append(np.c_[bx, by][m]); crease.append(k[m]); gated.append(g[m])
P, C, G = np.vstack(xs), np.concatenate(crease), np.concatenate(gated)


def phi0(x, y, z):
    sph = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.5) ** 2) - 0.15
    q = np.stack([np.abs(x - 0.5) - 0.025, np.abs(y - 0.725) - 0.125, np.abs(z - 0.5) - 0.10 + 0 * x])
    box = np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(0), 0)
    return np.maximum(sph, -box)


# region where the exact gradient jumps by > 0.5 within 0.02 (in-plane directions, z = 0.5)
n, e = 361, 1e-4
gx_, gy_ = np.meshgrid(np.linspace(0.32, 0.68, n), np.linspace(0.57, 0.93, n))
grad = lambda x, y: np.stack([(phi0(x + e, y, .5) - phi0(x - e, y, .5)) / (2 * e),
                              (phi0(x, y + e, .5) - phi0(x, y - e, .5)) / (2 * e),
                              (phi0(x, y, .5 + e) - phi0(x, y, .5 - e)) / (2 * e)])
g0, jump = grad(gx_, gy_), np.zeros_like(gx_)
for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
    jump = np.maximum(jump, np.linalg.norm(grad(gx_ + 0.02 * np.cos(ang), gy_ + 0.02 * np.sin(ang)) - g0, axis=0))
a.contourf(gx_, gy_, (jump > 0.5).astype(float), levels=[0.5, 1.5], colors=["#f3d3cf"])
a.scatter(*P[~C & ~G].T, s=3, c=GREY, lw=0, label="collocation point")
a.scatter(*P[C].T, s=9, c=ACCENT, lw=0, label="crease point (exact solution); shaded: crease region at z = 0.5")
a.scatter(*P[G].T, s=22, facecolors="none", edgecolors=INK, lw=0.7, label="excluded by the gate")
a.add_patch(Circle((0.5, 0.75), 0.15, fill=False, ec=INK, lw=0.8, ls="--"))
a.add_patch(Rectangle((0.475, 0.60), 0.05, 0.25, fill=False, ec=INK, lw=0.8, ls="--"))
a.set(xlim=(0.32, 0.68), ylim=(0.57, 0.93), aspect="equal", xlabel="$x$ (body frame)",
      ylabel="$y$ (body frame)")
a.set_title("(a) slotted sphere, $|z-0.5|<0.05$", fontsize=10, loc="left")
a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), fontsize=8, frameon=False, markerscale=1.5)

# (b) reversed vortex: precision of the gate against training progress
base = []
for i, s in enumerate((42, 43, 44)):
    D = [d for d in json.load(open(os.path.join(RES, "gate_diag_RV3D", f"RV3D_saw_seed{s}_q0.95_diag.json")))
         if d["call"] >= 0]
    c = np.array([d["call"] for d in D]); p = np.array([d["gate_dep0.5_prec"] for d in D])
    base += [d["gate_dep0.5_base"] for d in D]
    b.plot(c, 100 * p, color=ACCENT, lw=1.6, marker="o", ms=3.5, alpha=0.9,
           label="share of excluded points that depart (3 seeds)" if i == 0 else None)
b.axhline(100 * np.mean(base), color=INK, lw=1.0, ls="--",
          label=f"share of all points that depart ({100 * np.mean(base):.0f}%)")
b.axvline(2e4, color=GREY, lw=1.0)
b.text(2e4 + 400, 6, "end of Adam", fontsize=8, color="#555555")
b.set(ylim=(0, 105), xlabel="loss evaluation", ylabel="%")
b.set_title(r"(b) reversed vortex, departure $|\,\|\nabla\phi\|-1\,|>0.5$", fontsize=10, loc="left")
b.grid(alpha=0.25, lw=0.5)
b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=8, frameon=False)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print("saved", OUT, "points in slab", len(P), "crease", C.sum(), "gated", G.sum(), "both", (C & G).sum())
