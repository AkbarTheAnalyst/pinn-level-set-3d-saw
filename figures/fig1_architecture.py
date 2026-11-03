"""Figure 1 - architecture and loss schematic for the 3D level-set PINN."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
})

# hue for the reader, lightness for the photocopier
NAVY   = '#22364f'
MIDGY  = '#8c8c8c'
LTGY   = '#cfcfcf'
BLUE   = '#5b8cb8'
RED    = '#2e6ea6'   # deeper tint of the same hue, not a second colour
DKGY   = '#4a4a4a'
PANEL  = '#f2f2f2'
EDGE   = '#9a9a9a'

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

ax.add_patch(Rectangle((0.015, 0.275), 0.50, 0.70, fc=PANEL, ec='none', zorder=0))
ax.add_patch(Rectangle((0.545, 0.275), 0.44, 0.70, fc=PANEL, ec='none', zorder=0))
ax.text(0.265, 0.935, 'Neural network', ha='center', fontsize=11.5)
ax.text(0.765, 0.935, 'Physics-informed residuals', ha='center', fontsize=11.5)

R = 0.030
ASPECT = 7.2 / 4.4
def node(x, y, label, fc, tc='white', fs=9, r=R):
    ax.add_patch(Ellipse((x, y), 2*r, 2*r*ASPECT, fc=fc, ec='none', zorder=3))
    ax.text(x, y, label, ha='center', va='center', color=tc, fontsize=fs, zorder=4)

def arrow(p, q, color=EDGE, lw=1.1, ls='-', shrink=3, alpha=1.0, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=shrink, shrinkB=shrink, alpha=alpha,
                                 zorder=z))

def carrow(p, q, rad, color=EDGE, lw=1.1, ls='-', alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls, alpha=alpha,
                                 shrinkA=3, shrinkB=3, zorder=2,
                                 connectionstyle=f'arc3,rad={rad}'))

def box(x, y, w, h, label, fc, tc='white', fs=9):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=fc, ec='none', zorder=3))
    ax.text(x, y, label, ha='center', va='center', color=tc, fontsize=fs, zorder=4)

# ---------------------------------------------------------------- inputs
xin = 0.068
yin = [0.845, 0.730, 0.615, 0.500]
for y, lab in zip(yin, ['$x$', '$y$', '$z$', r'$\tau$']):
    node(xin, y, lab, NAVY)

# ------------------------------------------------- Fourier encoding (opt.)
xf = 0.163
ax.add_patch(FancyBboxPatch((xf - 0.028, 0.470), 0.056, 0.415,
                            boxstyle='round,pad=0.004,rounding_size=0.012',
                            fc='none', ec=MIDGY, lw=1.0, ls=(0, (3, 2)),
                            zorder=3))
ax.text(xf, 0.672, r'$\gamma(\cdot)$', ha='center', va='center', fontsize=11,
        zorder=4)
for y in yin:
    arrow((xin, y), (xf - 0.030, 0.672), alpha=0.55, lw=0.9)

# ---------------------------------------------------------------- hidden
xh = [0.268, 0.386]
yh = [0.870, 0.775, 0.680, 0.560, 0.465, 0.370]
for x in xh:
    for y in yh:
        node(x, y, '', LTGY, r=0.022)
    ax.text(x, 0.620, r'$\vdots$', ha='center', va='center', fontsize=13,
            color='0.45')
ax.text(0.327, 0.870, r'$\cdots$', ha='center', va='center', fontsize=11, color='0.5')
ax.text(0.327, 0.370, r'$\cdots$', ha='center', va='center', fontsize=11, color='0.5')
for y in yh:
    arrow((xf + 0.030, 0.672), (xh[0], y), alpha=0.45, lw=0.8)
for ya in yh:
    for yb in yh:
        arrow((xh[0], ya), (xh[1], yb), alpha=0.30, lw=0.7)

# ---------------------------------------------------------------- output
xo, yo = 0.478, 0.620
node(xo, yo, r'$\hat{\phi}$', NAVY, fs=10)
for y in yh:
    arrow((xh[1], y), (xo, yo), alpha=0.45, lw=0.9)

# ------------------------------------------------- automatic differentiation
xd = 0.632
yd = [0.855, 0.750, 0.645, 0.540]
dl = [r'$\partial_\tau$', r'$\partial_x$', r'$\partial_y$', r'$\partial_z$']
for y, lab in zip(yd, dl):
    node(xd, y, lab, MIDGY, fs=8.5, r=0.027)
    arrow((xo, yo), (xd, y), alpha=0.6, lw=1.0)


# ---------------------------------------------------------------- losses
box(0.858, 0.820, 0.175, 0.095, r'$\mathcal{L}_{\mathrm{pde}}$', BLUE, fs=11)
for y in yd:
    arrow((xd, y), (0.766, 0.820), color=EDGE, lw=0.9, alpha=0.7)

box(0.858, 0.625, 0.175, 0.095, r'$\mathcal{L}_{\mathrm{eik}}$', RED, fs=11)
for y in yd[1:]:
    arrow((xd, y), (0.766, 0.625), color=RED, lw=1.0, alpha=0.55)

# ---- initial-condition loss joins the stack ----------------------------
box(0.858, 0.430, 0.175, 0.095, r'$\mathcal{L}_{\mathrm{ic}}$', MIDGY, fs=11)
ax.plot([xo, xo], [yo - R*ASPECT, 0.430], color=EDGE, lw=1.1,
        ls=(0, (2.5, 2)), zorder=2)
arrow((xo, 0.430), (0.766, 0.430), color=EDGE, lw=1.1, ls=(0, (2.5, 2)))

# ------------------------------------------------------ training loop band
YB = 0.150
box(0.858, YB, 0.175, 0.095, 'Loss', DKGY, fs=11)
ax.add_patch(Ellipse((0.600, YB), 0.185, 0.185/ASPECT, fc=MIDGY, ec='none',
                     zorder=3))
ax.text(0.600, YB, 'Optimizer', ha='center', va='center', color='white',
        fontsize=9.5, zorder=4)
# θ moved right: sits under hidden layers (not encoding), so feedback arrow
# points clearly into the network weights, not γ(·)
box(0.355, YB, 0.140, 0.095, r'$\boldsymbol{\theta}\,(W, b)$', NAVY, fs=10)

# three loss terms → Loss: straight dashed verticals, x-offset so all three
# are visible as separate channels (boxes at zorder=3 mask the lines where
# they pass behind each lower box, naturally showing the merge toward Loss)
arrow((0.838, 0.773), (0.838, 0.200), color=BLUE, lw=1.1, ls=(0, (2.5, 2)))          # L_pde
arrow((0.858, 0.578), (0.858, 0.200), color=RED,  lw=1.2, ls=(0, (2.5, 2)), alpha=0.85)  # L_eik
arrow((0.878, 0.383), (0.878, 0.200), color=EDGE, lw=1.1, ls=(0, (2.5, 2)))          # L_ic

# Loss → Optimizer → θ → hidden-layer weights (not encoding block)
arrow((0.770, YB), (0.693, YB), color=EDGE, lw=1.2)
arrow((0.508, YB), (0.425, YB), color=EDGE, lw=1.2)
arrow((0.355, YB + 0.048), (0.355, 0.275), color=EDGE, lw=1.2)

fig.savefig('fig1_architecture.pdf', bbox_inches='tight')
fig.savefig('fig1_architecture.png', dpi=300, bbox_inches='tight')
print('rendered')