"""Figure: the adaptive weight trajectory, gated against ungated.

Reads the w_hist .npz files written by 02_saw_sweep.ipynb and produces a
four-panel figure, one per benchmark, with the three gated seeds and the three
ungated seeds overlaid. The .npz files are read from results/weight_history/;
run from figures/ to write the figure there.

Expected filenames (as written by run_one):
    {BENCH}_saw_whist_q{q:g}_seed{seed}.npz     e.g. TR3D_saw_whist_q0.95_seed42.npz
each containing a single array 'w_hist'.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------- settings
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'weight_history')
ROOT = {b: HISTORY_DIR for b in ('TR3D', 'RO3D', 'ZD3D', 'RV3D')}
BENCH = ['TR3D', 'RO3D', 'ZD3D', 'RV3D']
LABEL = {'TR3D': 'TR3D  (translating sphere)',
         'RO3D': 'RO3D  (rotating sphere)',
         'ZD3D': 'ZD3D  (slotted sphere)',
         'RV3D': 'RV3D  (reversed vortex)'}
SEEDS = [42, 43, 44]
Q_ON, Q_OFF = 0.95, 1.0

# weight selected by the fixed-weight sweep, drawn as a horizontal reference
SWEPT = {'TR3D': 1e-1, 'RO3D': 1e-1, 'ZD3D': 1e-3, 'RV3D': 1e-5}

ADAM_EPOCHS = 20000              # marks the Adam -> L-BFGS transition

PAPER_RC = {
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif', 'savefig.dpi': 300,
}

C_ON, C_OFF, C_REF = '#1f4e79', '#c0392b', '0.45'


def load(bench, q, seed):
    """Return the weight trajectory, or None if that run is absent."""
    path = os.path.join(ROOT[bench], f'{bench}_saw_whist_q{q:g}_seed{seed}.npz')
    if not os.path.exists(path):
        return None
    w = np.load(path)['w_hist'].astype(float)
    return w if w.size else None


def main():
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=False)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.945, bottom=0.115,
                        hspace=0.42, wspace=0.24)

    for ax, b in zip(axes.ravel(), BENCH):
        drew_on = drew_off = False

        for q, col, lw in ((Q_OFF, C_OFF, 1.1), (Q_ON, C_ON, 1.1)):
            for sd in SEEDS:
                w = load(b, q, sd)
                if w is None:
                    continue
                # the weight is clipped to [0, 1] and can reach 0; clip from
                # below so it is representable on a log axis
                w = np.clip(w, 1e-7, None)
                ax.semilogy(np.arange(w.size), w, color=col, lw=lw, alpha=0.75)
                if q == Q_ON:
                    drew_on = True
                else:
                    drew_off = True

        # the value the eighteen-run sweep selected
        ax.axhline(SWEPT[b], color=C_REF, ls='--', lw=1.0, zorder=1)

        # Adam -> L-BFGS. w_hist has one entry per loss evaluation, and L-BFGS
        # calls the closure several times per iteration, so the transition is
        # at ADAM_EPOCHS only if the closure is called once per Adam step.
        ax.axvline(ADAM_EPOCHS, color='0.6', ls=':', lw=0.9, zorder=1)

        ax.set_title(LABEL[b], pad=4)
        ax.set_xlabel('loss evaluation', labelpad=1)
        ax.set_ylabel(r'$w_{\mathrm{eik}}$', labelpad=1)
        ax.set_ylim(1e-6, 3.0)
        ax.grid(alpha=0.22, lw=0.5, which='major')
        ax.grid(alpha=0.10, lw=0.4, which='minor')

        if not (drew_on or drew_off):
            ax.text(0.5, 0.5, f'no {b} trajectories found',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='0.5')

    handles = [
        Line2D([], [], color=C_ON, lw=1.4, label=rf'SAW, $q={Q_ON}$'),
        Line2D([], [], color=C_OFF, lw=1.4, label='SAW, gate off'),
        Line2D([], [], color=C_REF, lw=1.0, ls='--', label='swept optimum'),
        Line2D([], [], color='0.6', lw=0.9, ls=':', label=r'Adam $\to$ L-BFGS'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=2.0,
               columnspacing=2.2)

    fig.savefig('saw_weight_trajectory.pdf', bbox_inches='tight')
    fig.savefig('saw_weight_trajectory.png', dpi=300, bbox_inches='tight')
    print('wrote saw_weight_trajectory.pdf / .png')


if __name__ == '__main__':
    main()
