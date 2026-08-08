# Eikonal Regularization in Physics-Informed Neural Networks for Three-Dimensional Level-Set Advection: Transferability of Two-Dimensional Design Principles

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Preprint%20in%20preparation-lightgrey)]()
<!-- TODO once available: swap the badge above for the DOI/journal badge,
     and add a Zenodo data/code-archive badge (see the \PH{...} placeholders
     in the manuscript's Code and Data Availability section). -->

> Direct 3D follow-up to [`pinn-level-set`](https://github.com/AkbarTheAnalyst/pinn-level-set)
> (Khan & Raees, *Machine Learning: Science and Technology*, 2026,
> [doi:10.1088/2632-2153/ae8b74](https://doi.org/10.1088/2632-2153/ae8b74)).
> That study identified the eikonal-regularisation weight as the most
> influential hyperparameter in 2D and left two questions open: does the
> design transfer to 3D, and do single-seed results survive run-to-run
> variance? This repository is the code for the paper answering both.

---

## Overview

Physics-informed neural networks applied to the level-set formulation of
interface advection

$$\frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi = 0, \qquad \mathbf{x}\in\Omega=[0,1]^3$$

commonly augment the loss with an eikonal regulariser penalising the
deviation of $\lVert\nabla\phi\rVert$ from unity. This repository sweeps that
weight across four 3D benchmarks (three random seeds each, full training
budget, a selection rule fixed in advance), and benchmarks the result
against a fifth-order WENO + TVD-RK3 classical solver on identical grids.

**Key findings:**
- The 2D weight-selection ordering transfers to 3D: the selected weight
  tracks how far the exact solution departs from a signed distance
  function, spanning four decades ($10^{-1}$ where the property holds
  exactly to $10^{-5}$ where the interface is stretched).
- The *specific* weights transfer only benchmark by benchmark: three of
  four carry over unchanged from the 2D study; one (translating sphere)
  does not.
- With three seeds per configuration, the seed-to-seed standard deviation
  at small weights is of the same order as the error itself; the
  regulariser reduces it by more than an order of magnitude — it buys
  reproducibility as well as accuracy.
- Against fifth-order WENO on identical grids/measures, the classical
  scheme is more accurate on all four benchmarks — by a factor of $122$
  on smooth rigid advection, narrowing to $4.3$ under deformation.
- The standard relative $L_2$ error cannot certify preservation of thin
  features: on the slotted sphere, removing the slot entirely from the
  exact solution changes it by only about one percentage point. We
  report a sign-based, feature-restricted measure that can.

---

## Benchmarks

Four benchmarks on the unit cube $\Omega=[0,1]^3$, ordered by how far their
exact solution departs from a signed distance function:

| Benchmark | Description | $T$ | Selected $w_\text{eik}$ |
|---|---|---|---|
| **TR3D** — Translating sphere | Uniform velocity, rigid | $5$ | $10^{-1}$ |
| **RO3D** — Rotating sphere | Solid-body rotation, rigid | $2\pi$ | $10^{-1}$ |
| **ZD3D** — Zalesak slotted sphere | Rotation + thin blind-pocket slot | $2\pi$ | $10^{-3}$ |
| **RV3D** — Reversed single vortex | Enright (2002) deformation field | $2$ | $10^{-5}$ |

### Reported configuration (selected weight, seed 42, 201³ grid, 6 snapshots)

| Benchmark | abs. $L_2$ | rel. $L_2$ (%) | volume MAPE (%) | eikonal dev. | time (min) |
|---|---|---|---|---|---|
| TR3D | 4.73e-4 | 0.122 | 1.143 | 1.31e-2 | 11.8 |
| RO3D | 6.76e-4 | 0.160 | 0.117 | 2.08e-2 | 15.6 |
| ZD3D | 2.03e-3 | 0.480 | 0.328 | 1.02e-1 | 27.1 |
| RV3D | 2.52e-3 | 0.591 | 0.564 | 5.97e-1 | 23.7 |

Timings are for a single NVIDIA T4. Full per-weight sweep tables (mean ± std
over 3 seeds) are tracked in benchmark-specific CSV files once each benchmark
folder is populated.

---

## Architecture & Training

Same architecture and optimiser protocol across all four benchmarks
(inherited from the 2D study), only the eikonal weight, sampling strategy,
and causal-weighting chunk count differ:

```
Input: (x, y, z, tau_norm) in R^4
  -> 8 x [Linear(256) -> Tanh]
       -> Linear(1)
Output: phi_hat(x, y, z, tau_norm) in R
```

| Hyperparameter | Value |
|---|---|
| Hidden layers | 8 |
| Neurons per layer | 256 |
| Activation | Tanh |
| Weight init | Xavier |
| $w_\text{pde}$, $w_\text{ic}$ | $1$, $10$ (fixed) |
| $w_\text{eik}$ | swept: 6 values, 4 decades, incl. 0 (control) |
| Optimiser | Adam (2e4 iters, cosine $10^{-3}\to10^{-5}$) then L-BFGS (strong-Wolfe) |
| Seeds | 42, 43, 44 (controls init *and* collocation sampling) |
| Eval grid | 201³ ($\Delta x = 0.005$), 6 snapshots $t=0,T/5,\dots,T$ |

ZD3D additionally uses a random Fourier feature encoding (128 features,
$\sigma=5$). ZD3D and RV3D additionally use residual-based adaptive
sampling (RAD/RAR) and causal loss weighting ($M=32$ and $M=10$ chunks
respectively).

## Embedded Figures (GitHub preview)

For quick browsing on GitHub we embed four high-impact paper figures below.
High-resolution PNGs for all paper figures are already available in
`figures/`. To show them inline from this README, we point to the existing
PNG files; if you prefer curated assets, copy the chosen PNGs into
`assets/`.

- Architecture & Training (priority 1):

  ![Figure 1 — Architecture](figures/fig1_architecture.png)

- RV3D interface (priority 2):

  ![Figure 9 — RV3D interface](figures/rv3d_interface.png)

- ZD3D cutaway (priority 3):

  ![Figure 6 — ZD3D cutaway](figures/zd3d_cutaway.png)

- TR3D verification (priority 4):

  ![Figure 3 — TR3D verification](figures/tr3d_verification.png)

If you want these assets consolidated for the repository front page, copy the
PNG files from `figures/` into `assets/` and commit the `assets/` folder.


---

## Installation

```bash
git clone https://github.com/AkbarTheAnalyst/pinn-level-set-3d.git
cd pinn-level-set-3d
pip install -r requirements.txt
```

<!-- TODO: add the Zenodo data/code archive link once minted, matching the
     \PH{URL} in the manuscript's Code and Data Availability section. -->

---

## Repository Structure

```
pinn-level-set-3d/
├── tr3d/                  # TR3D notebooks, core code, README
├── ro3d/                  # RO3D notebooks, core code, README
├── zd3d/                  # ZD3D notebooks, core code, README
├── rv3d/                  # RV3D notebooks, core code, README
├── weno/                  # WENO benchmark code and notebook
├── figures/               # all generated figure PDFs/PNGs
├── results/               # CSVs, checkpoints, and other result artifacts
├── assets/                # curated paper-ready assets
├── requirements.txt
├── LICENSE
└── README.md
```

The benchmark folders are intentionally small: each one keeps only its local
`.py`, `.ipynb`, and `README.md` files, while shared outputs live in
`figures/` and `results/`.

---

## Reproducing the eikonal sweep

Each benchmark folder contains the code and notebooks needed to run the local
ablation and evaluation workflow. Shared outputs are separated into central
folders, so the same layout can be used across TR3D, RO3D, ZD3D, and RV3D.

This section will be expanded with exact run commands for:

- eikonal-weight sweep (6 weights x 3 seeds)
- selected-weight training/evaluation run
- WENO baseline comparison

---

## Status

Current state:
- benchmark folders are present and organized by benchmark
- figure outputs are centralized in `figures/`
- result artifacts are centralized in `results/`
- WENO code and notebook live in `weno/`

This README now reflects the current on-disk structure.

---

## References

- Osher, S., & Sethian, J. A. (1988). Fronts propagating with
  curvature-dependent speed. *Journal of Computational Physics*, 79(1), 12–49.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
  neural networks. *Journal of Computational Physics*, 378, 686–707.
- Krishnapriyan, A., et al. (2021). Characterizing possible failure modes
  in physics-informed neural networks. *NeurIPS*, 34, 26548–26560.
- Enright, D., et al. (2002). A hybrid particle level set method for
  improved interface capturing. *Journal of Computational Physics*.
- Jiang, G.-S., & Shu, C.-W. (1996). Efficient implementation of weighted
  ENO schemes. *Journal of Computational Physics*, 126(1), 202–228.
- Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially
  non-oscillatory shock-capturing schemes. *Journal of Computational
  Physics*, 77(2), 439–471.
- Tancik, M., et al. (2020). Fourier features let networks learn high
  frequency functions in low dimensional domains. *NeurIPS*, 33, 7537–7547.
- Wang, S., Sankaran, S., & Perdikaris, P. (2024). Respecting causality
  for training physics-informed neural networks. *CMAME*, 421, 116813.
- Khan, M. A., & Raees, F. (2026). A systematic study of physics-informed
  neural networks for the level-set interface advection. *Machine
  Learning: Science and Technology*. doi:10.1088/2632-2153/ae8b74.

(Full reference list in the manuscript's `references.bib` / bibliography.)

---

## Citation

```bibtex
@article{khan2026eikonal3d,
  author  = {Muhammad Akbar Khan},
  title   = {Eikonal Regularization in Physics-Informed Neural Networks
             for Three-Dimensional Level-Set Advection: Transferability
             of Two-Dimensional Design Principles},
  year    = {2026},
  note    = {Preprint / in preparation}
}
```
<!-- TODO: replace with the final journal/preprint entry once assigned. -->

---

## Author

**Muhammad Akbar Khan**
MS Applied Mathematics, NED University of Engineering & Technology
akbar.bsma1337@gmail.com · [GitHub](https://github.com/AkbarTheAnalyst) ·
[ORCID 0009-0001-7956-0080](https://orcid.org/0009-0001-7956-0080) ·
[Website](https://akbarkhan.dev)

Supervisor: Dr. Fahim Raees, Associate Professor, NED University
