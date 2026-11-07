"""Shrink saw_ablation's settings so a full run takes seconds on a CPU (tests only)."""


def shrink(sa):
    for cfg in sa.BENCH.values():
        cfg.update(N_F=600, N_I=300, lbfgs=2)
    sa.ADAM_EPOCHS = 25
    sa.RAD.update(rad_freq=10, rad_candidates=800, rar_freq=12, rar_candidates=1000, rar_add=20)
    sa.GRID_N = 21
    sa.RK4_STEPS = 50
