"""CPU checks before any GPU time is spent.

1. Equivalence: SAWLoss(weight='ratio', gate='quantile') must reproduce the
   original core SAW exactly (same weight trajectory, same final loss) on
   TR3D, ZD3D and RV3D, with sizes shrunk so it runs in seconds.
2. Smoke test: every experiment mode runs end to end through run_one,
   including diagnostics and evaluation on a coarse grid.
"""
import importlib.util
import os
import sys
import tempfile

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.join(EXP, "..")
sys.path.insert(0, EXP)
import saw_ablation  # noqa: E402

torch.set_num_threads(4)
DEV = "cpu"


def core_of(bench):
    name = bench.lower() + "_core"
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, bench.lower(), name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(core, bench, cfg, seed=42):
    if cfg["rff"]:
        net = core.build_model(DEV, seed=seed, n_fourier=128, b_scale=5.0)
    else:
        net = core.build_model(DEV, seed=seed)
    X_f, X_i = core.sample_points(cfg["N_F"], cfg["N_I"], DEV, seed=seed)
    sampler = None
    if cfg["sampler"]:
        sampler = core.AdaptiveSampler(net, X_f, cfg["T"], DEV, rad_freq=10, rad_candidates=800,
                                       rad_k=1.0, rad_c=1.0, rar_freq=12, rar_candidates=1000, rar_add=20)
    return net, X_f, X_i, sampler


def equivalence(bench, adam=30, lbfgs=2):
    core = core_of(bench)
    cfg = dict(saw_ablation.BENCH[bench], N_F=600, N_I=300)
    out = []
    for which in ("core", "new"):
        net, X_f, X_i, sampler = build(core, bench, cfg)
        if which == "core":
            if cfg["sampler"]:
                fn = core.make_loss_saw(net, sampler, X_i, cfg["T"], beta=0.999, q=0.95,
                                        m_chunks=cfg["chunks"], eps_causal=1.0)
            else:
                fn = core.make_loss_saw(net, X_f, X_i, cfg["T"], beta=0.999, q=0.95)
        else:
            pf = sampler.points if sampler else (lambda: X_f)
            fn = saw_ablation.SAWLoss(core, bench, net, X_i, cfg["T"], pf, q=0.95,
                                m_chunks=cfg["chunks"], seed=42)
        kw = dict(adam_epochs=adam, lbfgs_epochs=lbfgs, verbose=False)
        if sampler:
            kw["sampler"] = sampler
        tr = core.train(net, fn, **kw)
        out.append((np.array(fn.w_hist), tr["loss_final"]))
    (w1, l1), (w2, l2) = out
    same_len = len(w1) == len(w2)
    maxdiff = float(np.max(np.abs(w1 - w2))) if same_len else float("inf")
    ok = same_len and maxdiff < 1e-6 and abs(l1 - l2) <= 1e-6 * max(1.0, abs(l1))
    print(f"  {bench}: calls {len(w1)} vs {len(w2)}, max weight diff {maxdiff:.2e}, "
          f"final loss {l1:.6e} vs {l2:.6e} -> {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok


def smoke(bench, run):
    core = core_of(bench)
    with tempfile.TemporaryDirectory() as tmp:
        grid = core.Grid(n=21)
        phi_e = None
        if bench == "RV3D":
            times, _ = core.snapshot_times(saw_ablation.BENCH[bench]["T"], saw_ablation.N_SNAP)
            phi_e = core.reference_fields(grid, times, saw_ablation.BENCH[bench]["T"], DEV, nsteps=50, verbose=False)
        row, tag = saw_ablation.run_one(core, bench, run, DEV, tmp, grid, phi_e)
        files = sorted(os.listdir(tmp))
    keys = [k for k in row if k.startswith("avg_")]
    print(f"  {tag}: {len(row)} columns, rel_L2={row['avg_rel_L2']:.3f}, w_final={row['w_eik_final']}, "
          f"files={files}")
    return row


print("1. Equivalence with the original SAW")
ok = all(equivalence(b) for b in ("TR3D", "ZD3D", "RV3D"))
print("   ALL IDENTICAL" if ok else "   PROBLEM")

from small import shrink  # noqa: E402
shrink(saw_ablation)
print("\n2. Smoke tests (tiny sizes)")
fake_replay = np.geomspace(1.0, 1e-3, 40)
smoke("ZD3D", dict(mode="saw", seed=42, q=0.95, diag=True))
smoke("RV3D", dict(mode="saw", seed=42, q=0.95, diag=True))
smoke("TR3D", dict(mode="saw", seed=42, q=0.95, diag=True))
smoke("RO3D", dict(mode="saw", seed=42, q=0.95, diag=True))
smoke("ZD3D", dict(mode="replay", seed=42, replay=fake_replay))
smoke("RV3D", dict(mode="random", seed=42, q=0.95))
smoke("ZD3D", dict(mode="fixed", seed=45, w_eik=1e-3))
smoke("ZD3D", dict(mode="saw", seed=42, q=0.99))
print("\nsmoke tests finished")
