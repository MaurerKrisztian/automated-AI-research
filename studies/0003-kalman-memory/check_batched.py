"""E2 check of the batched forms that the GPU benchmark times (kalman_batched.py, bl_triton.py), on the CPU.

Part P1: every batched form in float64 against kalman_memory.kalman_reference_np (the token-by-token filter in
  extended precision, the yardstick of check_chunk.py), head by head, over random inputs and edge cases with three
  heads each: the chunk-parallel form (triangular solves; explicit L^-1; GLA-style matmul Grams), the Bayesian Layer's
  chunk algorithm (gain loop with v5's sequential mean pass or v1's WY mean, in PyTorch and with the Triton kernels),
  and the batched token-by-token filter.
Part P2: gradients. Autograd through each differentiable batched form against autograd through the batched
  token-by-token filter, for every input, in float64.
Part P3: float32 against the float64 token-by-token filter on longer sequences (d = e = C = 64), as a check that the
  batched forms behave in float32 as the single-head forms of check_float32.py do.
Part P4: how much of a float32 error is the summation order. The token-by-token filter in float32, written three
  algebraically equal ways (kalman_memory.kalman_reference, one head; the batched filter, whose innovation variance
  k^T P k is an elementwise sum; the same with a batched matrix product), on four configurations of check_float32.py,
  two heads each. The GPU sums in yet other orders.

The Triton kernels run only under Triton's interpreter on the CPU, so run with TRITON_INTERPRET=1; without it they
are skipped. Run from this folder:
    TRITON_INTERPRET=1 ../../.venv/bin/python -u check_batched.py 2>&1 | tee /tmp/check_batched.log
(The recorded run is logs/check_batched.log; never tee into logs/, which is immutable. Each run adds a JSON to
results/; delete such copies, never the recorded one.)  --quick runs a small subset as a smoke test.
"""
import argparse
import math
import os

import numpy as np
import torch

import check_float32
import kalman_batched as kb
from kalman_memory import kalman_reference, kalman_reference_np, rel, stationary_noise

TRITON = os.environ.get("TRITON_INTERPRET") == "1"


# --------------------------------------------------------------------------- inputs
def make_case(seed, T, d, e, N=3, gates=(0.5, 1.0), per_channel=False, beta=(0.5, 5.0), lam=(0.5, 2.0),
              noise="stationary", prior_mean=False, dtype=torch.float64):
    """N heads of random inputs, as check_chunk.make_case draws them for one head. noise: "stationary"
    (Psi = (I - D^2)/lambda, P_init = Lambda^-1), "iso" (free l_t^2 I, P_init = 3I, as the Bayesian Layer) or "diag"
    (free per-channel noise, dense random P_init). Returns the keyword arguments of the kalman_batched functions."""
    rng = np.random.default_rng(seed)
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64)).to(dtype)
    K = rng.standard_normal((N, T, d)) / math.sqrt(d)
    V = rng.standard_normal((N, T, e))
    Q = rng.standard_normal((N, T, d)) / math.sqrt(d)
    logD = np.log(rng.uniform(*gates, (N, T, d if per_channel else 1)))
    b = np.exp(rng.uniform(math.log(beta[0]), math.log(beta[1]), (N, T)))
    c = dict(K=t(K), V=t(V), Q=t(Q), logD=t(logD), beta=t(b))
    if noise == "stationary":
        c["lam"] = t(np.exp(rng.uniform(math.log(lam[0]), math.log(lam[1]), (N, d))))
    elif noise == "iso":
        c["Psi"] = t(np.repeat(rng.uniform(0.01, 0.5, (N, T, 1)), d, axis=2))
        c["P_init"] = 3.0 * torch.eye(d, dtype=dtype).expand(N, d, d).clone()
    elif noise == "diag":
        c["Psi"] = t(rng.uniform(0.0, 0.5, (N, T, d)))
        A0 = rng.standard_normal((N, d, d)) / math.sqrt(d)
        c["P_init"] = t(A0 @ A0.transpose(0, 2, 1) + 0.1 * np.eye(d))
    else:
        raise ValueError(noise)
    if prior_mean:
        c["M_init"] = t(rng.standard_normal((N, e, d)))
    return c


def reference(c):
    """The extended-precision token-by-token filter of kalman_memory, head by head: O (N, T, e), M_T, P_T."""
    N, T, d = c["K"].shape
    e = c["V"].shape[-1]
    res = []
    for n in range(N):
        logD = c["logD"][n].expand(T, d)
        if "lam" in c:
            Psi, P0 = stationary_noise(logD, c["lam"][n]), torch.diag(1 / c["lam"][n])
        else:
            Psi, P0 = c["Psi"][n], c["P_init"][n]
        M0 = c["M_init"][n] if "M_init" in c else torch.zeros(e, d, dtype=c["K"].dtype)
        res.append(kalman_reference_np(c["K"][n], c["V"][n], c["Q"][n], logD, Psi, c["beta"][n], M0, P0))
    return [np.stack([r[i] for r in res]) for i in range(3)]


def methods(C, triton=TRITON):
    """name -> fn(case) -> (O, M_T, P_T)."""
    m = {"chunk, solve": lambda c: kb.kalman_chunkwise(C=C, **c),
         "chunk, explicit": lambda c: kb.kalman_chunkwise(C=C, inverse="explicit", **c),
         "chunk, matmul Grams": lambda c: kb.kalman_chunkwise(C=C, gram="matmul", **c),
         "BL, loop mean": lambda c: kb.bl_chunkwise(C=C, **c),
         "BL, WY mean": lambda c: kb.bl_chunkwise(C=C, mean="wy", **c)}
    if triton:
        m["BL Triton, loop mean"] = lambda c: kb.bl_chunkwise(C=C, gains="triton", mean="triton", **c)
        m["BL Triton, WY mean"] = lambda c: kb.bl_chunkwise(C=C, gains="triton", mean="wy", **c)
    m["token-by-token"] = lambda c: kb.kalman_recurrent(**c)
    return m


# --------------------------------------------------------------------------- part P1
CASES = [  # name, make_case kwargs, chunk size
    ("scalar gates 0.5-1", dict(seed=1, T=40, d=16, e=8), 8),
    ("per-channel gates 0.5-1", dict(seed=2, T=40, d=16, e=8, per_channel=True), 8),
    ("scalar gates 0.01-1", dict(seed=3, T=40, d=16, e=8, gates=(0.01, 1.0)), 8),
    ("per-channel gates 0.01-1", dict(seed=4, T=40, d=16, e=8, gates=(0.01, 1.0), per_channel=True), 8),
    ("scalar gates 0.999-1", dict(seed=5, T=40, d=16, e=8, gates=(0.999, 1.0)), 8),
    ("no forgetting (gates = 1)", dict(seed=6, T=40, d=16, e=8, gates=(1.0, 1.0)), 8),
    ("C does not divide T (T = 37, C = 8)", dict(seed=7, T=37, d=12, e=5, per_channel=True), 8),
    ("T < C (T = 5, C = 8)", dict(seed=8, T=5, d=8, e=4), 8),
    ("chunk size 1", dict(seed=9, T=24, d=8, e=4, per_channel=True), 1),
    ("d = 4 < C = 32", dict(seed=10, T=70, d=4, e=3, per_channel=True), 32),
    ("d = 48 > C = 8", dict(seed=11, T=40, d=48, e=6, per_channel=True), 8),
    ("mixed beta 1e-3 to 50", dict(seed=12, T=40, d=16, e=8, beta=(1e-3, 50.0), per_channel=True), 8),
    ("mixed ridge 1e-3 to 10", dict(seed=13, T=40, d=16, e=8, lam=(1e-3, 10.0), per_channel=True), 8),
    ("free isotropic noise, P_init = 3I (BL)", dict(seed=14, T=40, d=16, e=8, noise="iso"), 8),
    ("free diagonal noise, dense P_init, M_0 != 0", dict(seed=15, T=40, d=16, e=8, noise="diag", per_channel=True,
                                                        prior_mean=True), 8),
    ("long: T = 513, d = 32, C = 64", dict(seed=16, T=513, d=32, e=16, gates=(0.9, 1.0), per_channel=True), 64),
]
QUICK = {"scalar gates 0.5-1", "C does not divide T (T = 37, C = 8)", "free diagonal noise, dense P_init, M_0 != 0"}


def part_p1(quick):
    print("Part P1. Batched forms in float64 (3 heads per case) against the extended-precision token-by-token filter.")
    print("Errors are max|x - x_ref| / max|x_ref| over all heads: outputs O, then the final M_T and P_T.")
    rows = []
    for name, kw, C in CASES:
        if quick and name not in QUICK:
            continue
        c = make_case(**kw)
        o_ref, M_ref, P_ref = reference(c)
        r = dict(case=name, C=C, T=kw["T"], d=kw["d"], err={})
        for label, fn in methods(C).items():
            O, M, P = fn(c)
            r["err"][label] = dict(O=rel(O, o_ref), M_T=rel(M, M_ref), P_T=rel(P, P_ref))
        rows.append(r)
        print(f"{name}  (C = {C})")
        for label, x in r["err"].items():
            print(f"  {label:>22}: O {x['O']:.1e}   M_T {x['M_T']:.1e}   P_T {x['P_T']:.1e}")
    labels = list(rows[0]["err"])
    print("worst over all cases:")
    for label in labels:
        w = {k: max(r["err"][label][k] for r in rows) for k in ("O", "M_T", "P_T")}
        print(f"  {label:>22}: O {w['O']:.1e}   M_T {w['M_T']:.1e}   P_T {w['P_T']:.1e}")
    print()
    return rows


# --------------------------------------------------------------------------- part P2
def part_p2(quick):
    print("Part P2. Gradients in float64: autograd through each batched form against autograd through the batched")
    print("token-by-token filter (3 heads). Loss: random weights on O, M_T and P_T. Worst relative error per input.")
    configs = [("stationary, scalar gates", dict(seed=800, T=29, d=8, e=4), 8),
               ("stationary, per-channel gates", dict(seed=801, T=29, d=8, e=4, per_channel=True), 8),
               ("free diagonal noise, M_0 and P_init = S S^T leaves",
                dict(seed=802, T=29, d=8, e=4, noise="diag", per_channel=True, prior_mean=True), 6),
               ("ill-conditioned: gates 0.999-1, beta 50, ridge 1e-2",
                dict(seed=803, T=29, d=8, e=4, gates=(0.999, 1.0), beta=(50.0, 50.0), lam=(1e-2, 1e-2)), 8)]
    if quick:
        configs = configs[:1]
    rows = []
    for name, kw, C in configs:
        c = make_case(**kw)
        stationary = "lam" in c
        leaves = dict(K=c["K"], V=c["V"], Q=c["Q"], log_beta=torch.log(c["beta"]), logD=c["logD"])
        if stationary:
            leaves["log_lam"] = torch.log(c["lam"])
        else:
            leaves["log_Psi"] = torch.log(c["Psi"] + 1e-3)
            leaves["M_init"] = c["M_init"]
            leaves["S"] = torch.linalg.cholesky(c["P_init"])
        rng = np.random.default_rng(kw["seed"] + 50)
        N, T, d, e = 3, kw["T"], kw["d"], kw["e"]
        wO, wM, wP = (torch.from_numpy(rng.standard_normal(s)) for s in ((N, T, e), (N, e, d), (N, d, d)))

        def loss(fn, z):
            a = dict(K=z["K"], V=z["V"], Q=z["Q"], logD=z["logD"], beta=torch.exp(z["log_beta"]))
            if stationary:
                a["lam"] = torch.exp(z["log_lam"])
            else:
                a.update(Psi=torch.exp(z["log_Psi"]), M_init=z["M_init"], P_init=z["S"] @ z["S"].mT)
            O, M, P = fn(a)
            return (O * wO).sum() + (M * wM).sum() + (P * wP).sum()

        fns = {"token-by-token": lambda a: kb.kalman_recurrent(**a),
               "chunk, solve": lambda a: kb.kalman_chunkwise(C=C, **a),
               "chunk, explicit": lambda a: kb.kalman_chunkwise(C=C, inverse="explicit", **a),
               "BL, loop mean": lambda a: kb.bl_chunkwise(C=C, **a),
               "BL, WY mean": lambda a: kb.bl_chunkwise(C=C, mean="wy", **a)}
        grads = {}
        for label, fn in fns.items():
            z = {k: v.clone().requires_grad_() for k, v in leaves.items()}
            loss(fn, z).backward()
            grads[label] = {k: v.grad for k, v in z.items()}
        ref = grads.pop("token-by-token")
        errs = {label: {k: rel(g[k], ref[k]) for k in leaves} for label, g in grads.items()}
        rows.append(dict(config=name, C=C, grad_rel=errs))
        print(f"{name}  (C = {C})")
        for label, x in errs.items():
            print(f"  {label:>16}: worst {max(x.values()):.1e}   (" + ", ".join(f"{k} {v:.1e}" for k, v in x.items()) + ")")
    print()
    return rows


# --------------------------------------------------------------------------- part P3
F32 = [  # name, gates, per_channel, beta, ridge
    ("scalar gates 0.9-1", (0.9, 1.0), False, (0.5, 2.0), 1.0),
    ("per-channel gates 0.95-1", (0.95, 1.0), True, (0.5, 2.0), 1.0),
    ("strong writes: beta 50, ridge 1e-2, gates 0.999-1", (0.999, 1.0), False, (50.0, 50.0), 1e-2),
    ("no forgetting (gates = 1), beta 0.5-2, ridge 1", (1.0, 1.0), False, (0.5, 2.0), 1.0),
]


def part_p3(quick):
    T = 512 if quick else 2048
    print(f"Part P3. float32 against the float64 token-by-token filter: 2 heads, T = {T}, d = e = C = 64, stationary noise.")
    print("Errors are max|o32 - o64| / max|o64| over both heads and the whole sequence; P_T is the final covariance.")
    rows = []
    for i, (name, gates, pc, beta, lam) in enumerate(F32):
        c64 = make_case(seed=1000 + i, T=T, d=64, e=64, N=2, gates=gates, per_channel=pc, beta=beta, lam=(lam, lam))
        c32 = {k: v.float() for k, v in c64.items()}
        o64, _, P64 = kb.kalman_recurrent(**c64)
        r = dict(config=name, T=T, err={})
        for label, fn in methods(64).items():
            O, _, P = fn(c32)
            r["err"][label] = dict(O=rel(O, o64), P_T=rel(P, P64))
        rows.append(r)
        print(f"{name}")
        for label, x in r["err"].items():
            print(f"  {label:>22}: O {x['O']:.1e}   P_T {x['P_T']:.1e}")
    print()
    return rows


# --------------------------------------------------------------------------- part P4
def recurrent_bmm(K, V, Q, logD, beta, lam):
    """kalman_batched.kalman_recurrent with k^T P k as a batched matrix product instead of an elementwise sum."""
    M, P = K.new_zeros(K.shape[0], V.shape[-1], K.shape[-1]), torch.diag_embed(1 / lam)
    Psi, out = -torch.expm1(2 * logD) / lam[:, None, :], []
    for t in range(K.shape[1]):
        D = torch.exp(logD[:, t])
        M = M * D[:, None, :]
        P = D[:, :, None] * P * D[:, None, :] + torch.diag_embed(Psi[:, t])
        k = K[:, t]
        Pk = (P @ k[..., None])[..., 0]
        s = ((k[:, None, :] @ Pk[:, :, None])[:, 0, 0] + 1 / beta[:, t])[:, None]
        M = M + (V[:, t] - (M @ k[..., None])[..., 0])[..., None] * (Pk / s)[:, None, :]
        P = P - Pk[:, :, None] * Pk[:, None, :] / s[..., None]
        out.append((M @ Q[:, t, :, None])[..., 0])
    return torch.stack(out, 1)


def part_p4(quick):
    T = 1024 if quick else 4096
    print(f"Part P4. float32 error and summation order: 2 heads, T = {T}, d = e = C = 64, configurations of"
          " check_float32.py (seeds 1000 + i, and 1000 + i + 7919 for the second head).")
    print("Worst error over the eight segments of each head, against the float64 batched filter. Three float32")
    print("token-by-token filters that differ only in how they sum: single-head reference / batched elementwise sum /")
    print("batched matrix product; then the chunk form (triangular solves) and BL with v1's WY mean, for scale.")
    rows = []
    for i in (0, 3, 7, 8):
        name, _, gates, pc, beta, lam, keys = check_float32.CONFIGS[i]
        heads = [check_float32.make(1000 + i + 7919 * h, T, gates, pc, beta, lam, keys, torch.float64) for h in range(2)]
        K, V, Q, logD, Psi, b, M0, P0 = (torch.stack([hd[j] for hd in heads]) for j in range(8))
        x64 = dict(K=K, V=V, Q=Q, logD=logD if pc else logD[..., :1], beta=b, lam=1 / torch.diagonal(P0, dim1=-2, dim2=-1))
        x32 = {k: v.float() for k, v in x64.items()}
        o64 = kb.kalman_recurrent(**x64)[0]
        outs = {"single-head reference": torch.stack([kalman_reference(*(a.float() for a in hd))[0] for hd in heads]),
                "batched, sum": kb.kalman_recurrent(**x32)[0], "batched, bmm": recurrent_bmm(**x32),
                "chunk, solve": kb.kalman_chunkwise(C=64, **x32)[0], "BL, WY mean": kb.bl_chunkwise(C=64, mean="wy", **x32)[0]}
        err = {k: [max(check_float32.seg_errors(o[h], o64[h])) for h in range(2)] for k, o in outs.items()}
        tbt = [err[k][h] for k in ("single-head reference", "batched, sum", "batched, bmm") for h in range(2)]
        spread = max(max(err[k][h] for k in list(err)[:3]) / min(err[k][h] for k in list(err)[:3]) for h in range(2))
        rows.append(dict(config=name, T=T, err=err, token_by_token_range=[min(tbt), max(tbt)], spread_max=spread))
        print(f"{name}")
        for k, v in err.items():
            print(f"  {k:>22}: head 0 {v[0]:.1e}, head 1 {v[1]:.1e}")
        print(f"  largest ratio between the three token-by-token filters on one head: {spread:.1f}")
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    print(f"torch {torch.__version__}; Triton kernels {'under the interpreter' if TRITON else 'skipped (TRITON_INTERPRET unset)'}\n")
    results = dict(triton_interpreter=TRITON, part_p1=part_p1(args.quick), part_p2=part_p2(args.quick),
                   part_p3=part_p3(args.quick), part_p4=part_p4(args.quick))
    if not args.quick:
        from lab import save_results
        print(f"saved {save_results(results, 'results', 'check_batched')}")


if __name__ == "__main__":
    main()
