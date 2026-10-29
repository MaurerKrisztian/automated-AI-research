"""E2 check of study 0003 in float64 on the CPU.

Part A: the chunk forward pass (Propositions 2-3) against the token-by-token Kalman filter (Proposition 1b), over
  random inputs and edge cases. The reference runs in extended precision (numpy longdouble, eps 1.1e-19), so the
  float64 token-by-token filter is judged by the same yardstick. Four float64 forms are checked: triangular solves,
  an explicit L^-1, GLA-style "matmul" noise Grams, and Proposition 2's per-token prefix solves taken literally.
Part A2: a conditioning sweep (strong writes against ridges down to 1e-6, gates from 0.99-1 to exactly 1), reporting
  how the error grows, relative to cond(K) * eps and to the token-by-token filter.
Part B: special cases. The gate-free limit against the Mesa readout G_t (H_t + Lambda)^-1 q_t (Proposition 1c); the
  one-pair formulas of Proposition 1(d); the isotropic-covariance reduction to Gated DeltaNet and KDA
  (Proposition 5); the directional-noise Gram of the Remark; the stationary closed form of phi.
Part C: the bounds of Proposition 4(b) on random histories.
Part D: gradients. Proposition 6's closed-form adjoint against autograd; autograd through the chunk form against
  autograd through the token-by-token filter, for every input; central differences on the reference.

Run from this folder:  python3 -u check_chunk.py 2>&1 | tee /tmp/check_chunk.log
(The recorded run is logs/check_chunk_v5.log; never tee into logs/, which is immutable. Each run adds a JSON to
results/; delete such copies, never the recorded one.)  --quick runs a small subset as a smoke test.
"""
import argparse
import math

import numpy as np
import torch

from kalman_memory import (chunk_matrices, chunk_step, chunked, chunked_prefix, kalman_reference,
                           kalman_reference_np, max_cond, mesa_readout, noise_variance, rel, stationary_noise)

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- inputs
def make_case(seed, T, d, e, gates=(0.5, 1.0), per_channel=False, beta=(0.5, 5.0), lam=(0.5, 2.0),
              noise="stationary", keys="gauss", pinit=None, prior_mean=False):
    """Random inputs. gates: range of the forget gates (both ends 1.0 means no forgetting). noise: "stationary"
    (Psi = (I - D^2)/lambda, P_init = Lambda^-1), "iso" (free l_t^2 I, P_init = 3I, as the Bayesian Layer) or "diag"
    (free per-channel noise, dense random P_init). keys: "gauss", "unit", "decay" (coordinate scales i^-2) or
    "collinear" (one base key plus perturbations of size 1e-3)."""
    rng = np.random.default_rng(seed)
    if keys == "collinear":
        base = rng.standard_normal(d) / math.sqrt(d)
        K = base + 1e-3 * rng.standard_normal((T, d)) / math.sqrt(d)
    else:
        K = rng.standard_normal((T, d)) / math.sqrt(d)
        if keys == "unit":
            K /= np.linalg.norm(K, axis=1, keepdims=True)
        elif keys == "decay":
            K *= (np.arange(1, d + 1) ** -2.0) * math.sqrt(d)
    V = rng.standard_normal((T, e))
    Q = rng.standard_normal((T, d)) / math.sqrt(d)
    g = rng.uniform(*gates, (T, d) if per_channel else (T, 1)) * np.ones((1, d))
    logD = np.log(g)
    b = np.exp(rng.uniform(math.log(beta[0]), math.log(beta[1]), T))
    lm = np.exp(rng.uniform(math.log(lam[0]), math.log(lam[1]), d))
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
    logD_t, lam_t = t(logD), t(lm)
    if noise == "stationary":
        Psi = stationary_noise(logD_t, lam_t)
        P0 = torch.diag(1 / lam_t)
    elif noise == "iso":
        Psi = t(np.repeat(rng.uniform(0.01, 0.5, (T, 1)), d, axis=1))
        P0 = 3.0 * torch.eye(d)
    elif noise == "diag":
        Psi = t(rng.uniform(0.0, 0.5, (T, d)))
        A0 = rng.standard_normal((d, d)) / math.sqrt(d)
        P0 = t(A0 @ A0.T + 0.1 * np.eye(d))
    else:
        raise ValueError(noise)
    if pinit is not None:
        P0 = pinit
    M0 = t(rng.standard_normal((e, d))) if prior_mean else torch.zeros(e, d)
    return dict(K=t(K), V=t(V), Q=t(Q), logD=logD_t, Psi=Psi, beta=t(b), M0=M0, P0=P0, lam=lam_t)


def args_of(c):
    return c["K"], c["V"], c["Q"], c["logD"], c["Psi"], c["beta"], c["M0"], c["P0"]


# --------------------------------------------------------------------------- part A
CASES = [  # name, make_case kwargs, chunk size
    ("scalar gates 0.5-1", dict(seed=1, T=40, d=16, e=8), 8),
    ("per-channel gates 0.5-1", dict(seed=2, T=40, d=16, e=8, per_channel=True), 8),
    ("scalar gates 0.01-1", dict(seed=3, T=40, d=16, e=8, gates=(0.01, 1.0)), 8),
    ("per-channel gates 0.01-1", dict(seed=4, T=40, d=16, e=8, gates=(0.01, 1.0), per_channel=True), 8),
    ("scalar gates 0.999-1", dict(seed=5, T=40, d=16, e=8, gates=(0.999, 1.0)), 8),
    ("per-channel gates 0.999-1", dict(seed=6, T=40, d=16, e=8, gates=(0.999, 1.0), per_channel=True), 8),
    ("no forgetting (gates = 1)", dict(seed=7, T=40, d=16, e=8, gates=(1.0, 1.0)), 8),
    ("near resets (gates 1e-6 to 1e-3)", dict(seed=8, T=40, d=16, e=8, gates=(1e-6, 1e-3)), 8),
    ("chunk size 1", dict(seed=9, T=24, d=8, e=4), 1),
    ("one chunk (C = T = 33)", dict(seed=10, T=33, d=12, e=5, per_channel=True), 33),
    ("C does not divide T (T = 37, C = 8)", dict(seed=11, T=37, d=12, e=5, per_channel=True), 8),
    ("T = 1", dict(seed=12, T=1, d=8, e=4), 8),
    ("d = 4 < C = 32", dict(seed=13, T=70, d=4, e=3, per_channel=True), 32),
    ("d = 48 > C = 8", dict(seed=14, T=40, d=48, e=6, per_channel=True), 8),
    ("e = 1", dict(seed=15, T=40, d=16, e=1), 8),
    ("tiny write strengths (beta = 1e-3)", dict(seed=16, T=40, d=16, e=8, beta=(1e-3, 1e-3)), 8),
    ("large write strengths (beta = 50)", dict(seed=17, T=40, d=16, e=8, beta=(50.0, 50.0)), 8),
    ("mixed beta 1e-3 to 50", dict(seed=18, T=40, d=16, e=8, beta=(1e-3, 50.0), per_channel=True), 8),
    ("mixed ridge 1e-3 to 10", dict(seed=19, T=40, d=16, e=8, lam=(1e-3, 10.0), per_channel=True), 8),
    ("unit keys", dict(seed=20, T=40, d=16, e=8, keys="unit"), 8),
    ("decaying key spectrum (i^-2)", dict(seed=21, T=40, d=16, e=8, keys="decay", lam=(1e-3, 1e-2)), 8),
    ("near-collinear keys", dict(seed=22, T=40, d=16, e=8, keys="collinear", beta=(5.0, 50.0)), 8),
    ("free isotropic noise, P_init = 3I (BL)", dict(seed=23, T=40, d=16, e=8, noise="iso", per_channel=True), 8),
    ("free diagonal noise, dense P_init", dict(seed=24, T=40, d=16, e=8, noise="diag", per_channel=True), 8),
    ("nonzero prior mean", dict(seed=25, T=40, d=16, e=8, prior_mean=True, per_channel=True), 8),
    ("long: T = 513, d = 32, C = 64", dict(seed=26, T=513, d=32, e=16, gates=(0.9, 1.0), per_channel=True), 64),
]
QUICK = {"scalar gates 0.5-1", "per-channel gates 0.01-1", "C does not divide T (T = 37, C = 8)", "T = 1"}


def part_a(quick):
    print("Part A. Chunk forward pass against the extended-precision token-by-token filter (float64 inputs).")
    print("Errors are max|x - x_ref| / max|x_ref|; 'seq64' is the float64 token-by-token filter, for scale.")
    head = (f"{'case':>40} {'C':>3} {'cond(K) max':>11} {'solve':>8} {'explicit':>8} {'matmul':>8} {'prefix':>8}"
            f" {'seq64':>8} {'M_T':>8} {'P_T':>8}")
    print(head)
    rows = []
    for name, kw, C in CASES:
        if quick and name not in QUICK:
            continue
        c = make_case(**kw)
        a = args_of(c)
        o_ref, M_ref, P_ref = kalman_reference_np(*a)
        o_seq, _, _ = kalman_reference(*a)
        o_s, M_s, P_s = chunked(*a, C)
        o_x, _, _ = chunked(*a, C, inverse="explicit")
        o_m, _, _ = chunked(*a, C, gram="matmul")
        o_p = chunked_prefix(*a, C)
        r = dict(case=name, C=C, T=kw["T"], d=kw["d"], cond_max=max_cond(*a, C), solve=rel(o_s, o_ref),
                 explicit=rel(o_x, o_ref), matmul=rel(o_m, o_ref), prefix=rel(o_p, o_ref), seq64=rel(o_seq, o_ref),
                 M_T=rel(M_s, M_ref), P_T=rel(P_s, P_ref))
        rows.append(r)
        print(f"{name:>40} {C:>3} {r['cond_max']:>11.2e} {r['solve']:>8.1e} {r['explicit']:>8.1e} {r['matmul']:>8.1e}"
              f" {r['prefix']:>8.1e} {r['seq64']:>8.1e} {r['M_T']:>8.1e} {r['P_T']:>8.1e}")
    worst = {k: max(r[k] for r in rows) for k in ("solve", "explicit", "matmul", "prefix", "seq64", "M_T", "P_T")}
    print("worst over all cases: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    well = [r for r in rows if r["cond_max"] <= 1e4]
    print(f"worst over the {len(well)} cases with cond(K) <= 1e4: "
          + ", ".join(f"{k} {max(r[k] for r in well):.1e}" for k in ("solve", "explicit", "matmul", "prefix", "seq64")))
    print()
    return rows


def part_a2(quick):
    print("Part A2. Conditioning sweep: strong writes (beta = 50) against ever smaller ridges, scalar gates from 0.99-1 to 1.")
    print("unit and decay: d = C = 16; rank4: d = 4 keys in chunks of C = 32, so each chunk's key block has rank 4.")
    print("With gates below 1 the stationary noise (1 - gamma^2)/lambda grows as the ridge shrinks; with gates = 1 it is 0.")
    print("err/(cond eps): the chunk form's worst error over cond(K) max * 2^-52; chunk/seq: over the seq64 error.")
    head = (f"{'keys':>8} {'gates':>7} {'ridge':>7} {'cond(K) max':>11} {'solve':>8} {'explicit':>8} {'matmul':>8}"
            f" {'seq64':>8} {'P_T':>8} {'err/(cond eps)':>14} {'chunk/seq':>9}")
    print(head)
    rows = []
    sweep = [(k, g, lam) for k in (("unit", 16, 16), ("decay", 16, 16), ("rank4", 4, 32)) for g in ((0.99, 1.0), (0.999, 1.0), (0.9999, 1.0), (1.0, 1.0))
             for lam in (1.0, 1e-2, 1e-4, 1e-6)]
    if quick:
        sweep = sweep[2:3]
    for (keys, d, C), gates, lam in sweep:
        if True:
            c = make_case(seed=900, T=96, d=d, e=8, gates=gates, beta=(50.0, 50.0), lam=(lam, lam),
                          keys="gauss" if keys == "rank4" else keys)
            a = args_of(c)
            o_ref, _, P_ref = kalman_reference_np(*a)
            o_seq, _, _ = kalman_reference(*a)
            try:
                o_s, _, P_s = chunked(*a, C)
                o_x, _, _ = chunked(*a, C, inverse="explicit")
                o_m, _, _ = chunked(*a, C, gram="matmul")
                errs = dict(solve=rel(o_s, o_ref), explicit=rel(o_x, o_ref), matmul=rel(o_m, o_ref), P_T=rel(P_s, P_ref))
            except RuntimeError:                                      # a Cholesky that fails
                errs = dict(solve=float("nan"), explicit=float("nan"), matmul=float("nan"), P_T=float("nan"))
            r = dict(keys=keys, gates=list(gates), ridge=lam, cond_max=max_cond(*a, C), seq64=rel(o_seq, o_ref), **errs)
            worst = max(r["solve"], r["explicit"], r["matmul"])
            r["err_over_cond_eps"], r["chunk_over_seq"] = worst / (r["cond_max"] * 2.0 ** -52), worst / r["seq64"]
            rows.append(r)
            print(f"{keys:>8} {gates[0]:>7} {lam:>7.0e} {r['cond_max']:>11.2e} {r['solve']:>8.1e} {r['explicit']:>8.1e}"
                  f" {r['matmul']:>8.1e} {r['seq64']:>8.1e} {r['P_T']:>8.1e} {r['err_over_cond_eps']:>14.2f}"
                  f" {r['chunk_over_seq']:>9.1f}")
    print()
    return rows


# --------------------------------------------------------------------------- part B
def part_b():
    print("Part B. Special cases.")
    out = {}
    # B1: gate-free limit equals the Mesa ridge readout
    errs = []
    for seed in range(3):
        c = make_case(seed=100 + seed, T=48, d=12, e=6, gates=(1.0, 1.0), lam=(0.05, 2.0))
        o, _, _ = chunked(*args_of(c), 16)
        errs.append(rel(o, mesa_readout(c["K"], c["V"], c["Q"], c["beta"], c["lam"])))
    out["mesa_limit"] = max(errs)
    print(f"B1  gates = 1, stationary noise: chunk readout vs Mesa G_t (H_t + Lambda)^-1 q_t, 3 inputs: {max(errs):.1e}")
    # B2: one stored pair (Proposition 1d)
    rng = np.random.default_rng(200)
    e1, e2 = [], []
    for _ in range(20):
        d, gam, lam, beta, tau = 8, rng.uniform(0.3, 1.0), rng.uniform(0.1, 3.0), rng.uniform(0.1, 10.0), int(rng.integers(0, 30))
        k, v, q = rng.standard_normal(d), rng.standard_normal(3), rng.standard_normal(d)
        T = tau + 1
        K = np.zeros((T, d)); K[0] = k
        V = np.zeros((T, 3)); V[0] = v
        Q = np.zeros((T, d)); Q[-1] = q
        logD = torch.full((T, d), math.log(gam))
        lamv = torch.full((d,), lam)
        a = (torch.from_numpy(K), torch.from_numpy(V), torch.from_numpy(Q), logD, stationary_noise(logD, lamv),
             torch.full((T,), beta), torch.zeros(3, d), torch.eye(d) / lam)
        o, _, _ = chunked(*a, 8)
        aa = gam ** tau
        formula_k = aa * beta * (k @ q) / (beta * (k @ k) + lam) * v
        e1.append(rel(o[-1], formula_k))
        H = aa * beta * np.outer(k, k)
        G = aa * beta * np.outer(v, k)
        mesa = G @ np.linalg.solve(H + lam * np.eye(d), q)
        e2.append(rel(mesa, aa * beta * (k @ q) / (aa * beta * (k @ k) + lam) * v))
    out["one_pair_kalman"], out["one_pair_mesanet"] = max(e1), max(e2)
    print(f"B2  one stored pair, 20 draws of gate, ridge, beta and age 0-29 (Proposition 1d): Kalman formula {max(e1):.1e},"
          f" MesaNet formula {max(e2):.1e}")
    # B3: isotropic covariance gives the Gated DeltaNet / KDA recurrence (Proposition 5)
    worst, bmax = 0.0, 0.0
    for per_channel in (False, True):
        c = make_case(seed=300 + per_channel, T=50, d=16, e=8, per_channel=per_channel)
        p = torch.from_numpy(np.random.default_rng(301).uniform(0.1, 3.0, 50))
        Mk = Mg = torch.zeros(8, 16)
        for t in range(50):
            D, k, v, b = torch.exp(c["logD"][t]), c["K"][t], c["V"][t], c["beta"][t]
            Mm = Mk * D
            Mk = Mm + torch.outer(v - Mm @ k, p[t] * k / (p[t] * (k @ k) + 1 / b))
            bt = p[t] * b / (1 + p[t] * b * (k @ k))
            Mg = (Mg * D) @ (torch.eye(16) - bt * torch.outer(k, k)) + bt * torch.outer(v, k)
            bmax = max(bmax, float(bt * (k @ k)))
            worst = max(worst, rel(Mk, Mg))
    out["gdn_reduction"], out["b_knorm_max"] = worst, bmax
    print(f"B3  Kalman mean with covariance p_t I vs the Gated DeltaNet (scalar) and KDA (per-channel) recurrence:"
          f" {worst:.1e}; max b_t |k_t|^2 = {bmax:.4f} (< 1)")
    # B4: directional noise Gram B diag(psi) B^T (Remark)
    worst = 0.0
    for per_channel in (False, True):
        c = make_case(seed=400 + per_channel, T=12, d=6, e=2, per_channel=per_channel)
        K, logA = c["K"], torch.cumsum(c["logD"], 0)
        psi = torch.from_numpy(np.random.default_rng(401).uniform(0.1, 1.0, 12))
        C = 12
        direct = torch.zeros(C, C)
        for i in range(C):
            for ip in range(C):
                for j in range(min(i, ip) + 1):
                    ui = torch.exp(logA[i] - logA[j]) * K[i]
                    uip = torch.exp(logA[ip] - logA[j]) * K[ip]
                    direct[i, ip] += psi[j] * (ui @ K[j]) * (K[j] @ uip)
        B = torch.zeros(C, C)
        for i in range(C):
            for j in range(i + 1):
                B[i, j] = K[i] @ (torch.exp(logA[i] - logA[j]) * K[j])
        worst = max(worst, rel(B @ torch.diag(psi) @ B.T, direct))
    out["directional_gram"] = worst
    print(f"B4  directional-noise Gram, direct triple sum vs B diag(psi) B^T, scalar and per-channel gates: {worst:.1e}")
    # B5: stationary phi in closed form
    worst = 0.0
    for lo, hi in ((0.5, 1.0), (1 - 1e-6, 1.0), (1 - 1e-9, 1 - 1e-10)):
        c = make_case(seed=500, T=64, d=8, e=2, gates=(lo, hi), per_channel=True)
        phi = noise_variance(c["logD"], c["Psi"])
        closed = -torch.expm1(2 * torch.cumsum(c["logD"], 0)) / c["lam"]
        ref = -np.expm1(2 * np.cumsum(np.asarray(c["logD"], dtype=np.longdouble), 0)) / np.asarray(c["lam"], dtype=np.longdouble)
        worst = max(worst, rel(phi, ref), rel(closed, ref))
    out["phi_closed_form"] = worst
    print(f"B5  phi by recursion and by (I - A^2)/lambda with expm1, vs extended precision, gates down to 1 - 1e-10: {worst:.1e}")
    print()
    return out


# --------------------------------------------------------------------------- part C
def part_c():
    print("Part C. Bounds of Proposition 4(b), stationary noise, on random histories.")
    viol = dict(lmin=0, lmax_F=0, lmax_trace=0, cond_unit=0, P_le_prior=0)
    n, tight = 0, []
    for trial in range(120):
        rng = np.random.default_rng(600 + trial)
        per_channel, unit = bool(trial % 2), trial % 3 == 0
        H, C, d = int(rng.choice([0, 10, 50, 200])), int(rng.choice([4, 16, 64])), int(rng.choice([4, 16]))
        lam_r = (1.0, 1.0) if unit else (0.05, 2.0)
        c = make_case(seed=700 + trial, T=H + C, d=d, e=2, gates=(float(rng.choice([0.5, 0.9, 0.99])), 1.0),
                      per_channel=per_channel and not unit, beta=(0.1, 10.0), lam=lam_r, keys="unit" if unit else "gauss")
        a = args_of(c)
        if H:
            _, M0, P0 = kalman_reference(*(x[:H] for x in a[:6]), a[6], a[7])
        else:
            M0, P0 = a[6], a[7]
        Lam_inv = torch.diag(1 / c["lam"])
        if torch.linalg.eigvalsh(P0 - Lam_inv)[-1] > 1e-12 * float(Lam_inv.max()):
            viol["P_le_prior"] += 1
        tail = [x[H:] for x in a[:6]]
        Kmat = chunk_matrices(*tail, M0, P0)["Kmat"]
        ev = torch.linalg.eigvalsh(Kmat)
        b = tail[5]
        logA = torch.cumsum(tail[3], 0)
        idx = torch.arange(C)
        lo, hi = torch.minimum(idx[:, None], idx[None, :]), torch.maximum(idx[:, None], idx[None, :])
        F = torch.einsum('ic,jc,ijc->ij', tail[0], tail[0], torch.exp(logA[hi] - logA[lo]) / c["lam"])
        lmaxF = float(torch.linalg.eigvalsh(F)[-1])
        trace = float((tail[0] ** 2 / c["lam"]).sum())
        n += 1
        viol["lmin"] += float(ev[0]) < (1 - 1e-12) / float(b.max())
        viol["lmax_F"] += float(ev[-1]) > (1 + 1e-12) * (1 / float(b.min()) + lmaxF)
        viol["lmax_trace"] += float(ev[-1]) > (1 + 1e-12) * (1 / float(b.min()) + trace)
        if unit:
            gmax = float(torch.exp(tail[3][:, 0]).max())
            row = (1 + gmax) / (1 - gmax) if gmax < 1 else float("inf")
            bound = float(b.max() / b.min()) + float(b.max()) / float(c["lam"][0]) * min(C, row)
            viol["cond_unit"] += float(ev[-1] / ev[0]) > (1 + 1e-12) * bound
            tight.append(float(ev[-1] / ev[0]) / bound)
    print(f"{n} chunks (histories of 0-200 tokens, C = 4-64, d = 4-16, gates from 0.5-1 to 0.99-1, beta 0.1-10);"
          f" violations: " + ", ".join(f"{k} {v}" for k, v in viol.items()))
    print(f"unit-key condition bound: cond(K)/bound between {min(tight):.3f} and {max(tight):.3f} over {len(tight)} chunks")
    print()
    return dict(n=n, violations=viol, cond_over_bound=[min(tight), max(tight)])


# --------------------------------------------------------------------------- part D
def part_d(quick):
    print("Part D. Gradients.")
    out = {}
    # D1: Proposition 6's adjoint
    rows = []
    for C, e in ((9, 4), (17, 6), (33, 3)):
        g = torch.Generator().manual_seed(C)
        A0 = torch.randn(C, C, generator=g)
        Ksym = A0 @ A0.T / C + torch.eye(C)
        X = torch.triu(torch.randn(C, C, generator=g))
        R = torch.randn(C, e, generator=g)
        Delta = torch.randn(C, e, generator=g)
        Kc, Xc, Rc = (z.clone().requires_grad_() for z in (Ksym, X, R))
        L = torch.linalg.cholesky(Kc)
        Y = torch.linalg.solve_triangular(L, Xc, upper=False)
        Rt = torch.linalg.solve_triangular(L, Rc, upper=False)
        (torch.tril(Y.T) @ Rt * Delta).sum().backward()
        Kp = Ksym.clone().requires_grad_()                         # general leaf: prefix solves as written
        o = torch.stack([X[:t + 1, t] @ torch.linalg.solve(Kp[:t + 1, :t + 1], R[:t + 1]) for t in range(C)])
        (o * Delta).sum().backward()
        with torch.no_grad():
            U = torch.triu(torch.ones(C, C))
            L0 = torch.linalg.cholesky(Ksym)
            Y0 = torch.linalg.solve_triangular(L0, X, upper=False)
            R0 = torch.linalg.solve_triangular(L0, R, upper=False)
            Ab = torch.linalg.solve_triangular(L0.T, (R0 @ Delta.T) * U, upper=True)
            Wb = torch.linalg.solve_triangular(L0.T, Y0 * U, upper=True)
            gsym = -0.5 * (Ab @ Wb.T + Wb @ Ab.T)
        rows.append(dict(C=C, dX=rel(Ab, Xc.grad), dR=rel(Wb @ Delta, Rc.grad), dK_sym=rel(gsym, Kc.grad),
                         dK_unsym_written=rel(-Wb @ Ab.T, Kp.grad), dK_old_form=rel(-Ab @ Wb.T, Kp.grad)))
    w = {k: max(r[k] for r in rows) for k in rows[0] if k != "C"}
    out["adjoint"] = w
    print(f"D1  Proposition 6 against autograd (C = 9, 17, 33): dl/dX {w['dX']:.1e}, dl/dR {w['dR']:.1e},"
          f" symmetrized dl/dK {w['dK_sym']:.1e}.")
    print(f"    Unsymmetrized, for the readout written as kappa^T K^-1 R: -W A^T matches to {w['dK_unsym_written']:.1e};"
          f" the earlier -A W^T is off by {w['dK_old_form']:.1e} (relative).")
    # D2: autograd through the chunk form vs through the token-by-token filter
    configs = [("stationary, scalar gates", dict(seed=800, T=29, d=8, e=4), 8),
               ("stationary, per-channel gates", dict(seed=801, T=29, d=8, e=4, per_channel=True), 8),
               ("free diagonal noise, per-channel", dict(seed=802, T=29, d=8, e=4, noise="diag", per_channel=True), 6),
               ("ill-conditioned: gates 0.999-1, beta 50, ridge 1e-2, M_0 a leaf",
                dict(seed=803, T=29, d=8, e=4, gates=(0.999, 1.0), beta=(50.0, 50.0), lam=(1e-2, 1e-2), prior_mean=True), 8),
               ("free diagonal noise, M_0 and P_init = S S^T leaves",
                dict(seed=804, T=29, d=8, e=4, noise="diag", per_channel=True, prior_mean=True), 6)]
    if quick:
        configs = configs[:1]
    res = []
    for name, kw, C in configs:
        c = make_case(**kw)
        stationary = kw.get("noise", "stationary") == "stationary"
        leaves = dict(K=c["K"], V=c["V"], Q=c["Q"], log_beta=torch.log(c["beta"]), logD=c["logD"])
        if stationary:
            leaves["log_lam"] = torch.log(c["lam"])
        else:
            leaves["log_Psi"] = torch.log(c["Psi"] + 1e-3)
        if kw.get("prior_mean"):
            leaves["M0"] = c["M0"]
        if "S S^T" in name:
            leaves["S"] = torch.linalg.cholesky(c["P0"])
        rng = np.random.default_rng(kw["seed"] + 50)
        wO = torch.from_numpy(rng.standard_normal((kw["T"], kw["e"])))
        wM = torch.from_numpy(rng.standard_normal((kw["e"], kw["d"])))
        wP = torch.from_numpy(rng.standard_normal((kw["d"], kw["d"])))

        def loss(fn, z):
            lam = torch.exp(z["log_lam"]) if stationary else None
            Psi = stationary_noise(z["logD"], lam) if stationary else torch.exp(z["log_Psi"])
            P0 = torch.diag(1 / lam) if stationary else (z["S"] @ z["S"].T if "S" in z else c["P0"])
            O, M, P = fn(z["K"], z["V"], z["Q"], z["logD"], Psi, torch.exp(z["log_beta"]), z.get("M0", c["M0"]), P0)
            return (O * wO).sum() + (M * wM).sum() + (P * wP).sum()

        grads = {}
        for label, fn in (("reference", kalman_reference),
                          ("solve", lambda *a: chunked(*a, C)),
                          ("explicit", lambda *a: chunked(*a, C, inverse="explicit"))):
            z = {k: v.clone().requires_grad_() for k, v in leaves.items()}
            loss(fn, z).backward()
            grads[label] = {k: v.grad for k, v in z.items()}
        errs = {k: max(rel(grads["solve"][k], grads["reference"][k]), rel(grads["explicit"][k], grads["reference"][k]))
                for k in leaves}
        # D3: central differences on the reference along a random direction
        dirs = {k: torch.from_numpy(rng.standard_normal(tuple(v.shape))) for k, v in leaves.items()}
        h = 1e-5
        with torch.no_grad():
            fp = loss(kalman_reference, {k: leaves[k] + h * dirs[k] for k in leaves})
            fm = loss(kalman_reference, {k: leaves[k] - h * dirs[k] for k in leaves})
        fd = float((fp - fm) / (2 * h))
        ad = float(sum((grads["solve"][k] * dirs[k]).sum() for k in leaves))
        res.append(dict(config=name, grad_rel=errs, fd=fd, autograd=ad, fd_rel=abs(fd - ad) / abs(fd)))
        print(f"D2  {name}: chunk-form gradients vs token-by-token autograd, worst over inputs "
              + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
        print(f"D3  {name}: directional derivative, central differences on the reference {fd:.10e},"
              f" autograd through the chunk form {ad:.10e}, relative gap {abs(fd - ad) / abs(fd):.1e}")
    out["gradients"] = res
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    results = dict(part_a=part_a(args.quick), part_a2=part_a2(args.quick), part_b=part_b(), part_c=part_c(),
                   part_d=part_d(args.quick))
    if not args.quick:
        from lab import save_results
        print(f"saved {save_results(results, 'results', 'check_chunk')}")


if __name__ == "__main__":
    main()
