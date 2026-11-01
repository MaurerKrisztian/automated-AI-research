#!/usr/bin/env python3
"""E2 checks for study 0004, before any training. Run from this folder:
    ../../.venv/bin/python -u check_layers.py 2>&1 | tee logs/check_layers.log

A. The layer form (kalman_layer.kalman_chunk: whitened, safe Grams) in float64 against study 0003's token-by-token filter
   in extended precision (kalman_memory.kalman_reference_np), on the unwhitened inputs, over random and edge cases.
B. Gradients: autograd through kalman_chunk against autograd through the token-by-token filter, float64, every input.
C. The guard against 0003's claim 10: per-channel gates that overflow the matmul Grams, in float32. Forward and gradients
   stay finite here, and the matmul form of 0003 fails.
D. One key written twice, in closed form (README, Proposition 1), against the filters and recurrences in float64.
E. The task generators against brute force, and MQAR against study 0001's generator.
F. GPU only: fla's Gated DeltaNet and MesaNet kernels, as the models call them, against float64 references, forward and
   backward; and one training step through every model.
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kalman_layer import kalman_chunk, kalman_filter

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "0003-kalman-memory"))
import kalman_batched as kb                                              # noqa: E402
from kalman_memory import kalman_reference_np                            # noqa: E402

D64 = torch.float64


def make_case(seed, T, d, e, N=3, gates="scalar", gate_range=(0.5, 1.0), noise="stationary", beta=(0.5, 5.0),
              lam=(0.5, 2.0), psi=(0.01, 0.5), unit_keys=False, dtype=D64):
    rng = np.random.default_rng(seed)
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64)).to(dtype)
    K = rng.standard_normal((N, T, d))
    Q = rng.standard_normal((N, T, d))
    if unit_keys:
        K, Q = K / np.linalg.norm(K, axis=-1, keepdims=True), Q / np.linalg.norm(Q, axis=-1, keepdims=True)
    else:
        K, Q = K / math.sqrt(d), Q / math.sqrt(d)
    c = dict(K=t(K), V=t(rng.standard_normal((N, T, e))), Q=t(Q),
             beta=t(np.exp(rng.uniform(math.log(beta[0]), math.log(beta[1]), (N, T)))))
    if gates == "identity":
        c["logD"] = None
    else:
        lo, hi = np.log(gate_range[0]), np.log(gate_range[1])
        c["logD"] = t(rng.uniform(lo, hi, (N, T, 1 if gates == "scalar" else d)))
    if noise == "stationary":
        c["lam"] = t(np.exp(rng.uniform(math.log(lam[0]), math.log(lam[1]), (N, d))))
    else:
        c["psi"] = t(rng.uniform(psi[0], psi[1], (N, T)))
        c["P_init"] = t(rng.uniform(0.5, 3.0, (N, 1, 1))) * torch.eye(d, dtype=dtype)
    return c


def layer_args(c):
    """Whiten: sqrt(beta) k, sqrt(beta) v, unit observation noise."""
    s = c["beta"].sqrt()[..., None]
    out = dict(K=c["K"] * s, V=c["V"] * s, Q=c["Q"], logD=c["logD"])
    out.update({n: c[n] for n in ("lam", "psi", "P_init") if n in c})
    return out


def reference(c, h):
    """Head h by the extended-precision filter, on the unwhitened inputs."""
    T, d = c["K"].shape[1:]
    e = c["V"].shape[-1]
    logD = torch.zeros(T, 1, dtype=D64) if c["logD"] is None else c["logD"][h]
    logD = logD.expand(T, d)
    if "lam" in c:
        Psi = (-torch.expm1(2 * logD) / c["lam"][h]).detach()
        P0 = torch.diag(1 / c["lam"][h])
    else:
        Psi = c["psi"][h][:, None].expand(T, d)
        P0 = c["P_init"][h]
    M0 = torch.zeros(e, d, dtype=D64)
    return kalman_reference_np(c["K"][h], c["V"][h], c["Q"][h], logD, Psi, c["beta"][h], M0, P0)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.longdouble), np.asarray(b, dtype=np.longdouble)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def part_a(quick):
    print("Part A: kalman_chunk (float64) against the extended-precision token-by-token filter, relative to max |ref|")
    print(f"{'case':<46} {'C':>3} {'O':>9} {'M_T':>9} {'P_T':>9}")
    cases = [
        ("scalar 0.5-1, stationary", dict(T=40, d=8, e=8), 8, dict()),
        ("per-channel 0.01-1, stationary", dict(T=40, d=8, e=8), 8, dict(gates="channel", gate_range=(0.01, 1.0))),
        ("scalar 0.5-1, free", dict(T=40, d=8, e=8), 8, dict(noise="free")),
        ("per-channel 0.01-1, free", dict(T=40, d=8, e=8), 8, dict(gates="channel", gate_range=(0.01, 1.0), noise="free")),
        ("identity, free (BL's MQAR setting)", dict(T=40, d=8, e=8), 8, dict(gates="identity", noise="free")),
        ("identity, stationary (Mesa, no noise)", dict(T=40, d=8, e=8), 8, dict(gates="identity")),
        ("scalar near resets, e^-20..e^-1", dict(T=32, d=8, e=8), 8, dict(gate_range=(math.exp(-20), math.exp(-1)))),
        ("per-channel near resets, e^-20..e^-1", dict(T=32, d=8, e=8), 8, dict(gates="channel", gate_range=(math.exp(-20), math.exp(-1)))),
        ("per-channel near resets, free", dict(T=32, d=8, e=8), 8, dict(gates="channel", gate_range=(math.exp(-20), math.exp(-1)), noise="free")),
        ("T not a multiple of C (37, 8)", dict(T=37, d=8, e=5), 8, dict()),
        ("T < C (5, 8), per-channel", dict(T=5, d=8, e=5), 8, dict(gates="channel")),
        ("C = 1, per-channel, free", dict(T=9, d=4, e=4), 1, dict(gates="channel", noise="free")),
        ("d above C (d 32, C 8)", dict(T=24, d=32, e=32), 8, dict()),
        ("d below C (d 4, C 32), free", dict(T=64, d=4, e=4), 32, dict(noise="free")),
        ("mixed beta 1e-3..50", dict(T=40, d=8, e=8), 8, dict(beta=(1e-3, 50.0))),
        ("beta 1e-9..1 (a token that writes nothing)", dict(T=40, d=8, e=8), 8, dict(beta=(1e-9, 1.0))),
        ("study's size: T 96, d 32, C 64, unit keys", dict(T=96, d=32, e=32), 64, dict(unit_keys=True)),
        ("study's size: T 96, d 32, C 16, per-channel", dict(T=96, d=32, e=32), 16, dict(gates="channel", unit_keys=True)),
        ("study's size: T 96, d 32, C 64, per-channel", dict(T=96, d=32, e=32), 64, dict(gates="channel", unit_keys=True)),
        ("study's size: T 96, d 32, C 64, identity, free", dict(T=96, d=32, e=32), 64, dict(gates="identity", noise="free", unit_keys=True)),
    ]
    if quick:
        cases = cases[:6]
    worst = 0.0
    for i, (name, size, C, kw) in enumerate(cases):
        c = make_case(i, **size, **kw)
        O, M, P = kalman_chunk(C=C, **layer_args(c))
        eo = em = ep = 0.0
        for h in range(c["K"].shape[0]):
            ro, rm, rp = reference(c, h)
            eo, em, ep = max(eo, rel(O[h].numpy(), ro)), max(em, rel(M[h].numpy(), rm)), max(ep, rel(P[h].numpy(), rp))
        worst = max(worst, eo, em, ep)
        print(f"{name:<46} {C:>3} {eo:>9.1e} {em:>9.1e} {ep:>9.1e}")
    print(f"worst of {len(cases)} cases: {worst:.1e}\n")
    assert worst < 1e-11
    return worst


def part_b():
    print("Part B: autograd through kalman_chunk against autograd through the token-by-token filter (float64), every input")
    configs = [("scalar, stationary", dict()), ("per-channel, stationary", dict(gates="channel")),
               ("scalar, free", dict(noise="free")), ("per-channel, free", dict(gates="channel", noise="free")),
               ("identity, free", dict(gates="identity", noise="free"))]
    worst = 0.0
    for i, (name, kw) in enumerate(configs):
        c = make_case(100 + i, T=21, d=6, e=6, N=2, **kw)
        a = layer_args(c)
        R = torch.randn(2, 21, 6, dtype=D64, generator=torch.Generator().manual_seed(i))

        def grads(fn):
            ins = {n: (None if v is None else v.clone().requires_grad_(True)) for n, v in a.items()}
            (fn(**ins)[0] * R).sum().backward()
            out = {n: v.grad for n, v in ins.items() if v is not None}
            if "P_init" in out:                                          # P is symmetric, and only perturbed symmetrically
                out["P_init"] = 0.5 * (out["P_init"] + out["P_init"].mT)
            return out
        g1, g2 = grads(lambda **z: kalman_chunk(C=4, **z)), grads(kalman_filter)
        errs = {n: (g1[n] - g2[n]).abs().max().item() / max(g2[n].abs().max().item(), 1e-300) for n in g1}
        worst = max(worst, *errs.values())
        print(f"{name:<26}" + " ".join(f"{n}={v:.1e}" for n, v in errs.items()))
    print(f"worst: {worst:.1e}\n")
    assert worst < 1e-10
    return worst


def part_c():
    print("Part C: per-channel gates that overflow 0003's matmul Grams, float32, C = 64, d = 32 (T = 192)")
    print(f"{'gates':<16} {'min chunk log decay':>20} {'matmul Grams (0003)':>22} {'safe Grams':>12} {'grad finite':>12}")
    worst = 0.0
    for lo, hi in [(0.5, 1.0), (0.1, 0.3), (1e-3, 0.1), (1e-6, 1e-3)]:
        c = make_case(7, T=192, d=32, e=32, N=2, gates="channel", gate_range=(lo, hi), unit_keys=True, beta=(0.5, 2.0))
        a = layer_args(c)
        f32 = {n: v.float() for n, v in a.items()}
        ref = kalman_filter(**a)[0]
        safe = kalman_chunk(C=64, **f32)[0]
        safe_err = rel(safe.double().numpy(), ref.numpy())
        try:
            old = kb.kalman_chunkwise(f32["K"], f32["V"], f32["Q"], f32["logD"], torch.ones(2, 192), 64, lam=f32["lam"],
                                      gram="matmul")[0]
            old_err = rel(old.double().numpy(), ref.numpy()) if torch.isfinite(old).all() else float("nan")
        except Exception as exc:                                         # a failed factorization or a non-finite input
            old_err = float("nan")
            print("  matmul form raised", type(exc).__name__)
        g = {n: v.clone().requires_grad_(True) for n, v in f32.items()}
        (kalman_chunk(C=64, **g)[0] ** 2).sum().backward()
        finite = all(torch.isfinite(v.grad).all().item() for v in g.values())
        decay = a["logD"][:, :64].sum(1).min().item()
        worst = max(worst, safe_err)
        print(f"{lo:g}-{hi:g}".ljust(16) + f"{decay:>20.1f} {old_err:>22.1e} {safe_err:>12.1e} {str(finite):>12}")
        assert finite and safe_err < 1e-4
    print(f"worst safe-form error: {worst:.1e}\n")
    return worst


def part_d():
    print("Part D: one key written twice, closed forms (README, Proposition 1) against the recurrences, float64")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(50):
        d = 4
        k = rng.standard_normal(d)
        k = k / np.linalg.norm(k)
        v1, v2 = rng.standard_normal(3), rng.standard_normal(3)
        b1, b2, lam = np.exp(rng.uniform(-3, 3)), np.exp(rng.uniform(-3, 3)), np.exp(rng.uniform(-2, 1))
        gamma, tau, s = rng.uniform(0.5, 1.0), int(rng.integers(1, 20)), int(rng.integers(0, 10))
        a = gamma ** tau
        steps = tau + s + 1
        # Token 0 writes v1, token tau writes v2, the others write nothing (zero keys); read at the last token.
        K = np.zeros((steps, d))
        V = np.zeros((steps, 3))
        K[0], K[tau], V[0], V[tau] = k, k, v1, v2
        beta = np.ones(steps)
        beta[0], beta[tau] = b1, b2
        Kt, Vt = torch.tensor(K, dtype=D64), torch.tensor(V, dtype=D64)
        Qt = torch.zeros(steps, d, dtype=D64)
        Qt[-1] = torch.tensor(k, dtype=D64)
        # Kalman, stationary, gate gamma at every token after the first.
        logD = torch.full((steps, d), math.log(gamma), dtype=D64)
        logD[0] = 0.0
        Psi = -torch.expm1(2 * logD) / lam
        out, _, _ = kalman_reference_np(Kt, Vt, Qt, logD, Psi, torch.tensor(beta, dtype=D64), torch.zeros(3, d, dtype=D64),
                                        torch.eye(d, dtype=D64) / lam)
        den = (1 / lam + 1 / b1) * (1 / lam + 1 / b2) - a ** 2 / lam ** 2
        w1 = a / (lam * b2) / den
        w2 = ((1 - a ** 2) / lam ** 2 + 1 / (lam * b1)) / den
        pred = gamma ** s * (w1 * v1 + w2 * v2)
        e1 = rel(np.asarray(out[-1], dtype=np.float64), pred)
        ratio = a * lam * b1 / (b2 * ((1 - a ** 2) * b1 + lam))
        e1b = abs(w1 / w2 - ratio) / ratio
        # Kalman, free noise, identity dynamics: accumulated noise Psi over the tau steps between the writes.
        s0, psi = np.exp(rng.uniform(-2, 1)), np.exp(rng.uniform(-5, 0))
        Psi2 = torch.full((steps, d), psi, dtype=D64)
        Psi2[0] = 0.0
        out2, _, _ = kalman_reference_np(Kt, Vt, Qt, torch.zeros(steps, d, dtype=D64), Psi2, torch.tensor(beta, dtype=D64),
                                         torch.zeros(3, d, dtype=D64), s0 * torch.eye(d, dtype=D64))
        Ps = psi * tau
        den2 = s0 * Ps + (s0 + Ps) / b1 + s0 / b2 + 1 / (b1 * b2)
        u1, u2 = (s0 / b2) / den2, (s0 * Ps + (s0 + Ps) / b1) / den2
        e2 = rel(np.asarray(out2[-1], dtype=np.float64), u1 * v1 + u2 * v2)
        # MesaNet: H and G with gates, one dense solve.
        a1, a2 = gamma ** (tau + s), gamma ** s
        H = (a1 * b1 + a2 * b2) * np.outer(k, k) + lam * np.eye(d)
        G = a1 * b1 * np.outer(v1, k) + a2 * b2 * np.outer(v2, k)
        o3 = G @ np.linalg.solve(H, k)
        e3 = rel(o3, (a1 * b1 * v1 + a2 * b2 * v2) / (a1 * b1 + a2 * b2 + lam))
        # Gated DeltaNet: M <- gamma M; M <- M + beta (v - M k) k^T.
        M = np.zeros((3, d))
        for t in range(steps):
            M = M * (gamma if t else 1.0)
            if t in (0, tau):
                M = M + (beta[t] if beta[t] < 1 else 0.5) * np.outer(V[t] - M @ K[t], K[t])
        b_1, b_2 = (b1 if b1 < 1 else 0.5), (b2 if b2 < 1 else 0.5)
        pred4 = gamma ** s * (b_2 * v2 + (1 - b_2) * a * b_1 * v1)
        e4 = rel(M @ k, pred4)
        worst = max(worst, e1, e1b, e2, e3, e4)
    print(f"worst over 50 draws: {worst:.1e}\n")
    assert worst < 1e-11
    return worst


def part_e():
    import tasks
    print("Part E: task generators against brute force")
    spec = importlib.util.spec_from_file_location("mqar_0001", HERE.parent / "0001-prefix-cholesky" / "mqar_recall.py")
    m0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m0)
    dev = torch.device("cpu")
    for n in (32, 64, 128):
        x0, y0 = m0.mqar_batch(8, n, 1024, dev, torch.Generator().manual_seed(n))
        x1, y1, _ = tasks.mqar_batch(8, n, 1024, dev, torch.Generator().manual_seed(n))
        assert torch.equal(x0, x1) and torch.equal(y0, y1)
    print("MQAR: identical to study 0001's generator at N = 32, 64, 128")

    K, B = 16, 2000
    x, y, info = tasks.volatile_batch(B, K, 0.30, 1024, dev, torch.Generator().manual_seed(1))
    assert x.shape == (B, 8 * K)
    changes = 0
    for b in range(B):
        hidden, seen = {}, {}
        for r in range(3):
            pairs = x[b, 2 * K * r:2 * K * (r + 1)].tolist()
            ks, vs = pairs[::2], pairs[1::2]
            assert sorted(ks) == sorted(set(ks)) and len(ks) == K and all(0 < t < 512 for t in ks) and all(t >= 512 for t in vs)
            if r:
                assert set(ks) == set(hidden)
                changes += sum(hidden[k] != v for k, v in zip(ks, vs))
            hidden.update(zip(ks, vs))
            for k_, v_ in zip(ks, vs):
                seen.setdefault(k_, []).append(v_)
        q = x[b, 6 * K:].tolist()
        tgt = y[b, 6 * K:].tolist()
        assert all(t == -100 for t in tgt[1::2]) and sorted(q[::2]) == sorted(hidden)
        for j, (k_, t_) in enumerate(zip(q[::2], tgt[::2])):
            assert hidden[k_] == t_
            assert info["older"][b, j].tolist() == seen[k_][:2]
            assert bool(info["changed"][b, j]) == (len(set(seen[k_])) > 1)
    rate = changes / (B * K * 2)
    print(f"Volatile MQAR: rounds, keys, final values, stale values and the change flags match a Python replay of {B} "
          f"sequences; change rate per key and round {rate:.4f} (expected 0.30)")
    assert abs(rate - 0.30) < 0.01

    L, n, V = 256, 16, 16
    x, y, _ = tasks.selcopy_batch(B, L, n, V, dev, torch.Generator().manual_seed(2))
    blank, copy = V - 2, V - 1
    last_blank = 0
    for b in range(B):
        seq = x[b].tolist()
        c = seq.index(copy)
        assert c == L - n - 1 and seq[c + 1:] == [blank] * n and seq.count(copy) == 1
        data = [t for t in seq[:c] if t != blank]
        assert len(data) == n and y[b, c + 1:].tolist() == data and all(t == -100 for t in y[b, :c + 1].tolist())
        last_blank += seq[c - 1] == blank
    assert last_blank == 0
    print(f"Selective copying: length, token ids and targets match a Python scan of {B} sequences; the last data token "
          "always precedes the copy token, as in MAD's generator\n")


def part_f():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.mesa_net import chunk_mesa_net

    import diagnostics
    from models import ALT, MODELS, Net
    print("Part F (GPU): fla's kernels as the models call them, float32, against float64 references")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(5)
    B, T, H, d = 4, 150, 2, 32
    q, k, v = (torch.randn(B, T, H, d, device=dev, generator=g) for _ in range(3))
    a, b = torch.randn(B, T, H, device=dev, generator=g), torch.randn(B, T, H, device=dev, generator=g)
    A_log = torch.log(torch.tensor([3.0, 10.0], device=dev))
    dt_bias = torch.tensor([-4.0, -6.0], device=dev)
    R = torch.randn(B, T, H, d, device=dev, generator=g)

    # Gated DeltaNet: the kernel with its l2 norm, sigmoid and gate, against the recurrence in float64.
    def gdn_ref(q, k, v, a, b, A_log, dt_bias):
        qn, kn = F.normalize(q, dim=-1) * d ** -0.5, F.normalize(k, dim=-1)
        beta = torch.sigmoid(b)
        gate = -torch.exp(A_log) * F.softplus(a + dt_bias)
        h = torch.zeros(B, H, d, d, dtype=D64, device=dev)
        out = []
        for t in range(T):
            h = h * torch.exp(gate[:, t])[..., None, None]
            pred = (kn[:, t][..., None, :] @ h)[..., 0, :]
            h = h + kn[:, t][..., None] * ((v[:, t] - pred) * beta[:, t][..., None])[..., None, :]
            out.append((qn[:, t][..., None, :] @ h)[..., 0, :])
        return torch.stack(out, 1)
    ins = [t.clone().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    o = chunk_gated_delta_rule(*ins[:3], ins[3], ins[4], use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                               use_gate_in_kernel=True, A_log=ins[5], dt_bias=ins[6])[0]
    (o * R).sum().backward()
    ins2 = [t.double().clone().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    o2 = gdn_ref(*ins2)
    (o2 * R.double()).sum().backward()
    fw = ((o.double() - o2).abs().max() / o2.abs().max()).item()
    bw = {n: ((p.grad.double() - p2.grad).norm() / p2.grad.norm()).item() for n, p, p2 in zip("qkvab", ins, ins2)}
    bw.update(A_log=((ins[5].grad.double() - ins2[5].grad).norm() / ins2[5].grad.norm()).item(),
              dt_bias=((ins[6].grad.double() - ins2[6].grad).norm() / ins2[6].grad.norm()).item())
    print(f"Gated DeltaNet: forward error {fw:.1e}; backward, relative " + " ".join(f"{n}={e:.1e}" for n, e in bw.items()))
    assert fw < 1e-3 and max(bw.values()) < 2e-2

    # MesaNet: the CG kernel against a dense float64 solve per token, forward; and against autograd through that solve.
    gm, bm = F.logsigmoid(a + 2), torch.sigmoid(b)
    lamb = (F.softplus(torch.zeros(H, d, device=dev)) * 0.5 + 0.25)
    ins = [t.clone().requires_grad_(True) for t in (q, k, v, gm, bm, lamb)]
    o = chunk_mesa_net(*ins, max_CG_iteration=30, use_qk_l2norm_in_kernel=True)[0]
    (o * R).sum().backward()
    ins2 = [t.double().clone().requires_grad_(True) for t in (q, k, v, gm, bm, lamb)]
    cap = dict(zip(("q", "k", "v", "g", "beta", "lamb"), ins2))
    o2, cond = diagnostics._mesa_exact(cap)
    (o2 * R.double()).sum().backward()
    fw = ((o.double() - o2).abs().max() / o2.abs().max()).item()
    bw = {n: ((p.grad.double() - p2.grad).norm() / p2.grad.norm()).item() for n, p, p2 in zip(("q", "k", "v", "g", "beta", "lam"), ins, ins2)}
    print(f"MesaNet, 30 CG steps (cond(H + Lambda) up to {cond:.3g}): forward error {fw:.1e}; backward, relative "
          + " ".join(f"{n}={e:.1e}" for n, e in bw.items()))
    assert fw < 5e-2 and max(bw.values()) < 2e-1

    # One training step through every model: finite gradients on every parameter that takes part.
    print(f"{'model':<24} {'options':<30} {'params':>8} {'loss':>7}  gradients finite")
    x = torch.randint(1, 1024, (4, 96), device=dev, generator=g)
    y = torch.randint(0, 1024, (4, 96), device=dev, generator=g)
    for name, (kind, opts) in MODELS.items():
        for extra in ({}, ALT[name]):
            torch.manual_seed(0)
            kw = {**opts, **extra}
            if kind == "kalman":
                kw["chunk"] = 32 if kw.get("gates") == "channel" else 64
            m = Net(1024, 64, 2, 2, kind, **kw).to(dev)
            loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
            loss.backward()
            missing = [n for n, p in m.named_parameters() if p.grad is None and "conv" not in n]
            finite = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
            print(f"{name:<24} {str(extra):<30} {sum(p.numel() for p in m.parameters()):>8} {loss.item():>7.3f}  {finite}"
                  + (f"  no grad: {missing}" if missing else ""))
            assert finite and not missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--parts", default="ABCDEF")
    args = p.parse_args()
    torch.manual_seed(0)
    res = {}
    if "A" in args.parts:
        res["A"] = part_a(args.quick)
    if "B" in args.parts:
        res["B"] = part_b()
    if "C" in args.parts:
        res["C"] = part_c()
    if "D" in args.parts:
        res["D"] = part_d()
    if "E" in args.parts:
        part_e()
    if "F" in args.parts and torch.cuda.is_available():
        part_f()
    print("all checks passed:", {k: f"{v:.1e}" for k, v in res.items()})


if __name__ == "__main__":
    main()
