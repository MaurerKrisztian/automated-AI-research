#!/usr/bin/env python3
"""
Improvement test for exact chunkwise Mesa: carry the inverse instead of refactoring every chunk.

Current method ("parallel"): every chunk factors B_n = ridge + H_n (a d x d Cholesky), all chunks in parallel.
Carried inverse ("carry"): keep P = (ridge + H)^{-1} as the recurrent state. Per chunk (keys K, queries Q):
    A   = I + K P K^T = L L^T      (C x C Cholesky: the only factorization)
    U^T = L^{-1} K P               (columns of U = normalized recursive-least-squares gains)
    X   = Q P - tril(Q U) U^T      (row tau = exact (ridge + H_tau)^{-1} q_tau)
    P  <- P - U U^T                (exact inverse for the next chunk)
No d x d factorization or d x d triangular solve; the rest is matrix multiplication.

Part 1 - speed on whole sequences (forward, and forward+backward) vs the parallel method and CG.
Part 2 - float32 drift of the carried inverse on a long sequence, with optional periodic refresh.

    python bench_carry.py              # defaults: batch 8, heads 8, seq 2048, head dim 128
    python bench_carry.py --tf32       # TF32 matmuls
    python bench_carry.py --batch 1 --heads 2 --seq 512 --chunks 64 --drift-seq 4096 --reps 2 --warmup 1   # smoke test
"""
import argparse
import json
import os
import platform
import time

import torch

from bench_exact_chunk import KeyGen, bench, cg, exact_solve, max_rel_err, nested_matvec


# --------------------------------------------------------------------------- methods on whole sequences
def chunk_states(K, lam):
    """B_n = lam I + sum_{m<n} K_m^T K_m for every chunk n.  K: (nb, N, C, d) -> (nb, N, d, d)."""
    G = K.transpose(-1, -2) @ K
    S = torch.cumsum(G, dim=1) - G
    return S + lam * torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)


def parallel_exact(K, Q, lam):
    """Current method: a d x d Cholesky per chunk, all chunks in parallel."""
    nb, N, C, d = K.shape
    B = chunk_states(K, lam)
    X = exact_solve(B.reshape(-1, d, d), K.reshape(-1, C, d), Q.reshape(-1, C, d))
    return X.reshape(nb, N, C, d)


def carry_exact(K, Q, lam, refresh=0, sym=False):
    """Carried inverse: only a C x C Cholesky per chunk, sequential over chunks."""
    nb, N, C, d = K.shape
    eye = torch.eye(d, device=K.device, dtype=K.dtype)
    eyeC = torch.eye(C, device=K.device, dtype=K.dtype)
    P = (eye / lam).expand(nb, d, d).contiguous()
    H = torch.zeros(nb, d, d, device=K.device, dtype=K.dtype) if refresh else None
    out = []
    for n in range(N):
        Kn, Qn = K[:, n], Q[:, n]
        KP = Kn @ P                                                   # (nb, C, d)
        L, _ = torch.linalg.cholesky_ex(KP @ Kn.transpose(-1, -2) + eyeC)
        Ut = torch.linalg.solve_triangular(L, KP, upper=False)        # U^T = L^{-1} K P
        out.append(Qn @ P - torch.tril(Qn @ Ut.transpose(-1, -2)) @ Ut)
        P = P - Ut.transpose(-1, -2) @ Ut
        if sym:
            P = 0.5 * (P + P.transpose(-1, -2))
        if refresh:
            H = H + Kn.transpose(-1, -2) @ Kn
            if (n + 1) % refresh == 0:
                P = torch.cholesky_inverse(torch.linalg.cholesky(H + lam * eye))
    return torch.stack(out, dim=1)


class CarrySolve(torch.autograd.Function):
    """Carried-inverse forward + closed-form backward (no differentiation through the factorizations).

    With x_t = P_t q_t and P_t = (lam I + sum_{i<=t} k_i k_i^T)^{-1}:
        dq_t = g_t := P_t dx_t                                     (the same chunk solve, reusing P_n and U_n)
        dk_i = -sum_{t>=i} (g_t x_t^T + x_t g_t^T) k_i              (a suffix sum, chunk-parallel)
    Every chunk is processed at once in the backward; only matrix products and a cumulative sum.
    """

    @staticmethod
    def forward(ctx, K, Q, lam):
        nb, N, C, d = K.shape
        eye = torch.eye(d, device=K.device, dtype=K.dtype)
        eyeC = torch.eye(C, device=K.device, dtype=K.dtype)
        P = (eye / lam).expand(nb, d, d).contiguous()
        Ps, Us, Xs = [], [], []
        for n in range(N):
            Kn, Qn = K[:, n], Q[:, n]
            KP = Kn @ P
            L, _ = torch.linalg.cholesky_ex(KP @ Kn.transpose(-1, -2) + eyeC)
            Ut = torch.linalg.solve_triangular(L, KP, upper=False)
            Xs.append(Qn @ P - torch.tril(Qn @ Ut.transpose(-1, -2)) @ Ut)
            Ps.append(P)
            Us.append(Ut)
            P = P - Ut.transpose(-1, -2) @ Ut
        X = torch.stack(Xs, dim=1)
        ctx.save_for_backward(K, torch.stack(Ps, dim=1), torch.stack(Us, dim=1), X)
        return X

    @staticmethod
    def backward(ctx, dX):
        K, Ps, Us, X = ctx.saved_tensors
        G = dX @ Ps - torch.tril(dX @ Us.transpose(-1, -2)) @ Us        # g_t = P_t dx_t, all chunks at once
        T = G.transpose(-1, -2) @ X
        T = T + T.transpose(-1, -2)                                      # per-chunk sum of g x^T + x g^T
        S_next = torch.flip(torch.cumsum(torch.flip(T, [1]), 1), [1]) - T  # sum over later chunks
        dK = -(K @ S_next) - torch.triu(K @ X.transpose(-1, -2)) @ G - torch.triu(K @ G.transpose(-1, -2)) @ X
        return dK, G, None


def grad_check(device):
    """Closed-form backward vs autograd through the factorizations (float64)."""
    g = torch.Generator(device=device).manual_seed(1)
    nb, N, C, d, lam = 2, 4, 16, 32, 0.25
    K = torch.randn(nb, N, C, d, generator=g, device=device, dtype=torch.float64)
    K = K / K.norm(dim=-1, keepdim=True)
    Q = torch.randn(nb, N, C, d, generator=g, device=device, dtype=torch.float64)
    R = torch.randn(nb, N, C, d, generator=g, device=device, dtype=torch.float64)
    errs = []
    for f in (lambda K, Q: parallel_exact(K, Q, lam), carry_closed_form_lam(lam)):
        Kr, Qr = K.clone().requires_grad_(True), Q.clone().requires_grad_(True)
        (f(Kr, Qr) * R).sum().backward()
        errs.append((Kr.grad, Qr.grad))
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    return rel(errs[1][0], errs[0][0]), rel(errs[1][1], errs[0][1])


def carry_closed_form_lam(lam):
    return lambda K, Q: CarrySolve.apply(K, Q, lam)


def cg_seq(K, Q, lam, iters):
    """MesaNet-style CG on every chunk's nested systems, all chunks in parallel."""
    nb, N, C, d = K.shape
    B = chunk_states(K, lam).reshape(-1, d, d)
    Kf, Qf = K.reshape(-1, C, d), Q.reshape(-1, C, d)
    mv = nested_matvec(B, Kf)
    diag = torch.diagonal(B, dim1=-2, dim2=-1).unsqueeze(-2) + torch.cumsum(Kf * Kf, dim=-2)
    return cg(mv, Qf, Qf / diag, iters).reshape(nb, N, C, d)


def make_inputs(kg, nb, N, C, d):
    K = kg.keys(nb, N * C).reshape(nb, N, C, d) * torch.sqrt(0.2 + 0.8 * kg.uniform(nb, N, C, 1))
    Q = kg.keys(nb, N * C).reshape(nb, N, C, d)
    return K, Q


def peak_gb(device):
    return torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--lam", type=float, default=0.25)
    p.add_argument("--cg-iters", type=int, nargs="+", default=[5, 10, 30])
    p.add_argument("--ref-seqs", type=int, default=2, help="sequences checked against the float64 reference")
    p.add_argument("--drift-seq", type=int, default=65536)
    p.add_argument("--drift-heads", type=int, default=4)
    p.add_argument("--drift-chunk", type=int, default=64)
    p.add_argument("--refresh", type=int, default=64, help="refresh period (chunks) in the drift test")
    p.add_argument("--drift-alpha", type=float, default=1.0,
                   help="key spectrum decay in the drift test (larger = keys in fewer directions, worse conditioning)")
    p.add_argument("--skip-speed", action="store_true", help="run only the drift test")
    p.add_argument("--skip-drift", action="store_true", help="run only the speed test")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/carry")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        dev_name = torch.cuda.get_device_name(device)
    else:
        dev_name = platform.processor() or "cpu"
    print(f"torch {torch.__version__} | {dev_name} | fp32 | TF32 {'on' if args.tf32 else 'off'}")
    print(f"batch {args.batch} x heads {args.heads} x seq {args.seq}, head dim {args.d}, ridge {args.lam}")
    results = {"env": {"torch": torch.__version__, "device": dev_name, "tf32": args.tf32},
               "args": vars(args), "speed": [], "drift": {}}
    kg = KeyGen(args.d, device, dtype, args.seed)
    nb, d, lam = args.batch * args.heads, args.d, args.lam

    dk_err, dq_err = grad_check(device)
    print(f"closed-form backward vs autograd (float64): dK rel err {dk_err:.1e}, dQ rel err {dq_err:.1e}")
    results["grad_check"] = {"dK": dk_err, "dQ": dq_err}

    methods = [("parallel (per-chunk d x d Cholesky)", lambda K, Q: parallel_exact(K, Q, lam)),
               ("carry (carried inverse)", lambda K, Q: carry_exact(K, Q, lam)),
               ("carry + closed-form backward", carry_closed_form_lam(lam))]
    methods += [(f"CG {k}", (lambda k: lambda K, Q: cg_seq(K, Q, lam, k))(k)) for k in args.cg_iters]

    # ---------------- Part 1: speed
    for C in ([] if args.skip_speed else args.chunks):
        N = args.seq // C
        K, Q = make_inputs(kg, nb, N, C, d)
        m = min(args.ref_seqs, nb)
        ref = parallel_exact(K[:m].double(), Q[:m].double(), lam)
        R = torch.randn_like(Q)
        print(f"\n=== chunk {C}: {N} chunks per sequence ===")
        print(f"{'method':38s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'peak GB':>8s} {'max rel err':>12s}")
        for name, f in methods:
            with torch.no_grad():
                err = max_rel_err(f(K[:m], Q[:m]), ref)
                t_fwd = bench(lambda: f(K, Q), device, args.warmup, args.reps)
            Kr, Qr = K.clone().requires_grad_(True), Q.clone().requires_grad_(True)

            def step():
                (f(Kr, Qr) * R).sum().backward()
                Kr.grad = None
                Qr.grad = None

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t_bwd = bench(step, device, args.warmup, args.reps)
            mem = peak_gb(device)
            print(f"{name:38s} {t_fwd:8.2f} {t_bwd:11.2f} {mem:8.2f} {err:12.1e}")
            results["speed"].append({"chunk": C, "method": name, "fwd_ms": t_fwd, "fwd_bwd_ms": t_bwd,
                                     "peak_gb": mem, "err": err})
            del Kr, Qr
        del K, Q, ref, R

    # ---------------- Part 2: float32 drift on a long sequence (no forgetting: the hardest case)
    if args.skip_drift:
        return save(results, args)
    C = args.drift_chunk
    N = args.drift_seq // C
    nbd = args.drift_heads
    K, Q = make_inputs(KeyGen(d, device, dtype, args.seed + 1, alpha=args.drift_alpha), nbd, N, C, d)
    with torch.no_grad():
        ref = parallel_exact(K.double(), Q.double(), lam)
        Hend = (K.double().reshape(nbd, -1, d).transpose(-1, -2) @ K.double().reshape(nbd, -1, d))
        ev = torch.linalg.eigvalsh(Hend + lam * torch.eye(d, device=device, dtype=torch.float64))
        cond = (ev[:, -1] / ev[:, 0]).max().item()
        print(f"\n=== drift: {nbd} heads x {args.drift_seq} tokens, chunk {C} ({N} chunks), key spectrum "
              f"alpha {args.drift_alpha}, final condition number {cond:.2e} ===")
        variants = [("parallel fp32", lambda: parallel_exact(K, Q, lam)),
                    ("carry fp32", lambda: carry_exact(K, Q, lam)),
                    ("carry fp32 + symmetrize", lambda: carry_exact(K, Q, lam, sym=True)),
                    (f"carry fp32 + refresh every {args.refresh} chunks",
                     lambda: carry_exact(K, Q, lam, refresh=args.refresh))]
        variants += [(f"CG {k} fp32", (lambda k: lambda: cg_seq(K, Q, lam, k))(k)) for k in (10, 30)]
        marks = [n for n in (1, 4, 16, 64, 256, 1024, 4096) if n <= N]
        print(f"{'variant':42s} " + " ".join(f"{'chunk ' + str(n):>12s}" for n in marks))
        results["drift"] = {"heads": nbd, "seq": args.drift_seq, "chunk": C, "alpha": args.drift_alpha,
                            "cond": cond, "marks": marks, "rows": []}
        for name, f in variants:
            X = f()
            e = ((X.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).amax(dim=(0, 2))   # (N,)
            vals = [e[n - 1].item() for n in marks]
            print(f"{name:42s} " + " ".join(f"{v:12.1e}" for v in vals))
            results["drift"]["rows"].append({"variant": name, "err_at_marks": vals,
                                             "err_max": e.max().item()})
            del X
    save(results, args)


def save(results, args):
    os.makedirs(args.out, exist_ok=True)
    tag = ("tf32" if args.tf32 else "fp32") + (f"_alpha{args.drift_alpha:g}" if args.drift_alpha != 1.0 else "")
    tag += "_speed" if args.skip_drift else ""
    path = os.path.join(args.out, f"carry_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
