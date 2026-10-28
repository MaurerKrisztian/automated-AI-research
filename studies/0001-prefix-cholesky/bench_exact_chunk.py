#!/usr/bin/env python3
"""
GPU micro-benchmark for the "exact chunkwise Mesa" idea.

Section A - exact one pass:
    For every chunk instance, solve for all positions tau = 1..C
        x_tau = (B + K[:tau]^T K[:tau])^{-1} q_tau
    with ONE Cholesky per chunk plus two masked triangular solves,
    versus MesaNet-style conjugate gradient (CG) with k iterations.

Section B - MesaNet's own model (fixed ridge + forget gates):
    MesaNet-style CG versus CG preconditioned by the exact chunk solver.

Quick CPU smoke test:
    python bench_exact_chunk.py --device cpu --batch 1 --heads 2 --seq 256 --reps 2 --warmup 1
GPU run (defaults: batch 8, heads 8, seq 2048, head dim 128):
    python bench_exact_chunk.py
"""
import argparse
import json
import math
import os
import platform
import statistics
import time

import torch


# --------------------------------------------------------------------------- helpers
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3  # ms


def max_rel_err(X, ref):
    num = (X.double() - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    return (num / den).max().item()


# --------------------------------------------------------------------------- synthetic keys
class KeyGen:
    """Unit-norm keys with an anisotropic spectrum (eigenvalues ~ i^-alpha)."""

    def __init__(self, d, device, dtype, seed, alpha=1.0):
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device, self.dtype, self.d = device, dtype, d
        U, _ = torch.linalg.qr(torch.randn(d, d, generator=self.g, device=device, dtype=dtype))
        self.U = U
        self.scale = torch.arange(1, d + 1, device=device, dtype=dtype) ** (-alpha / 2)

    def keys(self, nb, n):
        x = torch.randn(nb, n, self.d, generator=self.g, device=self.device, dtype=self.dtype)
        x = (x * self.scale) @ self.U.T
        return x / x.norm(dim=-1, keepdim=True)

    def uniform(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device, dtype=self.dtype)

    def normal(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.device, dtype=self.dtype)

    def state(self, nb, prefix, gamma=1.0, block=256):
        """Data Gram at chunk start: sum_i gamma^(prefix-1-i) beta_i k_i k_i^T (no ridge)."""
        d = self.d
        H0 = torch.empty(nb, d, d, device=self.device, dtype=self.dtype)
        w = gamma ** torch.arange(prefix - 1, -1, -1, device=self.device, dtype=self.dtype)
        for s in range(0, nb, block):
            e = min(nb, s + block)
            Kp = self.keys(e - s, prefix)
            beta = 0.2 + 0.8 * self.uniform(e - s, prefix, 1)
            H0[s:e] = (Kp * (beta * w[None, :, None])).transpose(-1, -2) @ Kp
        return H0


# --------------------------------------------------------------------------- exact chunk solver
def exact_factor(B, K):
    """One d x d Cholesky + one C x C Cholesky per chunk instance."""
    LB, _ = torch.linalg.cholesky_ex(B)
    BiKt = torch.cholesky_solve(K.transpose(-1, -2), LB)  # B^{-1} K^T : (nb, d, C)
    A = K @ BiKt                                           # (nb, C, C)
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)
    L, _ = torch.linalg.cholesky_ex(A)
    return LB, BiKt, L


def exact_apply(fac, K, Q):
    """x_tau = (B + K[:tau]^T K[:tau])^{-1} q_tau for all tau, given the factors."""
    LB, BiKt, L = fac
    BiQt = torch.cholesky_solve(Q.transpose(-1, -2), LB)  # (nb, d, C)
    W = torch.triu(torch.linalg.solve_triangular(L, K @ BiQt, upper=False))
    Z = torch.linalg.solve_triangular(L.transpose(-1, -2), W, upper=True)
    return (BiQt - BiKt @ Z).transpose(-1, -2)             # (nb, C, d)


def exact_solve(B, K, Q):
    """Fused one-pass version (one cholesky_solve for both K and Q)."""
    C = K.shape[-2]
    LB, _ = torch.linalg.cholesky_ex(B)
    Sol = torch.cholesky_solve(torch.cat([K, Q], dim=-2).transpose(-1, -2), LB)
    BiKt, BiQt = Sol[..., :C], Sol[..., C:]
    A = K @ BiKt
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)
    L, _ = torch.linalg.cholesky_ex(A)
    W = torch.triu(torch.linalg.solve_triangular(L, K @ BiQt, upper=False))
    Z = torch.linalg.solve_triangular(L.transpose(-1, -2), W, upper=True)
    return (BiQt - BiKt @ Z).transpose(-1, -2)


# --------------------------------------------------------------------------- CG (MesaNet style)
def cg(matvec, Q, X0, iters, precond=None):
    X = X0
    R = Q - matvec(X)
    Z = precond(R) if precond is not None else R
    P = Z
    rz = (R * Z).sum(-1, keepdim=True)
    for _ in range(iters):
        AP = matvec(P)
        alpha = rz / (P * AP).sum(-1, keepdim=True).clamp_min(1e-30)
        X = X + alpha * P
        R = R - alpha * AP
        Z = precond(R) if precond is not None else R
        rz_new = (R * Z).sum(-1, keepdim=True)
        P = Z + (rz_new / rz.clamp_min(1e-30)) * P
        rz = rz_new
    return X


def nested_matvec(B, K):
    """P -> rows tau: (B + K[:tau]^T K[:tau]) p_tau, chunk-parallel (GLA-style)."""
    def mv(P):
        S = torch.triu(K @ P.transpose(-1, -2))  # S[i, tau] = k_i . p_tau for i <= tau
        return P @ B + S.transpose(-1, -2) @ K
    return mv


def mesa_matvec(H0, Kh, a, lam):
    """MesaNet's true systems: a_tau (H0 + Kh[:tau]^T Kh[:tau]) + lam I."""
    def mv(P):
        S = torch.triu(Kh @ P.transpose(-1, -2))
        return a.unsqueeze(-1) * (P @ H0 + S.transpose(-1, -2) @ Kh) + lam * P
    return mv


# --------------------------------------------------------------------------- references (fp64)
def reference_nested(B, K, Q):
    B, K, Q = B.double(), K.double(), Q.double()
    nb, C, d = K.shape
    X = torch.empty(nb, C, d, dtype=torch.float64, device=K.device)
    M = B.clone()
    for t in range(C):
        M = M + K[:, t, :, None] * K[:, t, None, :]
        X[:, t] = torch.linalg.solve(M, Q[:, t])
    return X


def reference_mesa(H0, Kh, a, Q, lam):
    H0, Kh, a, Q = H0.double(), Kh.double(), a.double(), Q.double()
    nb, C, d = Kh.shape
    I = torch.eye(d, dtype=torch.float64, device=Kh.device)
    X = torch.empty(nb, C, d, dtype=torch.float64, device=Kh.device)
    G = H0.clone()
    for t in range(C):
        G = G + Kh[:, t, :, None] * Kh[:, t, None, :]
        X[:, t] = torch.linalg.solve(a[:, t, None, None] * G + lam * I, Q[:, t])
    return X


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--chunks", type=int, nargs="+", default=[16, 64])
    p.add_argument("--lam", type=float, default=0.25)
    p.add_argument("--prefix", type=int, default=1024)
    p.add_argument("--cg-iters", type=int, nargs="+", default=[5, 10, 15, 30])
    p.add_argument("--pcg-iters", type=int, nargs="+", default=[1, 2])
    p.add_argument("--gates", type=float, nargs="+", default=[0.999, 0.9])
    p.add_argument("--ref-instances", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster, less accurate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/bench")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        dev_name = torch.cuda.get_device_name(device)
    else:
        dev_name = platform.processor() or "cpu"
    print(f"torch {torch.__version__} | device {dev_name} | fp32 | TF32 {'on' if args.tf32 else 'off'}")
    print(f"batch {args.batch} x heads {args.heads} x seq {args.seq}, head dim {args.d}, ridge {args.lam}")

    results = {"env": {"torch": torch.__version__, "device": dev_name, "tf32": args.tf32},
               "args": vars(args), "A": [], "B": []}
    token_heads = args.batch * args.heads * args.seq
    kg = KeyGen(args.d, device, dtype, args.seed)
    eye = torch.eye(args.d, device=device, dtype=dtype)

    for C in args.chunks:
        nb = token_heads // C
        print(f"\n=== chunk {C}: {nb} chunk instances ===")

        # ---------------- Section A: exact one pass vs CG on the same nested systems
        H0 = kg.state(nb, args.prefix, gamma=1.0)
        B = H0 + args.lam * eye
        K = kg.keys(nb, C) * torch.sqrt(0.2 + 0.8 * kg.uniform(nb, C, 1))
        Q = kg.keys(nb, C)
        del H0

        m = min(args.ref_instances, nb)
        ref = reference_nested(B[:m], K[:m], Q[:m])
        X_ex = exact_solve(B, K, Q)
        err_ex = max_rel_err(X_ex[:m], ref)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_ex = bench(lambda: exact_solve(B, K, Q), device, args.warmup, args.reps)
        t_chol = bench(lambda: torch.linalg.cholesky_ex(B), device, args.warmup, args.reps)
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")

        mv = nested_matvec(B, K)
        diag = torch.diagonal(B, dim1=-2, dim2=-1).unsqueeze(-2) + torch.cumsum(K * K, dim=-2)
        print(f"[A] exact one pass: {t_ex:8.2f} ms  (d x d Cholesky alone {t_chol:.2f} ms)"
              f"  max rel err {err_ex:.1e}  peak mem {mem:.2f} GB")
        rowA = {"chunk": C, "exact_ms": t_ex, "chol_dxd_ms": t_chol, "exact_err": err_ex, "cg": []}
        for k in args.cg_iters:
            Xk = cg(mv, Q, Q / diag, k)
            ek = max_rel_err(Xk[:m], ref)
            tk = bench(lambda: cg(mv, Q, Q / diag, k), device, args.warmup, args.reps)
            print(f"[A] CG {k:2d} iters  : {tk:8.2f} ms  max rel err {ek:.1e}"
                  f"  -> exact is {tk / t_ex:5.1f}x faster")
            rowA["cg"].append({"iters": k, "ms": tk, "err": ek})
        results["A"].append(rowA)
        del B, K, Q, X_ex, ref, mv, diag

        # ---------------- Section B: MesaNet's model (fixed ridge + forget gates)
        for gbar in args.gates:
            H0 = kg.state(nb, args.prefix, gamma=gbar)
            K = kg.keys(nb, C)
            Q = kg.keys(nb, C)
            beta = 0.2 + 0.8 * kg.uniform(nb, C)
            logit = math.log(gbar / (1 - gbar)) + 0.3 * kg.normal(nb, C)
            a = torch.cumprod(torch.sigmoid(logit), dim=-1)
            Kh = K * torch.sqrt(beta / a).unsqueeze(-1)
            del K

            m = min(args.ref_instances, nb)
            ref = reference_mesa(H0[:m], Kh[:m], a[:m], Q[:m], args.lam)
            mv = mesa_matvec(H0, Kh, a, args.lam)
            diag = a.unsqueeze(-1) * (torch.diagonal(H0, dim1=-2, dim2=-1).unsqueeze(-2)
                                      + torch.cumsum(Kh * Kh, dim=-2)) + args.lam
            abar = torch.sqrt(a[:, :1] * a[:, -1:])                       # (nb, 1)
            bound = torch.sqrt(a[:, 0] / a[:, -1]).max().item()

            def run_pcg(k):
                fac = exact_factor(H0 + (args.lam / abar).unsqueeze(-1) * eye, Kh)
                pre = lambda R: exact_apply(fac, Kh, R) / a.unsqueeze(-1)
                return cg(mv, Q, pre(Q), k, pre)

            print(f"[B] gate ~{gbar}: condition-number bound for preconditioned CG = {bound:.2f}")
            rowB = {"chunk": C, "gate": gbar, "bound": bound, "cg": [], "pcg": []}
            for k in args.cg_iters:
                ek = max_rel_err(cg(mv, Q, Q / diag, k)[:m], ref)
                tk = bench(lambda: cg(mv, Q, Q / diag, k), device, args.warmup, args.reps)
                print(f"[B]   MesaNet CG {k:2d} iters : {tk:8.2f} ms  max rel err {ek:.1e}")
                rowB["cg"].append({"iters": k, "ms": tk, "err": ek})
            for k in args.pcg_iters:
                ek = max_rel_err(run_pcg(k)[:m], ref)
                tk = bench(lambda: run_pcg(k), device, args.warmup, args.reps)
                print(f"[B]   exact-chunk PCG {k} iter: {tk:8.2f} ms  max rel err {ek:.1e}")
                rowB["pcg"].append({"iters": k, "ms": tk, "err": ek})
            results["B"].append(rowB)
            del H0, Kh, Q, a, ref, mv, diag

    os.makedirs(args.out, exist_ok=True)
    tag = "".join(ch if ch.isalnum() else "_" for ch in dev_name)[:40]
    path = os.path.join(args.out, f"bench_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
