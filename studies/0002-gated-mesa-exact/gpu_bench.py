#!/usr/bin/env python3
"""GPU micro-benchmark (E3) of the exact chunk solver against chunked CG, forward solve only (backlog I-011, small).

Both methods are plain PyTorch, batched over sequences, chunks and tokens:
- exact: start-anchor policy with capacitances padded to the chunk size C (one batched `eigh` of the whitened
  chunk-start state, then one batched C x C Cholesky per token). The two-anchor policy is not ported: it was less
  accurate in float32 (README, claim 4), and its variable sizes do not batch.
- cg: the chunked CG of flash-linear-attention's MesaNet kernel (x0 = 0, no preconditioner), and Jacobi PCG from
  MesaNet's diagonal start, each at fixed step counts. M_t p = Lambda p + a_t H_0 p + sum_{i<=t} (a_t/a_i) beta_i k_i (k_i . p).
Both need the chunk-start states H_0, which are included in every timing. The readout o_t = G_t x_t is the same
for every method and is not timed; output errors are measured on it, in float64, against a dense float64 solve.

Part A checks the port in float64 against gated_mesa.chunk_solve (NumPy). Part B times float32 solves on the GPU
(TF32 off) and measures their output error; it also times each stage of the exact solver. The eigendecomposition
runs on three backends: PyTorch's default (cuSOLVER), MAGMA, and the CPU (with copies to and from the GPU). Part C
times batched eigh alone against matrix size and batch. Version 3 adds the Triton Jacobi backend (jacobi_eigh.py) and
writes logs/gpu_bench_v3.log. Version 2 (after the review): eigh backends, a residual stop
for CG (--cg-tol), CG at 200 steps, and Part C; version 1 produced logs/gpu_bench.log.
Usage:
    ../../.venv/bin/python -u gpu_bench.py 2>&1 | tee logs/gpu_bench_v3.log
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import gated_mesa as gm
from lab import bench, peak_memory_gb, reset_peak_memory, save_results, seed_everything

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- shared pieces
def chunked(x, C):
    """(N, T, ...) -> (N, T // C, C, ...)."""
    N, T = x.shape[:2]
    return x.reshape(N, T // C, C, *x.shape[2:])


def chunk_logs(loggam, C):
    """Cumulative log gate inside each chunk, (N, nc, C)."""
    return chunked(loggam, C).cumsum(-1)


def decay_mask(L, beta_c):
    """W[t, i] = (a_t / a_i) beta_i for i <= t, else 0: (N, nc, C, C)."""
    diff = L[..., :, None] - L[..., None, :]
    C = L.shape[-1]
    causal = torch.ones(C, C, dtype=torch.bool, device=L.device).tril()
    return torch.where(causal, diff.clamp_max(0).exp(), torch.zeros((), dtype=L.dtype, device=L.device)) * beta_c[..., None, :]


def start_states(k, beta, loggam, C):
    """H_0 of every chunk, (N, nc, d, d), by the gated linear-attention state recurrence."""
    N, T, d = k.shape
    nc = T // C
    Kc, L = chunked(k, C), chunk_logs(loggam, C)
    w = (L[..., -1:] - L).exp() * chunked(beta, C)                       # a_C / a_i * beta_i
    upd = torch.einsum("nci,ncid,ncie->ncde", w, Kc, Kc)                 # all chunks' updates at once
    aC = L[..., -1].exp()
    H = torch.zeros(N, d, d, dtype=k.dtype, device=k.device)
    Hs = []
    for c in range(nc):
        Hs.append(H)
        H = aC[:, c, None, None] * H + upd[:, c]
    return torch.stack(Hs, 1)


def readout(X, k, v, beta, loggam, C):
    """o_t = G_t x_t from the token recurrence's chunk form (any dtype)."""
    N, T, d = k.shape
    dv = v.shape[-1]
    Kc, Vc, Xc, L = chunked(k, C), chunked(v, C), chunked(X, C), chunk_logs(loggam, C)
    w = (L[..., -1:] - L).exp() * chunked(beta, C)
    upd = torch.einsum("nci,nciv,ncid->ncvd", w, Vc, Kc)
    aC = L[..., -1].exp()
    G = torch.zeros(N, dv, d, dtype=k.dtype, device=k.device)
    Gs = []
    for c in range(T // C):
        Gs.append(G)
        G = aC[:, c, None, None] * G + upd[:, c]
    G0 = torch.stack(Gs, 1)
    W = decay_mask(L, chunked(beta, C))
    out = L.exp()[..., None] * torch.einsum("ncvd,nctd->nctv", G0, Xc)
    out = out + torch.einsum("ncti,ncti,nciv->nctv", W, torch.einsum("nctd,ncid->ncti", Xc, Kc), Vc)
    return out.reshape(N, T, dv)


# --------------------------------------------------------------------------- exact solver
def batched_eigh(M, backend="default"):
    """torch.linalg.eigh with a chosen backend: "default" (PyTorch's choice, cuSOLVER on CUDA), "magma", or "cpu"
    (eigh on the host, copied back), or "jacobi" (jacobi_eigh.py: batched Triton Jacobi with a convergence test, at most
    20 sweeps; checked in logs/check_jacobi_v2.log). Only eigh uses the chosen library;
    the previous preference is restored."""
    if backend == "jacobi":
        from jacobi_eigh import jacobi_eigh
        return jacobi_eigh(M)
    if backend == "cpu":
        h, V = torch.linalg.eigh(M.cpu())
        return h.to(M.device), V.to(M.device)
    prev = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library(backend)
    try:
        return torch.linalg.eigh(M)
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)


def exact_eig(H0, lam, backend="default"):
    """Eigendecomposition of the whitened chunk-start states S H_0 S, S = Lambda^-1/2."""
    s = lam.rsqrt()
    return batched_eigh(s[:, None] * H0 * s[None, :], backend)


def exact_solve(R, k, beta, loggam, lam, C, h, V):
    """Exact X[t] = (H_t + Lambda)^{-1} R[t], start anchor, capacitances padded to C (README, Proposition 2)."""
    N, T, d = R.shape
    s = lam.rsqrt()
    L = chunk_logs(loggam, C)
    Kh = (chunked(k, C) * s) @ V                                         # keys in the eigenbasis, (N, nc, C, d)
    Rh = (chunked(R, C) * s) @ V
    Dinv = 1.0 / (1.0 + L.exp()[..., None] * h[..., None, :])           # (N, nc, C_t, d)
    sw = decay_mask(L, chunked(beta, C)).sqrt()                          # sqrt(a_t/a_i beta_i), zero for i > t
    # A_t = I + U_t D_t^-1 U_t^T with U_t[i] = sw[t, i] khat_i; rows i > t are zero, which pads A_t with I
    Gram = torch.einsum("ncil,nctl,ncjl->nctij", Kh, Dinv, Kh)
    A = sw[..., :, None] * Gram * sw[..., None, :]
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)
    y0 = Dinv * Rh
    rhs = sw * torch.einsum("ncil,nctl->ncti", Kh, y0)
    z = torch.cholesky_solve(rhs[..., None], torch.linalg.cholesky(A))[..., 0]
    y = y0 - Dinv * torch.einsum("ncti,ncil->nctl", z * sw, Kh)
    return ((y @ V.transpose(-1, -2)) * s).reshape(N, T, d)


# --------------------------------------------------------------------------- CG baseline
def make_matvec(H0, k, beta, loggam, lam, C):
    L = chunk_logs(loggam, C)
    a = L.exp()[..., None]
    Kc = chunked(k, C)
    W = decay_mask(L, chunked(beta, C))

    def mv(P):                                                           # P: (N, nc, C, d)
        out = lam * P + a * torch.einsum("ncde,ncte->nctd", H0, P)
        return out + torch.einsum("ncti,ncid->nctd", W * torch.einsum("nctd,ncid->ncti", P, Kc), Kc)

    diag = lam + a * H0.diagonal(dim1=-2, dim2=-1)[..., None, :] + torch.einsum("ncti,ncid->nctd", W, Kc * Kc)
    return mv, diag


def cg_solve(R, H0, k, beta, loggam, lam, C, steps, jacobi=False, tol=0.0):
    """Fixed-step CG for every token at once. jacobi=False: x0 = 0, no preconditioner (fla's kernel).
    jacobi=True: x0 = diag(M)^-1 r (MesaNet's start) and Jacobi preconditioning. A token freezes on breakdown, or
    once its (preconditioned) residual norm falls to tol times its initial one, so float32 round-off after
    convergence cannot blow it up. Every step is still computed, so the time does not depend on tol."""
    N, T, d = R.shape
    mv, dg = make_matvec(H0, k, beta, loggam, lam, C)
    Rc = chunked(R, C)
    if jacobi:
        x = Rc / dg
        r = Rc - mv(x)
        z = r / dg
    else:
        x = torch.zeros_like(Rc)
        r = Rc.clone()
        z = r
    p = z.clone()
    rz = (r * z).sum(-1, keepdim=True)
    stop = tol * tol * rz
    zero = torch.zeros((), dtype=R.dtype, device=R.device)
    for _ in range(steps):
        Mp = mv(p)
        pMp = (p * Mp).sum(-1, keepdim=True)
        ok = (rz > stop) & (pMp > 0)
        alpha = torch.where(ok, rz / torch.where(ok, pMp, 1.0), zero)
        x = x + alpha * p
        r = r - alpha * Mp
        z = r / dg if jacobi else r
        rz_new = (r * z).sum(-1, keepdim=True)
        p = z + torch.where(ok, rz_new / torch.where(ok, rz, 1.0), zero) * p
        rz = torch.where(ok, rz_new, zero)
    return x.reshape(N, T, d)


# --------------------------------------------------------------------------- inputs and reference
def make_batch(seed, N, T, d, dv, gate_lo, gate_hi, lam_lo, lam_hi, device):
    """N independent heads drawn with gated_mesa.make_inputs (flat key spectrum); lam is shared by the batch."""
    rng = np.random.default_rng(seed)
    heads = [gm.make_inputs(rng, T, d, dv, gate_lo, gate_hi, lam_lo, lam_hi) for _ in range(N)]
    q, k, v, beta, loggam = (torch.tensor(np.stack([h[j] for h in heads]), device=device) for j in range(5))
    lam = torch.tensor(heads[0][5], device=device)
    return q, k, v, beta, loggam, lam


def dense_reference(q, k, beta, loggam, lam, C, block=4):
    """Float64 dense per-token solves of (H_t + Lambda) x = q_t, and cond(H_t + Lambda) max over every 8th token."""
    N, T, d = q.shape
    X = torch.empty_like(q)
    cond = 0.0
    H0 = start_states(k, beta, loggam, C)
    L = chunk_logs(loggam, C)
    W = decay_mask(L, chunked(beta, C))
    Kc = chunked(k, C)
    for n0 in range(0, N, block):
        sl = slice(n0, n0 + block)
        M = L[sl].exp()[..., None, None] * H0[sl, :, None] + torch.einsum("ncti,ncid,ncie->nctde", W[sl], Kc[sl], Kc[sl])
        M = M + torch.diag(lam)
        X[sl] = torch.linalg.solve(M, chunked(q[sl], C)[..., None])[..., 0].reshape(-1, T, d)
        ev = torch.linalg.eigvalsh(M[:, :, ::8])                     # every 8th token of each chunk, for speed
        cond = max(cond, (ev[..., -1] / ev[..., 0]).max().item())
    return X, cond


def out_err(o, o_ref):
    """Max-abs error relative to the max-abs reference value (gated_mesa.rel)."""
    return ((o.double() - o_ref).abs().max() / o_ref.abs().max()).item()


# --------------------------------------------------------------------------- parts
def part_a(device):
    print("Part A: float64 port of the start-anchor solver against gated_mesa.chunk_solve (NumPy) and a dense solve")
    rows = []
    for (T, d, dv, C, glo, ghi, llo, lhi) in [(128, 16, 8, 32, 0.9, 1.0, 0.05, 1.0), (256, 64, 16, 64, 0.5, 1.0, 0.05, 1.0),
                                             (128, 32, 8, 16, 0.99, 1.0, 1e-3, 1e-2), (128, 64, 8, 64, 0.01, 0.2, 0.05, 1.0)]:
        q, k, v, beta, loggam, lam = make_batch(1, 2, T, d, dv, glo, ghi, llo, lhi, device)
        H0 = start_states(k, beta, loggam, C)
        h, V = exact_eig(H0, lam)
        X = exact_solve(q, k, beta, loggam, lam, C, h, V)
        Xd, cond = dense_reference(q, k, beta, loggam, lam, C)
        args = [a[0].cpu().numpy() for a in (k, beta, loggam)]
        Hb, _ = gm.boundary_states(args[0], v[0].cpu().numpy(), *args[1:], C)
        Xn, _ = gm.chunk_solve(q[0].cpu().numpy(), *args, lam.cpu().numpy(), C, Hb, policy="start")
        e_np = gm.rel(X[0].cpu().numpy(), Xn)
        e_dense = (X - Xd).abs().max().item() / Xd.abs().max().item()
        e_out = out_err(readout(X, k, v, beta, loggam, C), readout(Xd, k, v, beta, loggam, C))
        rows.append(dict(T=T, d=d, dv=dv, C=C, gates=[glo, ghi], ridge=[llo, lhi], cond=cond,
                         x_err_vs_numpy=e_np, x_err_vs_dense=e_dense, out_err_vs_dense=e_out))
        print(f"  T={T:4d} d={d:3d} C={C:3d} gates {glo}-{ghi} ridge {llo:g}-{lhi:g}  cond {cond:8.1f}  "
              f"x vs numpy {e_np:.1e}  x vs dense {e_dense:.1e}  out vs dense {e_out:.1e}")
    return rows


def solvers(args):
    """name -> fn(R, H0, k, beta, loggam, lam, C): every method gets the same inputs and the same H_0."""
    out = {}
    for be in args.eigh:
        out[f"exact_{be}"] = lambda R, H, k, b, g, lam, C, be=be: exact_solve(R, k, b, g, lam, C, *exact_eig(H, lam, be))
    for st in args.cg_steps:
        out[f"cg{st}"] = lambda R, H, k, b, g, lam, C, st=st: cg_solve(R, H, k, b, g, lam, C, st, tol=args.cg_tol)
        out[f"jac{st}"] = lambda R, H, k, b, g, lam, C, st=st: cg_solve(R, H, k, b, g, lam, C, st, jacobi=True, tol=args.cg_tol)
    return out


def part_b(args, device):
    print(f"\nPart B: float32 forward solve on {torch.cuda.get_device_name() if device == 'cuda' else device}, TF32 off; "
          f"N={args.N} heads, T={args.T}, d_v={args.dv}; eigh backends {args.eigh} (CPU threads {torch.get_num_threads()}); "
          f"CG residual stop {args.cg_tol:g}. Times are medians in ms (warmup 3, reps {args.reps}), "
          "including the chunk-start states. Output error: max over seeds, float64 readout, against a dense float64 solve.")
    rows = []
    gates = [(0.99, 1.0), (0.9, 1.0), (0.5, 1.0)]
    ridges = [(0.05, 1.0), (1e-3, 1e-2)]
    methods = solvers(args)
    for d in args.d:
        for C in args.C:
            if args.T % C:
                continue
            for (glo, ghi) in gates:
                for (llo, lhi) in ridges:
                    errs, conds = {}, []
                    for seed in range(args.seeds):
                        q, k, v, beta, loggam, lam = make_batch(100 + seed, args.N, args.T, d, args.dv, glo, ghi, llo, lhi, device)
                        Xd, cond = dense_reference(q, k, beta, loggam, lam, C)
                        conds.append(cond)
                        o_ref = readout(Xd, k, v, beta, loggam, C)
                        f32 = [a.float() for a in (q, k, beta, loggam, lam)]
                        H0 = start_states(f32[1], f32[2], f32[3], C)
                        for name, fn in methods.items():
                            X = fn(f32[0], H0, *f32[1:], C)
                            e = out_err(readout(X.double(), k, v, beta, loggam, C), o_ref) if torch.isfinite(X).all() else float("inf")
                            errs[name] = max(errs.get(name, 0.0), e)
                        del q, k, v, beta, loggam, lam, Xd, o_ref
                    # timing on the last seed's float32 inputs; every method pays for H_0
                    q32, k32, beta32, lg32, lam32 = f32
                    times, mems = {}, {}
                    for name, fn in methods.items():
                        def full(fn=fn):
                            return fn(q32, start_states(k32, beta32, lg32, C), k32, beta32, lg32, lam32, C)
                        reset_peak_memory(device)
                        times[name] = bench(full, device, warmup=3, reps=args.reps)
                        mems[name] = peak_memory_gb(device)
                    h, V = exact_eig(H0, lam32)
                    stage = {"states": bench(lambda: start_states(k32, beta32, lg32, C), device, reps=args.reps),
                             "capacitance": bench(lambda: exact_solve(q32, k32, beta32, lg32, lam32, C, h, V), device, reps=args.reps)}
                    for be in args.eigh:
                        stage[f"eigh_{be}"] = bench(lambda be=be: exact_eig(H0, lam32, be), device, reps=args.reps)
                    row = dict(d=d, C=C, gates=[glo, ghi], ridge=[llo, lhi], cond_max=max(conds), err=errs, ms=times,
                               peak_gb=mems, exact_stage_ms=stage)
                    rows.append(row)
                    print(f"\n  d={d} C={C} gates {glo}-{ghi} ridge {llo:g}-{lhi:g}  cond max {max(conds):.2e}")
                    print("    exact stages (ms): " + "  ".join(f"{kk} {vv:.2f}" for kk, vv in stage.items()))
                    for name in methods:
                        print(f"    {name:14s} {times[name]:8.2f} ms   err {errs[name]:.2e}   peak {mems[name]:.2f} GB")
                    del q32, k32, beta32, lg32, lam32, f32, H0, h, V
                    torch.cuda.empty_cache()
    return rows


def part_c(args, device):
    """Batched eigh time against matrix size and batch, per backend (random SPD float32 matrices)."""
    print("\nPart C: torch.linalg.eigh, float32 SPD matrices, median ms (warmup 1, reps 3)")
    rows = []
    gen = torch.Generator(device=device).manual_seed(0)
    for n in (32, 64, 128):
        for B in (16, 256, 1024):
            A = torch.randn(B, n, n, device=device, generator=gen)
            A = A @ A.mT / n + torch.eye(n, device=device)
            ms = {be: bench(lambda be=be: batched_eigh(A, be), device, warmup=1, reps=3) for be in args.eigh}
            rows.append(dict(n=n, batch=B, ms=ms))
            print(f"  n={n:3d} batch={B:5d}  " + "  ".join(f"{be} {t:8.2f} ({t / B:.3f}/matrix)" for be, t in ms.items()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test at tiny sizes")
    p.add_argument("--N", type=int, default=16, help="independent heads (batch x heads)")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--dv", type=int, default=64)
    p.add_argument("--d", type=int, nargs="+", default=[64, 128])
    p.add_argument("--C", type=int, nargs="+", default=[32, 64])
    p.add_argument("--cg-steps", type=int, nargs="+", default=[10, 30, 100, 200])
    p.add_argument("--cg-tol", type=float, default=1e-6, help="freeze a CG token at this relative residual")
    p.add_argument("--eigh", nargs="+", default=["default", "magma", "cpu", "jacobi"], help="eigh backends for the exact solver")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.quick:
        args.N, args.T, args.d, args.C, args.seeds, args.reps = 2, 256, [64], [32], 1, 3
    seed_everything(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows_a = part_a(args.device)
    rows_b = part_b(args, args.device)
    rows_c = part_c(args, args.device)
    name = "gpu_bench_quick" if args.quick else "gpu_bench"
    path = save_results({"args": vars(args), "cpu_threads": torch.get_num_threads(), "part_a": rows_a, "part_b": rows_b,
                         "part_c": rows_c}, HERE / "results", name)
    print(f"\nsaved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
