#!/usr/bin/env python3
"""Checks jacobi_eigh.py against torch.linalg.eigh in float64, and measures float32 accuracy, sweeps used and speed.

Inputs: random symmetric matrices (Gaussian, and SPD with condition ~1e3), and the whitened chunk-start states
S H_0 S that the exact solver diagonalizes (gpu_bench.make_batch, gates 0.9-1 and 0.5-1, ridge 0.05-1 and 1e-3-1e-2).
Part A caps the sweeps at each value of --sweeps; the last value is the default cap of jacobi_eigh (20). Version 2
(threshold, early exit, zeroed pivots) writes logs/check_jacobi_v2.log; version 1 (fixed sweeps) wrote check_jacobi.log.
Metrics, worst over the batch: reconstruction ||V diag(h) V^T - M|| / ||M||, orthogonality ||V^T V - I||, and the
sorted eigenvalue error max|h - h_ref| / max|h_ref| against float64 torch.linalg.eigh.
Usage:
    ../../.venv/bin/python -u check_jacobi.py 2>&1 | tee logs/check_jacobi_v2.log
"""
import argparse
from pathlib import Path

import torch

from gpu_bench import batched_eigh, make_batch, start_states
from jacobi_eigh import jacobi_eigh
from lab import bench, save_results, seed_everything

HERE = Path(__file__).resolve().parent


def metrics(M, h, V):
    M64, h64, V64 = M.double(), h.double(), V.double()
    n = M.shape[-1]
    rec = (V64 * h64[..., None, :]) @ V64.mT - M64
    rec = (rec.norm(dim=(-2, -1)) / M64.norm(dim=(-2, -1))).max().item()
    orth = (V64.mT @ V64 - torch.eye(n, dtype=torch.float64, device=M.device)).norm(dim=(-2, -1)).max().item()
    ref = torch.linalg.eigvalsh(M64)
    ev = ((h64.sort(-1).values - ref).abs().amax(-1) / ref.abs().amax(-1)).max().item()
    return rec, orth, ev


def inputs(n, B, device, gen):
    G = torch.randn(B, n, n, dtype=torch.float64, device=device, generator=gen)
    out = {"gaussian symmetric": (G + G.mT) / 2}
    Qm, _ = torch.linalg.qr(torch.randn(B, n, n, dtype=torch.float64, device=device, generator=gen))
    ev = torch.logspace(0, 3, n, dtype=torch.float64, device=device)
    out["SPD, cond 1e3"] = (Qm * ev) @ Qm.mT
    for glo in (0.9, 0.5):
        for llo, lhi in ((0.05, 1.0), (1e-3, 1e-2)):
            q, k, v, beta, loggam, lam = make_batch(7, 4, 1024, n, 16, glo, 1.0, llo, lhi, device)
            H0 = start_states(k, beta, loggam, 32)[:, 1:]                   # skip the all-zero first state
            s = lam.rsqrt()
            out[f"S H0 S, gates {glo}-1, ridge {llo:g}-{lhi:g}"] = (s[:, None] * H0 * s[None, :]).reshape(-1, n, n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--sweeps", type=int, nargs="+", default=[6, 8, 10, 12, 20])
    p.add_argument("--n", type=int, nargs="+", default=[64, 128])
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.quick:
        args.sweeps, args.n, args.B = [8, 20], [64], 16
    seed_everything(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device=args.device).manual_seed(0)
    rows = []
    print("Part A: accuracy against sweeps (worst over the batch). torch.linalg.eigh rows use the same dtype.")
    for n in args.n:
        for name, M64 in inputs(n, args.B, args.device, gen).items():
            for dt in (torch.float64, torch.float32):
                M = M64.to(dt)
                ref = metrics(M, *torch.linalg.eigh(M))
                print(f"  n={n:3d} {name:38s} {str(dt)[6:]:7s} torch eigh: rec {ref[0]:.1e} orth {ref[1]:.1e} eig {ref[2]:.1e}")
                rows.append(dict(part="A", n=n, input=name, dtype=str(dt), method="torch", rec=ref[0], orth=ref[1], eig=ref[2]))
                for sw in args.sweeps:
                    h, V, used = jacobi_eigh(M, max_sweeps=sw, return_sweeps=True)
                    m = metrics(M, h, V)
                    print(f"  {'':5s} {'':38s} {'':7s} jacobi <={sw:2d}: rec {m[0]:.1e} orth {m[1]:.1e} eig {m[2]:.1e}"
                          f"  sweeps used {used.min().item()}-{used.max().item()}")
                    rows.append(dict(part="A", n=n, input=name, dtype=str(dt), method=f"jacobi{sw}", rec=m[0], orth=m[1], eig=m[2],
                                     sweeps_min=used.min().item(), sweeps_max=used.max().item()))
    print("\nPart B: float32 time for a batch of 1024 (median ms, warmup 2, reps 5)")
    for n in args.n:
        G = torch.randn(1024 if not args.quick else 64, n, n, device=args.device, generator=gen)
        M = G @ G.mT / n + torch.eye(n, device=args.device)
        t = {be: bench(lambda be=be: batched_eigh(M, be), args.device, warmup=2, reps=5) for be in ("default", "cpu")}
        for sw in args.sweeps:
            t[f"jacobi{sw}"] = bench(lambda sw=sw: jacobi_eigh(M, max_sweeps=sw), args.device, warmup=2, reps=5)
        print(f"  n={n:3d}  " + "  ".join(f"{k} {v:.2f}" for k, v in t.items()))
        rows.append(dict(part="B", n=n, batch=M.shape[0], ms=t))
    name = "check_jacobi_quick" if args.quick else "check_jacobi_v2"
    path = save_results({"args": vars(args), "rows": rows}, HERE / "results", name)
    print(f"\nsaved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
