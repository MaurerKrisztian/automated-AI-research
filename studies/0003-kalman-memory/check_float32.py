"""E2 check of study 0003 in float32 on the CPU: error against float64 over long sequences.

Part E1: the chunk form (triangular solves, explicit L^-1, GLA-style "matmul" Grams, a mixed form whose small
  C x C stage alone runs in float64, and one that also forms K in float64) and the token-by-token Kalman filter, all in
  float32, against the token-by-token filter in float64 (which part A of check_chunk.py
  matches to extended precision at short lengths). Errors are reported per eighth of the sequence, to show growth,
  together with the smallest eigenvalue ratio of the covariance over every chunk boundary (positive definiteness)
  and the final covariance's asymmetry. Version 2 adds the regimes without forgetting, where nothing refills the prior.
Part E2: Proposition 4(a)'s claim that the algebraically equal "P_0 - Lambda^-1" form of the chunk covariance cancels
  when the gates are near 1 and the boundary covariance is small, while the sum of positive semidefinite terms used
  here does not. Both forms in float32 against float64.

Run from this folder:  python3 -u check_float32.py 2>&1 | tee /tmp/check_float32.log
(The recorded run is logs/check_float32_v2.log; never tee into logs/. Each run adds a JSON to results/; delete such
copies, never the recorded one.)  --quick shortens every sequence to 2048 tokens as a smoke test.
"""
import argparse
import math
import time

import numpy as np
import torch

from kalman_memory import chunk_matrices, chunk_step, kalman_reference, max_cond, stationary_noise

CONFIGS = [  # name, T, gates, per_channel, beta, ridge, keys
    ("scalar gates 0.99-1", 65536, (0.99, 1.0), False, (0.5, 2.0), 1.0, "gauss"),
    ("scalar gates 0.9-1", 65536, (0.9, 1.0), False, (0.5, 2.0), 1.0, "gauss"),
    ("per-channel gates 0.95-1", 16384, (0.95, 1.0), True, (0.5, 2.0), 1.0, "gauss"),
    ("strong writes: beta 50, ridge 1e-2, gates 0.999-1", 16384, (0.999, 1.0), False, (50.0, 50.0), 1e-2, "gauss"),
    ("decaying key spectrum, beta 10, ridge 1e-2", 16384, (0.99, 1.0), False, (10.0, 10.0), 1e-2, "decay"),
    ("per-channel gates 0.1-0.3 (fast; log decay over a chunk about -105)", 4096, (0.1, 0.3), True, (0.5, 2.0), 1.0,
     "gauss"),
    # version 2: the regimes Proposition 4(c) flags, where nothing refills the prior (added after the E2 review)
    ("no forgetting (gates = 1), beta 0.5-2, ridge 1", 65536, (1.0, 1.0), False, (0.5, 2.0), 1.0, "gauss"),
    ("no forgetting, strong writes: beta 50, ridge 1e-2", 65536, (1.0, 1.0), False, (50.0, 50.0), 1e-2, "gauss"),
    ("no forgetting, decaying key spectrum, beta 10, ridge 1e-2", 65536, (1.0, 1.0), False, (10.0, 10.0), 1e-2, "decay"),
    ("gates 0.9999-1, strong writes: beta 50, ridge 1e-2", 65536, (0.9999, 1.0), False, (50.0, 50.0), 1e-2, "gauss"),
]
d = e = C = 64


def make(seed, T, gates, per_channel, beta, lam, keys, dtype):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((T, d)) / math.sqrt(d)
    if keys == "decay":
        K *= (np.arange(1, d + 1) ** -1.0) * math.sqrt(d) / 2
    V = rng.standard_normal((T, e))
    Q = rng.standard_normal((T, d)) / math.sqrt(d)
    g = rng.uniform(*gates, (T, d) if per_channel else (T, 1)) * np.ones((1, d))
    b = np.exp(rng.uniform(math.log(beta[0]), math.log(beta[1]), T))
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64)).to(dtype)
    logD, lamv = t(np.log(g)), t(np.full(d, lam))
    return (t(K), t(V), t(Q), logD, stationary_noise(logD, lamv), t(b), torch.zeros(e, d, dtype=dtype),
            torch.diag(1 / lamv))


def seg_errors(o, ref, parts=8):
    """Normwise relative error in each of `parts` equal segments of the sequence."""
    o, ref = o.double(), ref.double()
    T = ref.shape[0]
    cuts = [round(i * T / parts) for i in range(parts + 1)]
    return [float((o[a:b] - ref[a:b]).abs().max() / ref[a:b].abs().max()) for a, b in zip(cuts[:-1], cuts[1:])]


def eig_ratio(P):
    """Smallest over largest eigenvalue of the symmetric part of P (float64), or nan if P is not finite."""
    P = P.double()
    if not bool(torch.isfinite(P).all()):
        return float("nan")
    ev = torch.linalg.eigvalsh(0.5 * (P + P.T))
    return float(ev[0] / ev[-1])


def run_chunks(a, step):
    """Run `step` chunk by chunk, carrying (M, P); returns outputs, final P, and the smallest eigenvalue ratio of P
    over all chunk boundaries."""
    K, V, Q, logD, Psi, beta, M, P = a
    outs, worst = [], float("inf")
    for s in range(0, K.shape[0], C):
        sl = slice(s, s + C)
        o, M, P = step(K[sl], V[sl], Q[sl], logD[sl], Psi[sl], beta[sl], M, P)
        outs.append(o)
        worst = min(worst, eig_ratio(P))
    return torch.cat(outs), P, worst


def part_e1(quick):
    print("Part E1. float32 against float64 over long sequences (d = e = C = 64, stationary noise).")
    print("Per eighth of the sequence: max|o32 - o64| / max|o64|. P: smallest/largest eigenvalue at the worst chunk"
          " boundary; final asymmetry.")
    rows = []
    for i, (name, T, gates, pc, beta, lam, keys) in enumerate(CONFIGS):
        T = min(T, 2048) if quick else T
        t0 = time.time()
        a64 = make(1000 + i, T, gates, pc, beta, lam, keys, torch.float64)
        a32 = tuple(x.to(torch.float32) for x in a64)
        o64, _, P64 = kalman_reference(*a64)
        cond = max_cond(*a64, C)
        res = dict(config=name, T=T, cond_max=cond)
        print(f"{name}  (T = {T}; largest cond(K) over the chunks, in float64: {cond:.2e})")
        for label, step in (("chunk, triangular solves", chunk_step),
                            ("chunk, explicit L^-1", lambda *x: chunk_step(*x, inverse="explicit")),
                            ("chunk, matmul Grams", lambda *x: chunk_step(*x, gram="matmul")),
                            ("chunk, C x C stage in float64", lambda *x: chunk_step(*x, cc_dtype=torch.float64)),
                            ("chunk, K formed in float64", lambda *x: chunk_step(*x, kmat_dtype=torch.float64)),
                            ("token-by-token filter", kalman_reference)):
            try:
                o, P, lmin = run_chunks(a32, step)
                finite = bool(torch.isfinite(o).all())
                errs = seg_errors(o, o64) if finite else []
                Pd = P.double()
                asym = float((Pd - Pd.T).abs().max() / Pd.abs().max()) if finite else float("nan")
                Perr = float((Pd - P64).abs().max() / P64.abs().max()) if finite else float("nan")
                status = "ok" if finite else "non-finite output"
            except RuntimeError as exc:                                 # e.g. a Cholesky that fails
                errs, lmin, asym, Perr, status = [], float("nan"), float("nan"), float("nan"), f"error: {exc}"[:60]
            res[label] = dict(status=status, seg=errs, P_err=Perr, P_eig_ratio_min_over_boundaries=lmin, P_asym=asym)
            seg = " ".join(f"{x:.1e}" for x in errs) if errs else status
            print(f"  {label:>29}: {seg}   P err {Perr:.1e}, eig min/max (worst boundary) {lmin:.1e}, asym {asym:.1e}")
        res["reference_P_eig_ratio_final"] = eig_ratio(P64)
        print(f"  {'float64 reference':>29}: final P eig min/max {res['reference_P_eig_ratio_final']:.1e}")
        print(f"  ({time.time() - t0:.1f} s)")
        rows.append(res)
    print()
    return rows


def delta_form(K, V, Q, logD, Psi, beta, M0, P0, lam):
    """The chunk covariance written with P_0 - Lambda^-1 and the prior Gram F (algebraically equal, stationary)."""
    Cn = K.shape[0]
    logA = torch.cumsum(logD, 0)
    A = torch.exp(logA)
    idx = torch.arange(Cn)
    lo, hi = torch.minimum(idx[:, None], idx[None, :]), torch.maximum(idx[:, None], idx[None, :])
    F = torch.einsum('ic,jc,ijc->ij', K, K, torch.exp(logA[hi] - logA[lo]) / lam)
    KA = K * A
    return KA @ (P0 - torch.diag(1 / lam)) @ KA.T + F + torch.diag(1 / beta)


def part_e2(quick):
    print("Part E2. Chunk covariance in float32: the sum of PSD terms against the 'P_0 - Lambda^-1' form.")
    print("Boundary covariance from a float64 history; error is max|K32 - K64| / lambda_min(K64).")
    rows = []
    for gates in ((0.5, 1.0), (0.99, 1.0), (0.9999, 1.0)):
        for beta in (1.0, 50.0):
            H = 500 if quick else 2000
            a = make(2000, H + C, gates, False, (beta, beta), 1.0, "gauss", torch.float64)
            _, M0, P0 = kalman_reference(*(x[:H] for x in a[:6]), a[6], a[7])
            tail = [x[H:] for x in a[:6]]
            lam = torch.ones(d, dtype=torch.float64)
            K64 = chunk_matrices(*tail, M0, P0)["Kmat"]
            lmin = float(torch.linalg.eigvalsh(K64)[0])
            t32 = [x.float() for x in tail]
            K_psd = chunk_matrices(*t32, M0.float(), P0.float())["Kmat"].double()
            K_dlt = delta_form(*t32, M0.float(), P0.float(), lam.float()).double()
            r = dict(gates=list(gates), beta=beta, lambda_min=lmin,
                     psd=float((K_psd - K64).abs().max()) / lmin, delta=float((K_dlt - K64).abs().max()) / lmin,
                     P0_eig_max=float(torch.linalg.eigvalsh(P0)[-1]))
            rows.append(r)
            print(f"gates {gates[0]}-{gates[1]}, beta {beta:>4}: lambda_min(K) {lmin:.2e}, largest eig of P_0"
                  f" {r['P0_eig_max']:.2e}; error, PSD sum {r['psd']:.1e}, P_0 - Lambda^-1 form {r['delta']:.1e}")
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    results = dict(part_e1=part_e1(args.quick), part_e2=part_e2(args.quick))
    if not args.quick:
        from lab import save_results
        print(f"saved {save_results(results, 'results', 'check_float32')}")


if __name__ == "__main__":
    main()
