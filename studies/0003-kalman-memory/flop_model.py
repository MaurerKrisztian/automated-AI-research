"""FLOP model for study 0003 (E1): the chunkwise Kalman memory against MesaNet's CG solve and two references.

Counts one multiply-add as 2 FLOPs, per token and head, forward pass only. Products over a causal C x C mask are
counted dense (2C^2 per vector), which overstates the Kalman layer slightly. Formulas, with d the key dimension,
e the value dimension and C the chunk size (README, Method, "Cost"):

  kalman_chunk   per chunk: K^ P0, Q^ P0 and the covariance downdate 6Cd^2; the four C x C Gram matrices and
                 L^-1 Z^T 9C^2 d; residuals, readout from M0 and the mean update 6Cde; L^-1 R and the masked
                 output product 3C^2 e; Cholesky C^3/3 and L^-1 X C^3.
  kalman_decode  per token: predict P (d^2), P k (2d^2), rank-one downdate (2d^2), mean predict, residual,
                 update and readout (7de), scalars (4d).
  cg             MesaNet with a fixed number of CG steps, same formula as
                 studies/0002-gated-mesa-exact/gated_mesa.py:flops_cg (state update and readout included).
  cg_decode      the same at decoding: no intra-chunk term.
  carried_inv    study 0001's ungated carried-inverse layer, same formula as gated_mesa.py:flops_prefix_cholesky.
  gla            gated linear attention, no solve: state update 2de, readout 2de, intra-chunk C(d + e).
  bl_chunk       the Bayesian Layer's published chunk algorithm (Dowling et al. 2026, arXiv 2605.31163,
                 Appendix C.2-C.3), as read in this study: entry-covariance products 2Cd^2, the sequential gain loop
                 over the chunk's positions (history dot products and sums, 4td at step t, so 2C^2 d), the
                 sequential mean pass (predict, residual, rank-one update, readout: 7de per token) and the exit
                 covariance 2Cd^2. Same model as kalman_chunk; it counts fewer FLOPs because it never forms the
                 full C x C matrices.

The second table splits off the work done inside order-dependent steps over a chunk's positions (sequential work):
the Bayesian Layer's gain loop and mean pass (2C^2 d + 7Cde per chunk), against the Kalman chunk form's C x C
Cholesky and triangular solves (C^3/3 + C^2(e + C + d) per chunk). Blocked Cholesky and triangular solves are mostly
matrix products, so the second count overstates what stays sequential in practice. (Version 2; kept unchanged.)

The third table (version 3) separates the two kinds of order-dependent work, per token:
  kalman_inv     the Kalman chunk form with L^-1 formed explicitly (C^3/3 by forward substitution), so that L^-1 R,
                 L^-1 X and L^-1 Z^T become matrix products, as DeltaNet's UT transform does with its C x C inverse.
                 Total per chunk 6Cd^2 + 10C^2 d + 6Cde + 4C^2 e + (8/3)C^3. Order-dependent work: only the C x C
                 Cholesky and inverse, (2/3)C^3 per chunk, none of it scaling with d or e.
  bl (v5)        order-dependent work that scales with d or e: gain loop 2C^2 d and mean pass 7Cde per chunk.
  bl_wy          the strongest combination published: v5's compact gain loop (2C^2 d, scales with d) with v1's
                 WY-factored parallel mean pass (BL v1, Appendix D), whose order-dependent part is its C x C
                 triangular inverse (C^3/3). Its total FLOPs are not modeled.

Run: ../../.venv/bin/python -u flop_model.py 2>&1 | tee logs/flop_model_v3.log   (standard library only)
"""
import json
import time
from pathlib import Path


def kalman_chunk(d, e, C):
    per_chunk = 6 * C * d * d + 9 * C * C * d + 6 * C * d * e + 3 * C * C * e + C ** 3 / 3 + C ** 3
    return per_chunk / C


def kalman_decode(d, e):
    return 5 * d * d + 7 * d * e + 4 * d


def cg(d, e, C, steps):
    return steps * (2 * d * d + 2 * C * d + 11 * d) + (2 * d * d + 2 * d * e) + (2 * d * e + C * d + C * e)


def cg_decode(d, e, steps):
    return steps * (2 * d * d + 11 * d) + (2 * d * d + 2 * d * e) + 2 * d * e


def carried_inv(d, e, C):
    return (5 * C * d * d + 4 * C * C * d + C ** 3 / 3) / C + 4 * d * e + C * d + C * e


def gla(d, e, C):
    return 4 * d * e + C * (d + e)


def bl_chunk(d, e, C):
    return 4 * d * d + 2 * C * d + 7 * d * e


def seq_bl(d, e, C):
    return (2 * C * C * d + 7 * C * d * e) / C


def seq_kalman(d, e, C):
    return (C ** 3 / 3 + C * C * (e + C + d)) / C


def kalman_inv(d, e, C):
    return (6 * C * d * d + 10 * C * C * d + 6 * C * d * e + 4 * C * C * e + 8 * C ** 3 / 3) / C


def seq_split(d, e, C):
    """Order-dependent work per token: (part scaling with d or e, part on C x C matrices only)."""
    return {"kalman_inv": (0.0, 2 * C * C / 3), "bl": (2 * C * d + 7 * d * e, 0.0), "bl_wy": (2 * C * d, C * C / 3)}


def main():
    rows = []
    print("Forward FLOPs per token and head (e = d). Ratios are CG or reference over the Kalman layer.")
    print(f"{'d':>4} {'C':>4} {'kalman':>9} {'cg30':>10} {'cg10':>9} {'carried':>9} {'gla':>8}"
          f" {'cg30/k':>7} {'cg10/k':>7} {'k/carried':>9} {'k/gla':>6}")
    for d in (64, 128, 256):
        for C in (32, 64, 128):
            e = d
            r = dict(d=d, e=e, C=C, kalman=kalman_chunk(d, e, C), cg30=cg(d, e, C, 30), cg10=cg(d, e, C, 10),
                     carried=carried_inv(d, e, C), gla=gla(d, e, C))
            rows.append(r)
            print(f"{d:>4} {C:>4} {r['kalman']:>9.0f} {r['cg30']:>10.0f} {r['cg10']:>9.0f} {r['carried']:>9.0f}"
                  f" {r['gla']:>8.0f} {r['cg30'] / r['kalman']:>7.2f} {r['cg10'] / r['kalman']:>7.2f}"
                  f" {r['kalman'] / r['carried']:>9.2f} {r['kalman'] / r['gla']:>6.2f}")
    print()
    print("Kalman chunk form against the Bayesian Layer's sequential chunk algorithm, per token and head (e = d):")
    print("total forward FLOPs, and the part done inside order-dependent steps over the chunk's positions.")
    print(f"{'d':>4} {'C':>4} {'kalman':>9} {'bl':>9} {'k/bl':>6} {'seq_kalman':>11} {'seq_bl':>9} {'bl/k seq':>9}")
    for d in (64, 128, 256):
        for C in (32, 64, 128):
            e = d
            r = next(x for x in rows if x['d'] == d and x['C'] == C)
            r.update(bl=bl_chunk(d, e, C), seq_kalman=seq_kalman(d, e, C), seq_bl=seq_bl(d, e, C))
            print(f"{d:>4} {C:>4} {r['kalman']:>9.0f} {r['bl']:>9.0f} {r['kalman'] / r['bl']:>6.2f}"
                  f" {r['seq_kalman']:>11.0f} {r['seq_bl']:>9.0f} {r['seq_bl'] / r['seq_kalman']:>9.2f}")
    print()
    print("Version 3. Order-dependent work per token and head (e = d), split into the part that scales with d or e")
    print("and the part on C x C matrices only. kalman_inv forms L^-1 explicitly; bl_wy is BL's gain loop with a WY mean.")
    print(f"{'d':>4} {'C':>4} {'kalman_inv':>11} {'inv/bl':>7} {'k_dep':>6} {'k_CxC':>7} {'bl_dep':>8}"
          f" {'bl_wy_dep':>10} {'bl_wy_CxC':>10} {'bl_wy/k':>8}")
    for d in (64, 128, 256):
        for C in (32, 64, 128):
            e = d
            r = next(x for x in rows if x['d'] == d and x['C'] == C)
            s = seq_split(d, e, C)
            r.update(kalman_inv=kalman_inv(d, e, C), seq_split={k: list(v) for k, v in s.items()})
            k_all, wy_all = sum(s["kalman_inv"]), sum(s["bl_wy"])
            print(f"{d:>4} {C:>4} {r['kalman_inv']:>11.0f} {r['kalman_inv'] / r['bl']:>7.2f} {s['kalman_inv'][0]:>6.0f}"
                  f" {s['kalman_inv'][1]:>7.0f} {s['bl'][0]:>8.0f} {s['bl_wy'][0]:>10.0f} {s['bl_wy'][1]:>10.0f}"
                  f" {wy_all / k_all:>8.2f}")
    print()
    print("Decoding FLOPs per token and head (e = d): exact Kalman recursion against MesaNet with 9 or 30 CG steps.")
    print(f"{'d':>4} {'kalman':>8} {'cg9':>8} {'cg30':>9} {'cg9/k':>6} {'cg30/k':>7}")
    dec = []
    for d in (64, 128, 256):
        r = dict(d=d, kalman=kalman_decode(d, d), cg9=cg_decode(d, d, 9), cg30=cg_decode(d, d, 30))
        dec.append(r)
        print(f"{d:>4} {r['kalman']:>8.0f} {r['cg9']:>8.0f} {r['cg30']:>9.0f} {r['cg9'] / r['kalman']:>6.2f}"
              f" {r['cg30'] / r['kalman']:>7.2f}")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"flop_model_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "forward": rows, "decode": dec}, f, indent=1)
    print(f"\nsaved {path.relative_to(out.parent)}")


if __name__ == "__main__":
    main()
