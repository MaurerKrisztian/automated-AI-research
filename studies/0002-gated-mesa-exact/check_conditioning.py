"""Accuracy against conditioning, written after the review (REVIEW.md, issues 3, 4 and 7).

Part A (float64): condition numbers from ~1e1 to ~1e7, filling the gap that check_forward.py left between 8e2
and 4e4. Reported: forward error against the refined reference and normwise backward error, for the chunk solver
(start-anchor and two-anchor policies) and for a dense LU solve per token.
Part B (float32): the same sweep with T = 256, d = 64, C = 64, three seeds, against a dense solve and three CG
variants (zero start; MesaNet's diagonal start; Jacobi PCG), 30 steps each, plus 100 steps from zero.
Key spectra: flat; i^-2 aligned with the axes; i^-2 under a random rotation (no help for a diagonal preconditioner).

Run from this folder:  python3 -u check_conditioning.py 2>&1 | tee /tmp/check_conditioning.log
(The recorded run is logs/check_conditioning.log; never tee into logs/, which is immutable.)
"""
import numpy as np

from gated_mesa import (backward_error, boundary_states, cg_solve, chunk_readout, forward, make_inputs,
                        reference_forward, reference_states, refined_reference, rel, save_results)

RIDGES = ((0.05, 1.0), (1e-2, 1e-1), (1e-3, 1e-2), (1e-4, 1e-3), (1e-5, 1e-4), (1e-6, 1e-5))
KEYS = {"flat": dict(key_alpha=0.0), "i^-2 axis": dict(key_alpha=2.0), "i^-2 rot": dict(key_alpha=2.0, rotate=True)}


def part_a():
    print("== A. float64, T = 200, d = 64, C = 32, gates 0.9-1: forward error vs refined reference, normwise backward error")
    print(f"{'ridge':>12} {'keys':>10} {'seed':>4} {'cond max':>8} {'start x':>8} {'two x':>8} {'dense x':>8}"
          f" {'start bwd':>9} {'two bwd':>9} {'dense bwd':>9}")
    rows = []
    for (llo, lhi) in RIDGES:
        for kname, kw in KEYS.items():
            for seed in (0, 1):
                rng = np.random.default_rng(1000 + seed)
                args = make_inputs(rng, 200, 64, 16, 0.9, 1.0, llo, lhi, **kw)
                q, lam = args[0], args[5]
                _, x_ref, x_dense, cond, Hs = refined_reference(*args)
                r = dict(ridge=f"{llo:g}-{lhi:g}", keys=kname, seed=seed, cond_max=float(cond.max()),
                         err_x_dense=rel(x_dense, x_ref), bwd_dense=backward_error(Hs, lam, x_dense, q))
                for pol in ("start", "two"):
                    _, X, _ = forward(*args, 32, pol)
                    r[f"err_x_{pol}"] = rel(X, x_ref)
                    r[f"bwd_{pol}"] = backward_error(Hs, lam, X, q)
                rows.append(r)
                print(f"{r['ridge']:>12} {kname:>10} {seed:>4} {r['cond_max']:8.1e} {r['err_x_start']:8.1e} {r['err_x_two']:8.1e}"
                      f" {r['err_x_dense']:8.1e} {r['bwd_start']:9.1e} {r['bwd_two']:9.1e} {r['bwd_dense']:9.1e}")
    eps = np.finfo(np.float64).eps
    for lo, hi in ((0, 1e2), (1e2, 1e4), (1e4, 1e6), (1e6, 1e9)):
        sel = [r for r in rows if lo < r["cond_max"] <= hi]
        if sel:
            print(f"cond in ({lo:.0e}, {hi:.0e}]: {len(sel)} inputs; worst backward error / eps: "
                  f"start {max(r['bwd_start'] for r in sel) / eps:.0f}, two {max(r['bwd_two'] for r in sel) / eps:.0f}, "
                  f"dense {max(r['bwd_dense'] for r in sel) / eps:.1f}; worst forward error: start "
                  f"{max(r['err_x_start'] for r in sel):.1e}, two {max(r['err_x_two'] for r in sel):.1e}, "
                  f"dense {max(r['err_x_dense'] for r in sel):.1e}")
    return rows


def part_b():
    print("\n== B. float32 output error vs the float64 refined reference, T = 256, d = 64, dv = 32, C = 64, gates 0.9-1")
    cols = ("start", "two", "dense", "cg30", "cg30_diag", "pcg30_jacobi", "cg100")
    print(f"{'ridge':>12} {'keys':>10} {'seed':>4} {'cond max':>8} " + " ".join(f"{c:>12}" for c in cols))
    rows = []
    T, d, dv, C = 256, 64, 32, 64
    for (llo, lhi) in RIDGES[:5]:
        for kname, kw in KEYS.items():
            for seed in (0, 1, 2):
                rng = np.random.default_rng(2000 + seed)
                args = make_inputs(rng, T, d, dv, 0.9, 1.0, llo, lhi, **kw)
                o_ref, _, _, cond, _ = refined_reference(*args)
                f32 = [a.astype(np.float32) for a in args]
                res = {}
                for pol in ("start", "two"):
                    try:
                        o, _, _ = forward(*f32, C, pol)
                        res[pol] = rel(o.astype(np.float64), o_ref)
                    except np.linalg.LinAlgError:
                        res[pol] = None
                res["dense"] = rel(reference_forward(*f32)[0].astype(np.float64), o_ref)
                Hs = reference_states(f32[1], f32[3], f32[4])
                _, Gb = boundary_states(*f32[1:5], C)
                for name, steps, kwcg in (("cg30", 30, {}), ("cg30_diag", 30, dict(start="diag")),
                                          ("pcg30_jacobi", 30, dict(start="diag", jacobi=True)), ("cg100", 100, {})):
                    X = cg_solve(Hs, f32[5], f32[0], steps, **kwcg)
                    res[name] = rel(chunk_readout(X, f32[1], f32[2], f32[3], f32[4], C, Gb).astype(np.float64), o_ref)
                r = dict(ridge=f"{llo:g}-{lhi:g}", keys=kname, seed=seed, cond_max=float(cond.max()), **res)
                rows.append(r)
                fmtv = lambda x: f"{'failed':>12}" if x is None else f"{x:12.1e}"
                print(f"{r['ridge']:>12} {kname:>10} {seed:>4} {r['cond_max']:8.1e} " + " ".join(fmtv(res[c]) for c in cols))
    print("\nsummary by condition number (worst over inputs; ratio = chunk error / dense error on the same input):")
    for lo, hi in ((0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e7)):
        sel = [r for r in rows if lo < r["cond_max"] <= hi and r["start"] is not None and r["two"] is not None]
        if not sel:
            continue
        line = f"  cond ({lo:.0e}, {hi:.0e}], {len(sel)} inputs: "
        line += ", ".join(f"{c} {max(r[c] for r in sel):.1e}" for c in cols)
        line += (f"; worst ratio start/dense {max(r['start'] / r['dense'] for r in sel):.1f},"
                 f" two/dense {max(r['two'] / r['dense'] for r in sel):.1f}")
        print(line)
    both = [r for r in rows if r["start"] is not None and r["two"] is not None]
    worse = sum(r["two"] > r["start"] for r in both)
    print(f"two-anchor policy less accurate than start-only on {worse} of {len(both)} float32 inputs;"
          f" failures: {sum(r['start'] is None or r['two'] is None for r in rows)}")
    return rows


def main():
    rows_a = part_a()
    rows_b = part_b()
    path = save_results({"part_a_float64": rows_a, "part_b_float32": rows_b}, "check_conditioning")
    print(f"\nsaved {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
