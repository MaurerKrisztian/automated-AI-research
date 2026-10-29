"""How many CG steps do gated Mesa systems need, and what does the exact chunk solver cost in CG steps?

Version 2, written after the review (REVIEW.md, issues 1, 2 and 8). Version 1's log is logs/cost_and_cg.log.

Part A (float64, measured): output error of k-step CG against the refined reference, over gate regimes, ridges
and key spectra, for d = 64 and 128, for three CG variants:
  zero: x0 = 0, no preconditioner (flash-linear-attention's MesaNet kernel);
  diag: x0 = diag(H_t + Lambda)^-1 q_t (MesaNet's Appendix C);
  jacobi: the diag start plus Jacobi preconditioning.
The first 32 inputs use the same seed and draws as version 1, so its "zero" numbers reproduce version 1. Then
16 inputs with rotated i^-2 key spectra, whose anisotropy is not aligned with the coordinate axes (separate seed).
Part B (FLOP model, derived): solver FLOPs per token of the exact method, as CG-step equivalents, under several
assumptions (formulas in gated_mesa.solver_flops and cg_step_flops; multiply-add = 2 FLOPs).

Run from this folder:  python3 -u cost_and_cg.py 2>&1 | tee /tmp/cost_and_cg.log
(The recorded run is logs/cost_and_cg_v2.log; never tee into logs/, which is immutable.)
"""
import numpy as np

from gated_mesa import (boundary_states, capacitance_sizes, cg_solve, cg_step_flops, chunk_readout, make_inputs,
                        policy_sizes, refined_reference, rel, save_results, solver_flops)

STEPS = (5, 10, 15, 20, 30, 50, 100)
VARIANTS = {"zero": dict(start="zero"), "diag": dict(start="diag"), "jacobi": dict(start="diag", jacobi=True)}
GATES = (("1", (1.0, 1.0)), ("0.99-1", (0.99, 1.0)), ("0.9-1", (0.9, 1.0)), ("0.5-1", (0.5, 1.0)))


def run_case(args, C):
    q, k, v, b, lg, lam = args
    o_ref, _, _, cond, Hs = refined_reference(*args)
    _, Gb = boundary_states(k, v, b, lg, C)
    out = {}
    for name, kw in VARIANTS.items():
        errs = {s: rel(chunk_readout(cg_solve(Hs, lam, q, s, **kw), k, v, b, lg, C, Gb), o_ref) for s in STEPS}
        need = {tol: next((s for s in STEPS if errs[s] <= tol), None) for tol in (1e-4, 1e-6)}
        out[name] = dict(err={str(s): e for s, e in errs.items()}, steps_for_1e4=need[1e-4], steps_for_1e6=need[1e-6])
    return float(cond.max()), out


def part_a():
    print("== A. float64 output error of k-step CG against the refined reference, T = 256, C = 64")
    print(f"{'d':>4} {'gates':>7} {'ridge':>10} {'keys':>11} {'cond':>7} {'CG':>6} "
          + " ".join(f"{'k=' + str(s):>8}" for s in STEPS) + f" {'k@1e-4':>7} {'k@1e-6':>7}")
    rows, T, C = [], 256, 64
    fmt = lambda s: f">{STEPS[-1]}" if s is None else str(s)

    def report(d, gname, ridge, keys, args):
        cond, out = run_case(args, C)
        rows.append(dict(d=d, gates=gname, ridge=ridge, keys=keys, cond_max=cond, cg=out))
        for name, r in out.items():
            print(f"{d:>4} {gname:>7} {ridge:>10} {keys:>11} {cond:7.1e} {name:>6} "
                  + " ".join(f"{r['err'][str(s)]:8.1e}" for s in STEPS)
                  + f" {fmt(r['steps_for_1e4']):>7} {fmt(r['steps_for_1e6']):>7}")

    rng = np.random.default_rng(31)                      # version 1's seed and draw order
    for d in (64, 128):
        for gname, (glo, ghi) in GATES:
            for (llo, lhi, alpha) in ((0.05, 1.0, 0.0), (0.05, 1.0, 2.0), (1e-3, 1e-2, 0.0), (1e-3, 1e-2, 2.0)):
                args = make_inputs(rng, T, d, 32, glo, ghi, llo, lhi, key_alpha=alpha)
                if glo == ghi == 1.0:
                    args[4] = np.zeros(T)
                report(d, gname, f"{llo:g}-{lhi:g}", "flat" if alpha == 0 else "i^-2 axis", args)
    rng = np.random.default_rng(32)
    for d in (64, 128):
        for gname, (glo, ghi) in GATES:
            for (llo, lhi) in ((0.05, 1.0), (1e-3, 1e-2)):
                args = make_inputs(rng, T, d, 32, glo, ghi, llo, lhi, key_alpha=2.0, rotate=True)
                if glo == ghi == 1.0:
                    args[4] = np.zeros(T)
                report(d, gname, f"{llo:g}-{lhi:g}", "i^-2 rot", args)
    print("\nsteps to 1e-4 output error, pooled over the four gate regimes (min-max):")
    for d in (64, 128):
        for ridge in ("0.05-1", "0.001-0.01"):
            for keys in ("flat", "i^-2 axis", "i^-2 rot"):
                sel = [r for r in rows if r["d"] == d and r["ridge"] == ridge and r["keys"] == keys]
                conds = [r["cond_max"] for r in sel]
                parts = []
                for name in VARIANTS:
                    ks = [r["cg"][name]["steps_for_1e4"] for r in sel]
                    lo = min(k for k in ks if k is not None) if any(k is not None for k in ks) else None
                    hi = None if None in ks else max(ks)
                    parts.append(f"{name} {fmt(lo)}-{fmt(hi)}")
                print(f"  d={d:<4} ridge {ridge:<10} {keys:<10} cond {min(conds):.1e}-{max(conds):.1e}: " + ", ".join(parts))
    return rows


def part_b():
    print("\n== B. exact solver FLOPs per token as CG-step equivalents (solver only; state update and readout excluded)")
    print("   columns: exact policy / eigendecomposition cost; 'CG@64' divides by one CG step at chunk 64 (fla's default),")
    print("   'CG@16' by one CG step at chunk 16 (cheaper per step). Training = forward + backward solves, vs 60 CG steps.")
    rows = []
    rng = np.random.default_rng(5)
    hdr = (f"{'d':>4} {'C':>4} {'two ideal 4d3':>13} {'two ideal 9d3':>13} {'two, gates 0.5-1':>16}"
           f" {'start only':>10} {'two, padded':>11} {'train, two':>10}")
    for cg_C in (64, 16):
        print(f"\n   CG-step equivalents with CG at chunk {cg_C}")
        print(hdr)
        for d in (64, 128, 256):
            for C in (32, 64, 128):
                step = cg_step_flops(d, cg_C)
                m2, p2 = capacitance_sizes(C, d, "two")
                ms, ps = capacitance_sizes(C, d, "start")
                lg = np.log(rng.uniform(0.5, 1.0, 16 * C))
                mf, pf = policy_sizes(lg, C, d, 10.0)
                r = dict(d=d, C=C, cg_chunk=cg_C,
                         two_ideal_eig4=solver_flops(d, d, C, m2, p2, 4.0) / step,
                         two_ideal_eig9=solver_flops(d, d, C, m2, p2, 9.0) / step,
                         two_fast_gates_eig9=solver_flops(d, d, C, mf, pf, 9.0) / step,
                         start_only_eig9=solver_flops(d, d, C, ms, ps, 9.0) / step,
                         two_padded_eig9=solver_flops(d, d, C, m2, p2, 9.0, pad=True) / step,
                         train_two_ideal_eig9=solver_flops(d, d, C, m2, p2, 9.0, backward=True) / step)
                rows.append(r)
                print(f"{d:>4} {C:>4} {r['two_ideal_eig4']:13.1f} {r['two_ideal_eig9']:13.1f} {r['two_fast_gates_eig9']:16.1f}"
                      f" {r['start_only_eig9']:10.1f} {r['two_padded_eig9']:11.1f} {r['train_two_ideal_eig9']:10.1f}")
    print("\n   'two, gates 0.5-1' uses the policy's actual sizes (rho <= 10) on random gates in 0.5-1; 'train' counts")
    print("   CG-step equivalents for forward + backward, to compare with 30 + 30 = 60 steps.")
    return rows


def main():
    rows_a = part_a()
    rows_b = part_b()
    path = save_results({"version": 2, "part_a_cg_accuracy": rows_a, "part_b_cg_equivalents": rows_b}, "cost_and_cg_v2")
    print(f"\nsaved {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
