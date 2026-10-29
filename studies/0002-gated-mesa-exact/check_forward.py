"""E2 check of the exact chunk solver's forward pass against the token-by-token recurrence.

Part A (float64): random inputs over gate regimes, ridges, key spectra, sizes and chunk sizes.
Part A2 (float64): edge cases.
  The reference is a dense solve per token refined with extended-precision residuals, so that on
  ill-conditioned systems the chunk solver and a plain float64 dense solve can both be compared against it.
  Also reported: cond(H_t + Lambda) and the chunk solver's normwise backward error.
Part B (float32): error of the chunk solver, a dense per-token solve and fixed-step CG, each against float64.

Run from this folder:  python3 -u check_forward.py 2>&1 | tee /tmp/check_forward.log
(The recorded run is logs/check_forward_v2.log; never tee into logs/, which is immutable.)
"""
import numpy as np

from gated_mesa import (backward_error, boundary_states, cg_solve, chunk_readout, forward, make_inputs,
                        reference_forward, reference_states, refined_reference, rel, save_results)

GATES = {"none (1)": (1.0, 1.0), "0.999-1": (0.999, 1.0), "0.99-1": (0.99, 1.0), "0.9-1": (0.9, 1.0),
         "0.5-1": (0.5, 1.0), "0.01-0.2": (0.01, 0.2)}
HEAD = (f"{'case':>44} {'C':>3} {'policy':>6} {'cond max':>8} {'err o':>8} {'err x':>8} {'dense x':>8}"
        f" {'bwd err':>8} {'end':>4} {'primal':>6} {'max m':>5}")


def compare(name, args, C, pol, rows):
    q, k, v, b, lg, lam = args
    o_ref, x_ref, x_dense, cond, Hs = refined_reference(*args)
    o, X, info = forward(*args, C, pol)
    row = dict(case=name, C=C, policy=pol, cond_max=float(cond.max()), err_o=rel(o, o_ref), err_x=rel(X, x_ref),
               err_x_dense=rel(x_dense, x_ref), bwd_err=backward_error(Hs, lam, X, q),
               end_used=int(info["used_end"].sum()), primal=int(info["primal"].sum()), max_m=int(info["m"].max()))
    rows.append(row)
    print(f"{name:>44} {C:>3} {pol:>6} {row['cond_max']:8.1e} {row['err_o']:8.1e} {row['err_x']:8.1e}"
          f" {row['err_x_dense']:8.1e} {row['bwd_err']:8.1e} {row['end_used']:>4} {row['primal']:>6} {row['max_m']:>5}")


def part_a(rng):
    print("== A. float64: chunk solver vs refined token-by-token reference (max error / max |reference|)")
    print("   err o, err x: chunk solver; dense x: plain float64 dense solve per token; bwd err: chunk solver's")
    print("   normwise backward error; end: tokens that downdated from the chunk-end anchor; primal: tokens that solved")
    print("   the d x d system because their capacitance would have had m >= d rows; max m: largest system solved")
    print(HEAD)
    rows = []
    sizes = [(200, 16, 8, 32), (200, 64, 16, 16), (150, 8, 4, 64), (257, 32, 32, 64), (96, 128, 16, 32)]
    ridges = {"0.05-1": (0.05, 1.0), "1e-6-1e-4": (1e-6, 1e-4)}
    for gname, (glo, ghi) in GATES.items():
        for (T, d, dv, C) in sizes:
            for rname, (llo, lhi) in ridges.items():
                for alpha in (0.0, 2.0):
                    args = make_inputs(rng, T, d, dv, glo, ghi, llo, lhi, key_alpha=alpha)
                    if glo == ghi == 1.0:
                        args[4] = np.zeros(T)
                    name = f"gates {gname} T{T} d{d} dv{dv} ridge {rname} a{alpha:g}"
                    for pol in ("start", "two"):
                        compare(name, args, C, pol, rows)
    return rows


def part_a_edges(rng):
    print("\n== A2. float64 edge cases")
    print(HEAD)
    rows = []
    args = make_inputs(rng, 1, 8, 4, 0.5, 1.0)
    compare("T = 1", args, 16, "two", rows)
    args = make_inputs(rng, 40, 8, 4, 0.5, 1.0)
    compare("chunk size 1", args, 1, "two", rows)
    compare("one chunk longer than the sequence", args, 64, "two", rows)
    compare("T = 40, C = 7 (does not divide)", args, 7, "two", rows)
    compare("always downdate (policy end), gates 0.5-1", args, 7, "end", rows)
    q, k, v, b, lg, lam = args
    k2 = k.copy()
    k2[10:30] = k2[10]
    compare("one key repeated 20 times", [q, k2, v, b, lg, lam], 8, "two", rows)
    b2 = b.copy()
    b2[::3] = 0.0
    compare("beta = 0 on every third token", [q, k, v, b2, lg, lam], 8, "two", rows)
    lg2 = lg.copy()
    lg2[[5, 17, 33]] = np.log(1e-12)
    compare("three near-resets (gate 1e-12)", [q, k, v, b, lg2, lam], 16, "two", rows)
    args = make_inputs(rng, 128, 16, 8, 0.95, 1.0, beta_hi=50.0)
    compare("large write strengths (beta up to 50)", args, 32, "two", rows)
    args = make_inputs(rng, 128, 16, 8, 0.95, 1.0, 1e-8, 1e-8, key_alpha=4.0)
    compare("ridge 1e-8, key spectrum alpha 4", args, 32, "start", rows)
    compare("ridge 1e-8, key spectrum alpha 4", args, 32, "two", rows)
    return rows


def summarize(rows, label):
    well = [r for r in rows if r["cond_max"] <= 1e4]
    ill = [r for r in rows if r["cond_max"] > 1e4]
    out = {"cases": len(rows), "well_conditioned_cases": len(well), "worst_err_o_well": max(r["err_o"] for r in well),
           "worst_err_x_well": max(r["err_x"] for r in well), "worst_bwd_err_well": max(r["bwd_err"] for r in well)}
    if ill:
        out.update(ill_conditioned_cases=len(ill), worst_cond=max(r["cond_max"] for r in ill),
                   worst_err_x_ill=max(r["err_x"] for r in ill),
                   worst_err_x_dense_ill=max(r["err_x_dense"] for r in ill),
                   worst_bwd_err_ill=max(r["bwd_err"] for r in ill),
                   worst_err_x_over_cond_eps=max(r["err_x"] / (r["cond_max"] * 2.2e-16) for r in ill))
    print(f"\n{label}: {out}")
    return out


def part_b(rng):
    print("\n== B. float32 output error against the float64 reference (max error / max |reference|), "
          "T = 512, d = 64, dv = 32, C = 64")
    print(f"{'gates':>10} {'ridge':>10} {'alpha':>5} {'cond max':>8} {'chunk start':>11} {'chunk two':>9} {'dense':>9}"
          f" {'CG 10':>9} {'CG 30':>9} {'CG 100':>9}")
    rows = []
    T, d, dv, C = 512, 64, 32, 64
    for gname in ("none (1)", "0.99-1", "0.9-1", "0.5-1"):
        glo, ghi = GATES[gname]
        for (llo, lhi, alpha) in ((0.05, 1.0, 0.0), (1e-3, 1e-2, 2.0)):
            args = make_inputs(rng, T, d, dv, glo, ghi, llo, lhi, key_alpha=alpha)
            if glo == ghi == 1.0:
                args[4] = np.zeros(T)
            ref, _, _, cond, _ = refined_reference(*args)
            f32 = [a.astype(np.float32) for a in args]
            res = {}
            for pol in ("start", "two"):
                o, _, _ = forward(*f32, C, pol)
                res[pol] = rel(o.astype(np.float64), ref)
            o_dense, _ = reference_forward(*f32)
            res["dense"] = rel(o_dense.astype(np.float64), ref)
            Hs = reference_states(f32[1], f32[3], f32[4])
            _, Gb = boundary_states(*f32[1:5], C)
            for steps in (10, 30, 100):
                Xc = cg_solve(Hs, f32[5], f32[0], steps)
                oc = chunk_readout(Xc, f32[1], f32[2], f32[3], f32[4], C, Gb)
                res[f"cg{steps}"] = rel(oc.astype(np.float64), ref)
            ridge = f"{llo:g}-{lhi:g}"
            rows.append(dict(gates=gname, ridge=ridge, key_alpha=alpha, cond_max=float(cond.max()), **res))
            print(f"{gname:>10} {ridge:>10} {alpha:>5.1f} {cond.max():8.1e} {res['start']:11.1e} {res['two']:9.1e}"
                  f" {res['dense']:9.1e} {res['cg10']:9.1e} {res['cg30']:9.1e} {res['cg100']:9.1e}")
    return rows


def main():
    rng = np.random.default_rng(20261029)
    rows_a = part_a(rng)
    rows_e = part_a_edges(rng)
    sum_a = summarize(rows_a, "A summary")
    sum_e = summarize(rows_e, "A2 summary")
    rows_b = part_b(rng)
    path = save_results({"part_a": rows_a, "part_a_summary": sum_a, "part_a_edges": rows_e, "part_a_edges_summary": sum_e,
                         "part_b_float32": rows_b}, "check_forward")
    print(f"\nsaved {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
