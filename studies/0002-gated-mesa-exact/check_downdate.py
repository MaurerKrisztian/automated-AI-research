"""How much accuracy does downdating from the chunk-end anchor cost, as a function of rho_t = a_t / a_C?

The two-anchor policy lets token t downdate from the chunk-end eigendecomposition when that capacitance is
smaller and rho_t <= kappa_max. This sweeps kappa_max (1 allows only tokens with rho_t = 1, inf allows all) and
also runs policy "end" (every token downdates), in float64 and float32, against the refined reference.
"Cholesky failed" means a float32 downdate capacitance I - U D^-1 U^T lost positive definiteness to cancellation.

Run from this folder:  python3 -u check_downdate.py 2>&1 | tee /tmp/check_downdate.log
(The recorded run is logs/check_downdate_v2.log; never tee into logs/, which is immutable.)
"""
import numpy as np

from gated_mesa import chunk_readout, boundary_states, chunk_solve, make_inputs, refined_reference, rel, save_results


def main():
    rng = np.random.default_rng(7)
    T, d, dv, C = 256, 64, 16, 64
    kappas = (1.0, 10.0, 100.0, 1e4, np.inf)
    print(f"float64 solution error and float32 output error vs the refined reference; T={T}, d={d}, C={C}, ridge 0.05-1")
    print(f"{'gates':>9} {'policy':>14} {'end tokens':>10} {'mean m':>7} {'max rho used':>12} {'f64 err x':>10} {'f32 err o':>10}")
    rows = []
    for gname, (glo, ghi) in (("0.99-1", (0.99, 1.0)), ("0.9-1", (0.9, 1.0)), ("0.5-1", (0.5, 1.0))):
        args = make_inputs(rng, T, d, dv, glo, ghi)
        q, k, v, b, lg, lam = args
        o_ref, x_ref, _, _, _ = refined_reference(*args)
        f32 = [a.astype(np.float32) for a in args]
        Hb, _ = boundary_states(k, v, b, lg, C)
        Hb32, Gb32 = boundary_states(*f32[1:5], C)
        L = np.concatenate([np.cumsum(lg[c0:c0 + C]) for c0 in range(0, T, C)])
        Lend = np.concatenate([np.full(min(C, T - c0), np.cumsum(lg[c0:c0 + C])[-1]) for c0 in range(0, T, C)])
        rho = np.exp(L - Lend)
        for pol, kap in [("two", kk) for kk in kappas] + [("end", np.inf)]:
            X, info = chunk_solve(q, k, b, lg, lam, C, Hb, pol, kap)
            try:
                X32, _ = chunk_solve(f32[0], f32[1], f32[3], f32[4], f32[5], C, Hb32, pol, np.float32(kap))
                f32_err = rel(chunk_readout(X32, *f32[1:5], C, Gb32).astype(np.float64), o_ref)
            except np.linalg.LinAlgError:          # a downdate capacitance lost positive definiteness in float32
                f32_err = None
            used = info["used_end"]
            row = dict(gates=gname, policy=pol, kappa_max=None if np.isinf(kap) else kap, end_tokens=int(used.sum()),
                       mean_m=float(info["m"].mean()), max_rho_used=float(rho[used].max()) if used.any() else 1.0,
                       f64_err_x=rel(X, x_ref), f32_err_o=f32_err)
            rows.append(row)
            label = f"{pol} k<={kap:g}" if pol == "two" else "end (all)"
            print(f"{gname:>9} {label:>14} {row['end_tokens']:>10} {row['mean_m']:7.1f} {row['max_rho_used']:12.3g}"
                  f" {row['f64_err_x']:10.1e} {'Cholesky failed' if f32_err is None else f'{f32_err:10.1e}':>10}")
    path = save_results({"rows": rows}, "check_downdate")
    print(f"saved {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
