"""E2 check of the closed-form backward pass (README, Method: "Backward pass").

The closed-form gradients come from `gated_mesa.backward`, which uses two exact chunk solves (x_t and
g_t = (H_t + Lambda)^{-1} G_t^T dO_t) and the formulas in its docstring, including the gate-gradient identity
    dl/dlog(gamma_j) = sum_{t>=j} ( dO_t . o_t - g_t . q_t + x_t^T Lambda g_t - beta_t dl/dbeta_t ).
The reference is the complex-step derivative of the token-by-token recurrence (`reference_forward` on complex
inputs): d l / d theta = Im l(theta + i h) / h with h = 1e-30, exact to round-off, with no subtraction error.

Run from this folder:  python3 -u check_backward.py 2>&1 | tee /tmp/check_backward.log
(The recorded run is logs/check_backward.log; never tee into logs/, which is immutable.)
"""
import numpy as np

from gated_mesa import backward, make_inputs, reference_forward, rel, save_results

NAMES = ("q", "k", "v", "beta", "loggam", "lam")


def complex_step_grads(args, dO, h=1e-30):
    grads = {}
    base = [a.astype(complex) for a in args]
    for n, name in enumerate(NAMES):
        G = np.zeros(args[n].shape)
        for idx in np.ndindex(args[n].shape):
            pert = [a.copy() for a in base]
            pert[n][idx] += 1j * h
            o, _ = reference_forward(*pert)
            G[idx] = np.imag((dO * o).sum()) / h
        grads[name] = G
    return grads


def main():
    rng = np.random.default_rng(29102026)
    cases = [  # (T, d, dv, C, gate_lo, gate_hi, ridge_lo, ridge_hi, policy)
        (20, 6, 4, 6, 1.0, 1.0, 0.05, 1.0, "two"),
        (20, 6, 4, 6, 0.99, 1.0, 0.05, 1.0, "two"),
        (20, 6, 4, 6, 0.9, 1.0, 0.05, 1.0, "start"),
        (20, 6, 4, 6, 0.9, 1.0, 0.05, 1.0, "two"),
        (24, 10, 3, 8, 0.5, 1.0, 0.05, 1.0, "two"),
        (24, 10, 3, 8, 0.01, 0.2, 0.05, 1.0, "two"),
        (17, 4, 3, 16, 0.8, 1.0, 0.05, 1.0, "two"),       # chunk longer than d: primal d x d form
        (18, 8, 4, 5, 0.9, 1.0, 1e-4, 1e-3, "two"),       # small ridge: ill-conditioned systems
    ]
    print("closed-form gradients vs complex-step derivatives of the token-by-token recurrence (float64)")
    print("error = max |closed form - complex step| / max |complex step|, per input")
    print(f"{'T':>3} {'d':>3} {'dv':>3} {'C':>3} {'gates':>10} {'ridge':>10} {'policy':>6} "
          + " ".join(f"{n:>8}" for n in NAMES))
    rows, worst = [], 0.0
    for (T, d, dv, C, glo, ghi, llo, lhi, pol) in cases:
        args = make_inputs(rng, T, d, dv, glo, ghi, llo, lhi)
        if glo == ghi == 1.0:
            args[4] = np.zeros(T)
        dO = rng.standard_normal((T, dv))
        ref = complex_step_grads(args, dO)
        got = backward(dO, *args, C, pol)
        errs = {n: rel(got[n], ref[n]) for n in NAMES}
        worst = max(worst, max(errs.values()))
        rows.append(dict(T=T, d=d, dv=dv, C=C, gates=f"{glo:g}-{ghi:g}", ridge=f"{llo:g}-{lhi:g}", policy=pol, **errs))
        print(f"{T:>3} {d:>3} {dv:>3} {C:>3} {f'{glo:g}-{ghi:g}':>10} {f'{llo:g}-{lhi:g}':>10} {pol:>6} "
              + " ".join(f"{errs[n]:8.1e}" for n in NAMES))
    print(f"worst relative error over all cases and inputs: {worst:.2e}")
    path = save_results({"cases": rows, "worst": worst}, "check_backward")
    print(f"saved {path.relative_to(path.parent.parent)}")


if __name__ == "__main__":
    main()
