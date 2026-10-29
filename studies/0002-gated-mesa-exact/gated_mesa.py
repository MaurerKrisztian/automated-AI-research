"""Exact chunkwise MesaNet (scalar forget gates, fixed diagonal ridge) in NumPy, and the references it is checked against.

Model, per head, for t = 1..T (H_0 = 0, G_0 = 0):

    H_t = gamma_t H_{t-1} + beta_t k_t k_t^T,   G_t = gamma_t G_{t-1} + beta_t v_t k_t^T
    x_t = (H_t + Lambda)^{-1} q_t,              o_t = G_t x_t

Inside a chunk with boundary state H_0 and cumulative gate a_t = prod_{j<=t} gamma_j,

    H_t + Lambda = a_t (H_0 + sum_{i<=t} (beta_i / a_i) k_i k_i^T + Lambda / a_t),

so the gates act as a per-token ridge Lambda / a_t on nested data. One eigendecomposition of the whitened
boundary state diagonalizes every shift, and Woodbury with a per-token capacitance matrix gives each token's
exact solve (see README.md, Method). Gates are handled in log space so that float32 does not underflow.

Everything here is written for clarity, not speed: loops over chunks and tokens, NumPy only.
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- references
def reference_forward(q, k, v, beta, loggam, lam):
    """Token-by-token recurrence with a dense solve per token. Works for complex inputs (complex-step checks)."""
    T, d = q.shape
    dv = v.shape[1]
    dt = np.result_type(q, k, v, beta, loggam, lam)
    H = np.zeros((d, d), dtype=dt)
    G = np.zeros((dv, d), dtype=dt)
    L = np.diag(lam).astype(dt)
    out = np.zeros((T, dv), dtype=dt)
    xs = np.zeros((T, d), dtype=dt)
    gam = np.exp(loggam)
    for t in range(T):
        H = gam[t] * H + beta[t] * np.outer(k[t], k[t])
        G = gam[t] * G + beta[t] * np.outer(v[t], k[t])
        x = np.linalg.solve(H + L, q[t])
        xs[t] = x
        out[t] = G @ x
    return out, xs


def reference_states(k, beta, loggam):
    """H_t for every token (T, d, d): the matrices the CG baseline and the dense per-token solver use."""
    T, d = k.shape
    H = np.zeros((d, d), dtype=k.dtype)
    Hs = np.zeros((T, d, d), dtype=k.dtype)
    gam = np.exp(loggam).astype(k.dtype)
    for t in range(T):
        H = gam[t] * H + beta[t] * np.outer(k[t], k[t])
        Hs[t] = H
    return Hs


def refined_reference(q, k, v, beta, loggam, lam, iters=3):
    """A sharper reference for ill-conditioned cases: dense float64 solves refined with residuals computed in
    extended precision (np.longdouble, 64-bit mantissa on x86). Returns o_ref, x_ref, the plain float64 dense
    solution x_dense, cond(H_t + Lambda) per token, and the state matrices."""
    T, d = q.shape
    Hs = reference_states(k, beta, loggam)
    ld = np.longdouble
    x_ref = np.zeros((T, d), dtype=ld)
    x_dense = np.zeros((T, d))
    cond = np.zeros(T)
    for t in range(T):
        M = Hs[t] + np.diag(lam)
        x = np.linalg.solve(M, q[t])
        x_dense[t] = x
        xl = x.astype(ld)
        Ml, ql = M.astype(ld), q[t].astype(ld)
        for _ in range(iters):
            r = ql - Ml @ xl
            xl = xl + np.linalg.solve(M, r.astype(np.float64)).astype(ld)
        x_ref[t] = xl
        cond[t] = np.linalg.cond(M)
    # o_t = G_t x_t with G_t accumulated in extended precision
    G = np.zeros((v.shape[1], d), dtype=ld)
    o_ref = np.zeros((T, v.shape[1]), dtype=ld)
    gam = np.exp(loggam.astype(ld))
    for t in range(T):
        G = gam[t] * G + ld(beta[t]) * np.outer(v[t].astype(ld), k[t].astype(ld))
        o_ref[t] = G @ x_ref[t]
    return o_ref.astype(np.float64), x_ref.astype(np.float64), x_dense, cond, Hs


def backward_error(Hs, lam, X, R):
    """Max over tokens of the normwise backward error ||M x - r|| / (||M||_2 ||x|| + ||r||)."""
    worst = 0.0
    for t in range(X.shape[0]):
        M = Hs[t] + np.diag(lam)
        res = np.linalg.norm(M @ X[t] - R[t])
        worst = max(worst, res / (np.linalg.norm(M, 2) * np.linalg.norm(X[t]) + np.linalg.norm(R[t])))
    return float(worst)


def cg_solve(Hs, lam, R, steps, x0=None, start="zero", jacobi=False):
    """Conjugate gradient on (H_t + Lambda) x = r_t for every token: at most `steps` steps, stopping early only on
    breakdown (zero residual or non-positive curvature, which happens in float32 after convergence).

    start "zero": x = 0, as in flash-linear-attention's MesaNet kernel. start "diag": x = diag(H_t + Lambda)^-1 r_t,
    the initial guess in MesaNet's Appendix C. jacobi=True preconditions with diag(H_t + Lambda) (Jacobi PCG).
    Each step costs one product with H_t + Lambda; the diagonal costs d per token more."""
    T, d = R.shape
    X = np.zeros_like(R) if x0 is None else x0.copy()
    for t in range(T):
        M = Hs[t] + np.diag(lam).astype(R.dtype)
        dg = np.diag(M).copy()
        x = X[t] if start == "zero" else R[t] / dg
        r = R[t] - M @ x
        z = r / dg if jacobi else r
        p = z.copy()
        rz = r @ z
        for _ in range(steps):
            if not rz > 0:
                break
            Mp = M @ p
            pMp = p @ Mp
            if not pMp > 0:
                break
            alpha = rz / pMp
            x = x + alpha * p
            r = r - alpha * Mp
            z = r / dg if jacobi else r
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        X[t] = x
    return X


# --------------------------------------------------------------------------- chunk algorithm
def boundary_states(k, v, beta, loggam, C):
    """H and G at chunk boundaries: lists of length n_chunks + 1, the state before chunk c and after the last."""
    T, d = k.shape
    dv = v.shape[1]
    H = np.zeros((d, d), dtype=k.dtype)
    G = np.zeros((dv, d), dtype=k.dtype)
    Hb, Gb = [H], [G]
    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        L = np.cumsum(loggam[c0:c1])
        w = np.exp(L[-1] - L) * beta[c0:c1]                     # a_C / a_i * beta_i
        K, V = k[c0:c1], v[c0:c1]
        H = np.exp(L[-1]) * H + (K * w[:, None]).T @ K
        G = np.exp(L[-1]) * G + (V * w[:, None]).T @ K
        Hb.append(H)
        Gb.append(G)
    return Hb, Gb


def _chol_solve(A, b):
    Lc = np.linalg.cholesky(A)
    return np.linalg.solve(Lc.T, np.linalg.solve(Lc, b))


def chunk_solve(R, k, beta, loggam, lam, C, Hb, policy="two", kappa_max=10.0):
    """Exact X[t] = (H_t + Lambda)^{-1} R[t] for every token, chunk by chunk, with no iteration.

    policy "start": every token updates from the chunk-start anchor (capacitance size t, eigenvalues >= 1).
    policy "two":   a token may instead downdate from the chunk-end anchor when that capacitance is smaller and
                    rho_t = a_t / a_C <= kappa_max, which bounds the cancellation (README, Method).
    policy "end":   always downdate (for testing only; loses accuracy with fast forgetting).
    When a token's capacitance would have m_t >= d rows (chunks longer than the key dimension), it solves the
    d x d system diag(D) + U^T U from the start anchor instead (the "primal" form), which is cheaper and avoids
    Woodbury's cancellation there.
    Returns X and per-token info: the anchor used, the system size m_t, and whether the primal form was used.
    """
    T, d = R.shape
    s = 1.0 / np.sqrt(lam)
    X = np.empty_like(R)
    used_end = np.zeros(T, dtype=bool)
    primal = np.zeros(T, dtype=bool)
    m_used = np.zeros(T, dtype=int)
    eigs = [np.linalg.eigh((s[:, None] * H) * s[None, :]) for H in Hb]   # one per boundary, shared by two chunks
    for c, c0 in enumerate(range(0, T, C)):
        c1 = min(c0 + C, T)
        n = c1 - c0
        Ks = k[c0:c1] * s
        Rs = R[c0:c1] * s
        B = beta[c0:c1]
        L = np.cumsum(loggam[c0:c1])                            # log a_t within the chunk
        (h0, V0), (h1, V1) = eigs[c], eigs[c + 1]
        K0, K1 = Ks @ V0, Ks @ V1                               # keys in each anchor's eigenbasis
        for t in range(n):
            m_start, m_end = t + 1, n - 1 - t
            rho = np.exp(L[t] - L[-1])                          # a_t / a_C >= 1
            if policy == "end":
                end = True
            elif policy == "two":
                end = m_end < m_start and rho <= kappa_max and m_end < d
            else:
                end = False
            if not end:
                D = 1.0 + np.exp(L[t]) * h0                     # whitened (Lambda + a_t H_0), diagonal
                w = np.exp(L[t] - L[: t + 1]) * B[: t + 1]      # a_t / a_i * beta_i <= beta_i
                U = K0[: t + 1] * np.sqrt(w)[:, None]
                r = Rs[t] @ V0
                if m_start >= d and policy != "end":
                    y = _chol_solve(np.diag(D) + U.T @ U, r)    # primal d x d form
                    primal[c0 + t] = True
                    m_used[c0 + t] = d
                else:
                    y = r / D
                    UD = U / D
                    A = np.eye(m_start, dtype=R.dtype) + UD @ U.T  # I + U D^-1 U^T, eigenvalues >= 1
                    y = y - UD.T @ _chol_solve(A, U @ y)
                    m_used[c0 + t] = m_start
                X[c0 + t] = (V0 @ y) * s
            else:
                D = 1.0 + rho * h1                              # whitened (Lambda + (a_t/a_C) H_C)
                w = np.exp(L[t] - L[t + 1:]) * B[t + 1:]        # a_t / a_i * beta_i, i > t
                U = K1[t + 1:] * np.sqrt(w)[:, None]
                y = (Rs[t] @ V1) / D
                if m_end:
                    UD = U / D
                    A = np.eye(m_end, dtype=R.dtype) - UD @ U.T  # I - U D^-1 U^T, positive definite
                    y = y + UD.T @ _chol_solve(A, U @ y)
                X[c0 + t] = (V1 @ y) * s
                used_end[c0 + t] = True
                m_used[c0 + t] = m_end
    return X, {"used_end": used_end, "m": m_used, "primal": primal}


def chunk_readout(X, k, v, beta, loggam, C, Gb):
    """o_t = G_t x_t = a_t G_0 x_t + sum_{i<=t in chunk} (a_t / a_i) beta_i v_i (k_i . x_t), the gated linear-attention form."""
    T = X.shape[0]
    out = np.empty((T, v.shape[1]), dtype=X.dtype)
    for c, c0 in enumerate(range(0, T, C)):
        c1 = min(c0 + C, T)
        L = np.cumsum(loggam[c0:c1])
        Wm = np.tril(np.exp(np.minimum(L[:, None] - L[None, :], 0.0))) * beta[c0:c1][None, :]
        Xc = X[c0:c1]
        out[c0:c1] = np.exp(L)[:, None] * (Xc @ Gb[c].T) + (Wm * (Xc @ k[c0:c1].T)) @ v[c0:c1]
    return out


def chunk_transpose_readout(D, k, v, beta, loggam, C, Gb):
    """e_t = G_t^T delta_t, the same structure as the readout with the roles of v and k swapped."""
    T = D.shape[0]
    out = np.empty((T, k.shape[1]), dtype=D.dtype)
    for c, c0 in enumerate(range(0, T, C)):
        c1 = min(c0 + C, T)
        L = np.cumsum(loggam[c0:c1])
        Wm = np.tril(np.exp(np.minimum(L[:, None] - L[None, :], 0.0))) * beta[c0:c1][None, :]
        Dc = D[c0:c1]
        out[c0:c1] = np.exp(L)[:, None] * (Dc @ Gb[c]) + (Wm * (Dc @ v[c0:c1].T)) @ k[c0:c1]
    return out


def forward(q, k, v, beta, loggam, lam, C, policy="two", kappa_max=10.0):
    Hb, Gb = boundary_states(k, v, beta, loggam, C)
    X, info = chunk_solve(q, k, beta, loggam, lam, C, Hb, policy, kappa_max)
    return chunk_readout(X, k, v, beta, loggam, C, Gb), X, info


def backward(dO, q, k, v, beta, loggam, lam, C, policy="two", kappa_max=10.0):
    """Closed-form gradients of l = sum_t dO_t . o_t. Two exact chunk solves (x and g); everything else is sums.

    dq_t = g_t,  dLambda = -sum_t x_t * g_t,
    dk_i = beta_i sum_{t>=i} (a_t/a_i) [ (dO_t . v_i) x_t - (x_t . k_i) g_t - (g_t . k_i) x_t ],
    dv_i = beta_i sum_{t>=i} (a_t/a_i) (k_i . x_t) dO_t,
    dbeta_i = sum_{t>=i} (a_t/a_i) [ (dO_t . v_i) - (g_t . k_i) ] (k_i . x_t),
    dloggam_j = sum_{t>=j} ( dO_t . o_t - g_t . q_t + x_t^T Lambda g_t - beta_t dbeta_t ).
    The pair sums use a dense T x T decay mask here for clarity; a chunked version is the usual gated
    linear-attention backward.
    """
    Hb, Gb = boundary_states(k, v, beta, loggam, C)
    X, _ = chunk_solve(q, k, beta, loggam, lam, C, Hb, policy, kappa_max)
    O = chunk_readout(X, k, v, beta, loggam, C, Gb)
    E = chunk_transpose_readout(dO, k, v, beta, loggam, C, Gb)
    Gr, _ = chunk_solve(E, k, beta, loggam, lam, C, Hb, policy, kappa_max)
    Lg = np.cumsum(loggam)
    W = np.tril(np.exp(np.minimum(Lg[:, None] - Lg[None, :], 0.0)))   # W[t, i] = a_t / a_i for i <= t
    XK = X @ k.T                                                      # x_t . k_i
    P = W * (dO @ v.T)                                                # (a_t/a_i) dO_t . v_i
    S = W * XK
    Rg = W * (Gr @ k.T)                                               # (a_t/a_i) g_t . k_i
    dbeta = ((P - Rg) * XK).sum(axis=0)
    dk = beta[:, None] * (P.T @ X - S.T @ Gr - Rg.T @ X)
    dv = beta[:, None] * (S.T @ dO)
    dlam = -(X * Gr).sum(axis=0)
    r = (dO * O).sum(1) - (Gr * q).sum(1) + (X * lam * Gr).sum(1)
    dloggam = np.cumsum((r - beta * dbeta)[::-1])[::-1]
    return {"q": Gr, "k": dk, "v": dv, "beta": dbeta, "loggam": dloggam, "lam": dlam}


# --------------------------------------------------------------------------- cost model (FLOPs, multiply-add = 2)
def capacitance_sizes(C, d, policy):
    """System size and kind per token of a full chunk, assuming every token may use the end anchor ("two") or
    none may ("start"). Returns (m, primal) arrays as chunk_solve would choose them."""
    m, primal = [], []
    for t in range(C):
        m_start, m_end = t + 1, C - 1 - t
        if policy == "two" and m_end < m_start and m_end < d:
            m.append(m_end)
            primal.append(False)
        elif m_start >= d:
            m.append(m_start)
            primal.append(True)
        else:
            m.append(m_start)
            primal.append(False)
    return np.array(m, dtype=float), np.array(primal)


def flops_exact(d, dv, C, policy="two", eig_coef=9.0):
    """Forward FLOPs per token for the exact chunk solver.

    Per chunk: boundary state update 2Cd^2 + 2Cd*dv; whitening d^2 and one eigendecomposition eig_coef*d^3 per
    boundary (shared by two chunks, so one per chunk); keys into one anchor basis 2Cd^2, queries 2Cd^2, solutions
    back 2Cd^2. Per token with an m x m capacitance: forming it m^2 d (symmetric), Cholesky m^3/3, two triangular
    solves 2m^2, U^T z and U y 4md, diagonal work 4d. Per token in the primal d x d form (m >= d keys):
    U^T U m d^2, Cholesky d^3/3, solves 2d^2. Readout 2d*dv + intra-chunk products (C d + C dv on average).
    """
    m, primal = capacitance_sizes(C, d, policy)
    per_chunk = 2 * C * d * d + 2 * C * d * dv + d * d + eig_coef * d ** 3 + 6 * C * d * d
    woodbury = m ** 2 * d + m ** 3 / 3 + 2 * m ** 2 + 4 * m * d + 4 * d
    dense_in_basis = m * d * d + d ** 3 / 3 + 2 * d * d + 2 * d
    per_tok = np.where(primal, dense_in_basis, woodbury)
    readout = 2 * d * dv + C * d + C * dv
    return per_chunk / C + per_tok.mean() + readout


def policy_sizes(loggam, C, d, kappa_max=10.0):
    """System size m and kind (primal or not) per token, as chunk_solve's "two" policy chooses them for these gates
    (the rho cap included). Needs only the gates, not a solve."""
    T = len(loggam)
    m, primal = np.zeros(T), np.zeros(T, dtype=bool)
    for c0 in range(0, T, C):
        L = np.cumsum(loggam[c0:c0 + C])
        n = len(L)
        for t in range(n):
            m_start, m_end = t + 1, n - 1 - t
            if m_end < m_start and np.exp(L[t] - L[-1]) <= kappa_max and m_end < d:
                m[c0 + t] = m_end
            elif m_start >= d:
                m[c0 + t], primal[c0 + t] = m_start, True
            else:
                m[c0 + t] = m_start
    return m, primal


def solver_flops(d, dv, C, m, primal, eig_coef=9.0, pad=False, backward=False):
    """Solver FLOPs per token for the exact method (state update and readout excluded: every method pays those),
    given per-token system sizes. pad=True charges every token of a chunk the chunk's largest capacitance, as a
    batched GPU implementation that pads would. backward=True adds the backward pass's second solve: it reuses the
    eigendecompositions and the keys in each basis, transforms its right-hand side in and the solution out (4Cd^2
    per chunk), and refactors each capacitance (the Cholesky factors are not stored)."""
    m, primal = np.asarray(m, dtype=float), np.asarray(primal)
    if pad:
        m = np.concatenate([np.full(len(m[c0:c0 + C]), m[c0:c0 + C][~primal[c0:c0 + C]].max(initial=0))
                            for c0 in range(0, len(m), C)])
    woodbury = m ** 2 * d + m ** 3 / 3 + 2 * m ** 2 + 4 * m * d + 4 * d
    dense_in_basis = m * d * d + d ** 3 / 3 + 2 * d * d + 2 * d
    per_tok = np.where(primal, dense_in_basis, woodbury).mean()
    fwd = (d * d + eig_coef * d ** 3 + 6 * C * d * d) / C + per_tok
    return fwd + (4 * d * d + per_tok if backward else 0.0)


def cg_step_flops(d, C):
    """One CG step per token in chunked form (see flops_cg); Jacobi preconditioning adds d."""
    return 2 * d * d + 2 * C * d + 11 * d


def flops_cg(d, dv, C, steps):
    """Forward FLOPs per token for k CG steps in chunked form: each step applies H_t + Lambda to one vector per token
    (a_t H_0 p: 2d^2; intra-chunk decayed key sum: 2Cd on average; Lambda p and the vector updates: about 11d),
    plus the same state update and readout as the exact solver."""
    state = 2 * d * d + 2 * d * dv
    readout = 2 * d * dv + C * d + C * dv
    return steps * (2 * d * d + 2 * C * d + 11 * d) + state + readout


def flops_dense(d, dv):
    """Forward FLOPs per token for a dense per-token solve: rank-one state update 3d^2, Cholesky d^3/3,
    two triangular solves 2d^2, readout and value-state update 5d*dv."""
    return 3 * d * d + d ** 3 / 3 + 2 * d * d + 5 * d * dv


def flops_prefix_cholesky(d, dv, C):
    """Forward FLOPs per token for study 0001's carried-inverse chunk solver without gates (for scale):
    P K^T 2Cd^2, A = I + K P K^T C^2 d, Cholesky C^3/3, U C^2 d, X = QP - tril(QU)U^T 2Cd^2 + 2C^2 d,
    P - U U^T Cd^2, plus the same state and readout terms."""
    per_chunk = 5 * C * d * d + 4 * C * C * d + C ** 3 / 3
    return per_chunk / C + 2 * d * dv + 2 * d * dv + C * d + C * dv


# --------------------------------------------------------------------------- inputs and results
def make_inputs(rng, T, d, dv, gate_lo, gate_hi, lam_lo=0.05, lam_hi=1.0, key_alpha=0.0, beta_hi=1.0, dtype=np.float64,
                rotate=False):
    """Random inputs. key_alpha > 0 gives keys a decaying spectrum (scale i^-alpha), as in study 0001's drift test.
    That spectrum is aligned with the coordinate axes, which favors a diagonal (Jacobi) preconditioner; rotate=True
    applies a random orthogonal rotation to the keys so it is not. Draws for rotate=True come after all others, so
    the default draws are unchanged."""
    q = rng.standard_normal((T, d))
    scale = np.arange(1, d + 1) ** (-key_alpha)
    k = rng.standard_normal((T, d)) * scale / np.sqrt(d)
    v = rng.standard_normal((T, dv))
    beta = rng.uniform(0.1, beta_hi, T)
    loggam = np.log(rng.uniform(gate_lo, gate_hi, T))
    lam = np.exp(rng.uniform(np.log(lam_lo), np.log(lam_hi), d))
    if rotate:
        Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
        k = k @ Qr
    return [a.astype(dtype) for a in (q, k, v, beta, loggam, lam)]


def rel(a, b):
    """Max-abs error relative to the max-abs reference value."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def save_results(data, name):
    """Like lab.results.save_results, without the torch dependency: results/<name>_<timestamp>.json."""
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(),
           "numpy": np.__version__, "machine": platform.machine(), "processor": platform.processor() or "cpu",
           "git_commit": _git("rev-parse", "--short", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}
    with open(path, "w") as f:
        json.dump({"env": env, **data}, f, indent=1)
    return path
