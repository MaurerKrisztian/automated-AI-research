"""Study 0003: the chunkwise Kalman memory (README, Propositions 1-3 and 6) and its token-by-token reference.

One head. Keys and queries K, Q: (T, d); values V: (T, e); write strengths beta: (T,); log gates logD: (T, d),
with a scalar gate written as a row of equal entries; diagonal process-noise variances Psi: (T, d). The memory
W_t (e x d) follows W_t = W_{t-1} D_t + E_t with the rows of E_t ~ N(0, diag(Psi_t)); v_t = W_t k_t + N(0, I/beta_t);
the readout is o_t = E[W_t | v_1..v_t] q_t. The posterior is MN(M_t, P_t): mean M (e x d), covariance P (d x d).

Torch is used so that the same code gives autograd gradients; the numpy version of the reference runs in extended
precision (np.longdouble) and needs no linear solves, so it can judge float64 results on badly conditioned inputs.
"""
import numpy as np
import torch


def stationary_noise(logD, lam):
    """Psi_t = (I - D_t^2) Lambda^-1 for a diagonal ridge lam (d,): the prior stays N(0, Lambda^-1) (Proposition 1a)."""
    return -torch.expm1(2 * logD) / lam


def kalman_reference(K, V, Q, logD, Psi, beta, M0, P0):
    """Token-by-token Kalman filter (Proposition 1b), in the inputs' dtype. Returns outputs (T, e), M_T, P_T."""
    M, P, out = M0, P0, []
    for t in range(K.shape[0]):
        D = torch.exp(logD[t])
        M = M * D
        P = D[:, None] * P * D[None, :] + torch.diag(Psi[t])
        Pk = P @ K[t]
        s = K[t] @ Pk + 1 / beta[t]
        M = M + torch.outer(V[t] - M @ K[t], Pk / s)
        P = P - torch.outer(Pk, Pk) / s
        out.append(M @ Q[t])
    return torch.stack(out), M, P


def kalman_reference_np(K, V, Q, logD, Psi, beta, M0, P0, dtype=np.longdouble):
    """The same filter in numpy at the given precision (default: x87 extended, eps 1.1e-19)."""
    K, V, Q, logD, Psi, beta, M, P = (np.asarray(np.asarray(_np(a), dtype=np.float64), dtype=dtype)
                                      for a in (K, V, Q, logD, Psi, beta, M0, P0))
    out = np.empty((K.shape[0], V.shape[1]), dtype=dtype)
    for t in range(K.shape[0]):
        D = np.exp(logD[t])
        M = M * D
        P = D[:, None] * P * D[None, :]
        P[np.diag_indices_from(P)] += Psi[t]
        Pk = P @ K[t]
        s = K[t] @ Pk + 1 / beta[t]
        M = M + np.outer(V[t] - M @ K[t], Pk / s)
        P = P - np.outer(Pk, Pk) / s
        out[t] = M @ Q[t]
    return out, M, P


def noise_variance(logD, Psi):
    """phi_m = D_m^2 phi_{m-1} + Psi_m with phi_0 = 0: the in-chunk noise variance at token m (rows m = 1..C)."""
    D2, acc, out = torch.exp(2 * logD), torch.zeros_like(Psi[0]), []
    for m in range(Psi.shape[0]):
        acc = D2[m] * acc + Psi[m]
        out.append(acc)
    return torch.stack(out)


def chunk_matrices(K, V, Q, logD, Psi, beta, M0, P0, gram="logspace"):
    """The chunk's covariance Kmat (C x C), cross terms X (C x C, X[i, t] = kappa_ti for i <= t, else 0), residuals R,
    rows of Z, and the quantities the boundary update needs (Propositions 2-3).

    gram="logspace" forms the in-chunk noise Grams from pairwise decays exp(log A_max - log A_min), which never
    overflow. gram="matmul" uses GLA-style scaled keys (K * A) and (K * phi / A), which divide by cumulative decays
    and overflow in float32 for fast per-channel forgetting (Proposition 4a).
    """
    C = K.shape[0]
    logA = torch.cumsum(logD, 0)
    A = torch.exp(logA)
    phi = noise_variance(logD, Psi)
    KA, QA = K * A, Q * A
    idx = torch.arange(C)
    lo, hi = torch.minimum(idx[:, None], idx[None, :]), torch.maximum(idx[:, None], idx[None, :])
    if gram == "logspace":
        ratio = torch.exp(logA[hi] - logA[lo])                               # A_{max/min}: (C, C, d), entries <= 1
        Nkk = torch.einsum('ic,jc,ijc->ij', K, K, ratio * phi[lo])
        Nqk = torch.einsum('tc,ic,tic->ti', Q, K, ratio * phi[None, :, :])   # used for i <= t
    elif gram == "matmul":
        low = idx[:, None] >= idx[None, :]
        Nlow = KA @ (K * phi / A).T                                          # [i, j], i >= j
        Nkk = torch.where(low, Nlow, Nlow.T)
        Nqk = QA @ (K * phi / A).T                                           # [t, i], i <= t
    else:
        raise ValueError(gram)
    KP = KA @ P0
    Kmat = KP @ KA.T + Nkk + torch.diag(1 / beta)
    upper = idx[:, None] <= idx[None, :]                                     # [i, t]: i <= t
    X = torch.where(upper, (QA @ P0 @ KA.T).T + Nqk.T, torch.zeros((), dtype=K.dtype))
    R = V - KA @ M0.T
    AC = A[-1]
    Zt = KP * AC + K * torch.exp(logA[-1] - logA) * phi                      # rows z_i^T
    return dict(Kmat=Kmat, X=X, R=R, Zt=Zt, QA=QA, AC=AC, phiC=phi[-1])


def chunk_step(K, V, Q, logD, Psi, beta, M0, P0, inverse="solve", gram="logspace", cc_dtype=None, kmat_dtype=None):
    """One chunk from one Cholesky (Propositions 2-3). inverse="solve" uses triangular solves; inverse="explicit"
    forms L^-1 first, so every step that scales with d or e is a matrix product. cc_dtype (e.g. torch.float64) runs
    only the C x C stage, the Cholesky and the solves, at that precision. kmat_dtype also forms Kmat itself at that
    precision, from the same inputs and state; everything else stays at the inputs' precision. Returns outputs
    (C, e), M_C, P_C."""
    m = chunk_matrices(K, V, Q, logD, Psi, beta, M0, P0, gram)
    C, e = K.shape[0], V.shape[1]
    if kmat_dtype is not None:
        up = [x.to(kmat_dtype) for x in (K, V, Q, logD, Psi, beta, M0, P0)]
        m["Kmat"] = chunk_matrices(*up, gram)["Kmat"]
        cc_dtype = cc_dtype or kmat_dtype
    dt = cc_dtype or K.dtype
    L = torch.linalg.cholesky(m["Kmat"].to(dt))
    B = torch.cat([m["R"], m["X"], m["Zt"]], dim=1).to(dt)
    if inverse == "solve":
        Bt = torch.linalg.solve_triangular(L, B, upper=False)
    elif inverse == "explicit":
        Bt = torch.linalg.solve_triangular(L, torch.eye(C, dtype=dt), upper=False) @ B
    else:
        raise ValueError(inverse)
    Bt = Bt.to(K.dtype)
    Rt, Y, Ztt = Bt[:, :e], Bt[:, e:e + C], Bt[:, e + C:]
    O = m["QA"] @ M0.T + torch.tril(Y.T) @ Rt
    MC = M0 * m["AC"] + Rt.T @ Ztt
    PC = m["AC"][:, None] * P0 * m["AC"][None, :] + torch.diag(m["phiC"]) - Ztt.T @ Ztt
    return O, MC, PC


def chunk_prefix(K, V, Q, logD, Psi, beta, M0, P0):
    """Proposition 2's first formula taken literally: a separate solve with the leading t x t block for each token."""
    m = chunk_matrices(K, V, Q, logD, Psi, beta, M0, P0)
    out = []
    for t in range(K.shape[0]):
        w = torch.linalg.solve(m["Kmat"][:t + 1, :t + 1], m["X"][:t + 1, t])
        out.append(M0 @ m["QA"][t] + m["R"][:t + 1].T @ w)
    return torch.stack(out)


def chunked(K, V, Q, logD, Psi, beta, M0, P0, C, **kw):
    """Run chunk_step over a sequence in chunks of C tokens (the last chunk may be shorter)."""
    outs, M, P = [], M0, P0
    for s in range(0, K.shape[0], C):
        sl = slice(s, s + C)
        O, M, P = chunk_step(K[sl], V[sl], Q[sl], logD[sl], Psi[sl], beta[sl], M, P, **kw)
        outs.append(O)
    return torch.cat(outs), M, P


def max_cond(K, V, Q, logD, Psi, beta, M0, P0, C):
    """Largest condition number of the chunk covariances along the sequence (float64 eigenvalues)."""
    worst, M, P = 0.0, M0, P0
    for s in range(0, K.shape[0], C):
        sl = slice(s, s + C)
        args = (K[sl], V[sl], Q[sl], logD[sl], Psi[sl], beta[sl], M, P)
        ev = torch.linalg.eigvalsh(chunk_matrices(*args)["Kmat"].double())
        worst = max(worst, float(ev[-1] / ev[0]))
        _, M, P = chunk_step(*args)
    return worst


def mesa_readout(K, V, Q, beta, lam):
    """Ungated Mesa readout o_t = G_t (H_t + Lambda)^-1 q_t by a dense solve per token (study 0001's layer)."""
    d, e = K.shape[1], V.shape[1]
    H, G, out = torch.zeros(d, d, dtype=K.dtype), torch.zeros(e, d, dtype=K.dtype), []
    for t in range(K.shape[0]):
        H = H + beta[t] * torch.outer(K[t], K[t])
        G = G + beta[t] * torch.outer(V[t], K[t])
        out.append(G @ torch.linalg.solve(H + torch.diag(lam), Q[t]))
    return torch.stack(out)


def chunked_prefix(K, V, Q, logD, Psi, beta, M0, P0, C):
    """chunk_prefix over a sequence, with the boundary states from chunk_step."""
    outs, M, P = [], M0, P0
    for s in range(0, K.shape[0], C):
        sl = slice(s, s + C)
        args = (K[sl], V[sl], Q[sl], logD[sl], Psi[sl], beta[sl], M, P)
        outs.append(chunk_prefix(*args))
        _, M, P = chunk_step(*args)
    return torch.cat(outs)


def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a


def rel(a, b):
    """Normwise relative error max|a - b| / max|b|, computed in extended precision."""
    a, b = np.asarray(_np(a), dtype=np.longdouble), np.asarray(_np(b), dtype=np.longdouble)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
