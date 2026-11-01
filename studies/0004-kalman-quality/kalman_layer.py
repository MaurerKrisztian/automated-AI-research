"""Study 0004: the Kalman memory in the form that trains, through autograd, for any gates.

The chunk-parallel form of study 0003 (Propositions 2-3, E2) with three changes, none of which changes the function
(check_layers.py verifies it in float64 against the extended-precision token-by-token filter):

- Whitened observations. v_t = W_t k_t + noise of precision beta_t is the same observation as
  sqrt(beta_t) v_t = W_t (sqrt(beta_t) k_t) + unit noise, so the caller scales K and V by s = sqrt(beta) and the C x C
  matrix is I + (noiseless part). Its eigenvalues are >= 1 and a token that writes nothing (s -> 0) needs no 1/beta.
- Safe Grams. Study 0003's per-channel matmul Grams divide by cumulative decays and overflow in float32 once a
  channel's log decay within a chunk falls below about -88 (0003, claim 10). Here relative decays exp(logA_t - logA_i)
  are built in log space before anything is multiplied, for scalar and per-channel gates and for both kinds of noise.
  With per-channel gates that is a C x C x d tensor per chunk, so per-channel layers use a small C.
- on_chunk(Kmat, info) reports every chunk's C x C matrix and Cholesky status, for the diagnostics of cond(K).

Shapes: K, Q (N, T, d) with K already whitened; V (N, T, e), already whitened; logD (N, T, g) with g = 1 or d, or None
for identity dynamics. Noise: lam (N, d) for the stationary choice Psi_t = (I - D_t^2) Lambda^-1, P_init = Lambda^-1, or
psi (N, T) for free isotropic noise Psi_t = psi_t I with P_init (N, d, d). Returns O (N, T, e), M_T (N, e, d),
P_T (N, d, d).
"""
import math

import torch
import torch.nn.functional as F

solve_triangular = torch.linalg.solve_triangular


def pad_inputs(K, V, Q, logD, psi, C):
    """Pad T to a multiple of C. A padded token has k = q = v = 0, gate 1 and no noise: it neither informs nor moves
    the state (0003, Implementation)."""
    pad = (-K.shape[1]) % C
    if pad:
        K, V, Q = (F.pad(x, (0, 0, 0, pad)) for x in (K, V, Q))
        logD = None if logD is None else F.pad(logD, (0, 0, 0, pad))
        psi = None if psi is None else F.pad(psi, (0, pad))
    return K, V, Q, logD, psi


def chunk_prep(K, Q, logD, C, lam, psi):
    """The parts of the chunk matrices that do not depend on the carried state, for all chunks at once, with relative
    decays in log space: Kmat0 = I + in-chunk noise Gram, XN[i, t] = q_t^T A_{t/i} phi_i k_i (used for i <= t),
    Zn[i] = A_{C/i} phi_i k_i, AC = A_C, phiC = phi_C, KA = K A, KQA = [K A; Q A]."""
    N, T, d = K.shape
    nc = T // C
    Kc, Qc = K.reshape(N, nc, C, d), Q.reshape(N, nc, C, d)
    g = 1 if logD is None else logD.shape[-1]
    logA = torch.zeros(N, nc, C, 1, dtype=K.dtype, device=K.device) if logD is None else logD.reshape(N, nc, C, g).cumsum(2)
    if lam is not None:
        f = -torch.expm1(2 * logA)                                    # (1 - A_m^2): phi_m = f_m w with w = 1 / lambda
        w = (1 / lam)[:, None, None, :]
    else:
        D2 = torch.ones_like(logA) if logD is None else torch.exp(2 * logD.reshape(N, nc, C, g))
        ps, acc, out = psi.reshape(N, nc, C, 1), torch.zeros_like(logA[:, :, 0]), []
        for m in range(C):                                            # phi_m = D_m^2 phi_{m-1} + psi_m, within the chunk
            acc = D2[:, :, m] * acc + ps[:, :, m]
            out.append(acc)
        f = torch.stack(out, 2)
        w = torch.ones(N, 1, 1, 1, dtype=K.dtype, device=K.device)
    A = torch.exp(logA)
    phi = f * w                                                       # (N, nc, C, d) or (N, nc, C, 1)
    eye = torch.eye(C, dtype=K.dtype, device=K.device)
    if g == 1:                                                        # phi_m = f_m w factors into token and channel
        la, fs = logA[..., 0], f[..., 0]
        E = torch.exp(-(la[..., :, None] - la[..., None, :]).abs())   # A_{m'/m}, with m = min(i, j), m' = max(i, j)
        S = E * (fs[..., None, :].expand_as(E).tril() + fs[..., :, None].expand_as(E).triu(1))  # A_{m'/m} f_m: symmetric
        Kw = Kc * w
        Kmat0 = (Kw @ Kc.mT) * S + eye
        XN = (Kw @ Qc.mT) * S
    else:                                                             # one relative decay per pair and channel
        E = torch.exp((logA[..., :, None, :] - logA[..., None, :, :]).clamp_max(0.0))     # [t, i, c]: A_{t/i}, t >= i
        H = E * (phi * Kc)[..., None, :, :]                           # [t, i, c]: A_{t/i} phi_i k_i
        low = (Kc[..., :, None, :] * H).sum(-1)                       # [t, i], valid for t >= i
        Kmat0 = low.tril() + low.tril(-1).mT + eye
        XN = (Qc[..., :, None, :] * H).sum(-1).mT                     # [i, t], valid for i <= t
    Zn = Kc * torch.exp(logA[:, :, -1:] - logA) * phi                 # rows A_{C/i} phi_i k_i
    KA, QA = Kc * A, Qc * A
    return dict(KA=KA, KQA=torch.cat([KA, QA], 2), Kmat0=Kmat0, XN=XN, Zn=Zn, AC=A[:, :, -1],
                phiC=phi[:, :, -1].expand(N, nc, d))


def kalman_chunk(K, V, Q, logD, C, lam=None, psi=None, P_init=None, on_chunk=None):
    """Every readout and the final posterior from one C x C Cholesky per chunk (0003, Propositions 2-3), for whitened
    K and V. Differentiable by autograd."""
    if (lam is None) == (psi is None):
        raise ValueError("give either lam (stationary noise) or psi with P_init (free noise)")
    T, e = K.shape[1], V.shape[-1]
    C = math.ceil(T / math.ceil(T / C))                               # the same number of chunks, with the least padding
    P = torch.diag_embed(1 / lam) if lam is not None else P_init
    M = K.new_zeros(K.shape[0], e, K.shape[-1])
    K, V, Q, logD, psi = pad_inputs(K, V, Q, logD, psi, C)
    N, Tp, d = K.shape
    p = chunk_prep(K, Q, logD, C, lam, psi)
    Vc = V.reshape(N, Tp // C, C, e)
    outs = []
    for n in range(Tp // C):
        KQA, AC = p["KQA"][:, n], p["AC"][:, n]                       # rows: A K, then A Q
        KP = p["KA"][:, n] @ P                                        # (A K) P_0
        KPKQ = KP @ KQA.mT                                            # (A K) P_0 [A K; A Q]^T
        MKQ = KQA @ M.mT                                              # [A K; A Q] M_0^T
        Kmat = KPKQ[..., :C] + p["Kmat0"][:, n]
        X = (KPKQ[..., C:] + p["XN"][:, n]).triu()                    # X[i, t] = kappa_ti, i <= t
        R = Vc[:, n] - MKQ[:, :C]                                     # residuals r_i
        Zt = KP * AC[:, None, :] + p["Zn"][:, n]                      # rows z_i^T
        L, info = torch.linalg.cholesky_ex(Kmat)
        if on_chunk:
            on_chunk(Kmat, info)
        Bt = solve_triangular(L, torch.cat([R, X, Zt], -1), upper=False)
        Rt, Y, Zz = Bt.split([e, C, d], -1)
        outs.append(torch.baddbmm(MKQ[:, C:], Y.mT.tril(), Rt))
        M = torch.baddbmm(M * AC[:, None, :], Rt.mT, Zz)
        P = torch.baddbmm(AC[:, :, None] * P * AC[:, None, :] + torch.diag_embed(p["phiC"][:, n]), Zz.mT, Zz, alpha=-1)
    return torch.cat(outs, 1)[:, :T], M, P


def kalman_filter(K, V, Q, logD, lam=None, psi=None, P_init=None):
    """The token-by-token filter for the same whitened inputs (0003, Proposition 1b), the direct reference."""
    N, T, d = K.shape
    P = torch.diag_embed(1 / lam) if lam is not None else P_init
    M = K.new_zeros(N, V.shape[-1], d)
    out = []
    for t in range(T):
        D = None if logD is None else torch.exp(logD[:, t])
        if D is not None:
            M = M * D[:, None, :]
            P = D[:, :, None] * P * D[:, None, :]
        noise = (-torch.expm1(2 * logD[:, t]) / lam if logD is not None else 0 * lam) if lam is not None else psi[:, t, None]
        P = P + torch.diag_embed(noise.expand(N, d))
        k = K[:, t]
        Pk = (P @ k[..., None])[..., 0]
        s = ((k * Pk).sum(-1) + 1)[:, None]
        M = M + (V[:, t] - (M @ k[..., None])[..., 0])[..., None] * (Pk / s)[:, None, :]
        P = P - Pk[:, :, None] * Pk[:, None, :] / s[..., None]
        out.append((M @ Q[:, t, :, None])[..., 0])
    return torch.stack(out, 1), M, P
