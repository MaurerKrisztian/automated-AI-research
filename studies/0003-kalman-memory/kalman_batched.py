"""Study 0003: the Kalman memory batched over heads, for the GPU benchmark (E3).

The same model and notation as kalman_memory.py (README, Method), for N independent heads at once and on any device.
Keys and queries K, Q: (N, T, d); values V: (N, T, e); write strengths beta: (N, T); log gates logD: (N, T, g),
with g = d for per-channel gates and g = 1 for a scalar gate. The noise is either the stationary choice, given by
the diagonal ridge lam: (N, d) (Psi_t = (I - D_t^2) Lambda^-1 and P_init = Lambda^-1, Proposition 1a), or free
diagonal noise Psi: (N, T, d) with P_init: (N, d, d). M_init: (N, e, d) defaults to zero. Every function returns the
readouts O: (N, T, e) and the final posterior M_T: (N, e, d), P_T: (N, d, d).

Three algorithms for the same filter:
- kalman_chunkwise: the chunk-parallel form of this study (Propositions 2-3). One C x C Cholesky per chunk and head;
  the chunk's d- and e-dependent work is matrix products, plus triangular solves (inverse="solve") or a product with
  an explicit L^-1 (inverse="explicit").
- bl_chunkwise: the Bayesian Layer's chunk algorithm (Dowling et al. 2026, arXiv 2605.31163; v5 Appendix C.2,
  eqs. 23-27, and v1 Appendix D, as read in this study: README, Cost, and NOTEBOOK.md). The entry-covariance
  products Pi_t k_t come from one matrix product; a sequential loop over the chunk's positions then computes each
  update direction u_t = P_t^- k_t and innovation variance s_t from the earlier directions, transported by the decays;
  the exit covariance is P_C = Pi_C - W diag(1/s) W^T (BL eq. 27). The mean is then carried token by token (v5,
  mean="loop") or computed per chunk in the WY form once the gains are known (v1, mean="wy"). gains="triton" and
  mean="triton" run the two sequential loops as Triton kernels (bl_triton.py), as BL's own implementation does.
- kalman_recurrent: the token-by-token filter (Proposition 1b), one step per token.

Everything is plain PyTorch and differentiable by autograd, except the Triton kernels (forward only). The chunk forms
pad T to a multiple of C with k = q = v = 0, gates 1, no noise and beta = 1: such a token neither informs nor moves the
state, so the outputs and final state are exact. check_batched.py checks every form in float64 against
kalman_memory.kalman_reference_np in extended precision.
"""
import torch
import torch.nn.functional as F

solve_triangular = torch.linalg.solve_triangular


def _pad(K, V, Q, logD, beta, Psi, C):
    pad = (-K.shape[1]) % C
    if pad:
        K, V, Q, logD = (F.pad(x, (0, 0, 0, pad)) for x in (K, V, Q, logD))
        beta = F.pad(beta, (0, pad), value=1.0)
        Psi = None if Psi is None else F.pad(Psi, (0, 0, 0, pad))
    return K, V, Q, logD, beta, Psi


def _state(K, V, lam, P_init, M_init):
    """Initial posterior: P_init defaults to Lambda^-1 (stationary choice), M_init to zero."""
    if (lam is None) == (P_init is None):
        raise ValueError("give either lam (stationary noise) or Psi with P_init (free noise)")
    P = torch.diag_embed(1 / lam) if P_init is None else P_init
    M = K.new_zeros(K.shape[0], V.shape[-1], K.shape[-1]) if M_init is None else M_init
    return M, P


def _noise_variance(logDc, Psic):
    """phi_m = D_m^2 phi_{m-1} + Psi_m within each chunk (phi_0 = 0), all chunks at once: (N, nc, C, d)."""
    D2 = torch.exp(2 * logDc)
    acc, out = torch.zeros_like(Psic[:, :, 0]), []
    for m in range(Psic.shape[2]):
        acc = D2[:, :, m] * acc + Psic[:, :, m]
        out.append(acc)
    return torch.stack(out, 2)


def _chunk_logs(K, logD, C, lam, Psi):
    """Chunked keys, log cumulative gates log A_i (N, nc, C, g) and in-chunk noise variances phi_i (N, nc, C, d)."""
    N, T, d = K.shape
    nc = T // C
    logA = logD.reshape(N, nc, C, -1).cumsum(2)
    if lam is not None:
        phi = -torch.expm1(2 * logA) / lam[:, None, None, :]                     # (I - A_i^2) Lambda^-1
    else:
        phi = _noise_variance(logD.reshape(N, nc, C, -1), Psi.reshape(N, nc, C, d))
    return K.reshape(N, nc, C, d), logA, phi


# --------------------------------------------------------------------------- chunk-parallel form (Propositions 2-3)
def chunk_prep(K, Q, logD, beta, C, lam=None, Psi=None, gram="auto"):
    """The parts of the chunk matrices that do not depend on the carried state, for all chunks at once.

    Kmat0 = in-chunk noise Gram + diag(1/beta) (C x C), XN[i, t] = q_t^T A_{t/i} phi_i k_i (used for i <= t),
    Zn[i] = A_{C/i} phi_i k_i, AC = A_C, phiC = phi_C. gram="hadamard" (scalar gates with stationary noise) forms
    the noise Grams as (K Lambda^-1 K^T) times a C x C matrix of decays built in log space, which never overflows.
    gram="matmul" is the GLA-style product (K A)(K phi / A)^T, for any diagonal gates and noise; it divides by
    cumulative decays and overflows in float32 once a channel's log decay within a chunk falls below about -88
    (Proposition 4a). "auto" picks hadamard when it applies.
    """
    N, T, d = K.shape
    nc = T // C
    Kc, logA, phi = _chunk_logs(K, logD, C, lam, Psi)
    Qc = Q.reshape(N, nc, C, d)
    A = torch.exp(logA)
    if gram == "auto":
        gram = "hadamard" if logD.shape[-1] == 1 and lam is not None else "matmul"
    KA, QA = Kc * A, Qc * A
    if gram == "hadamard":
        if logD.shape[-1] != 1 or lam is None:
            raise ValueError("the Hadamard form needs a scalar gate and stationary noise")
        la = logA[..., 0]                                     # nonincreasing along the chunk
        hi = torch.maximum(la[..., :, None], la[..., None, :])  # log A at the earlier of the two tokens
        lo = torch.minimum(la[..., :, None], la[..., None, :])  # ... at the later one
        S = torch.exp(lo - hi) * -torch.expm1(2 * hi)           # A_{m'/m} (1 - A_m^2): symmetric, in [0, 1]
        Kl = Kc / lam[:, None, None, :]
        Kmat0 = (Kl @ Kc.mT) * S
        XN = (Kl @ Qc.mT) * S
    elif gram == "matmul":
        B = Kc * phi * torch.exp(-logA)                       # K phi / A
        low = KA @ B.mT                                       # [i, j], valid for i >= j
        Kmat0 = low.tril() + low.tril(-1).mT
        XN = B @ QA.mT                                        # [i, t], valid for i <= t
    else:
        raise ValueError(gram)
    Kmat0 = Kmat0 + torch.diag_embed(1 / beta.reshape(N, nc, C))
    Zn = Kc * torch.exp(logA[:, :, -1:] - logA) * phi          # A_{C/i} phi_i k_i, decays <= 1
    return dict(KA=KA, KQA=torch.cat([KA, QA], 2), Kmat0=Kmat0, XN=XN, Zn=Zn, AC=A[:, :, -1], phiC=phi[:, :, -1])


def kalman_chunkwise(K, V, Q, logD, beta, C, lam=None, Psi=None, P_init=None, M_init=None, inverse="solve",
                     gram="auto", mark=None, on_state=None):
    """The chunk-parallel form: per chunk, one Cholesky Kmat = L L^T and L^-1 [R, X, Z^T] give all C readouts
    O = (Q A) M_0^T + tril(Y^T) R~ (Proposition 2) and the next state M_C = M_0 A_C + R~^T Z~,
    P_C = A_C P_0 A_C + phi_C - Z~^T Z~ (Proposition 3). `mark(stage)`, if given, is called after each stage
    (for timing with CUDA events); `on_state(P)` after each chunk, with the boundary covariance."""
    T, e = K.shape[1], V.shape[-1]
    M, P = _state(K, V, lam, P_init, M_init)
    K, V, Q, logD, beta, Psi = _pad(K, V, Q, logD, beta, Psi, C)
    N, Tp, d = K.shape
    p = chunk_prep(K, Q, logD, beta, C, lam, Psi, gram)
    Vc = V.reshape(N, Tp // C, C, e)
    eye = torch.eye(C, dtype=K.dtype, device=K.device)
    if mark:
        mark("prep")
    outs = []
    for n in range(Tp // C):
        KQA, AC = p["KQA"][:, n], p["AC"][:, n]                        # rows: A K, then A Q
        KP = p["KA"][:, n] @ P                                         # (A K) P_0
        KPKQ = KP @ KQA.mT                                             # (A K) P_0 [A K; A Q]^T
        MKQ = KQA @ M.mT                                               # [A K; A Q] M_0^T
        Kmat = KPKQ[..., :C] + p["Kmat0"][:, n]
        X = (KPKQ[..., C:] + p["XN"][:, n]).triu()                     # X[i, t] = kappa_ti, i <= t
        R = Vc[:, n] - MKQ[:, :C]                                      # residuals r_i
        Zt = KP * AC[:, None, :] + p["Zn"][:, n]                       # rows z_i^T
        if mark:
            mark("matrices")
        L = torch.linalg.cholesky_ex(Kmat).L
        if mark:
            mark("cholesky")
        Bm = torch.cat([R, X, Zt], -1)
        if inverse == "solve":
            Bt = solve_triangular(L, Bm, upper=False)
        elif inverse == "explicit":
            Bt = solve_triangular(L, eye, upper=False) @ Bm
        else:
            raise ValueError(inverse)
        if mark:
            mark("solve")
        Rt, Y, Zz = Bt.split([e, C, d], -1)
        outs.append(torch.baddbmm(MKQ[:, C:], Y.mT.tril(), Rt))
        M = torch.baddbmm(M * AC[:, None, :], Rt.mT, Zz)
        P = torch.baddbmm(AC[:, :, None] * P * AC[:, None, :] + torch.diag_embed(p["phiC"][:, n]), Zz.mT, Zz, alpha=-1)
        if mark:
            mark("update")
        if on_state:
            on_state(P)
    return torch.cat(outs, 1)[:, :T], M, P


# --------------------------------------------------------------------------- the Bayesian Layer's chunk algorithm
def bl_gains_torch(Pik, Kc, Dc, binv, floor):
    """BL's sequential gain loop over one chunk. Pik: Pi_t k_t (N, C, d); Kc: keys; Dc: decays (N, C, g);
    binv: 1/beta (N, C). Step t transports the earlier update directions to token t, w_j <- D_t w_j, and computes
    u_t = Pi_t k_t - sum_{j<t} w_j (w_j^T k_t) / s_j and s_t = max(k_t^T u_t + 1/beta_t, floor).
    Returns the directions transported to the chunk's last token and divided by sqrt(s_j), Ws (N, C, d), so that the
    exit covariance is one Gram matrix; s (N, C); and U = (u_t) (N, C, d)."""
    W, rs, us, ss = Pik[:, :0], binv[:, :0], [], []
    for t in range(Kc.shape[1]):
        k = Kc[:, t]
        u = Pik[:, t]
        if t:
            W = W * Dc[:, t, None, :]
            c = (W @ k[..., None])[..., 0] * rs                        # (w_j^T k_t) / s_j
            u = u - (c[:, None, :] @ W)[:, 0]
        s = ((k * u).sum(-1) + binv[:, t]).clamp_min(floor)
        us.append(u)
        ss.append(s)
        W = torch.cat([W, u[:, None]], 1)
        rs = torch.cat([rs, (1 / s)[:, None]], 1)
    return W * rs.sqrt()[..., None], torch.stack(ss, 1), torch.stack(us, 1)


def bl_mean_torch(K, V, Q, G, Dec, M):
    """BL v5's mean pass, token by token: M <- M D_t, r_t = v_t - M k_t, M <- M + r_t g_t^T, o_t = M q_t."""
    out = []
    for t in range(K.shape[1]):
        M = M * Dec[:, t, None, :]
        r = V[:, t] - (M @ K[:, t, :, None])[..., 0]
        M = M + r[..., None] * G[:, t, None, :]
        out.append((M @ Q[:, t, :, None])[..., 0])
    return torch.stack(out, 1), M


def bl_mean_wy(Kc, Vc, Qc, KQA, logA, G, M):
    """BL v1's mean pass for one chunk once its gains G = (g_t) are known, in the WY form of DeltaNet's chunkwise
    algorithm. With r_t the innovations, M_t = M_0 A_t + sum_{j<=t} r_j (A_{t/j} g_j)^T, so
    (I + Lg) Rinn = V - (K A) M_0^T with Lg[t, j] = g_j^T A_{t/j} k_t (j < t), then O = (Q A) M_0^T + Gq Rinn with
    Gq[t, j] = g_j^T A_{t/j} q_t (j <= t), and M_C = M_0 A_C + Rinn^T (A_{C/j} g_j). One unit-triangular C x C solve.
    KQA stacks A K over A Q, as in kalman_chunkwise."""
    C = Kc.shape[1]
    KA, QA = KQA[:, :C], KQA[:, C:]
    if logA.shape[-1] == 1:                                            # scalar gate: decays in log space
        la = logA[..., 0]
        Dm = torch.exp((la[:, :, None] - la[:, None, :]).tril()).tril()  # A_{t/j}, j <= t
        Lg, Gq = (Kc @ G.mT) * Dm, (Qc @ G.mT) * Dm
    else:                                                              # per-channel: GLA-style, divides by A
        Gs = G * torch.exp(-logA)
        Lg, Gq = KA @ Gs.mT, (QA @ Gs.mT).tril()
    MKQ = KQA @ M.mT
    Rinn = solve_triangular(Lg, Vc - MKQ[:, :C], upper=False, unitriangular=True)
    O = torch.baddbmm(MKQ[:, C:], Gq, Rinn)
    M = torch.baddbmm(M * torch.exp(logA[:, -1])[:, None, :], Rinn.mT, G * torch.exp(logA[:, -1:] - logA))
    return O, M


def bl_chunkwise(K, V, Q, logD, beta, C, lam=None, Psi=None, P_init=None, M_init=None, gains="torch", mean="loop",
                 floor=1e-6, mark=None, on_state=None):
    """The Bayesian Layer's chunk algorithm (see the module docstring). gains: "torch" or "triton"; mean: "loop"
    (v5, torch), "triton" (v5, one Triton kernel over the whole sequence) or "wy" (v1, per chunk). The innovation
    variance is floored at `floor`, as BL does (1e-6; it never binds when beta is bounded). mark and on_state as in
    kalman_chunkwise."""
    T, e = K.shape[1], V.shape[-1]
    M, P = _state(K, V, lam, P_init, M_init)
    K, V, Q, logD, beta, Psi = _pad(K, V, Q, logD, beta, Psi, C)
    N, Tp, d = K.shape
    nc = Tp // C
    Kc, logA, phi = _chunk_logs(K, logD, C, lam, Psi)
    A, Dec = torch.exp(logA), torch.exp(logD)
    KA, PhiK, Decc = Kc * A, phi * Kc, Dec.reshape(N, nc, C, -1)
    binv = (1 / beta).reshape(N, nc, C)
    if gains == "triton" or mean == "triton":
        import bl_triton
    if mean == "wy":
        Vc, Qc = V.reshape(N, nc, C, e), Q.reshape(N, nc, C, d)
        KQA = torch.cat([KA, Qc * A], 2)
    if mark:
        mark("prep")
    Gs, outs = [], []
    for n in range(nc):
        AC = A[:, n, -1]
        Pik = A[:, n] * (KA[:, n] @ P) + PhiK[:, n]                     # Pi_t k_t = A_t P_0 A_t k_t + phi_t k_t
        if mark:
            mark("entry")
        gain_loop = bl_triton.gain_loop if gains == "triton" else bl_gains_torch
        Ws, s, U = gain_loop(Pik, Kc[:, n], Decc[:, n], binv[:, n], floor)
        if mark:
            mark("gains")
        P = torch.baddbmm(AC[:, :, None] * P * AC[:, None, :] + torch.diag_embed(phi[:, n, -1]), Ws.mT, Ws,
                          alpha=-1)                                    # P_C = Pi_C - W diag(1/s) W^T (BL eq. 27)
        G = U / s[..., None]                                           # gains g_t
        if mark:
            mark("exit")
        if on_state:
            on_state(P)
        if mean == "wy":
            O, M = bl_mean_wy(Kc[:, n], Vc[:, n], Qc[:, n], KQA[:, n], logA[:, n], G, M)
            outs.append(O)
            if mark:
                mark("mean")
        else:
            Gs.append(G)
    if mean != "wy":
        G = torch.cat(Gs, 1)
        if mean == "triton":
            O, M = bl_triton.mean_pass(K, V, Q, G, Dec, M)
        elif mean == "loop":
            O, M = bl_mean_torch(K, V, Q, G, Dec, M)
        else:
            raise ValueError(mean)
        outs = [O]
        if mark:
            mark("mean")
    return torch.cat(outs, 1)[:, :T], M, P


# --------------------------------------------------------------------------- token-by-token filter (Proposition 1b)
def kalman_recurrent(K, V, Q, logD, beta, lam=None, Psi=None, P_init=None, M_init=None, on_state=None, state_every=1):
    """The Kalman filter one token at a time, batched over heads: kalman_memory.kalman_reference for N heads.
    `on_state(P)`, if given, is called after every `state_every` tokens."""
    M, P = _state(K, V, lam, P_init, M_init)
    if lam is not None:
        Psi = -torch.expm1(2 * logD) / lam[:, None, :]
    out = []
    for t in range(K.shape[1]):
        D = torch.exp(logD[:, t])
        M = M * D[:, None, :]
        P = D[:, :, None] * P * D[:, None, :] + torch.diag_embed(Psi[:, t])
        k = K[:, t]
        Pk = (P @ k[..., None])[..., 0]
        s = ((k * Pk).sum(-1) + 1 / beta[:, t])[:, None]
        M = M + (V[:, t] - (M @ k[..., None])[..., 0])[..., None] * (Pk / s)[:, None, :]
        P = P - Pk[:, :, None] * Pk[:, None, :] / s[..., None]               # exactly symmetric, as the reference
        out.append((M @ Q[:, t, :, None])[..., 0])
        if on_state and (t + 1) % state_every == 0:
            on_state(P)
    return torch.stack(out, 1), M, P
