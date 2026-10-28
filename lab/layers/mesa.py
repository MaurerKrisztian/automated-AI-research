"""Exact least-squares (Mesa) memory without forgetting, chunkwise with a carried inverse.

Promoted from studies/0001-prefix-cholesky (mqar_recall.py: _CarrySolve, mesa_fast, mesa_reference).
Evidence E2: matches the token-by-token recurrence to ~1e-15 in float64, and the closed-form backward
matches autograd (tests/test_mesa.py). Speed is E3 on an RTX 5060 Ti; see the study for numbers.

Layer: o_t = G_t (lam I + H_t)^{-1} q_t,  H_t = sum_{i<=t} beta_i k_i k_i^T,  G_t = sum_{i<=t} beta_i v_i k_i^T
(G_t is stored transposed below, as d_k x d_v). Shapes: q, k (B, H, T, d_k), v (B, H, T, d_v),
beta (B, H, T), lam (H,), one ridge per head.
"""
import torch
import torch.nn.functional as F


class CarrySolve(torch.autograd.Function):
    """x_t = (lam I + sum_{i<=t} kh_i kh_i^T)^{-1} q_t for every t, chunked inputs (B, H, N, C, d), lam (H,).

    Forward carries the inverse P across chunks (one C x C Cholesky per chunk, no d x d factorization).
    Backward is closed-form and runs on all chunks at once:
        dq_t = g_t = P_t dx_t,   dk_i = -sum_{t>=i} (g_t x_t^T + x_t g_t^T) k_i,   dlam = -sum_t g_t . x_t
    """

    @staticmethod
    def forward(ctx, Kh, Q, lam):
        B, H, N, C, d = Kh.shape
        eye = torch.eye(d, device=Kh.device, dtype=Kh.dtype)
        eyeC = torch.eye(C, device=Kh.device, dtype=Kh.dtype)
        P = (eye / lam.view(1, H, 1, 1)).expand(B, H, d, d).contiguous()
        Ps, Us, Xs = [], [], []
        for n in range(N):
            Kn, Qn = Kh[:, :, n], Q[:, :, n]
            KP = Kn @ P
            L = torch.linalg.cholesky(KP @ Kn.transpose(-1, -2) + eyeC)
            Ut = torch.linalg.solve_triangular(L, KP, upper=False)          # U^T = L^{-1} K P
            Xs.append(Qn @ P - torch.tril(Qn @ Ut.transpose(-1, -2)) @ Ut)
            Ps.append(P)
            Us.append(Ut)
            P = P - Ut.transpose(-1, -2) @ Ut                                # exact inverse for the next chunk
        X = torch.stack(Xs, dim=2)
        ctx.save_for_backward(Kh, torch.stack(Ps, dim=2), torch.stack(Us, dim=2), X)
        return X

    @staticmethod
    def backward(ctx, dX):
        Kh, Ps, Us, X = ctx.saved_tensors
        G = dX @ Ps - torch.tril(dX @ Us.transpose(-1, -2)) @ Us              # g_t = P_t dx_t, all chunks at once
        T = G.transpose(-1, -2) @ X
        T = T + T.transpose(-1, -2)                                          # per-chunk sum of g x^T + x g^T
        S_next = torch.flip(torch.cumsum(torch.flip(T, [2]), 2), [2]) - T   # sum over later chunks
        dK = (-(Kh @ S_next) - torch.triu(Kh @ X.transpose(-1, -2)) @ G
              - torch.triu(Kh @ G.transpose(-1, -2)) @ X)
        return dK, G, -(G * X).sum(dim=(0, 2, 3, 4))


def mesa_carry(q, k, v, beta, lam, chunk):
    """The Mesa layer: carried-inverse solve + closed-form backward; value path for all chunks at once."""
    B, H, T, dk = k.shape
    C = chunk
    pad = (-T) % C
    if pad:  # zero keys/values/queries after the end change nothing for earlier tokens
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
    N = (T + pad) // C
    Kc, Vc, Qc = (t.reshape(B, H, N, C, t.shape[-1]) for t in (k, v, q))
    b = beta.reshape(B, H, N, C)
    X = CarrySolve.apply(Kc * b.unsqueeze(-1).sqrt(), Qc, lam)
    Gc = (Kc * b.unsqueeze(-1)).transpose(-1, -2) @ Vc        # each chunk's sum of beta k v^T
    G0 = torch.cumsum(Gc, dim=2) - Gc                          # memory at each chunk start
    S = torch.tril((X @ Kc.transpose(-1, -2)) * b.unsqueeze(-2))
    return (X @ G0 + S @ Vc).reshape(B, H, N * C, -1)[:, :, :T]


def mesa_reference(q, k, v, beta, lam):
    """Token-by-token reference: one d x d solve per token. Slow; for tests only."""
    B, H, T, dk = k.shape
    Hs = lam.view(1, H, 1, 1) * torch.eye(dk, device=q.device, dtype=q.dtype).expand(B, H, dk, dk)
    G = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None, None]
        Hs = Hs + bt * kt.unsqueeze(-1) * kt.unsqueeze(-2)
        G = G + bt * kt.unsqueeze(-1) * vt.unsqueeze(-2)
        x = torch.linalg.solve(Hs, q[:, :, t])
        outs.append((x.unsqueeze(-2) @ G).squeeze(-2))
    return torch.stack(outs, dim=2)
