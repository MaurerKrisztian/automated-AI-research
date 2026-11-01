"""Triton kernels for the two sequential loops of the Bayesian Layer's chunk algorithm (kalman_batched.bl_chunkwise),
forward only. BL's own implementation runs these loops in Triton (Dowling et al. 2026, v5 Appendix C.3), and its code
is not public, so these kernels are this study's version of that baseline.

- gain_loop: one program per head runs the gain loop over one chunk's positions, keeping the transported update
  directions (C x d) and the inverse innovation variances in registers. Each step costs O(C d) on the full block
  (rows j >= t are zero), so the loop is C dependent steps of width C x d. It returns the directions at the chunk's
  end already scaled by 1/sqrt(s_j), so the exit covariance is one Gram matrix.
- mean_pass: one program per head and block of value rows carries M (BLOCK_E x d) through the whole sequence,
  token by token (BL v5's mean pass), once all gains are known.

Both kernels load the next token's vectors before computing the current step, so the loads overlap the dependent
arithmetic. On a GPU, num_warps is autotuned per shape and number of heads (triton.autotune over 2, 4, 8 and 16 warps;
n_heads is passed only to key the tuning, since the grid size changes how many programs share an SM); the tuning runs
on the first call, which the benchmark leaves out of its timings. Both take float32 or float64 tensors on a CUDA device,
or CPU tensors when TRITON_INTERPRET=1 is set before triton is imported (the interpreter that check_batched.py uses
to test them without a GPU). The loop bounds C and T are compile-time constants: one compilation per chunk size and
sequence length, and the interpreter needs them as ints.
"""
import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _gain_kernel(pik, k, dec, binv, u, ws, s, d, sdn, sdc, sdd, floor, n_heads,
                 C: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_C)
    cols = tl.arange(0, BLOCK_D)
    cm = cols < d
    base = n * C * d
    W = tl.zeros((BLOCK_C, BLOCK_D), dtype=pik.dtype.element_ty)    # transported directions w_j, rows j < t
    rs = tl.zeros((BLOCK_C,), dtype=pik.dtype.element_ty)           # 1 / s_j
    kt = tl.load(k + base + cols, mask=cm, other=0.0)
    Dt = tl.load(dec + n * sdn + cols * sdd, mask=cm, other=1.0)
    pt = tl.load(pik + base + cols, mask=cm, other=0.0)
    bt = tl.load(binv + n * C)
    for t in range(0, C):
        nm = cm & (t + 1 < C)                                        # prefetch token t + 1
        kn = tl.load(k + base + (t + 1) * d + cols, mask=nm, other=0.0)
        Dn = tl.load(dec + n * sdn + (t + 1) * sdc + cols * sdd, mask=nm, other=1.0)
        pn = tl.load(pik + base + (t + 1) * d + cols, mask=nm, other=0.0)
        bn = tl.load(binv + n * C + t + 1, mask=t + 1 < C, other=1.0)
        W = W * Dt[None, :]
        c = tl.sum(W * kt[None, :], axis=1) * rs
        ut = pt - tl.sum(c[:, None] * W, axis=0)
        st = tl.maximum(tl.sum(kt * ut, axis=0) + bt, floor)
        tl.store(u + base + t * d + cols, ut, mask=cm)
        tl.store(s + n * C + t, st)
        W = tl.where(rows[:, None] == t, ut[None, :], W)
        rs = tl.where(rows == t, 1.0 / st, rs)
        kt, Dt, pt, bt = kn, Dn, pn, bn
    tl.store(ws + base + rows[:, None] * d + cols[None, :], W * tl.sqrt(rs)[:, None],
             mask=(rows[:, None] < C) & cm[None, :])


@triton.jit
def _mean_kernel(k, q, v, g, dec, m_in, m_out, o, d, e, sdn, sdt, sdd, n_heads,
                 T: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    er = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    cols = tl.arange(0, BLOCK_D)
    em, cm = er < e, cols < d
    mask = em[:, None] & cm[None, :]
    moff = n * e * d + er[:, None] * d + cols[None, :]
    M = tl.load(m_in + moff, mask=mask, other=0.0)
    kt = tl.load(k + n * T * d + cols, mask=cm, other=0.0)
    qt = tl.load(q + n * T * d + cols, mask=cm, other=0.0)
    gt = tl.load(g + n * T * d + cols, mask=cm, other=0.0)
    Dt = tl.load(dec + n * sdn + cols * sdd, mask=cm, other=1.0)
    vt = tl.load(v + n * T * e + er, mask=em, other=0.0)
    for t in range(0, T):
        row = (n * T + t + 1) * d                                    # prefetch token t + 1
        nm, nv = cm & (t + 1 < T), em & (t + 1 < T)
        kn = tl.load(k + row + cols, mask=nm, other=0.0)
        qn = tl.load(q + row + cols, mask=nm, other=0.0)
        gn = tl.load(g + row + cols, mask=nm, other=0.0)
        Dn = tl.load(dec + n * sdn + (t + 1) * sdt + cols * sdd, mask=nm, other=1.0)
        vn = tl.load(v + (n * T + t + 1) * e + er, mask=nv, other=0.0)
        M = M * Dt[None, :]
        r = vt - tl.sum(M * kt[None, :], axis=1)
        M = M + r[:, None] * gt[None, :]
        tl.store(o + (n * T + t) * e + er, tl.sum(M * qt[None, :], axis=1), mask=em)
        kt, qt, gt, Dt, vt = kn, qn, gn, Dn, vn
    tl.store(m_out + moff, M, mask=mask)


def _warps(elements):
    return 4 if elements <= 2048 else 8 if elements <= 8192 else 16


def _tuned(kernel, key):
    return triton.autotune(configs=[triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)], key=key)(kernel)


if INTERPRET:
    _gain, _mean = _gain_kernel, _mean_kernel
else:
    _gain = _tuned(_gain_kernel, ["d", "C", "BLOCK_C", "BLOCK_D", "n_heads"])
    _mean = _tuned(_mean_kernel, ["d", "e", "T", "BLOCK_E", "BLOCK_D", "n_heads"])


def tuned_configs():
    """The num_warps the autotuner chose for each kernel and shape so far (empty under the interpreter)."""
    if INTERPRET:
        return {}
    return {f"{name} {key}": str(cfg) for name, fn in (("gain", _gain), ("mean", _mean)) for key, cfg in fn.cache.items()}


def _dec_strides(Dec):
    """Strides of a decay tensor (N, L, g); a scalar gate (g = 1) is read with stride 0 along the channels."""
    return Dec.stride(0), Dec.stride(1), (Dec.stride(2) if Dec.shape[2] > 1 else 0)


def gain_loop(Pik, Kc, Dc, binv, floor=1e-6):
    """Same contract as kalman_batched.bl_gains_torch: returns Ws (N, C, d), the update directions at the chunk's
    end divided by sqrt(s); s (N, C); U = (u_t) (N, C, d)."""
    N, C, d = Kc.shape
    Pik, Kc, Dc, binv = (x.contiguous() for x in (Pik, Kc, Dc, binv))
    U, Ws, s = torch.empty_like(Kc), torch.empty_like(Kc), torch.empty_like(binv)
    BC, BD = triton.next_power_of_2(C), triton.next_power_of_2(d)
    extra = dict(num_warps=_warps(BC * BD)) if INTERPRET else {}
    _gain[(N,)](Pik, Kc, Dc, binv, U, Ws, s, d, *_dec_strides(Dc), floor, N, C=C, BLOCK_C=BC, BLOCK_D=BD, **extra)
    return Ws, s, U


def mean_pass(K, V, Q, G, Dec, M, block_e=16):
    """Same contract as kalman_batched.bl_mean_torch: K, Q, G (N, T, d), V (N, T, e), Dec (N, T, g), M (N, e, d)."""
    N, T, d = K.shape
    e = V.shape[-1]
    K, V, Q, G, Dec, M = (x.contiguous() for x in (K, V, Q, G, Dec, M))
    O, M_out = torch.empty(N, T, e, dtype=K.dtype, device=K.device), torch.empty_like(M)
    BE, BD = min(triton.next_power_of_2(e), block_e), triton.next_power_of_2(d)
    extra = dict(num_warps=_warps(BE * BD)) if INTERPRET else {}
    _mean[(N, triton.cdiv(e, BE))](K, Q, V, G, Dec, M, M_out, O, d, e, *_dec_strides(Dec), N, T=T, BLOCK_E=BE,
                                   BLOCK_D=BD, **extra)
    return O, M_out
