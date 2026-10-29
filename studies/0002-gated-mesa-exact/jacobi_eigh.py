"""Batched cyclic Jacobi eigendecomposition of small symmetric matrices in Triton, one program per matrix.

PyTorch's CUDA eigh runs one matrix at a time for n > 32 (logs/gpu_bench_v2.log, part C), which dominates the exact
chunk solver's time. This kernel diagonalizes every matrix of a batch at once with the classical two-sided Jacobi
method in parallel (round-robin) order: each sweep has n - 1 rounds, and each round applies n/2 rotations on disjoint
index pairs, so they commute and can be applied together (rows, then columns, then the eigenvector columns).

For a pair (p, q) with entries a_pp, a_qq, a_pq the rotation zeroes a_pq (Golub & Van Loan, Algorithm 8.5.2):
    theta = (a_qq - a_pp) / (2 a_pq),  t = sign(theta) / (|theta| + sqrt(theta^2 + 1)),  c = 1 / sqrt(t^2 + 1),  s = t c,
    rows:    A_p <- c A_p - s A_q,  A_q <- s A_p + c A_q;   columns and V: the same on columns.
The matrix and its eigenvectors live in global memory scratch, which stays in the SM's caches at n <= 128; a CTA
barrier separates the phases of a round. A rotation is skipped when |a_pq| <= eps sqrt(|a_pp a_qq|), the classical
threshold, or when |a_pq| <= eps max|A|, the round-off level that rotations of other pairs leave behind (without this floor,
pairs of small eigenvalues never pass the relative test and every sweep keeps rotating). After each rotation a_pq
and a_qp are set to exactly zero, as in the classical algorithm, and a matrix stops after the first sweep that rotates nothing, or after max_sweeps. Without the threshold,
sweeps past convergence slowly erode orthogonality in float32. check_jacobi.py measures accuracy and the sweeps used.
Eigenvalues come out unsorted.
n must be a power of two (64 and 128 here).
"""
import torch
import triton
import triton.language as tl


def round_robin_pairs(n):
    """(n - 1, n/2) index tables P, Q: in each round every index appears exactly once (circle method)."""
    idx = list(range(n))
    P, Q = [], []
    for _ in range(n - 1):
        P.append([idx[i] for i in range(n // 2)])
        Q.append([idx[n - 1 - i] for i in range(n // 2)])
        idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return torch.tensor(P, dtype=torch.int32), torch.tensor(Q, dtype=torch.int32)


@triton.jit
def _jacobi_kernel(A, V, P, Q, used, max_sweeps, eps, N: tl.constexpr, H: tl.constexpr):
    b = tl.program_id(0)
    A = A + b.to(tl.int64) * N * N
    V = V + b.to(tl.int64) * N * N
    cols = tl.arange(0, N)
    half = tl.arange(0, H)
    floor = eps * tl.max(tl.max(tl.abs(tl.load(A + cols[:, None] * N + cols[None, :])), 1), 0)
    sweep = 0
    active = 1
    while (sweep < max_sweeps) & (active > 0):
        active = 0
        for r in range(N - 1):
            p = tl.load(P + r * H + half)
            q = tl.load(Q + r * H + half)
            app = tl.load(A + p * N + p)
            aqq = tl.load(A + q * N + q)
            apq = tl.load(A + p * N + q)
            nz = (tl.abs(apq) > eps * tl.sqrt(tl.abs(app * aqq))) & (tl.abs(apq) > floor)
            active = tl.maximum(active, tl.max(nz.to(tl.int32), 0))
            theta = (aqq - app) / (2.0 * tl.where(nz, apq, 1.0))
            sgn = tl.where(theta >= 0.0, 1.0, -1.0)
            t = sgn / (tl.abs(theta) + tl.sqrt(theta * theta + 1.0))
            t = tl.where(nz, t, 0.0)
            c = 1.0 / tl.sqrt(t * t + 1.0)
            s = t * c
            c2 = c[:, None]
            s2 = s[:, None]
            tl.debug_barrier()
            # rows p and q of A
            rp = A + p[:, None] * N + cols[None, :]
            rq = A + q[:, None] * N + cols[None, :]
            Ap = tl.load(rp)
            Aq = tl.load(rq)
            tl.debug_barrier()
            tl.store(rp, c2 * Ap - s2 * Aq)
            tl.store(rq, s2 * Ap + c2 * Aq)
            tl.debug_barrier()
            # columns p and q of A (as transposed tiles: element [j, k] = A[k, p_j])
            cp = A + cols[None, :] * N + p[:, None]
            cq = A + cols[None, :] * N + q[:, None]
            Ap = tl.load(cp)
            Aq = tl.load(cq)
            tl.debug_barrier()
            tl.store(cp, c2 * Ap - s2 * Aq)
            tl.store(cq, s2 * Ap + c2 * Aq)
            tl.debug_barrier()
            tl.store(A + p * N + q, tl.where(nz, 0.0, apq))
            tl.store(A + q * N + p, tl.where(nz, 0.0, apq))
            # columns p and q of V
            vp = V + cols[None, :] * N + p[:, None]
            vq = V + cols[None, :] * N + q[:, None]
            Vp = tl.load(vp)
            Vq = tl.load(vq)
            tl.debug_barrier()
            tl.store(vp, c2 * Vp - s2 * Vq)
            tl.store(vq, s2 * Vp + c2 * Vq)
            tl.debug_barrier()
        sweep += 1
    tl.store(used + b, sweep)


_PAIRS = {}


def jacobi_eigh(M, max_sweeps=20, num_warps=8, return_sweeps=False):
    """Eigenvalues (unsorted) and eigenvectors of a batch of symmetric matrices (..., n, n), n a power of two.
    With return_sweeps=True also returns the number of sweeps each matrix used."""
    n = M.shape[-1]
    shape = M.shape
    A = M.reshape(-1, n, n).contiguous().clone()
    V = torch.eye(n, dtype=M.dtype, device=M.device).expand_as(A).contiguous()
    key = (n, M.device)
    if key not in _PAIRS:
        P, Q = round_robin_pairs(n)
        _PAIRS[key] = (P.to(M.device), Q.to(M.device))
    P, Q = _PAIRS[key]
    used = torch.zeros(A.shape[0], dtype=torch.int32, device=M.device)
    eps = torch.finfo(M.dtype).eps
    _jacobi_kernel[(A.shape[0],)](A, V, P, Q, used, max_sweeps, eps, N=n, H=n // 2, num_warps=num_warps)
    h = A.diagonal(dim1=-2, dim2=-1).reshape(shape[:-1])
    return (h, V.reshape(shape), used.reshape(shape[:-2])) if return_sweeps else (h, V.reshape(shape))
