"""Check the gain form and the inverse-carrying chunk update against per-token solves (numpy)."""
import numpy as np
rng = np.random.default_rng(0)
d, T, lam = 32, 512, 0.25

def keys(n):
    x = rng.standard_normal((n, d)) * (np.arange(1, d + 1) ** -0.5)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

K = keys(T) * np.sqrt(0.2 + 0.8 * rng.random((T, 1)))
Q = keys(T)

# reference: x_t = (lam I + sum_{i<=t} k_i k_i^T)^{-1} q_t
ref = np.zeros((T, d)); H = lam * np.eye(d)
for t in range(T):
    H += np.outer(K[t], K[t]); ref[t] = np.linalg.solve(H, Q[t])

for C in (16, 64):
    for dtype in (np.float64, np.float32):
        P = (np.eye(d) / lam).astype(dtype); X = np.zeros((T, d), dtype)
        gains_ok = True
        for c in range(0, T, C):
            Kc, Qc = K[c:c+C].astype(dtype), Q[c:c+C].astype(dtype)
            KP = Kc @ P
            A = np.eye(C, dtype=dtype) + KP @ Kc.T
            L = np.linalg.cholesky(A)
            Ut = np.linalg.solve(L, KP)                    # U^T = L^{-1} K P
            X[c:c+C] = Qc @ P - np.tril(Qc @ Ut.T) @ Ut
            if dtype == np.float64 and c == 0:             # columns of U = normalized RLS gains
                Pp = P.copy()
                for i in range(C):
                    g = Pp @ Kc[i]; s = np.sqrt(1 + Kc[i] @ g)
                    gains_ok &= np.allclose(Ut[i], g / s) and np.isclose(L[i, i], s)
                    Pp = Pp - np.outer(g, g) / s**2
            P = P - Ut.T @ Ut
        err = np.max(np.linalg.norm(X - ref, axis=1) / np.linalg.norm(ref, axis=1))
        print(f"C={C:3d} {np.dtype(dtype).name:8s}: carried-inverse vs per-token solve, max rel err {err:.1e}"
              + (f" | gains = P_(i-1) k_i / L_ii: {gains_ok}" if dtype == np.float64 else ""))
