---
id: "0002"
title: Exact chunkwise MesaNet: forget gates as a per-token ridge
status: review
evidence: E2
track: A
builds_on: ["0001"]
tags: [mesa, mesanet, least-squares-memory, forget-gates, ridge-regression, woodbury, eigendecomposition, closed-form-backward]
authors: ["@MaurerKrisztian", "Claude Opus 5.5"]
summary: MesaNet's forget gates act as a per-token ridge, so one eigendecomposition per chunk plus small Woodbury solves gives exact readouts without CG (float64 error ≤ 4.5e-14 when well-conditioned) and closed-form gradients; FLOPs ≈ 4–19 CG steps forward, 8–16 vs fla's 60 per training pass; GPU untested.
updated: 2026-10-29
---

# Exact chunkwise MesaNet: forget gates as a per-token ridge

Oct 29, 2026 · @MaurerKrisztian with Claude Opus 5.5

## Summary

MesaNet reads its memory by solving a d×d ridge regression at every token. With forget gates and a fixed ridge, MesaNet and its flash-linear-attention kernels solve these systems with conjugate gradient (CG): 30 steps in training, in both the forward and the backward pass. In the classical literature on the same recursion (regularized exponentially weighted RLS), the exact updates found cost O(d³) per step. One closed-access paper claims a cheaper exact solution, probably for special inputs, and could not be checked. The literature search of 2026-10-29 found no exact chunk-parallel algorithm. Study 0001 made the ungated case exact. This study shows the gated case can be exact too. Inside a chunk, the forget gates act exactly as a **per-token ridge Λ/a_t** on nested data, where a_t is the cumulative gate. One eigendecomposition of the whitened chunk-boundary state therefore diagonalizes every token's shift, and a small Woodbury solve per token adds the in-chunk keys. Every token's readout comes out exact, with no iteration and no tolerance. The study also derives a closed-form backward pass, with the gate gradient in a state-free form. Everything was checked on a CPU (E2). Nothing ran on a GPU.

- **Exact in float64.** Over 129 well-conditioned cases (condition number ≤ 800; 6 gate regimes from no forgetting to gates 0.01–0.2, d = 8–128, chunks 1–64, plus edge cases), the output matches a token-by-token reference to ≤ 4.5e-14 (`logs/check_forward_v2.log`). On ill-conditioned systems the error grows with the condition number, faster than for a dense LU solve per token. The solve's normwise backward error rises from ≤ 15·eps at condition ≤ 1e2 to about 1.2e3·eps at 1e6, against ≤ 0.5·eps for dense LU (`logs/check_conditioning.log`).
- **Float32 (CPU).** At condition ≤ 1e2 the output error is ≤ 4.9e-6, within 5× of a dense solve. 30 CG steps reach up to 2.3e-4 there, from either start or with Jacobi preconditioning. From condition 1e2 to 1e4 the exact solver stays ≤ 4.1e-4, while 30-step CG's worst errors are 7e-2–1.0. Above 1e4 the exact solver degrades (up to 2.2e-3, and up to 2000× a dense solve's error on some inputs), and 100-step CG or Jacobi PCG is sometimes more accurate (`logs/check_conditioning.log`).
- **Closed-form backward.** Gradients for q, k, v, β, the gates and the ridge match complex-step derivatives to ≤ 1.4e-14, or 4.8e-12 with ridge 1e-4–1e-3 (`logs/check_backward.log`). The q and Λ gradients are MesaNet's own (its Appendix D). The gate gradient is written as a reverse cumulative sum of per-token scalars, δ_t·o_t − g_t·q_t + x_tᵀΛg_t − β_t ∂ℓ/∂β_t. That is GLA's trick for log-decay gradients plus a ridge term; MesaNet's own gate gradient uses the states instead.
- **Cost (derived, E1).** With CG costed at flash-linear-attention's chunk size of 64, the exact forward solve costs as many FLOPs as 4–19 CG steps. The figure depends on how fast the gates forget, on d (64–256) and on the eigendecomposition's assumed cost (4d³–9d³). A training pass costs 8–16 CG-step equivalents, against fla's fixed 30 + 30 steps (`logs/cost_and_cg_v2.log`). At 1e-4 float64 accuracy, CG from zero needs 20–30 steps. MesaNet's diagonal starting guess needs 15–30, or 5–15 when the keys' anisotropy lies along the coordinate axes. Every CG variant needs 50 to over 100 steps on flat key spectra with ridge 1e-3–1e-2 (condition ~1e3).

**Verdict:** the identity and the closed-form backward hold at E2. No prior publication of an exact chunk-parallel solver for this system was found; the ingredients are classical. The cost case is conditional. Against fla's fixed 30-step CG in training, the FLOP model gives 3.8–7.8× fewer solver FLOPs per training pass, and in float32 the exact solver is far more accurate once the condition number passes ~1e2. Against CG stopped at 1e-4 with MesaNet's diagonal start, the forward FLOP advantage ranges from none, on easy well-preconditioned inputs, to about 8×, depending on d, the gates and the eigendecomposition's cost. It is largest on ill-conditioned systems where CG needs 50–100+ steps. GPU speed is untested, and GPU eigendecompositions and small factorizations run far below matmul speed.

## Claims

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | Inside a chunk, H_t + Λ = a_t (H_0 + Σ_{i≤t} (β_i/a_i) k_i k_iᵀ + Λ/a_t): MesaNet's forget gates act as a per-token ridge Λ/a_t on nested, reweighted data | E2 (algebra; implied by claim 2) | Method, Proposition 1; `logs/check_forward_v2.log` |
| 2 | One eigendecomposition per chunk boundary plus a per-token Woodbury capacitance (or the d×d system when it is smaller) gives every token's exact solve. Over 129 well-conditioned float64 cases (condition ≤ 800; 251 cases in all), the output error is ≤ 4.5e-14 and the solve's normwise backward error ≤ 1.3e-14 | E2 | `logs/check_forward_v2.log`, `results/check_forward_20261029_004818.json` |
| 3 | The solve's normwise backward error grows with conditioning: ≤ 15·eps at condition ≤ 1e2, ≤ 19·eps at 1e2–1e4, ≤ 172·eps at 1e4–1e6 (sweep), and ≤ 2.6e-13 ≈ 1.2e3·eps at 4e4–1.4e6. Dense LU stays ≤ 0.5·eps. The worst forward error is 1.1e-9, against 1.7e-11 for dense LU | E2 | `logs/check_conditioning.log`, `logs/check_forward_v2.log` |
| 4 | Downdating from the chunk-end anchor only while ρ_t = a_t/a_C ≤ 10 keeps float64 error ≤ 1.4e-14 and float32 error ≤ 2.2e-6 on three inputs with ridge 0.05–1, and halves the mean capacitance for slowly forgetting gates. It is less accurate than start-only on 33 of 45 float32 inputs. With gates 0.5–1, unrestricted downdating reached 2.6e-3 in float32, and downdating every token made a float32 Cholesky fail | E2 | `logs/check_downdate_v2.log`, `logs/check_conditioning.log` |
| 5 | Float32 on a CPU (d = 64, C = 64): at condition ≤ 1e2 the output error is ≤ 4.9e-6 (≤ 5× a dense solve), against ≤ 2.3e-4 for 30 CG steps from either start or with Jacobi. At 1e2–1e4 it is ≤ 4.1e-4, against up to 1.0 for 30-step CG. At 1e4–1e5 it is ≤ 1.6e-3, up to 2000× a dense solve's (≤ 2.2e-3 at 1e5–1e7), and 100-step CG or Jacobi PCG is sometimes more accurate | E2 (CPU numerics, not GPU) | `logs/check_conditioning.log`; `logs/check_forward_v2.log` part B |
| 6 | Closed-form gradients for q, k, v, β, log γ and Λ match complex-step derivatives to ≤ 1.4e-14 on well-conditioned inputs and 4.8e-12 with ridge 1e-4–1e-3. The q and Λ gradients are MesaNet's (its Appendix D). The gate gradient takes the state-free form dℓ/dlogγ_j = Σ_{t≥j} (δ_t·o_t − g_t·q_t + x_tᵀΛg_t − β_t ∂ℓ/∂β_t) | E2 | `logs/check_backward.log`, `results/check_backward_20261029_004856.json` |
| 7 | Steps to 1e-4 output error in float64 (d = 64 and 128): CG from zero 20–30. MesaNet's diagonal start: 5–15 on i⁻² keys along the axes, 15–30 otherwise. Jacobi PCG: 5–20, or 15–50 on rotated i⁻² keys with ridge 1e-3–1e-2. Every variant needs 50 to over 100 on flat keys with ridge 1e-3–1e-2 (condition ~1e3) | E2 (accuracy of the baseline, not speed) | `logs/cost_and_cg_v2.log`, part A |
| 8 | At the best chunk size for each d = 64–256, with CG costed at chunk 64, the exact forward solve costs this many CG-step equivalents: 3.8–8.0 (eigendecomposition 4d³) or 5.3–11.9 (9d³) with slowly forgetting gates; 6.6–17.6 with gates 0.5–1 under the ρ cap; 7.6–19.0 start-only. Padding capacitances to the chunk maximum gives 7.1–16.7. Against CG at chunk 16 these are 1.2–1.6× larger. A training pass (two solves) costs 7.7–15.9, against 60 for fla | E1 (FLOP model) | `logs/cost_and_cg_v2.log`, part B |

## Problem

A least-squares (Mesa) memory layer keeps the decayed key and value sums H_t and G_t and reads them with a ridge-regression solve. For one head with keys and queries in ℝ^d, values in ℝ^{d_v}, write strengths β_t > 0, scalar forget gates γ_t ∈ (0, 1] and a fixed diagonal ridge Λ = diag(λ_1, …, λ_d) ≻ 0, MesaNet computes:

```math
H_t = \gamma_t H_{t-1} + \beta_t k_t k_t^\top, \qquad G_t = \gamma_t G_{t-1} + \beta_t v_t k_t^\top, \qquad x_t = (H_t + \Lambda)^{-1} q_t, \qquad o_t = G_t\, x_t
```

The matrix M_t = H_t + Λ changes at every token, so each token needs its own d×d solve. MesaNet solves them all in parallel with CG. It trains with a fixed 30 steps and stops at a relative tolerance of 1e-4 at inference, which takes about 9 steps on average (MesaNet paper, Sections 4–5). The flash-linear-attention kernels run a fixed 30 CG steps from x = 0, without a preconditioner, in the forward pass and again for the backward pass's adjoint solve. The same recursion is known in adaptive filtering as regularized or exponential-resetting RLS. There, the forgetting step and the fixed regularizer are both full-rank changes, so the matrix inversion lemma doesn't help and exact updates cost O(d³) per step (Lai & Bernstein 2023).

Study 0001 made the solve exact for the ungated layer: one C×C Cholesky per chunk gives every prefix solve. That is its Claim 1, at E2. With gates, that identity is exact only if the ridge decays along with the gates (0001's "Mesa-P"). For MesaNet as published, with a fixed ridge, 0001 used the exact chunk solve only as a preconditioner for CG. The question here is whether MesaNet as published, with fixed ridge and forget gates, also has an exact chunk-parallel solve, and at what cost.

## Method

Take one chunk of C tokens, renumbered 1…C, with boundary states H_0 and G_0 carried in from the previous chunk. Let a_t = γ_1 ⋯ γ_t be the cumulative gate (a_0 = 1), and let S = Λ^{-1/2}.

### Proposition 1: gates are a per-token ridge

Unrolling the recurrence inside the chunk gives

```math
H_t + \Lambda \;=\; a_t\Big(H_0 + \sum_{i\le t} \tfrac{\beta_i}{a_i}\, k_i k_i^\top + \tfrac{1}{a_t}\Lambda\Big),
\qquad
o_t \;=\; \hat G_t\,\Big(\hat H_t + \tfrac{1}{a_t}\Lambda\Big)^{-1} q_t ,
```

where Ĥ_t = H_0 + Σ_{i≤t} (β_i/a_i) k_i k_iᵀ and Ĝ_t = G_0 + Σ_{i≤t} (β_i/a_i) v_i k_iᵀ. The factor a_t cancels in o_t. Each readout is therefore an *ungated* ridge regression on nested data: the boundary state plus the chunk's pairs weighted by β_i/a_i, with ridge Λ/a_t. That ridge grows as the gates close. If the ridge decayed with the gates (Mesa-P), the shift would be the same for every token, and one factorization per chunk would serve all of them (study 0001). MesaNet's fixed ridge instead gives a different scalar multiple of Λ for each token.

### Proposition 2: exact solves from one eigendecomposition per boundary

A different shift for every token rules out the shared factorization. But all the shifts are multiples of the same Λ, so one eigendecomposition makes all of them diagonal. Whiten and diagonalize the boundary state, S H_0 S = V diag(h) Vᵀ, and set k̂_i = Vᵀ S k_i and q̂_t = Vᵀ S q_t. Then

```math
H_t + \Lambda = S^{-1} V \big(D_t + U_t^\top U_t\big) V^\top S^{-1}, \qquad D_t = I + a_t\,\mathrm{diag}(h), \qquad
U_t = \begin{bmatrix} \sqrt{\beta_i\, a_t / a_i}\;\hat k_i^\top \end{bmatrix}_{i \le t}
```

and Woodbury gives every token's solution:

```math
x_t = S V y_t, \qquad y_t = D_t^{-1}\hat q_t - D_t^{-1} U_t^\top A_t^{-1} U_t D_t^{-1}\hat q_t, \qquad A_t = I_t + U_t D_t^{-1} U_t^\top \succeq I .
```

D_t is diagonal, so every shift costs only elementwise work. The weights a_t/a_i ≤ 1 never overflow, and the capacitance A_t has all its eigenvalues ≥ 1, so its Cholesky factorization always exists. When the capacitance would have m_t ≥ d rows, the solver uses the d×d system D_t + U_tᵀU_t instead. It is cheaper there, and it avoids Woodbury's cancellation, which gave backward errors up to 3.5e-10 at d = 8 and C = 64 before this switch was added (`logs/check_forward.log`, superseded).

**Second anchor: downdating from the chunk end.** The chunk-end state H_C is the next chunk's start state, so its eigendecomposition S H_C S = W diag(η) Wᵀ is computed anyway. Let ρ_t = a_t/a_C ≥ 1. Then H_t = ρ_t (H_C − Σ_{i>t} (a_C/a_i) β_i k_i k_iᵀ), and

```math
H_t + \Lambda = S^{-1} W \big(D'_t - U'^\top_t U'_t\big) W^\top S^{-1}, \qquad D'_t = I + \rho_t\,\mathrm{diag}(\eta), \qquad
U'_t = \begin{bmatrix} \sqrt{\beta_i\, a_t / a_i}\;\tilde k_i^\top \end{bmatrix}_{t < i \le C},
```

```math
y_t = D'^{-1}_t\tilde q_t + D'^{-1}_t U'^\top_t \big(I - U'_t D'^{-1}_t U'^\top_t\big)^{-1} U'_t D'^{-1}_t \tilde q_t .
```

In exact arithmetic the downdate capacitance is positive definite, with smallest eigenvalue at least 1/(1 + ρ_t η_max). This holds because the whitened H_t + Λ is ⪰ I, so D'^{-1/2}(D' − U'ᵀU')D'^{-1/2} ⪰ D'^{-1}. The ratio ρ_t also measures how much cancellation the downdate suffers. With gates 0.5–1 and chunk 64, downdating every token (ρ_t up to 1.4e9) left a float64 error of 4.4e-7, and in float32 a capacitance lost positive definiteness (`logs/check_downdate_v2.log`). The two-anchor policy therefore lets token t downdate only when its downdate capacitance is smaller than its update capacitance (m_end < m_start and m_end < d) and ρ_t ≤ 10. With slowly forgetting gates this caps every capacitance at about C/2. With fast forgetting, more tokens fall back to the start anchor. Downdating costs accuracy, though: in float32 it was less accurate than the start anchor alone on most inputs (Results).

**Algorithm.** (1) Compute the boundary states H and G at chunk starts with the usual gated linear-attention state recurrence. (2) Compute one symmetric eigendecomposition of S H S per boundary, shared by the two chunks it borders. (3) For each token, choose the anchor and solve the m_t×m_t capacitance system, or the d×d system when m_t ≥ d. (4) Read out o_t = a_t G_0 x_t + Σ_{i≤t} (a_t/a_i) β_i v_i (k_iᵀx_t), the gated linear-attention chunk product with x_t in place of the query. Once the boundaries are known, every token in every chunk is independent: the work is small batched matrix products and small Cholesky factorizations, with no iteration. Gates enter as log a_t, so a weight a_t/a_i is never computed by dividing two products that have underflowed. Float32 was tested only with gates down to 0.5.

### Proposition 3: closed-form backward

Let δ_t = ∂ℓ/∂o_t and M_t = H_t + Λ, and define g_t = M_t^{-1} G_tᵀ δ_t. This is the same chunk solve with right-hand side G_tᵀδ_t, which comes from a transposed gated linear-attention product. Implicit differentiation of M_t x_t = q_t then gives

```math
\frac{\partial \ell}{\partial q_t} = g_t, \qquad \frac{\partial \ell}{\partial \Lambda} = -\sum_t x_t \odot g_t, \qquad
\frac{\partial \ell}{\partial v_i} = \beta_i \sum_{t\ge i} \tfrac{a_t}{a_i}\,(k_i^\top x_t)\,\delta_t ,
```

```math
\frac{\partial \ell}{\partial k_i} = \beta_i \sum_{t\ge i} \tfrac{a_t}{a_i}\Big[(\delta_t^\top v_i)\,x_t - (x_t^\top k_i)\,g_t - (g_t^\top k_i)\,x_t\Big], \qquad
\frac{\partial \ell}{\partial \beta_i} = \sum_{t\ge i} \tfrac{a_t}{a_i}\big[\delta_t^\top v_i - g_t^\top k_i\big](k_i^\top x_t) .
```

For the gates, write c_{ti} = (a_t/a_i) β_i (δ_tᵀv_i − g_tᵀk_i)(k_iᵀx_t). Gate γ_j scales pair (t, i) exactly when i < j ≤ t, so ∂ℓ/∂log γ_j = Σ_{t≥j} Σ_{i<j} c_{ti}. Split this double sum into row sums R_t = Σ_{i≤t} c_{ti} minus column sums B_i = Σ_{t≥i} c_{ti} = β_i ∂ℓ/∂β_i. The row sum collapses through M_t x_t = q_t: R_t = δ_tᵀG_t x_t − g_tᵀH_t x_t = δ_t·o_t − g_t·q_t + x_tᵀΛg_t. Hence

```math
\frac{\partial \ell}{\partial \log\gamma_j} = \sum_{t \ge j} \Big(\delta_t^\top o_t - g_t^\top q_t + x_t^\top \Lambda\, g_t - \beta_t \frac{\partial \ell}{\partial \beta_t}\Big).
```

The term x_tᵀΛg_t is the gradient signal of the non-decaying ridge. The q and Λ gradients are the ones MesaNet's Appendix D gives. MesaNet also gives the gate gradient in closed form, but as traces against the states G_{s−1} and H_{s−1}. The form above needs only per-token scalars. It is GLA's reverse-cumulative-sum trick for log-decay gradients (Yang et al. 2023, Section 4.3), extended by the ridge term. The backward pass needs one more exact solve per token, using the same eigendecompositions, plus decayed sums of the kind the gated linear-attention backward already computes. `gated_mesa.backward` evaluates these sums with a dense T×T decay mask for clarity. A chunked version would be the standard one and was not written here.

### Cost

Counting a multiply-add as 2 FLOPs, per chunk and head the solver needs:
- the boundary state update: 2Cd² + 2Cd·d_v;
- one eigendecomposition per boundary, assumed to cost 4d³–9d³;
- moving keys and queries into an anchor basis and the solutions back: 6Cd²;
- per token, either a capacitance of size m (forming it m²d, Cholesky m³/3, solves 2m² + 4md) or the d×d form (md² + d³/3).

One CG step costs 2d² + 2Cd + 11d per token: a_t H_0 p, the in-chunk decayed key sum, and the vector updates. The readout and state update are the same for every method. `gated_mesa.solver_flops`, `cg_step_flops` and `flops_*` hold the formulas, and the tables under Results evaluate them. CG is costed at flash-linear-attention's chunk size of 64, and also at 16, where each CG step is cheaper. The exact solver's per-token capacitance sizes are taken as they come, and also padded to each chunk's largest. Its backward solve reuses the eigendecompositions and the transformed keys. Memory is the boundary state, its eigenvectors, and one m×m capacitance per token in flight. A dense per-token solve would instead materialize a d×d matrix per token.

## Results

All runs are NumPy on a CPU, float64 unless stated. Errors are the maximum absolute error divided by the maximum absolute reference value.

**Exactness, float64.** The reference solves each token densely and refines the result with residuals computed in extended precision (`np.longdouble`, 64-bit mantissa). That gives a sharper reference than float64 alone on ill-conditioned systems, against which both the chunk solver and a plain float64 dense solve are measured. The inputs cover gates of 1 (none), 0.999–1, 0.99–1, 0.9–1, 0.5–1 and 0.01–0.2; (T, d, d_v, C) of (200, 16, 8, 32), (200, 64, 16, 16), (150, 8, 4, 64), (257, 32, 32, 64) and (96, 128, 16, 32); ridges 0.05–1 and 1e-6–1e-4; key spectra flat or decaying as i⁻²; and both policies.

| Cases | Condition number of H_t + Λ | Output error | Solution error | Dense solve's solution error | Backward error |
|---|---|---|---|---|---|
| 120 random, well-conditioned | 9.5–120 | ≤ 4.5e-14 | ≤ 6.7e-14 | – | ≤ 1.3e-14 |
| 120 random, ridge 1e-6–1e-4 | 4.3e4–1.4e6 | ≤ 2.3e-9 | ≤ 1.1e-9 (≤ 4.6× cond·eps) | ≤ 1.7e-11 | ≤ 2.6e-13 |
| 9 edge cases, well-conditioned | 18–800 | ≤ 3.0e-14 | ≤ 5.0e-14 | – | ≤ 4.4e-15 |
| 2 edge cases, ridge 1e-8, key spectrum i⁻⁴ | 1.3e8 | 1.0e-9 | 1.9e-10 | 8.6e-14 | 5.8e-14 |

The edge cases were T = 1; chunk size 1; a chunk longer than the sequence; a chunk size that doesn't divide T; downdating only; one key repeated 20 times; β = 0 on every third token; three near-resets (gate 1e-12); β up to 50; and ridge 1e-8 with a steep key spectrum. *Source: `logs/check_forward_v2.log` (summary lines and per-case rows), `results/check_forward_20261029_004818.json`.*

On ill-conditioned systems the chunk solver's normwise backward error stays small (≤ 2.6e-13, about 1.2e3·eps) but grows with the condition number, and its forward error exceeds a dense LU solve's. Over these cases its worst forward error was 1.1e-9, against 1.7e-11 for the dense solve. On single inputs the gap can be much wider: on one case with keys decaying as i⁻², the dense solve was accurate to 1.2e-15 where the chunk solver reached 1.1e-9. The sweep below quantifies both effects. One step of iterative refinement would probably close the gap, at the cost of about one CG step, but it was not tried.

**Downdating against ρ** (T = 256, d = 64, C = 64, ridge 0.05–1). Each row allows token t to use the chunk-end anchor when its capacitance is smaller and ρ_t = a_t/a_C ≤ κ.

| Gates | Policy | Tokens downdating | Mean capacitance size | Largest ρ used | float64 solution error | float32 output error |
|---|---|---|---|---|---|---|
| 0.99–1 | start only (κ = 1) | 4 | 31.5 | 1 | 1.1e-15 | 7.0e-7 |
| 0.99–1 | κ = 10 (default) to ∞ | 128 | 16.0 | 1.18 | 6.5e-15 | 6.0e-7 |
| 0.9–1 | start only (κ = 1) | 4 | 31.5 | 1 | 1.6e-15 | 5.9e-7 |
| 0.9–1 | κ = 10 (default) to ∞ | 128 | 16.0 | 6.77 | 1.4e-14 | 1.9e-6 |
| 0.9–1 | every token downdates | 256 | 31.5 | 37.3 | 9.9e-14 | 1.1e-5 |
| 0.5–1 | start only (κ = 1) | 4 | 31.5 | 1 | 1.0e-15 | 7.2e-7 |
| 0.5–1 | κ = 10 (default) | 34 | 25.0 | 9.17 | 5.8e-15 | 2.2e-6 |
| 0.5–1 | κ = 100 | 70 | 19.6 | 92 | 5.5e-14 | 1.8e-5 |
| 0.5–1 | κ = ∞ | 128 | 16.0 | 3.1e4 | 7.6e-12 | 2.6e-3 |
| 0.5–1 | every token downdates | 256 | 31.5 | 1.4e9 | 4.4e-7 | Cholesky failed |

*Source: `logs/check_downdate_v2.log`; with gates 0.99–1, every token downdating gave 3.3e-14 and 1.3e-6.* In the "start only" rows the 4 downdating tokens are the last token of each chunk, which has nothing to downdate (ρ = 1, m = 0). The cap ρ ≤ 10 keeps float32 errors within about 3× of the start-anchor-only solver while cutting the mean capacitance by up to half.

**Float32, CPU** (T = 512, d = 64, d_v = 32, C = 64; output error against the float64 reference).

| Gates | Ridge, key spectrum | Condition number | Exact, start anchor | Exact, two anchors | Dense per-token solve | CG 10 | CG 30 | CG 100 |
|---|---|---|---|---|---|---|---|---|
| none | 0.05–1, flat | 30 | 1.3e-6 | 7.0e-7 | 6.9e-7 | 2.0e-2 | 2.0e-6 | 6.6e-7 |
| none | 1e-3–1e-2, i⁻² | 3.8e3 | 1.2e-6 | 7.3e-7 | 5.8e-7 | 1.6e-1 | 1.6e-4 | 6.0e-7 |
| 0.99–1 | 0.05–1, flat | 28 | 7.5e-7 | 6.0e-7 | 6.7e-7 | 1.7e-2 | 1.4e-6 | 6.3e-7 |
| 0.99–1 | 1e-3–1e-2, i⁻² | 1.7e3 | 3.8e-6 | 1.2e-6 | 7.5e-7 | 1.1e-1 | 3.3e-5 | 7.2e-7 |
| 0.9–1 | 0.05–1, flat | 31 | 7.8e-7 | 1.3e-6 | 3.6e-7 | 1.2e-2 | 9.8e-7 | 5.6e-7 |
| 0.9–1 | 1e-3–1e-2, i⁻² | 260 | 2.3e-6 | 7.5e-7 | 2.5e-7 | 2.1e-2 | 8.2e-7 | 7.7e-7 |
| 0.5–1 | 0.05–1, flat | 33 | 9.1e-7 | 1.4e-6 | 4.6e-7 | 2.0e-2 | 1.3e-6 | 1.1e-6 |
| 0.5–1 | 1e-3–1e-2, i⁻² | 140 | 5.6e-7 | 2.0e-6 | 1.9e-7 | 8.7e-3 | 8.0e-7 | 8.0e-7 |

*Source: `logs/check_forward_v2.log`, part B.*

**Accuracy against conditioning** (added after the review; gates 0.9–1, ridge ranges from 0.05–1 down to 1e-6–1e-5, key spectra flat, i⁻² along the coordinate axes, or i⁻² under a random rotation). In float64 (T = 200, d = 64, C = 32, 36 inputs, two seeds), each cell is the worst value in that condition-number bin.

| Condition number | Inputs | Backward error / eps: start, two, dense LU | Forward error: start, two, dense LU |
|---|---|---|---|
| ≤ 1e2 | 11 | 9, 15, 0.5 | 8.8e-15, 1.9e-14, 1.3e-15 |
| 1e2–1e4 | 12 | 11, 19, 0.4 | 8.1e-13, 1.5e-12, 1.4e-13 |
| 1e4–1e6 | 12 | 172, 85, 0.5 | 9.1e-11, 1.7e-10, 1.1e-11 |
| 1e6–1e9 | 1 | 74, 17, 0.4 | 2.1e-11, 1.7e-10, 1.5e-11 |

The chunk solver's backward error grows with conditioning. On the same inputs it was 3–895× dense LU's, so the solver should not be called backward stable. Comparing each bin's worst cases, its forward error is 1.4–15× the dense solve's. On single inputs, where dense LU happens to be far more accurate than its bound, the gap reaches 1.3e4× (start anchor) and 3.5e4× (two anchors).

In float32 (T = 256, d = 64, d_v = 32, C = 64; 45 inputs over 5 ridge ranges, 3 key spectra and 3 seeds), each cell is the worst output error in the bin. The last column is the worst ratio of the start-anchor solver's error to the dense solve's on the same input.

| Condition number | Inputs | Exact, start | Exact, two | Dense | CG 30, x₀ = 0 | CG 30, diagonal start | Jacobi PCG 30 | CG 100, x₀ = 0 | start / dense |
|---|---|---|---|---|---|---|---|---|---|
| ≤ 1e2 | 16 | 3.2e-6 | 4.9e-6 | 1.4e-6 | 2.3e-4 | 1.9e-4 | 6.2e-5 | 2.4e-6 | 4.5 |
| 1e2–1e3 | 9 | 2.2e-5 | 4.6e-5 | 8.2e-6 | 8.0e-2 | 8.6e-2 | 7.2e-2 | 2.0e-5 | 19 |
| 1e3–1e4 | 9 | 2.7e-4 | 4.1e-4 | 5.3e-5 | 1.0 | 0.62 | 0.66 | 2.5e-2 | 103 |
| 1e4–1e5 | 9 | 1.6e-3 | 1.6e-3 | 3.4e-4 | 3.3 | 3.5 | 2.2 | 0.34 | 2005 |
| 1e5–1e7 | 2 | 2.2e-3 | 2.0e-3 | 3.9e-4 | 2.8 | 3.1 | 3.3 | 0.49 | 5.8 |

*Source: `logs/check_conditioning.log`, `results/check_conditioning_20261029_093722.json`.* The worst CG errors come from flat key spectra. The exact solver's error grows roughly in proportion to the condition number, while a dense solve's grows more slowly. CG fixed at 30 steps, whatever its start or preconditioner, is much less accurate than the exact solver once the condition number passes about 1e2. The ordering is not universal, though. On i⁻² keys along the axes at condition ~2.5e4, Jacobi PCG reached 4.1e-7–1.6e-6 and 100 CG steps 6e-7–1.3e-6, where the exact solver had 2.6e-5–8.9e-4. On rotated i⁻² keys at condition ~2.2e4, the exact solver (4.6e-5–1.7e-4) matched the dense solve (7.1e-5–9.3e-5) and 100 CG steps (6.5e-5–9.2e-5), and beat 30 CG steps (3.0e-3–5.1e-3) and Jacobi PCG (0.13–0.17). The two-anchor policy was less accurate than start-only on 33 of the 45 float32 inputs. Downdating saves capacitance work at a cost in accuracy, and in float32 start-only is the safer default.

**Closed-form backward.** Each entry is the error against complex-step derivatives of the token-by-token recurrence (step 1e-30, exact to round-off).

| T, d, d_v, C | Gates | Ridge | Policy | q | k | v | β | log γ | Λ |
|---|---|---|---|---|---|---|---|---|---|
| 20, 6, 4, 6 | none | 0.05–1 | two | 1.6e-15 | 1.6e-15 | 1.7e-15 | 4.6e-15 | 7.3e-15 | 2.2e-15 |
| 20, 6, 4, 6 | 0.99–1 | 0.05–1 | two | 2.8e-15 | 2.9e-15 | 2.1e-15 | 2.4e-15 | 8.8e-15 | 3.5e-15 |
| 20, 6, 4, 6 | 0.9–1 | 0.05–1 | start | 8.1e-16 | 7.9e-16 | 7.1e-16 | 1.9e-15 | 7.3e-16 | 4.6e-16 |
| 20, 6, 4, 6 | 0.9–1 | 0.05–1 | two | 6.4e-16 | 1.7e-15 | 1.1e-15 | 1.7e-15 | 8.6e-15 | 1.6e-15 |
| 24, 10, 3, 8 | 0.5–1 | 0.05–1 | two | 2.2e-15 | 2.0e-15 | 1.7e-15 | 2.3e-15 | 2.1e-15 | 5.2e-15 |
| 24, 10, 3, 8 | 0.01–0.2 | 0.05–1 | two | 1.5e-15 | 4.6e-15 | 7.6e-16 | 1.6e-15 | 1.4e-14 | 9.9e-16 |
| 17, 4, 3, 16 | 0.8–1 | 0.05–1 | two (d×d form) | 8.0e-16 | 4.6e-16 | 1.7e-15 | 2.1e-16 | 1.4e-15 | 6.7e-16 |
| 18, 8, 4, 5 | 0.9–1 | 1e-4–1e-3 | two | 2.3e-13 | 4.8e-12 | 1.8e-13 | 1.2e-13 | 5.5e-13 | 6.7e-13 |

*Source: `logs/check_backward.log`.*

**How many CG steps the baseline needs** (float64, T = 256, C = 64; steps needed for an output error of 1e-4, out of 5, 10, 15, 20, 30, 50 and 100). Three CG variants are compared. "Zero" starts from x = 0 without a preconditioner, as the flash-linear-attention kernel does. "Diag" starts from diag(H_t + Λ)⁻¹q_t, as the MesaNet paper does. "Jacobi" is the diag start plus Jacobi preconditioning. Each cell pools the four gate regimes (none, 0.99–1, 0.9–1, 0.5–1). The i⁻² "axis" keys have their anisotropy along the coordinate axes, which favors a diagonal preconditioner. The "rot" keys are randomly rotated, so it does not.

| d | Ridge | Keys | Condition numbers | Zero | Diag | Jacobi |
|---|---|---|---|---|---|---|
| 64 | 0.05–1 | flat | 29–33 | 20–30 | 20 | 15 |
| 64 | 0.05–1 | i⁻² axis | 16–54 | 20–30 | 5–10 | 5 |
| 64 | 0.05–1 | i⁻² rot | 17–50 | 20–30 | 15–20 | 5–10 |
| 64 | 1e-3–1e-2 | flat | 1.1e3–1.4e3 | 50–100 | 50–100 | 50–100 |
| 64 | 1e-3–1e-2 | i⁻² axis | 160–2.1e3 | 20–30 | 10–15 | 5 |
| 64 | 1e-3–1e-2 | i⁻² rot | 150–2.1e3 | 20 | 15–20 | 15–50 |
| 128 | 0.05–1 | flat | 28–31 | 30 | 20–30 | 15–20 |
| 128 | 0.05–1 | i⁻² axis | 19–20 | 20–30 | 5–10 | 5 |
| 128 | 0.05–1 | i⁻² rot | 19–31 | 20–30 | 15–20 | 5 |
| 128 | 1e-3–1e-2 | flat | 1.0e3–1.3e3 | 50 to > 100 | 50 to > 100 | 100 to > 100 |
| 128 | 1e-3–1e-2 | i⁻² axis | 69–1.0e3 | 20 | 10–15 | 5 |
| 128 | 1e-3–1e-2 | i⁻² rot | 120–1.3e3 | 20–30 | 15–20 | 15–50 |

*Source: `logs/cost_and_cg_v2.log`, part A. Its "zero" rows on the first 32 inputs reproduce version 1 (`logs/cost_and_cg.log`) exactly.*

**FLOP model** (derived, E1). Each cell gives the exact solver's FLOPs per token, as a number of CG steps. Only the solver is counted: the state update and readout, which every method pays, are excluded. CG is costed at chunk 64, flash-linear-attention's default. The columns are:
- two-anchor sizes when every token may downdate, as with slowly forgetting gates, at eigendecomposition costs 4d³ and 9d³;
- the sizes the policy actually picks (ρ ≤ 10) on random gates in 0.5–1;
- start-anchor only;
- two-anchor sizes padded to the chunk's largest capacitance, as a batched kernel would;
- a training pass (forward and backward solves; the backward reuses the eigendecompositions and transformed keys and refactors the capacitances), to compare with 60 CG steps.

| d | C | Two, 4d³ | Two, 9d³ | Gates 0.5–1 | Start only | Two, padded | Training pass |
|---|---|---|---|---|---|---|---|
| 64 | 32 | 3.8 | 6.2 | 6.6 | 7.6 | 7.1 | 7.7 |
| 64 | 64 | 4.1 | 5.3 | 8.4 | 10.8 | 8.7 | 8.0 |
| 128 | 64 | 5.6 | 8.9 | 10.7 | 12.2 | 11.0 | 11.3 |
| 128 | 128 | 7.5 | 9.1 | 19.7 | 23.2 | 17.8 | 14.7 |
| 256 | 64 | 9.3 | 17.2 | 18.2 | 19.0 | 18.4 | 19.4 |
| 256 | 128 | 8.0 | 11.9 | 17.6 | 19.5 | 16.7 | 15.9 |

*Source: `logs/cost_and_cg_v2.log`, part B, which also has C = 32 to 128 for every d. With CG at chunk 16, where each CG step is cheaper, every entry grows by 1.2–1.6×.*

In absolute terms, from `logs/cost_and_cg.log` part B (version 1, two-anchor sizes, d_v = d): at d = 128 and C = 64, a forward pass costs 400–564 thousand FLOPs per token. A dense Cholesky per token costs 863 thousand, study 0001's ungated solver 198 thousand, and 30 CG steps 1631 thousand. At d = 64 the exact solver and a dense per-token Cholesky cost about the same (103–124 vs 128 thousand). At d = 256 the exact solver is 2.6–3.5× cheaper.

Taken together:
- Against fla's fixed 30 + 30 CG steps, a training pass needs 3.8–7.8× fewer solver FLOPs.
- Against CG stopped at 1e-4 with MesaNet's diagonal start (15–30 steps on most inputs here, against 3.8–19 CG-step equivalents), the forward advantage ranges from about 1× to 8×.
- Where the diagonal start already converges in 5–15 steps, the exact solver can cost more.
- On flat-spectrum systems with condition ~1e3, where every CG variant needs 50–100+ steps, the advantage is largest.

These are FLOP counts, not times.

## Novelty check

Searched on 2026-10-29 by two `literature-scout` agents, one on the ML literature and one on numerical linear algebra and adaptive filtering. Sources: arXiv (API and HTML), OpenReview search, Semantic Scholar citations, OpenAlex, the flash-linear-attention repository, and web search. NOTEBOOK.md holds the query log. MesaNet, the flash-linear-attention kernel and Lai & Bernstein were also opened in this session. The other rows rest on what the agents read, as noted.

| Work | What it does | Difference |
|---|---|---|
| MesaNet [1] | The same layer: fixed diagonal Λ and scalar gates. CG: 30 steps in training, tolerance 1e-4 (about 9 steps) at inference, started from diag(H_t + Λ)⁻¹q_t. Closed-form dℓ/dq and dℓ/dΛ, the same as here, and a state-based closed form for dℓ/dγ (Appendix D) | Iterative. No rescaling of the gates into a ridge, no eigendecomposition and no Woodbury |
| flash-linear-attention MesaNet kernels [2] | Triton CG with a fixed 30 steps from x = 0, no preconditioner, in both passes; chunk 64 | Iterative. This study's CG baseline follows the same algorithm |
| Lai & Bernstein, ER-RLS [3] | The same recursion R_{k+1} = λR_k + (1−λ)R_∞ + φᵀφ (MesaNet's system when R_0 = R_∞ = Λ, with a constant forgetting factor). Because both terms are rank n, the matrix inversion lemma doesn't help, so exact updates cost O(n³) per step. Its CR-RLS variant reaches O(pn²) by cycling through the eigencomponents of R_∞, which changes the recursion | Sequential. No block or eigendecomposition method for the exact recursion |
| Regularized / leaky RLS in signal processing [4–6] (read by the agent: Schier's paper and Gay's patent in full; Horita et al. and Tsakiris et al. as abstracts) | The same objective. Exact methods cost O(d³) per step (Tsakiris et al.) or track eigendecompositions per step (Horita et al.). O(d²) methods approximate the regularizer: a cyclic rank-one refresh (Gay) or block-accumulated regularization (Schier) | Sequential, and approximate where cheaper. Schier's block scheme gets the in-block solutions wrong; here they are exact |
| Chernov & Zhdanov [7] (agent) | The same matrix in online learning (discounted aggregating algorithm for regression), stated as O(n³) per step | No efficient algorithm |
| Gated KalmaNet [8] (abstract opened here; method read by the agent) | The same gated sums with a per-token ridge that the agent read as λ_t = a‖H_t‖_F, solved by Chebyshev iteration. Its abstract calls the exact Kalman-filter equations hard to parallelize on GPUs | Iterative. If its ridge is a per-token scalar multiple of I, Proposition 1's form covers it; this study did not test that |
| Memory by Design [9] (agent) | An exact chunked full-covariance Kalman memory | Process noise added to the covariance, not the precision, so a different model; sequential within chunks |
| Preconditioned DeltaNet [12] (abstract opened here) | Derives equivalences between linear attention and the delta rule under exact preconditioning; in practice uses a diagonal approximation, with chunkwise algorithms for preconditioned DeltaNet, Gated DeltaNet and KDA | An approximate preconditioner for delta-rule layers; no exact solve of MesaNet's system |
| GLA [10] (agent) | Closed-form log-decay gradient as a reverse cumulative sum (Section 4.3) | Linear attention, no solve; Proposition 3's gate form extends it with the ridge term |
| Golub, Heath & Wahba [11] (agent) | One SVD gives ridge solutions for every λ | Classical ingredient: one matrix, no data updates |

**Conclusion.** No prior publication was found, as of 2026-10-29, of an exact block or chunk-parallel algorithm for exponentially weighted least squares with a fixed regularizer, which is MesaNet's system. Searched: MesaNet and its citing works, flash-linear-attention, Gated KalmaNet, Kalman-style and test-time-regression layers, and the regularized, leaky and exponential-resetting RLS literature. The method is a classical combination applied in a new setting: one eigendecomposition serving many ridge shifts, as for ridge paths, together with Woodbury capacitance updates. The q and Λ gradients are already published (MesaNet, Appendix D). The state-free gate-gradient form is GLA's trick carried over to MesaNet.

One candidate remains unresolved: "An Alternative Solution to the Dynamically Regularized RLS Algorithm" (APSIPA 2019, doi 10.1109/APSIPAASC47483.2019.9023073). Its abstract claims an exact solution at about standard-RLS cost by "exploiting the special structure of the correlation matrix". That probably means tapped-delay-line inputs, which would not cover general keys, but the paper is closed access and was not read. A web search also cannot rule out unpublished or concurrent work.

## Limitations

- **No GPU timing.** The study stops at E2, and every cost figure is a FLOP count. GPU eigendecompositions and small batched Cholesky factorizations run far below matmul throughput, so the 3.8–7.8× training-pass advantage over fla's 30 + 30 CG steps may shrink, or vanish, in wall-clock time. The eigendecomposition's cost (4d³–9d³) is an assumption. The FLOP model also assumes per-token capacitance sizes that vary; padding them, as a batched kernel would, is in the table and costs up to 1.5× more.
- **CG baseline.** Three variants were measured: from zero as in fla, MesaNet's diagonal start, and Jacobi PCG. All are this study's NumPy implementations, at fixed step counts. With the diagonal start or Jacobi, CG converges in 5–15 steps when the keys' anisotropy lies along the coordinate axes, and then it can be cheaper than the exact solver. Real keys may sit anywhere between the axis-aligned and rotated cases tested here. At inference MesaNet stops at tolerance 1e-4 after about 9 steps, so any gain there is small; it lies mainly in training. Study 0001's exact-chunk preconditioner, the strongest CG baseline in this lab for slowly forgetting heads (its claim at E3, under review), was not rerun here.
- **Accuracy depends on conditioning.** The solve's backward error grows with the condition number: about 1.2e3·eps at 1e6, against ≤ 0.5·eps for dense LU. On single float64 inputs, dense LU's forward error was up to 1.3e4× smaller. In float32 the exact solver is within 5× of a dense solve up to condition 1e2, but up to 103× worse at 1e3–1e4 and up to 2000× at 1e4–1e5. There, 100-step CG or Jacobi PCG was sometimes more accurate. Iterative refinement (backlog I-012) is the obvious remedy, but it was not tried.
- **Two-anchor policy.** Downdating halves the capacitance work for slowly forgetting gates but was less accurate than start-only on 33 of 45 float32 inputs. Its cap ρ_t ≤ 10 rests on a sweep of three inputs. Only the capacitance's conditioning has a proven bound, not the whole downdate. In float32, start-only is the safer choice.
- **Scalar gates only.** With per-channel gates, as in GLA or KDA, the per-token shift becomes a diagonal matrix that varies per token. The boundary eigenbasis doesn't diagonalize it, and no exact chunk method is known here for that case.
- **Sizes.** The float64 checks used d ≤ 128, T ≤ 257 and chunks ≤ 64. The float32 checks used T ≤ 512, d = 64 and gates down to 0.5. The FLOP model goes to d = 256.
- **Value of exactness.** Study 0001 found 5 CG steps as good as the exact solve on a small recall task (its claim 7: E4, but 2 seeds only). Exact solves may not improve model quality. Their case is cost against fixed-step CG, predictability (no step count or tolerance to tune), exact gradients, and float32 accuracy at moderate conditioning.
- **Backward implementation.** The closed-form sums use a dense T×T decay mask. A chunked version would be the usual gated linear-attention backward and was not written here.

## Next steps

- **E3: GPU benchmark (backlog I-011).** Batched `eigh` per boundary plus batched capacitance Cholesky, against flash-linear-attention's MesaNet CG kernels and a diagonal-start or Jacobi CG, at equal accuracy, across gate regimes and key statistics taken from a trained MesaNet.
- **Iterative refinement (backlog I-012).** One refinement step, costing about one CG step, to bring the float32 error at condition 1e3–1e5 toward the dense solve's.
- **Per-head policy (backlog I-003).** Exact solver against preconditioned CG, chosen per head from its gates and conditioning.
- **Per-channel gates (backlog I-013).** An exact chunk method, or a proof that none of this kind exists.
- **Promotion.** Move the solver to `lab/` with tests after an independent review and a GPU test.

## References

Opened in this session: [1]–[3] directly, [8] and [12] as their abstracts, and [11] as its bibliographic record; the rest by the literature-search agents, to the extent stated in the Novelty check.

1. J. von Oswald et al., "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training", ICLR 2026. https://arxiv.org/abs/2506.05233
2. flash-linear-attention, MesaNet kernels (`fla/ops/mesa_net/chunk_cg_solver_fwd.py`). https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net
3. B. Lai, D. S. Bernstein, "Exponential Resetting and Cyclic Resetting Recursive Least Squares", IEEE Control Systems Letters / ACC 2023. https://doi.org/10.1109/LCSYS.2022.3229868
4. J. Schier, "Inverse updated systolic RLS algorithm with regularized exponential forgetting", Kybernetika 32(3), 1996. https://www.kybernetika.cz/content/1996/3/209/paper.pdf
5. S. L. Gay, "Dynamically regularized fast RLS with application to echo cancellation", ICASSP 1996, https://doi.org/10.1109/ICASSP.1996.543281; read through US patent 5677951, https://patents.google.com/patent/US5677951
6. E. Horita, K. Sumiya, H. Urakami, S. Mitsuishi, "A Leaky RLS Algorithm: Its Optimality and Implementation", IEEE Trans. Signal Processing 52(10), 2004, https://doi.org/10.1109/TSP.2004.834212 (abstract); M. Tsakiris, C. Lopes, V. Nascimento, "An array RLS algorithm with generic nonfading regularization matrix", IEEE Signal Processing Letters 17(12), 2010, https://doi.org/10.1109/LSP.2010.2083652 (abstract)
7. A. Chernov, F. Zhdanov, "Prediction with Expert Advice under Discounted Loss", ALT 2010. https://arxiv.org/abs/1005.1918
8. L. Peng, A. Chattopadhyay, L. Zancato, E. Nunez, W. Xia, S. Soatto, "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression", 2025. https://arxiv.org/abs/2511.21016
9. Dowling et al., "Memory by Design: Bayesian Layers", 2026. https://arxiv.org/abs/2605.31163
10. S. Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", ICML 2024. https://arxiv.org/abs/2312.06635
11. G. Golub, M. Heath, G. Wahba, "Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter", Technometrics 21(2), 1979. https://doi.org/10.1080/00401706.1979.10489751
12. N. Tumma, N. Loo, D. Rus, "Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences", 2026. https://arxiv.org/abs/2604.21100

## Reproduce

These scripts need only NumPy (`pip install numpy`); run them from this folder with `python3` or `../../.venv/bin/python`. Seeds are fixed. Each command below writes a fresh log to `/tmp`, to compare with the recorded one in `logs/`. Never `tee` into `logs/`: those files are the immutable record. Each rerun also adds a new JSON to `results/`; delete those copies, never the recorded ones.

| Command | Recorded log | What it checks | Time |
|---|---|---|---|
| `python3 -u check_forward.py 2>&1 \| tee /tmp/check_forward.log` | `logs/check_forward_v2.log` | Float64 exactness (parts A, A2), float32 error (part B) | ~20 s |
| `python3 -u check_backward.py 2>&1 \| tee /tmp/check_backward.log` | `logs/check_backward.log` | Closed-form gradients against complex-step derivatives | ~3 s |
| `python3 -u check_downdate.py 2>&1 \| tee /tmp/check_downdate.log` | `logs/check_downdate_v2.log` | Downdating accuracy against the cap on ρ | ~5 s |
| `python3 -u cost_and_cg.py 2>&1 \| tee /tmp/cost_and_cg.log` | `logs/cost_and_cg_v2.log` | CG variants' accuracy against steps (part A), CG-step equivalents (part B) | tens of minutes |
| `python3 -u check_conditioning.py 2>&1 \| tee /tmp/check_conditioning.log` | `logs/check_conditioning.log` | Forward and backward error against conditioning, float64 and float32 | minutes |

The study's code was never committed while these runs were made: each results JSON records `git_dirty: true` at a commit without the study. The scripts in this folder are the versions that produced the current logs. NOTEBOOK.md lists which script version produced each superseded log.

## Files

| File | Contents |
|---|---|
| `gated_mesa.py` | Reference recurrence, refined reference, CG, the exact chunk solver, readouts, closed-form backward, FLOP model |
| `check_forward.py`, `check_backward.py`, `check_downdate.py`, `cost_and_cg.py`, `check_conditioning.py` | The checks above |
| `logs/`, `results/` | Their outputs; superseded runs are listed in NOTEBOOK.md |
