---
id: "0003"
title: Chunk-parallel Kalman memory: one Cholesky per chunk
status: review
evidence: E3
track: A
builds_on: ["0001", "0002"]
tags: [kalman-filter, bayesian-layer, least-squares-memory, forget-gates, per-channel-gates, cholesky, chunkwise, innovations]
authors: ["@MaurerKrisztian", "Claude Opus 5.5"]
summary: For the Bayesian Layer's Kalman memory, one C×C Cholesky per chunk gives every exact readout and gradient (float64, E2), less accurately than the token-by-token filter when cond(𝒦) is large. On an RTX 5060 Ti (E3) its plain-PyTorch forward pass is mostly not faster than BL's algorithm with Triton loops: at each one's best chunk size, BL takes 0.66–1.08× its time eager, 0.68–0.99× as CUDA graphs.
updated: 2026-11-01
---

# Chunk-parallel Kalman memory: one Cholesky per chunk

Oct 29 – Nov 1, 2026 · @MaurerKrisztian with Claude Opus 5.5

## Summary

Least-squares (Mesa-type) memory layers read their memory with a ridge regression at every token. MesaNet's forget gates down-weight old key–value pairs relative to the reading token, so the regularization differs from token to token (study 0002, claim 1, E2). That is why 0002's exact chunk solver uses one d×d eigendecomposition per chunk, and why MesaNet itself uses conjugate gradient (CG). A Kalman-filter memory forgets differently: the memory itself decays with the gates and is refreshed by process noise. That model is published as the **Bayesian Layer** (BL) of Dowling et al. (2026). It keeps a dense d×d covariance. Its authors report MQAR recall that matches MesaNet at width 256 and exceeds it at width 128, and higher associative recall than Gated DeltaNet on WikiText-103 (BL, Section 4.2). Its exact chunk algorithm, though, computes the Kalman gains in a sequential loop over the chunk's positions (BL v5, Appendix C.2). Its current version also carries the mean token by token; its first version computed the mean in parallel with DeltaNet's WY form once the gains were known (BL v1, Appendix D).

This study derives a chunk-parallel form of that computation. In a state-space model, the joint covariance of a chunk's observations does not depend on which token is reading. So the leading-block property of Cholesky, which study 0001 used for the ungated layer (its claim 1, E2), applies again. One C×C Cholesky of that covariance, formed by matrix products, gives every token's exact readout, and the d×e memory mean is touched only by triangular solves and matrix products.

- **Chunk identity (E2, Propositions 2–3).** For diagonal transitions (scalar or per-channel gates) and diagonal process noise, which includes BL with diagonal transitions, every readout and the next boundary state come from one C×C Cholesky per chunk and head. No d×d factorization, no iteration, and no loop over the chunk's positions on d-dimensional vectors. The identity is classical: it is the causal Wiener filter computed from a triangular factorization of the observations' covariance (Kailath, Sayed & Hassibi, Section 4.1.2), here per chunk, conditional on the carried posterior. Slock and Maouche's block RLS (1994) uses the same per-block mechanism for exponentially weighted RLS, without process noise.
- **Verified in float64 on the CPU (E2; `logs/check_chunk_v5.log`).** Over 26 random and edge cases, the chunk readouts are within 1.2e-15 of an extended-precision token-by-token filter, and the boundary states within 3.0e-15. The cases cover scalar and per-channel gates from 1e-6 to 1, stationary and free noise, C = 1 to 64, d below and above C, T not a multiple of C, β from 1e-3 to 50, and ridges from 1e-3 to 10. All four forms agree: triangular solves, explicit L⁻¹, matrix-product Grams, and per-token prefix solves. The float64 token-by-token filter scores 4.8e-16 on the same yardstick. Gradients for every input match autograd through the token-by-token filter to ≤ 1.3e-15. Proposition 6's adjoint matches to ≤ 8.4e-16, and the bounds of Proposition 4(b) held in all 120 chunks tested.
- **Accuracy depends on conditioning (E2).** The chunk form is exact, but its rounding error is about cond(𝒦)·eps: at most 0.46·cond(𝒦)·eps in 46 of 48 float64 sweep rows. The token-by-token filter is usually more accurate. In float64 the chunk form's error is 2–8× the filter's with unit keys, 11–56× with a decaying key spectrum, and 12–195× with rank-4 key blocks. cond(𝒦) stays ≤ 1e4 with gates 0.99–1, but reaches 1e6 at 0.9999–1 and 5e8 without forgetting. In float32 over sequences of up to 65,536 tokens (`logs/check_float32_v2.log`), well-conditioned chunks give 2.3–9.3e-7, like the token-by-token filter with forgetting (1.4–6.0e-7). Without forgetting, that filter drifts to 6.8e-6 while the chunk form stays ≤ 9.3e-7. In the hard regimes, with strong writes or a decaying key spectrum under slow or no forgetting (cond(𝒦) 430–1.35e4), the chunk forms reach up to 3.7e-4. That is 2–56× the token-by-token filter's error. Forming 𝒦 in float64 cuts it 2–10×.
- **Backward pass (E2, Proposition 6).** The readouts' gradients with respect to the chunk's C×C matrices come from two more triangular solves with the same factor.
- **The stationary parametrization (E2, Propositions 1 and 4).** Tying the process noise to the gates, (I − D_t²)Λ⁻¹, keeps the prior of the memory at N(0, Λ⁻¹) at every token, which is MesaNet's fixed ridge. For a scalar gate this prior is published (Van Vaerenbergh et al. 2012; Titsias et al. 2024), and Kalman Linear Attention discretizes an OU prior exactly for a diagonal state. Here it is per channel, with a dense covariance. The noise sums then telescope, the factored matrix has its eigenvalues between 1/β_max and 1/β_min + Σ_i k_iᵀΛ⁻¹k_i whatever the history, and without gates the readout is exactly Mesa's G_t(H_t + Λ)⁻¹q_t.
- **Cost (E1, FLOP model; e = d = 64–256, C = 32–128).** Computing L⁻¹ explicitly, as DeltaNet's chunk form does with its C×C inverse, makes every step that scales with d or e a matrix product. Then the only order-dependent work in a chunk is the C×C Cholesky and inverse. This form costs 1.23–3.38× the FLOPs of BL's published chunk algorithm (v5, as read from its paper). Its order-dependent work is 2.0–24.5× smaller than that of the strongest combination of BL's published pieces, v5's gain loop with v1's parallel mean pass (6.5× at d = 128, C = 64). Both still take about 2C dependent steps per chunk. BL has C gain steps plus its C×C WY inverse; here there are the C×C Cholesky and its inverse. So the ratio compares the width of those steps (d-vectors against rows of a C×C matrix), not their latency. With triangular solves instead of an explicit inverse, the form needs fewer FLOPs (1.20–2.76× BL v5), and 4.7–5.6× fewer than MesaNet with 30 CG steps (`logs/flop_model_v3.log`). On the GPU, the smaller order-dependent work did not make the plain-PyTorch form faster than BL's Triton loops (GPU benchmark, below).
- **The benchmarked code, checked on the CPU and the GPU (E2; `logs/check_batched.log`, `logs/gpu_bench_v2.log` part A).** The code the GPU benchmark times is batched over heads. It covers the chunk form, BL's chunk algorithm as read here (its gain loop with v5's token-by-token mean pass or v1's WY mean pass, in PyTorch and with this study's Triton kernels), and the token-by-token filter. In float64 all of them are within 8.9e-16 of the extended-precision filter over 16 cases with 3 heads each on the CPU. On the GPU, with the Triton kernels compiled for it, they are within 1.1e-15 over five cases. Their gradients match the filter's to ≤ 9.1e-16, or 1.6e-13 when ill-conditioned. In float32, three algebraically equal token-by-token filters differ by up to 5.1× in the hardest configuration, so error ratios between methods smaller than that may come from the summation order alone.
- **GPU benchmark (E3; RTX 5060 Ti, float32 with TF32 off; `logs/aggregate_gpu_bench_v5.log`).** The grid: T = 2048, N = 16 or 64 heads, d = e = 64 or 128, C = 16–128, and scalar or per-channel gates. In the forward pass the chunk form, in plain PyTorch, is mostly not faster than BL's chunk algorithm with its two loops as this study's Triton kernels:
  - with each method at its own fastest chunk size, BL takes 0.66–1.08× the chunk form's time eager and 0.68–0.99× as CUDA graphs;
  - in backlog I-016's hypothesis cells (d = 128, C = 32–64), BL takes 0.73–0.92× the chunk form's time eager and 0.79–1.09× as graphs. By the decision rule written down before the run, BL is faster beyond the spread between runs in 7 of 8 cells eager and 4 of 8 as graphs, and the chunk form in none.

  So claim 6's 6.5–12.5× smaller order-dependent work does not make the plain-PyTorch chunk form faster. No single stage dominates its eager time (Cholesky 8–15%, triangular solve 13–22%), and each stage is one or more separate kernels, while each of BL's loops is one fused kernel.

  Through autograd, BL's algorithm in PyTorch loops takes 1.8–6.0× the chunk form's forward + backward time as CUDA graphs, and 2.9–47× eager, where those loops are bound by kernel launches. The chunk form also needs less memory. This compares autograd paths only, because BL's own fused backward is not public. If that backward costs k times BL's Triton forward, BL's training step beats the chunk form's autograd step at every tested size for any k < 3.6 eager and k < 4.2 as graphs. Fused kernels in this benchmark range from k = 2.7–3.6 (fla's KDA and MesaNet) to k = 12.7–24.8 (fla's Gated DeltaNet), so training speed against BL is open.

  Against MesaNet with 30 CG steps, a different model, the hypothesis holds only in plain PyTorch. 0002's plain-PyTorch CG is slower than the chunk form in 14 of 16 cells in the forward pass, while flash-linear-attention's fused MesaNet kernel takes 0.04–0.44× its forward time and 0.04–0.25× its forward + backward time (d = 64). A rerun in fresh processes moved the per-cell ratios by 0.96–1.07× eager and 0.85–1.03× as CUDA graphs, and changed the faster family in no cell. In float32 on the GPU (part D, 4 heads per configuration), on hard inputs (cond(𝒦) 430–1.35e4), the chunk form's worst error is 1.8–30× the token-by-token filter's with triangular solves and 2.0–41× with the explicit L⁻¹. BL's chunk algorithm's is 2.2–33×.
- **Known, not claimed here.** The model and its Kalman recursion (BL, Definition 1 and Proposition 1; the target model of Kalman Delta Networks; the stationary scalar case in Van Vaerenbergh et al. 2012 and Titsias et al. 2024). The reduction to DeltaNet, Gated DeltaNet and KDA with an isotropic covariance (Gated KalmaNet, Section 6.4; BL, Proposition 2). The chunk's exit covariance as one matrix product (BL, eq. 27), and, without dynamics, the block update of a matrix memory through one observation covariance (the Kanerva Machine, 2018). The causal estimator from one triangular factorization (Kailath, Sayed & Hassibi). Block-exact RLS with a carried inverse (Slock & Maouche 1994).

**Verdict:** partly known; every ingredient is classical. The Kalman memory, its relation to the delta-rule family, its chunked exit covariance, the stationary prior and the identity behind Proposition 2 are all published. No prior publication was found, as of 2026-10-29, of their combination: a chunk-parallel exact Kalman memory in which one C×C Cholesky serves every in-chunk readout, with a carried dense posterior and per-channel decay. It is derived here and verified in float64 (E2), including its gradients. On an RTX 5060 Ti (E3), its forward pass in plain PyTorch is mostly not faster than BL's chunk algorithm with BL's loops as Triton kernels. With each method at its fastest chunk size, it is ahead only eager, at N = 16 and d = 128, by 4–11%. Backlog I-016's hypothesis was that its 6.5–12.5× smaller order-dependent work would make it faster at d = 128 and C = 32–64, and faster than MesaNet with 30 CG steps. The first part holds in no cell, and BL is faster beyond the spread between runs in 7 of 8 cells eager and 4 of 8 as CUDA graphs. So it is refuted at these sizes, for the plain-PyTorch form against Triton loops. The second part holds only against CG in plain PyTorch, in the forward pass, and not against flash-linear-attention's fused MesaNet kernel. Through autograd the chunk form needs no custom kernel to train, and as CUDA graphs its forward + backward takes 1.8–6.0× less time than BL's algorithm in PyTorch loops. That is not a comparison with BL's own fused backward, which this study does not have: BL's training step would be faster at every tested size if that backward costs under about 3.6× BL's forward. Training speed against BL is therefore open. In floating point it is usually less accurate than the token-by-token filter. On well-conditioned float32 chunks with forgetting, its worst error is 0.84–3.5× the filter's on the CPU. Once cond(𝒦) reaches 430–1.35e4, with slow or no forgetting and strong writes or anisotropic keys, its worst error with the explicit L⁻¹ (the faster variant) is 2.3–56× the filter's on the CPU and 2.0–41× on the GPU. With triangular solves on the GPU it is 1.8–30×, and BL's chunk algorithm with Triton loops 2.2–33×. Without forgetting, though, the float32 token-by-token filter drifts and the chunk form does not. Its interest is practical. Three recent ML papers name the sequential covariance recursion as the reason they changed the model or approximated the covariance: Gated KalmaNet keeps the memory static, and Kalman Delta Networks and Voltic use diagonal, isotropic or quasi-diagonal covariances. The classical block form removes that obstacle in principle, and trains through autograd without a custom kernel. Whether a fused kernel would make its forward pass faster than BL's is open.

## Claims

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | For diagonal transitions and diagonal process noise, the covariance 𝒦 of a chunk's observations is the same for every reading token, so one Cholesky 𝒦 = LLᵀ gives all C exact readouts: O = (Q ⊙ 𝒜)M_0ᵀ + tril(Yᵀ)R̃ with Y = L⁻¹X and R̃ = L⁻¹R. (The classical causal-filter identity, applied per chunk with a carried posterior.) Over 26 float64 cases, the four implemented forms are within 1.2e-15 of an extended-precision token-by-token filter | E2 | Method, Proposition 2; `logs/check_chunk_v5.log` part A |
| 2 | The boundary posterior follows by matrix products: M_C = M_0𝒜_C + R̃ᵀZ̃ and P_C = 𝒜_CP_0𝒜_C + φ_C − Z̃ᵀZ̃ (the covariance part is equivalent to BL's eq. 27). Final M and P are within 3.0e-15 and 5.5e-16 over the same cases | E2 | Method, Proposition 3; `logs/check_chunk_v5.log` part A |
| 3 | The readouts' gradients with respect to 𝒦, X and R come from two more triangular solves with the same factor: ∂ℓ/∂X = Ā, ∂ℓ/∂R = W̄Δ, and, 𝒦 being symmetric, ∂ℓ/∂𝒦 = −½(ĀW̄ᵀ + W̄Āᵀ). These match autograd to ≤ 8.4e-16. On four well-conditioned configurations, autograd through the chunk form matches autograd through the token-by-token filter to ≤ 1.3e-15 for every input. The inputs include M_0 and P_init = SSᵀ in one of them. Central differences agree to ≤ 3.2e-9. On an ill-conditioned configuration (gates 0.999–1, β = 50, ridge 1e-2) the two autograd paths differ by up to 2.0e-13, mirroring the forward pass, and the central-difference check is inconclusive (gap 6.3e-6 at h = 1e-5) | E2 | Method, Proposition 6; `logs/check_chunk_v5.log` part D |
| 4 | With stationary noise (I − D_t²)Λ⁻¹, the prior of every W_t is N(0, Λ⁻¹) (published for a scalar gate: Van Vaerenbergh et al. 2012, Titsias et al. 2024). With gates equal to 1 the readout is the Mesa readout G_t(H_t + Λ)⁻¹q_t (2.9e-15). With gates below 1 it differs from MesaNet's: a stored pair fades by the gate product a, while MesaNet keeps it near full strength as long as aβ‖k‖² ≫ λ (both formulas to ≤ 2.2e-14) | E2 | Method, Proposition 1; `logs/check_chunk_v5.log` part B |
| 5 | 𝒦 is a sum of three positive semidefinite terms, formed without cancellation. With stationary noise its eigenvalues lie in [1/β_max, 1/β_min + Σ_i k_iᵀΛ⁻¹k_i]; with scalar gates and unit keys, cond(𝒦) ≤ β_max/β_min + (β_max/λ)·min(C, (1 + γ_max)/(1 − γ_max)). No violation in 120 chunks with histories of 0–200 tokens; with unit keys, cond(𝒦) was 0.018–0.41 of the bound | E2 | Method, Proposition 4; `logs/check_chunk_v5.log` part C |
| 6 | Forward FLOPs per token and head (e = d = 64–256, C = 32–128). With L⁻¹ formed explicitly: 1.23–3.38× BL v5's chunk algorithm, with order-dependent work only on C×C matrices, 2.0–24.5× less than BL v5's gain loop with v1's parallel mean pass. With triangular solves: 1.20–2.76× BL v5. 4.72–5.57× fewer than MesaNet with 30 CG steps; 1.39–1.85× study 0001's ungated carried inverse. Exact decoding: 2.03–2.12× fewer FLOPs than 9 CG steps | E1 (FLOP model) | `logs/flop_model_v3.log` |
| 7 | With the covariance replaced by a multiple of the identity, the mean update is Gated DeltaNet's recurrence (scalar gates) or KDA's (per-channel), to 7.2e-16 | E2; published (Gated KalmaNet, Section 6.4; BL, Proposition 2) | Method, Proposition 5; `logs/check_chunk_v5.log` part B |
| 8 | In float64, across a 48-row conditioning sweep, the chunk form's error is ≤ 0.46·cond(𝒦)·eps in 46 rows. The two exceptions (unit keys, gates equal to 1, ridge 1e-4 and 1e-6: 11 and 951 × cond·eps) are driven by the prior-to-posterior downdate, which also degrades the token-by-token filter. The chunk form's error is 2.1–195× the token-by-token filter's, depending on the keys: 2.1–8.2× for unit keys, 11–56× for a decaying spectrum, 12–195× for rank-4 key blocks. cond(𝒦) stays ≤ 9.7e3 with gates 0.99–1, but reaches 1.0e5 at 0.999–1, 1.0e6 at 0.9999–1 and 5.2e8 with gates equal to 1 (errors 1.6–4.0e-8 there, token-by-token 2.7e-9) | E2 | `logs/check_chunk_v5.log` part A2 |
| 9 | In float32 (CPU, d = e = C = 64, ten configurations, sequences up to 65,536 tokens), P stays positive definite at every chunk boundary in every form. With cond(𝒦) ≤ 7 the chunk forms give 2.3–9.3e-7. The token-by-token filter gives 1.4–6.0e-7 with forgetting, but without it drifts from 4.6e-7 to 6.8e-6 (the chunk form there: 4.5e-7 to 7.9e-7). With cond(𝒦) 430–1.35e4 the chunk forms reach 9.0e-6–3.7e-4, 2.3–56× the token-by-token filter's error (whose own error moves by up to 5.1× with the summation order; claim 12). Running the C×C stage in float64 barely helps; forming 𝒦 in float64 cuts the error 2–10× | E2 (CPU numerics, not GPU) | `logs/check_float32_v2.log` part E1 |
| 10 | Proposition 4(a)'s numerics. With per-channel gates of 0.1–0.3, the GLA-style matrix-product Grams overflow in float32 and the Cholesky fails, while the log-space Grams are unaffected. At gates 0.9999–1 with β = 50, the algebraically equal "P_0 − Λ⁻¹" form of 𝒦 has 52× the float32 error of the sum of PSD terms (1.7e-5 against 3.3e-7, relative to λ_min(𝒦)); in the other five settings the ratio is 0.69–5.4× | E2 | `logs/check_float32_v2.log` parts E1–E2 |
| 11 | The batched code that the GPU benchmark times computes the filter exactly. In float64, over 16 cases with 3 heads each, the outputs are within 8.9e-16 of the extended-precision token-by-token filter and the final states within 1.2e-15. The cases cover scalar and per-channel gates from 0.01 to 1, stationary and free noise, T not a multiple of C, T < C, C = 1, d = 4–48, β from 1e-3 to 50 and ridges from 1e-3 to 10. The forms are the chunk form (triangular solves, explicit L⁻¹, matrix-product Grams), BL's chunk algorithm as read here (gain loop with v5's token-by-token mean or v1's WY mean, in PyTorch and with the Triton kernels under Triton's interpreter) and the batched filter. Autograd through each PyTorch form matches autograd through the filter to ≤ 9.1e-16 on three configurations, and to ≤ 1.6e-13 on an ill-conditioned one (gates 0.999–1, β = 50, ridge 1e-2). On the GPU, in float64, every form, the Triton kernels compiled for it included, is within 1.1e-15 of the extended-precision filter over five cases | E2 | Method, Implementation for the GPU benchmark; `logs/check_batched.log` parts P1–P2; `logs/gpu_bench_v2.log` part A |
| 12 | In float32 the token-by-token filter's error depends on its summation order. Three algebraically equal versions (the single-head reference, and the batched filter with kᵀPk as an elementwise sum or as a matrix product) differ on one head by up to 1.3× with forgetting, 1.7× without forgetting and with strong writes, and 5.1× without forgetting and with a decaying key spectrum (T = 4096, d = e = C = 64). Error ratios between methods smaller than this spread can come from the summation order alone | E2 (CPU numerics) | `logs/check_batched.log` part P4 |
| 13 | GPU forward pass against BL's chunk algorithm: the same model and float32 inputs, TF32 off, on an RTX 5060 Ti. T = 2048, N = 16 and 64 heads, d = e = 64 and 128, C = 16–128, scalar and per-channel gates 0.9–1; the chunk form in plain PyTorch, BL's two loops as this study's Triton kernels. Best BL / best chunk form per cell: 0.63–1.69 eager and 0.59–1.91 as CUDA graphs. The chunk form is faster in 4 and 7 of 32 cells, 9 of these 11 at C = 128. There, at d = 128, BL's Triton form slows down: 13.86 ms at C = 128 against 8.90 ms at C = 64, for N = 16 with scalar gates. With each method at its own fastest chunk size: 0.66–1.08 eager and 0.68–0.99 as graphs, over 8 settings of N, d and gates. In backlog I-016's hypothesis cells (d = 128, C = 32–64): 0.73–0.92 eager and 0.79–1.09 as graphs. A rerun of part B in fresh processes moved the per-cell ratios by 0.955–1.073× eager and 0.848–1.032× as graphs, and changed the faster family in no cell. It gives 0.67–1.11 and 0.68–0.99 with each method at its fastest chunk size, and 0.73–0.99 eager and 0.78–1.08 as graphs in the hypothesis cells. By the decision rule written down before the run, BL is faster beyond that spread in 7 of the 8 hypothesis cells eager and 4 of 8 as graphs. The rest show no measurable difference, and the hypothesis holds in none | E3 (forward; BL's kernels are this study's) | Results, GPU benchmark; `logs/gpu_bench_v2.log`, `logs/gpu_bench_rerun_n16.log`, `logs/gpu_bench_rerun_n64.log`, `logs/aggregate_gpu_bench_v5.log` |
| 14 | GPU forward + backward through autograd, same grid. As CUDA graphs, BL's chunk algorithm in PyTorch loops with the WY mean, its faster PyTorch form, takes 1.83–6.02× the chunk form's time over 28 cells, and 2.26–6.02× with each at its fastest chunk size, over the 6 of 8 settings timed at every C. The token-by-token filter takes 8.69–27.49× over 8 cells (N = 16, d = 64). Captures needing over 5.6 GiB were skipped. Eager, where those loops are bound by kernel launches, the same ratios are 2.93–47.38×, 9.01–47.38× and 11.85–83.84×. Peak memory above the inputs: 0.23–3.13 GiB for the chunk form, 0.42–10.11 GiB for BL's loops and 3.08–12.44 GiB for the token-by-token filter. Where autograd would keep 25–48 GiB (BL's loop mean and the filter at N = 64, d = 128), they were not run. This compares autograd paths only: BL's own backward, a fused kernel, is not public. If it costs k times BL's Triton forward, BL's training step beats the chunk form's autograd step in all 8 settings for any k < 3.64 eager and k < 4.17 as graphs. Fused kernels in this benchmark have k = 2.7–3.1 (fla's KDA), 3.1–3.6 (its MesaNet) and 12.7–24.8 (its Gated DeltaNet), so training speed against BL is open | E3 (autograd only; BL's own backward not available) | Results, GPU benchmark; `logs/gpu_bench_v2.log`, `logs/gpu_bench_graph_bwd_n16.log`, `logs/gpu_bench_graph_bwd_n64.log`, `logs/aggregate_gpu_bench_v5.log` |
| 15 | References from other models at the same sizes, time over the chunk form's. Forward: study 0001's ungated carried-inverse layer 0.55–1.04× (the chunk form faster in 2 of 32 cells). MesaNet with 30 CG steps in 0002's plain PyTorch 0.71–11.23× (chunk form faster in 14 of 16; scalar gates). 0002's exact solver 1.74–67.86×. flash-linear-attention's fused Triton kernels in float32: 0.02–0.62× (Gated DeltaNet), 0.02–1.40× (KDA; chunk form faster in 4 of 16) and 0.04–0.44× (MesaNet with 30 CG steps; d = 64 only, since at d = 128 it needs more shared memory than this GPU has). Forward + backward, against the chunk form through autograd: 0001's closed-form backward 0.15–0.28×, fla's Gated DeltaNet 0.10–1.18× (chunk form faster in 2 of 16), KDA 0.02–0.79× and MesaNet with 30 + 30 CG steps 0.04–0.25×. So backlog I-016's claim that the chunk form beats 30-step CG holds against CG in plain PyTorch, forward only, and fails against fla's fused MesaNet kernel | E3 | Results, GPU benchmark; same logs |
| 16 | Float32 on the GPU. In part B (gates 0.9–1, unit keys, ridge 1), the output errors are 2.8e-7–8.2e-7 for the chunk form (growing with C), 1.7e-7–3.0e-7 for BL's Triton form and 2.4e-7–4.0e-7 for the token-by-token filter. Part D (check_float32.py's ten configurations, 4 heads, up to 65,536 tokens) reproduces claim 9: P stays positive definite at every chunk boundary of every run with finite output, and without forgetting the token-by-token filter drifts from 5.6e-7 to 6.3e-6 while the chunk form stays ≤ 7.9e-7. With cond(𝒦) 430–1.35e4, the chunk form's worst error is 1.77–30.01× the filter's, and BL's Triton form's 2.18–33.27×. Without forgetting, BL's token-by-token mean pass drifts too (4.7e-7 to 4.8e-6), although its covariance error (1.7e-6) equals the chunk form's. With the same PyTorch gains, that mean pass reaches 5.1e-6 while the WY mean stays ≤ 7.7e-7, both with covariance error 1.6e-6 | E3 (GPU numerics) | Results, GPU benchmark; `logs/gpu_bench_v2.log`, `logs/gpu_bench_cd.log` part D, `logs/gpu_bench_drift_bl.log`, `logs/aggregate_gpu_bench_v5.log` |

## Problem

A least-squares memory keeps key and value statistics and reads them with a ridge regression. For one head, with keys and queries in ℝ^d, values in ℝ^e, write strengths β_t > 0 and a diagonal ridge Λ ≻ 0, MesaNet computes

```math
H_t = \gamma_t H_{t-1} + \beta_t k_t k_t^\top, \qquad G_t = \gamma_t G_{t-1} + \beta_t v_t k_t^\top, \qquad o_t = G_t\,(H_t + \Lambda)^{-1} q_t .
```

Equivalently, o_t applies to q_t the minimizer of Σ_{i≤t} w_{ti} β_i ‖W k_i − v_i‖² + tr(W Λ Wᵀ), with w_{ti} = γ_{i+1}⋯γ_t. The gates down-weight old pairs **relative to the reading token t**, while the ridge does not decay. Study 0002 wrote this as a per-token ridge: inside a chunk, every token solves an ungated system on nested data with its own ridge Λ/a_t, where a_t is the cumulative gate (its claim 1, E2). Two consequences:

- MesaNet and its flash-linear-attention kernels run 30 CG steps in training, in the forward and the backward pass (0002, Problem).
- 0002's exact chunk solver needs one d×d eigendecomposition per chunk boundary. On an RTX 5060 Ti it took 27–96% of the forward solve, which stayed 2.1–8.8× slower than 30-step CG (0002, claim 9, E3).

Without gates, study 0001 showed that one C×C Cholesky per chunk yields every prefix solve, because the Cholesky factor of a leading block is the leading block of the factor (its claim 1, E2). That needs the systems of successive tokens to be nested, and MesaNet's gates break the nesting.

The Bayesian Layer (Dowling et al. 2026) is a least-squares memory of a different kind: a Kalman filter on the memory, whose forgetting is state dynamics plus process noise. Its exact chunk algorithm needs no d×d factorization. It does, however, compute each token's gain from the previous gains in a loop over the chunk's positions. Its current version then carries the mean through the chunk token by token (BL v5, Appendix C.2: "the gain and mean passes retain sequential dependencies over the chunk's positions"). Its first version computed the mean in parallel with DeltaNet's WY form once the gains were known (BL v1, Appendix D). Each step of the gain loop works on d-dimensional vectors. Chunkwise algorithms for linear attention and DeltaNet exist to remove exactly such loops. Other recent Kalman layers give up the dense covariance instead. Kalman Delta Networks calls the exact recursion "not directly compatible with fully parallel linear attention" (its Section 4), and Voltic's abstract says a dense covariance "would have to be propagated token by token". They train diagonal, isotropic or quasi-diagonal approximations. The question here is whether the Kalman memory's in-chunk filter can be computed in parallel, exactly.

## Method

### Model

One head, in this lab's notation. The memory W_t ∈ ℝ^{e×d} is the transpose of BL's S_t ∈ ℝ^{D×m} (so d = D, e = m). Gates D_t = γ_tI (scalar) or D_t = diag(α_t) (per-channel) have entries in (0, 1]; Ψ_t ⪰ 0 is a diagonal process-noise covariance. The model is BL's design model (its Definition 1) with diagonal transitions and diagonal noise:

```math
W_0 \sim \mathcal{MN}\big(0,\ P_{\text{init}}\big), \qquad
W_t = W_{t-1} D_t + E_t,\quad E_t \sim \mathcal{MN}(0,\ \Psi_t),
```

```math
v_t = W_t k_t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal N\big(0,\ \beta_t^{-1} I_e\big), \qquad
o_t = \mathbb E\,[\,W_t \mid v_1, \dots, v_t\,]\; q_t .
```

Here 𝓜𝓝(0, Σ) means the e rows are independent N(0, Σ). β_t, MesaNet's write strength, is the precision of observation t, and the readout is the posterior mean of the memory applied to the query. BL uses isotropic noise ℓ_t²I and P_init = σ_0²I, with ℓ_t and σ_0 free (BL, Definition 1 and Appendix C.1).

This study's default is the **stationary** choice P_init = Λ⁻¹ and Ψ_t = (I − D_t²)Λ⁻¹, with Λ diagonal. Its noise refills, channel by channel, exactly the prior variance that the decay removed. For scalar gates and Λ = λI it is BL's model with ℓ_t² = (1 − γ_t²)/λ and σ_0² = 1/λ. For per-channel gates the noise is diagonal instead of isotropic. The scalar version is published elsewhere: it is the "back-to-the-prior" forgetting of the kernel RLS tracker, shown to equal a Gaussian process with covariance λ^{|t−t′|/2}k(x, x′) (Van Vaerenbergh et al. 2012), and the variance-preserving drift N(γ_iw_{i−1}, (1 − γ_i²)σ²I) of Titsias et al. (2024), whose output columns also share one covariance.

### Proposition 1: stationary prior, Kalman recursion, and Mesa as the gate-free case

**(a)** With the stationary choice, every W_t has prior 𝓜𝓝(0, Λ⁻¹), since D_tΛ⁻¹D_t + (I − D_t²)Λ⁻¹ = Λ⁻¹ (diagonal matrices commute). The ridge is Λ at every token: it neither decays, as in Mesa-P, nor accumulates.

**(b)** All rows see the same keys and isotropic observation noise, so the posterior is 𝓜𝓝(M_t, P_t), with mean M_t ∈ ℝ^{e×d} and one covariance P_t ∈ ℝ^{d×d} shared by the rows. The Kalman filter gives it token by token (BL, Proposition 1):

```math
M_t^- = M_{t-1} D_t, \qquad P_t^- = D_t P_{t-1} D_t + \Psi_t,
```

```math
s_t = k_t^\top P_t^- k_t + \beta_t^{-1}, \quad g_t = P_t^- k_t / s_t, \quad
M_t = M_t^- + (v_t - M_t^- k_t)\, g_t^\top, \quad P_t = P_t^- - s_t\, g_t g_t^\top, \quad o_t = M_t q_t .
```

This is exact recurrent decoding with O(d² + de) work per token and no linear solve. MesaNet decodes with CG, about 9 steps on average at its inference tolerance (0002, Problem).

**(c)** With D_t = I and Ψ_t = 0, the filter is recursive least squares with prior precision P_init⁻¹ = Λ: P_t = (Λ + H_t)⁻¹ and M_t = G_tP_t, with H_t and G_t the ungated sums. So o_t = G_t(H_t + Λ)⁻¹q_t, the Mesa readout of study 0001.

**(d)** With gates below 1 it is not MesaNet. Take one pair (k, v) written τ tokens before the readout, with the stationary choice, a constant gate γ, Λ = λI and a = γ^τ. Then

```math
o^{\text{Kalman}} = a\,\frac{\beta\,(k^\top q)}{\beta\|k\|^2 + \lambda}\, v, \qquad
o^{\text{MesaNet}} = \frac{a\beta\,(k^\top q)}{a\beta\|k\|^2 + \lambda}\, v .
```

The two agree at a = 1. The Kalman memory scales the association by a, as gated linear attention and Gated DeltaNet do. MesaNet keeps it near full strength while aβ‖k‖² ≫ λ and lets it fade only once the down-weighted pair is weaker than the ridge. Which behavior trains better is an empirical question that this study does not answer.

### Proposition 2: one Cholesky per chunk gives every readout

Take a chunk of C tokens, renumbered 1…C, entered with the posterior (M_0, P_0). Write 𝒜_i = D_1⋯D_i for the cumulative decay (𝒜_0 = I) and 𝒜_{i/j} = D_{j+1}⋯D_i for i ≥ j. Let φ_m be the in-chunk noise variance at token m:

```math
\varphi_0 = 0, \qquad \varphi_m = D_m^2\, \varphi_{m-1} + \Psi_m \qquad \big(\text{stationary choice: } \varphi_m = (I - \mathcal A_m^2)\,\Lambda^{-1}\big).
```

All of these are diagonal. For tokens i, i′ let m = min(i, i′) and m′ = max(i, i′). Define the C×C matrix 𝒦, the cross terms κ_{ti} for i ≤ t, and the residuals r_i:

```math
\mathcal K_{ii'} = (\mathcal A_i k_i)^\top P_0\, (\mathcal A_{i'} k_{i'})
\;+\; k_i^\top \mathcal A_{m'/m}\, \varphi_m\, k_{i'}
\;+\; \beta_i^{-1}\,\delta_{ii'},
```

```math
\kappa_{ti} = (\mathcal A_t q_t)^\top P_0\, (\mathcal A_i k_i) \;+\; q_t^\top \mathcal A_{t/i}\, \varphi_i\, k_i,
\qquad r_i = v_i - M_0 \mathcal A_i k_i .
```

Then, for every token t of the chunk,

```math
o_t \;=\; M_0 \mathcal A_t q_t \;+\; \sum_{i \le t} r_i \,\big[\mathcal K_{[t]}^{-1} \kappa_{t,[t]}\big]_i ,
```

where 𝒦_{[t]} is the leading t×t block of 𝒦 and κ_{t,[t]} = (κ_{t1}, …, κ_{tt}). With one Cholesky factorization 𝒦 = LLᵀ, all C readouts come at once. Stack the residuals as the rows of R ∈ ℝ^{C×e}, let X ∈ ℝ^{C×C} hold X_{it} = κ_{ti} for i ≤ t and 0 below, and let R̃ = L⁻¹R and Y = L⁻¹X. Then

```math
O = (Q \odot \mathcal A)\, M_0^\top + \operatorname{tril}\!\big(Y^\top\big)\, \tilde R ,
```

where the rows of Q ⊙ 𝒜 are (𝒜_t q_t)ᵀ and tril keeps the entries (t, j) with j ≤ t.

*Proof.* Unrolling the dynamics gives W_i = W_0𝒜_i + Σ_{j≤i} E_j 𝒜_{i/j}. Take one row of W; the rows are independent and identically structured. Given the data before the chunk, that row of W_0 is N(m_0, P_0), with m_0 the matching row of M_0, and the observations v_i and the target y_t = W_t q_t (that row's entries) are jointly Gaussian. Their means are M_0𝒜_i k_i and M_0𝒜_t q_t. The covariance of two observations is

```math
\operatorname{Cov}(v_i, v_{i'}) = (\mathcal A_i k_i)^\top P_0 (\mathcal A_{i'} k_{i'}) + \sum_{j \le m} (\mathcal A_{i/j} k_i)^\top \Psi_j (\mathcal A_{i'/j} k_{i'}) + \beta_i^{-1}\delta_{ii'} .
```

All matrices in the sum are diagonal, and 𝒜_{m′/j} = 𝒜_{m′/m}𝒜_{m/j}, so the sum is k_iᵀ𝒜_{m′/m}(Σ_{j≤m}𝒜_{m/j}²Ψ_j)k_{i′} = k_iᵀ𝒜_{m′/m}φ_mk_{i′}. For the stationary choice, (1 − D_j²)/𝒜_j² = 𝒜_j⁻² − 𝒜_{j−1}⁻² channel by channel, so the sum telescopes to φ_m = 𝒜_m²(𝒜_m⁻² − I)Λ⁻¹ = (I − 𝒜_m²)Λ⁻¹. This gives 𝒦_{ii′}. The same computation with (t, q_t) in place of (i′, k_{i′}) gives κ_{ti}. Gaussian conditioning then yields E[y_t | v_1, …, v_t] = M_0𝒜_tq_t + κ_{t,[t]}ᵀ𝒦_{[t]}⁻¹(r_1, …, r_t).

The key fact is that Cov(v_i, v_{i′}) involves only tokens i and i′, never the reading token t. So the covariance of the first t observations is the leading block 𝒦_{[t]} for every t. For a lower-triangular L, the first t entries of L⁻¹x depend only on the first t entries of x, and L_{[t]} is the Cholesky factor of 𝒦_{[t]}. Hence κ_{t,[t]}ᵀ𝒦_{[t]}⁻¹R_{[t]} = (L⁻¹X_{:,t})_{[t]}ᵀ(L⁻¹R)_{[t]}, which is row t of the matrix formula. ∎

**Where the identity comes from, and how it relates to BL's loop.** Sequential Gaussian conditioning is a triangular factorization of the observations' covariance. L⁻¹ maps the residuals to normalized innovations: row t of L⁻¹R is the innovation of v_t given v_1, …, v_{t−1}, divided by its standard deviation L_tt. The general statement is the causal Wiener filter. If R_Y = LDL*, the best estimate of each X_i from Y_1, …, Y_i is K_fY with K_f = [R_XYL^{−*}D^{−1}]_L L^{−1}, where [·]_L keeps the lower triangle (Kailath, Sayed & Hassibi, Section 4.1.2, as given in Jiao's lecture notes [8]). Proposition 2 is that formula applied to one chunk, with X_t = W_tq_t, Y_i = v_i and all covariances conditional on the carried posterior. Slock and Maouche's subsampled-updating RLS [9] uses the same per-block mechanism for exponentially weighted RLS without process noise. It carries R⁻¹ across blocks with one L×L matrix per block, and recovers every in-block RLS error from that matrix's UDL factorization. BL's gain loop computes the same factor column by column, with d-dimensional vectors and a history of update directions. Proposition 2 forms 𝒦 first, by matrix products, and leaves only the C×C factorization order-dependent. Chunkwise linear attention and DeltaNet's WY/UT form apply the same reorganization to their recurrences.

**Why MesaNet's gates lack this structure.** In MesaNet's least-squares form (Problem), pair i enters token t's problem with weight w_{ti}β_i: an observation with noise variance 1/(w_{ti}β_i) against the fixed prior Λ. That variance changes with t, so the first t observations have a different joint covariance for every reading token. That is 0002's per-token ridge. In a state-space model, the observations' joint distribution is fixed once they are made, and only the cross terms κ_{ti} depend on t. Those enter linearly, so they cost matrix products, not factorizations.

**With all gates equal to 1 and no noise**, 𝒜 = I and φ = 0, so 𝒦 = KP_0Kᵀ + diag(β)⁻¹. After scaling the keys by √β this is study 0001's capacitance I + KB⁻¹Kᵀ (its claim 1, E2). Proposition 2 extends that identity to decaying memories with process noise.

### Proposition 3: the boundary update is matrix products

Let z_i = 𝒜_C P_0 𝒜_i k_i + 𝒜_{C/i}φ_i k_i be the covariance between a row of W_C and observation i, stacked as the columns of Z ∈ ℝ^{d×C}, and let Z̃ = L⁻¹Zᵀ. Then

```math
M_C = M_0\, \mathcal A_C + \tilde R^\top \tilde Z, \qquad
P_C = \mathcal A_C P_0\, \mathcal A_C + \varphi_C - \tilde Z^\top \tilde Z .
```

*Proof.* The same conditioning, applied to W_C instead of y_t. Its prior mean is M_0𝒜_C and, by the same sum, its prior covariance is 𝒜_CP_0𝒜_C + φ_C. ∎

The covariance part equals BL's exit covariance P_L = Π_L − W diag(β)Wᵀ (their eq. 27), with Z̃ᵀZ̃ = Z𝒦⁻¹Zᵀ in place of their sum over transported update directions. The chunk's only factorization is the C×C Cholesky of 𝒦.

### Proposition 4: numerics

**(a) No cancellation in 𝒦.** Its three terms are positive semidefinite: the boundary Gram (𝒜K)P_0(𝒜K)ᵀ, the in-chunk noise Gram (the covariance of Σ_j E_j𝒜_{i/j}k_i), and diag(β)⁻¹. φ_m is built by a recursion of nonnegative terms. With scalar gates, the decay factors are formed in log space, 𝒜_{m′/m} = exp(log 𝒜_{m′} − log 𝒜_m) ∈ (0, 1], and multiply the Gram matrices entrywise, so no product of gates is divided by another. With per-channel gates, the matrix-product form that the FLOP count assumes scales keys by 𝒜_i and by (1 − 𝒜_i²)/𝒜_i, as GLA's chunk form does. In float32 that overflows once a channel's log cumulative decay within the chunk falls below about −88.7; per-channel gates of 0.2 over 64 tokens give about −103. GLA's remedies apply: secondary chunking, or log-space products. With the stationary choice, I − 𝒜_m² = −expm1(2 log 𝒜_m) stays accurate when the gates are close to 1. An algebraically equal form writes the first two terms as (𝒜K)(P_0 − Λ⁻¹)(𝒜K)ᵀ plus the prior Gram F below. It cancels when the gates are near 1 and the boundary covariance is small, so it is avoided.

**(b) Bounded conditioning (stationary choice).** λ_min(𝒦) ≥ 1/β_max. Conditioning on data can only shrink a Gaussian covariance, so the noiseless part of 𝒦 is bounded by the prior covariance of the observations, F_{ii′} = k_iᵀΛ⁻¹𝒜_{m′/m}k_{i′}. Hence λ_max(𝒦) ≤ 1/β_min + λ_max(F) ≤ 1/β_min + Σ_i k_iᵀΛ⁻¹k_i. With scalar gates, F = (KΛ⁻¹Kᵀ) ∘ S, where S_{ii′} = 𝒜_{m′/m} is the correlation matrix of a unit-variance AR(1) process. By Schur's inequality for Hadamard products of positive semidefinite matrices, λ_max(F) ≤ max_i k_iᵀΛ⁻¹k_i · λ_max(S), and λ_max(S) ≤ min(C, (1 + γ_max)/(1 − γ_max)) by its row sums. For unit keys and Λ = λI:

```math
\operatorname{cond}(\mathcal K) \;\le\; \frac{\beta_{\max}}{\beta_{\min}} + \frac{\beta_{\max}}{\lambda}\,\min\!\Big(C,\ \frac{1 + \gamma_{\max}}{1 - \gamma_{\max}}\Big).
```

The bound does not depend on how long the sequence has run. It is of the same order as for MesaNet's own d×d system, cond(H_t + Λ) ≤ 1 + β_max/(λ(1 − γ_max)), but only a C×C matrix is factored.

**(c) What is not covered.** The covariance update P_C = (…) − Z̃ᵀZ̃ subtracts, as every covariance-form Kalman filter does. In float32, over many chunks with precise observations, P can drift or lose positive semidefiniteness. Square-root (array) Kalman forms avoid that, but their chunk form was not derived here. BL keeps its recurrent quantities in float32 and floors innovation denominators at 1e-6 (BL, Appendix C.3). In exact arithmetic, the stationary choice gives P_t ⪯ Λ⁻¹, so the state cannot grow without bound. Rounding was not analyzed.

### Proposition 5: Gated DeltaNet and KDA keep the mean and drop the covariance (published)

Replace the predicted covariance P_t⁻ in the gain of Proposition 1(b) by p_tI. Then g_t = b_tk_t with b_t = p_tβ_t/(1 + p_tβ_t‖k_t‖²), and

```math
M_t = M_{t-1} D_t \big(I - b_t\, k_t k_t^\top\big) + b_t\, v_t k_t^\top .
```

With D_t = γ_tI this is Gated DeltaNet's recurrence. With per-channel D_t it is KDA's, in the convention o_t = M_tq_t. Gated KalmaNet derived the DeltaNet and Gated DeltaNet cases with a fixed identity covariance (its Section 6.4), and BL's Proposition 2 and Table 1 give this form, including KDA and Longhorn. It is restated here because it places the chunk-parallel form: those layers keep the Kalman memory's mean dynamics with an isotropic covariance, and Proposition 2 makes the full covariance cost one C×C Cholesky per chunk.

### Proposition 6: the backward pass reuses the factor

Let Δ ∈ ℝ^{C×e} hold the incoming gradients δ_t = ∂ℓ/∂o_t as rows, and let U be the upper-triangular C×C mask, U_{jt} = 1 for j ≤ t. Two more triangular solves with Lᵀ give

```math
\bar A = L^{-\top}\big((\tilde R\, \Delta^\top) \odot U\big), \qquad \bar W = L^{-\top}\,(Y \odot U),
```

whose columns are a_t = 𝒦_{[t]}⁻¹R_{[t]}δ_t and w_t = 𝒦_{[t]}⁻¹κ_{t,[t]}, padded with zeros below row t. The readouts' gradients with respect to the chunk's intermediate matrices are then

```math
\frac{\partial \ell}{\partial X} = \bar A, \qquad \frac{\partial \ell}{\partial R} = \bar W \Delta, \qquad \frac{\partial \ell}{\partial \mathcal K} = -\tfrac12\big(\bar A\, \bar W^\top + \bar W \bar A^\top\big).
```

*Proof.* Row t of the readout is o_tᵀ = (M_0𝒜_tq_t)ᵀ + κ_{t,[t]}ᵀ𝒦_{[t]}⁻¹R_{[t]}. With d(𝒦⁻¹) = −𝒦⁻¹(d𝒦)𝒦⁻¹, a change d𝒦 changes token t's term δ_tᵀo_t by −w_tᵀ(d𝒦_{[t]})a_t. 𝒦 is symmetric and is only ever perturbed symmetrically, so its gradient is the symmetric part, −½(a_tw_tᵀ + w_ta_tᵀ). (Without that convention, the unsymmetrized form would depend on how o_t is written: −w_ta_tᵀ for κᵀ𝒦⁻¹R, −a_tw_tᵀ for Rᵀ𝒦⁻¹κ.) Token t also contributes a_t to ∂ℓ/∂κ_{t,[t]} and w_tδ_tᵀ to ∂ℓ/∂R_{[t]}. Summing over t with zero padding gives the matrix forms. For the columns: the leading block of L⁻¹ is L_{[t]}⁻¹, so L_{[t]}⁻ᵀb_{[t]} is the first t entries of L⁻ᵀ applied to b with its entries after t set to zero, and the remaining entries of that product vanish because L⁻ᵀ is upper triangular. ∎

The boundary outputs M_C and P_C add the ordinary inverse-derivative terms, with the full 𝒦⁻¹ = L⁻ᵀL⁻¹. From there to K, Q, V, β, the gates, the noise, M_0 and P_0 it is the chain rule through the products that formed 𝒦, X, R and Z. The extra C×C work is two triangular solves with C right-hand sides and one C×C product (4C³ FLOPs), plus 4C²e for R̃Δᵀ and W̄Δ. No step of the backward pass needs a d×d factorization or a loop over positions on d-dimensional vectors. The chain rule through the products was not written out here, and the training-pass cost was not modeled.

### Remark: other process noise

D_t = I with Ψ_t ≻ 0 is a random walk: associations do not decay, but their uncertainty grows, so new writes overwrite them more readily, which is closer to the ungated delta rule. Noise along the keys, Ψ_t = ψ_tk_tk_tᵀ, is a form of directional forgetting. It is not diagonal, but its in-chunk noise Gram is still a product of C×C matrices, B diag(ψ)Bᵀ with B_{ij} = k_iᵀ𝒜_{i/j}k_j for j ≤ i and 0 otherwise: O(C²d + C³) per chunk. BL's rotational transitions (blocks of damped 2×2 rotations) are not covered by the formulas as written. The key fact, that the observations' covariance does not depend on the reading token, holds for any linear-Gaussian transition; only the diagonal products and the φ_m shortcut assume diagonal transitions. Damped 2×2 rotations commute and are diagonal over the complex numbers (BL v1 says as much), so the formulas probably carry over with complex cumulative products. That was not checked. None of these variants is analyzed further here.

### Algorithm, one head

1. Carry (M, P), starting from (0, P_init).
2. For each chunk, compute the log cumulative gates and φ, then the matrices of Proposition 2: two d×d products (𝒜K)P_0 and (𝒜Q)P_0, the decayed Gram matrices of keys with keys and queries with keys (with per-channel gates these are GLA's intra-chunk products), and the residuals.
3. Factor 𝒦 = LLᵀ (C×C) and solve L⁻¹[R, X, Zᵀ].
4. Read out O; update (M, P) by Proposition 3.

Within a chunk every token is handled at once. The decayed Gram matrices without P_0 depend only on the chunk's own inputs and can be computed for all chunks in parallel. The terms with P_0, the Cholesky and the update form a sequential loop over chunks, as in 0001's carried-inverse layer and in BL. The state is P (d×d) and M (e×d), the same size as MesaNet's H and G.

### Cost

Counting a multiply-add as 2 FLOPs, per chunk and head, with triangular solves: 6Cd² for (𝒜K)P_0, (𝒜Q)P_0 and the covariance update; 9C²d for the four C×C Gram products and L⁻¹Zᵀ; 6Cde for the residuals, the readout from M_0 and the mean update; 3C²e for L⁻¹R and the masked output product; and (4/3)C³ for the Cholesky and L⁻¹X. Masked products are counted dense. Decoding costs 5d² + 7de + 4d per token.

**Order-dependent work.** A triangular solve is a loop over the chunk's positions, and here its right-hand sides have e + C + d columns. Forming L⁻¹ explicitly by forward substitution on the identity (C³/3) turns L⁻¹R, L⁻¹X and L⁻¹Zᵀ into matrix products. DeltaNet's chunk form does the same with its C×C inverse. The total becomes 6Cd² + 10C²d + 6Cde + 4C²e + (8/3)C³ per chunk. The only order-dependent work left is the Cholesky and the inverse, (2/3)C³ per chunk, and none of it scales with d or e.

For BL's chunk algorithm, as read from its Appendix C.2–C.3 (v5): 2Cd² for the entry-covariance products, a gain loop with 4td work at position t (2C²d in all), a mean pass of 7de per token (prediction, residual, rank-one update and readout) and 2Cd² for the exit covariance, so 4d² + 2Cd + 7de per token. BL's paper states the same orders: O(LD²) per chunk for the covariances, O(L²D) for the gain history and O(LDm) for the mean and readouts. Its gain loop and mean pass are order-dependent and scale with d (2C²d + 7Cde per chunk). BL's first version computed the mean pass in parallel with DeltaNet's WY form once the gains were known (BL v1, Appendix D). The strongest combination of BL's published pieces therefore keeps only the gain loop (2C²d) and a C×C inverse (C³/3) order-dependent; its total FLOPs are not modeled here. BL counts fewer FLOPs than Proposition 2 because it never forms the full C×C matrices.

`flop_model.py` evaluates these formulas, together with the CG formula of study 0002 (MesaNet's state update and readout plus k CG steps), 0002's formula for 0001's ungated carried inverse, and gated linear attention without a solve. Comparisons with CG are between different models: they compare what each costs to read out exactly (Kalman) or as MesaNet trains (fixed-step CG). For scale, in 0002's float64 tests CG from zero needed 20–30 steps to reach 1e-4 output error, and 5–30 with MesaNet's diagonal starting guess, depending on the keys (0002, claim 7, E2).

### Implementation for the GPU benchmark

`kalman_batched.py` computes the same filter three ways, for N heads at once, on any device, in plain PyTorch that autograd can differentiate.

- **The chunk-parallel form** (`kalman_chunkwise`). The parts of 𝒦, X and Z that do not depend on the carried state are formed for all chunks at once. The in-chunk noise Gram is a Hadamard product with decays built in log space for scalar gates with stationary noise, and the GLA-style product of Proposition 4(a) otherwise. Each chunk then takes:
  - three batched products: (𝒜K)P_0, then (𝒜K)P_0[𝒜K; 𝒜Q]ᵀ and [𝒜K; 𝒜Q]M_0ᵀ;
  - one batched C×C Cholesky (`torch.linalg.cholesky_ex`, which runs cuSOLVER. PyTorch 2.11 has deprecated MAGMA and sends the call to cuSOLVER even when MAGMA is preferred; part A of the benchmark finds the outputs bitwise equal, so MAGMA is not timed);
  - one triangular solve with e + C + d right-hand sides, or a product with an explicit L⁻¹;
  - three fused multiply-adds, for O, M_C and P_C.
- **The Bayesian Layer's chunk algorithm** (`bl_chunkwise`), as this study reads BL v5's Appendix C.2–C.3 and v1's Appendix D (NOTEBOOK.md, 2026-10-29). Π_tk_t = 𝒜_tP_0𝒜_tk_t + φ_tk_t comes from one product for the whole chunk. A loop over the chunk's positions then computes, with the earlier update directions w_j transported to token t,

```math
w_j \leftarrow D_t\, w_j \ \ (j < t), \qquad u_t = \Pi_t k_t - \sum_{j<t} \frac{w_j\,(w_j^\top k_t)}{s_j}, \qquad s_t = \max\!\big(k_t^\top u_t + \beta_t^{-1},\ 10^{-6}\big), \qquad w_t = u_t .
```

  Here u_t = P_t⁻k_t and s_t is the innovation variance; the floor is BL's, and with β ≤ 50 it never binds. The exit covariance is P_C = Π_C − Σ_j w_jw_jᵀ/s_j (BL eq. 27), formed as the Gram matrix of the w_j/√s_j. With the gains g_t = u_t/s_t, the mean is carried token by token (v5: M ← MD_t, r_t = v_t − Mk_t, M ← M + r_tg_tᵀ, o_t = Mq_t), or computed per chunk in a WY form (v1). For the WY form, the innovations r_t, stacked as R_inn, solve a unit lower-triangular C×C system, and give the readouts and the next mean:

```math
(I + L_g)\, R_{\text{inn}} = V - (\mathcal A K)\, M_0^\top, \qquad (L_g)_{tj} = g_j^\top \mathcal A_{t/j}\, k_t \ \ (j < t),
```

```math
O = (\mathcal A Q)\, M_0^\top + G_q R_{\text{inn}}, \qquad (G_q)_{tj} = g_j^\top \mathcal A_{t/j}\, q_t \ \ (j \le t), \qquad M_C = M_0 \mathcal A_C + R_{\text{inn}}^\top \big(\mathcal A_{C/j}\, g_j\big)_j .
```

  This has the structure of DeltaNet's UT transform. It is derived here, because BL v1's own formulas could not be re-read in this session. `bl_triton.py` runs BL's two loops as Triton kernels, the way BL's own implementation does (its code is not public). One program per head runs a chunk's gain loop with the transported directions in registers. One program per head and block of 16 value rows carries M through the whole sequence. Both prefetch the next token's vectors, and on the GPU num_warps is autotuned for each shape and number of heads. They run forward only.
- **The token-by-token filter** (`kalman_recurrent`): Proposition 1(b), one step per token, with the exactly symmetric downdate P − PkPkᵀ/s of `kalman_memory.kalman_reference`.

The chunk forms pad T to a multiple of C with zero keys, queries and values, gates 1, no noise and β = 1. Such a token's observation carries no information about the memory, and the state does not move, so the outputs and the final state stay exact.

`gpu_bench.py` times every method on the same float32 inputs with TF32 off. `run_gpu.sh` runs the GPU probe, a smoke test, then the benchmark in five runs (parts A–B for N = 16, part B for N = 64, parts C–D, then part B again for each N in fresh processes, to measure how stable the times are), and `aggregate_gpu_bench.py`'s tables. The grid: N = 16 or 64 independent heads, T = 2048, d = e = 64 or 128, C = 16, 32, 64 or 128, scalar or per-channel gates uniform on 0.9–1, unit keys, ridge 1, β log-uniform on 0.5–2. Each method's time includes its own chunk-local precomputation. Forward passes are timed under `no_grad`, and also as CUDA-graph replays, which take the CPU's kernel-launch overhead out of the comparison: in eager PyTorch, loops over positions are launch-bound. Forward + backward passes are timed by autograd for the PyTorch forms, eager and as CUDA-graph replays. A capture is skipped when the eager pass needs over 40% of the memory cap, because a capture that fails leaves PyTorch's allocator holding on to memory. For scale, references from other models run at the same sizes: study 0001's carried-inverse Mesa layer (no gates, closed-form backward; timed eager only, since its Cholesky's error check synchronizes with the host and cannot be captured in a CUDA graph); MesaNet with 30 CG steps and study 0002's exact chunk solver, both from 0002's plain-PyTorch benchmark (forward only, scalar gates); and, when installed, flash-linear-attention's Gated DeltaNet, KDA and MesaNet kernels. Part A first checks every Kalman method in float64 on the GPU against the extended-precision filter. Part C splits the time into stages with CUDA events. Part D repeats `check_float32.py`'s ten long-sequence configurations on the GPU, with 4 heads each.

## Results

### Exactness in float64 (E2)

`check_chunk.py` (`logs/check_chunk_v5.log`, `results/check_chunk_20261029_221909.json`) compares the chunk form with the token-by-token Kalman filter of Proposition 1(b). The reference runs in extended precision (x87 long double, eps 1.1e-19) on the float64 inputs, so the float64 token-by-token filter can be scored on the same yardstick. Errors are max|o − o_ref| / max|o_ref|.

| Case (selection; 26 in the log) | C | cond(𝒦) max | triangular solves | explicit L⁻¹ | matmul Grams | per-token prefix solves | float64 token-by-token |
|---|---|---|---|---|---|---|---|
| per-channel gates 0.01–1 | 8 | 4.53 | 3.2e-16 | 3.0e-16 | 2.8e-16 | 4.0e-16 | 3.3e-16 |
| no forgetting (gates = 1) | 8 | 7.16 | 6.2e-16 | 4.6e-16 | 6.2e-16 | 5.7e-16 | 4.8e-16 |
| near resets (gates 1e-6 to 1e-3) | 8 | 3.54 | 6.4e-16 | 5.0e-16 | 6.4e-16 | 6.4e-16 | 1.4e-16 |
| C does not divide T (37, 8) | 8 | 4.27 | 6.7e-16 | 5.5e-16 | 3.9e-16 | 3.0e-16 | 3.7e-16 |
| mixed β 1e-3 to 50 | 8 | 607 | 3.2e-16 | 2.6e-16 | 2.8e-16 | 2.8e-16 | 4.6e-16 |
| free isotropic noise, P_init = 3I (BL) | 8 | 5.52 | 6.7e-16 | 6.1e-16 | 4.4e-16 | 3.7e-16 | 2.2e-16 |
| long: T = 513, d = 32 | 64 | 12.3 | 5.4e-16 | 4.7e-16 | 5.1e-16 | 1.2e-15 | 2.6e-16 |
| **worst of all 26** | | | 8.9e-16 | 7.4e-16 | 8.6e-16 | 1.2e-15 | 4.8e-16 |

The final states agree too: M_T within 3.0e-15 and P_T within 5.5e-16. The other checks in the same log:
- **Gate-free limit.** It equals the Mesa readout G_t(H_t + Λ)⁻¹q_t to 2.9e-15.
- **Proposition 1(d).** Both one-pair formulas hold to ≤ 2.2e-14 over 20 draws of gate, ridge, β and age.
- **Proposition 5.** The isotropic-covariance mean equals the Gated DeltaNet and KDA recurrences to 7.2e-16, with b_t‖k_t‖² ≤ 0.92 < 1.
- **The Remark.** The directional-noise Gram B diag(ψ)Bᵀ matches the direct triple sum to 1.5e-16.
- **The stationary φ.** By recursion and in closed form, it matches extended precision to 3.8e-16 for gates down to 1 − 1e-10.
- **Proposition 4(b).** No violation of any bound in 120 chunks: λ_min ≥ 1/β_max, both upper bounds on λ_max, the unit-key condition bound, and P_0 ⪯ Λ⁻¹.

**Gradients.** Proposition 6's adjoint matches autograd to ≤ 8.4e-16. The unsymmetrized gradient of the readout written as κᵀ𝒦⁻¹R is −W̄Āᵀ, to 8.4e-16, as the review found; the earlier −ĀW̄ᵀ is off by 1.4 (relative). Autograd through the chunk form, with triangular solves or an explicit inverse, matches autograd through the token-by-token filter on four well-conditioned configurations (gates 0.5–1, β 0.5–5, ridge 0.5–2, T = 29, d = 8). The inputs are K, V, Q, log β, the log gates, and either the ridge or the free noise; one configuration also has M_0 and P_init = SSᵀ. The match is ≤ 1.3e-15. Central differences on the reference agree to 7.6e-10–3.2e-9, the truncation error of differences with h = 1e-5. On an ill-conditioned configuration (gates 0.999–1, β = 50, ridge 1e-2, M_0 an input), the two autograd paths differ by up to 2.0e-13, as the forward passes do. The central-difference gap there is 6.3e-6, so at h = 1e-5 that check is inconclusive.

**Conditioning (part A2).** Strong writes (β = 50) against ridges from 1 down to 1e-6, scalar gates 0.99–1, 0.999–1, 0.9999–1 or exactly 1. The keys are d = C = 16 unit keys, a key spectrum decaying as i⁻², or d = 4 keys in chunks of 32, whose key block has rank 4. Selected rows of the 48:

| keys | gates | ridge | cond(𝒦) max | chunk form, worst of three | float64 token-by-token | error / (cond(𝒦)·eps) | chunk / token-by-token |
|---|---|---|---|---|---|---|---|
| decay | 0.99–1 | 1e-4 | 8.79e3 | 7.4e-14 | 2.2e-15 | 0.04 | 32.7 |
| rank 4 | 0.99–1 | 1e-6 | 9.68e3 | 4.4e-14 | 4.6e-16 | 0.02 | 96.1 |
| rank 4 | 0.999–1 | 1e-6 | 1.00e5 | 2.0e-13 | 3.0e-15 | 0.01 | 66.2 |
| rank 4 | 0.9999–1 | 1e-6 | 1.00e6 | 3.2e-12 | 1.7e-14 | 0.01 | 194.9 |
| decay | 0.9999–1 | 1e-6 | 1.59e5 | 4.3e-12 | 1.3e-13 | 0.12 | 33.9 |
| unit | 1 | 1e-4 | 6.81e3 | 1.7e-11 | 3.6e-12 | 11.00 | 4.7 |
| unit | 1 | 1e-6 | 6.84e3 | 1.4e-9 | 5.5e-10 | 950.89 | 2.6 |
| decay | 1 | 1e-6 | 4.03e7 | 4.1e-9 | 1.1e-10 | 0.46 | 37.3 |
| rank 4 | 1 | 1e-6 | 5.15e8 | 4.0e-8 | 2.7e-9 | 0.35 | 15.0 |

Here eps is 2⁻⁵². With gates 0.99–1, the stationary noise grows as the ridge shrinks, and cond(𝒦) stays below 1e4. Slower gates refill less, so cond(𝒦) reaches 1e5 at 0.999–1 and 1e6 at 0.9999–1. Without forgetting, nothing bounds it but the β/λ term of Proposition 4(b).

In 46 of the 48 rows the chunk form's error is at most 0.46·cond(𝒦)·eps. The two exceptions have unit keys, no forgetting and ridges of 1e-4 and 1e-6. There, the update from a prior variance of 1/λ = 1e4–1e6 down to the posterior dominates, and the token-by-token filter degrades with it (ratios 4.7× and 2.6×). So a fallback chosen by cond(𝒦) would not catch those two cases. The token-by-token filter, which conditions on one scalar innovation at a time, is otherwise more accurate. The size of that gap depends on the keys more than on cond(𝒦): 2.1–8.2× for unit keys, 11–56× for the decaying spectrum, and 12–195× for rank-4 key blocks.

### Float32 over long sequences (E2, CPU numerics)

`check_float32.py` (`logs/check_float32_v2.log`, `results/check_float32_20261029_222118.json`) runs everything in float32 and compares it with the float64 token-by-token filter, with d = e = C = 64 and stationary noise. The table gives the largest error over the eight segments of each sequence. Version 2 added four configurations, the regimes where nothing, or almost nothing, refills the prior: the three without forgetting, and strong writes at gates 0.9999–1. It also added the variant that forms 𝒦 itself in float64. The six earlier configurations reproduce version 1 exactly.

| Configuration | T | cond(𝒦) max | chunk, triangular solves | chunk, explicit L⁻¹ | chunk, C×C stage in float64 | chunk, 𝒦 formed in float64 | float32 token-by-token |
|---|---|---|---|---|---|---|---|
| scalar gates 0.99–1 | 65,536 | 5.97 | 5.1e-7 | 4.1e-7 | 4.5e-7 | 5.0e-7 | 6.0e-7 |
| scalar gates 0.9–1 | 65,536 | 4.32 | 5.1e-7 | 4.6e-7 | 4.2e-7 | 3.7e-7 | 2.9e-7 |
| per-channel gates 0.95–1 | 16,384 | 5.18 | 4.2e-7 | 3.5e-7 | 3.6e-7 | 3.1e-7 | 3.1e-7 |
| per-channel gates 0.1–0.3 | 4,096 | 3.17 | 8.4e-7 | 8.4e-7 | 8.3e-7 | 8.3e-7 | 2.4e-7 |
| no forgetting (gates = 1), β 0.5–2, ridge 1 | 65,536 | 6.72 | 7.9e-7 (from 4.5e-7) | 7.9e-7 | 8.3e-7 | 9.3e-7 | 6.8e-6 (from 4.6e-7) |
| strong writes: β 50, ridge 1e-2, gates 0.999–1 | 16,384 | 430 | 8.2e-6 | 9.0e-6 | 6.5e-6 | 1.5e-6 | 3.9e-6 |
| strong writes, gates 0.9999–1 | 65,536 | 2.03e3 | 5.3e-5 | 6.5e-5 | 3.5e-5 | 1.4e-5 | 1.9e-5 |
| decaying key spectrum, β 10, ridge 1e-2, gates 0.99–1 | 16,384 | 2.71e3 | 1.5e-5 | 1.5e-5 | 1.4e-5 | 1.5e-6 | 6.1e-7 |
| no forgetting, strong writes | 65,536 | 1.29e4 | 1.9e-4 | 3.0e-4 | 1.4e-4 | 9.4e-5 | 6.6e-5 |
| no forgetting, decaying key spectrum | 65,536 | 1.35e4 | 1.8e-4 | 3.7e-4 | 2.1e-4 | 7.2e-5 | 6.5e-6 |

- **Well-conditioned rows (cond(𝒦) ≤ 7).** The chunk forms give 2.3–9.3e-7. The token-by-token filter gives 1.4–6.0e-7 with forgetting. Without forgetting, though, it drifts from 4.6e-7 to 6.8e-6 over 65,536 tokens, with a final covariance error of 1.6e-5, while the chunk form rises only from 4.5e-7 to 7.9e-7 (covariance error 1.3e-6). There, the chunk form is the more accurate of the two.
- **Ill-conditioned rows (cond(𝒦) 430–1.35e4).** The chunk forms' worst errors are 2.3–56× the token-by-token filter's (56× with the explicit inverse and a decaying spectrum without forgetting). The largest errors sit in the first segment, while the covariance still holds the prior's large variance, and they fall along the sequence.
- **Where the error enters.** Running the C×C stage in float64 barely helps, but forming 𝒦 itself in float64 cuts the worst error 2–10×. That brings four of the five ill-conditioned rows to within 2.5× of the token-by-token filter, two of them below it. The row without forgetting and with a decaying spectrum stays 11× above it. The remaining gap must come from the other chunk matrices and products, which were not isolated here.
- **Positive definiteness.** At every chunk boundary of every run, the covariance's smallest-to-largest eigenvalue ratio is positive, at least 7.4e-5, and the same for every form, so P stays positive definite.
- **Overflow.** With per-channel gates of 0.1–0.3, the matrix-product Grams overflow and their Cholesky fails, as Proposition 4(a) predicts; the log-space Grams are unaffected. Elsewhere the matrix-product Grams match the triangular-solve form to within 1.3×.

In part E2, the "P_0 − Λ⁻¹" form of 𝒦 has 52× the error of the sum of PSD terms at gates 0.9999–1 with β = 50 (1.7e-5 against 3.3e-7, relative to λ_min(𝒦)). Over the other five settings the ratio is 0.69–5.4× (`results/check_float32_20261029_222118.json`).

### Batched implementations for the GPU benchmark (E2)

`check_batched.py` (`logs/check_batched.log`, `results/check_batched_20261029_225300.json`) checks, on a CPU, the code that `gpu_bench.py` times, with the Triton kernels run by Triton's interpreter. Part P1 runs 16 cases with 3 heads each. They cover scalar and per-channel gates from 0.01 to 1 and no forgetting; stationary noise, free isotropic noise, and free diagonal noise with a dense P_init and a nonzero M_0; T not a multiple of C, T < C and C = 1; d = 4–48, β from 1e-3 to 50 and ridges from 1e-3 to 10. The worst errors against the extended-precision token-by-token filter, in float64:

| Form | outputs O | M_T | P_T |
|---|---|---|---|
| chunk, triangular solves | 8.9e-16 | 1.2e-15 | 9.1e-16 |
| chunk, explicit L⁻¹ | 8.9e-16 | 6.9e-16 | 5.8e-16 |
| chunk, matrix-product Grams | 8.9e-16 | 1.2e-15 | 9.1e-16 |
| BL, PyTorch, v5 mean pass | 7.7e-16 | 6.9e-16 | 5.8e-16 |
| BL, PyTorch, v1 WY mean pass | 7.2e-16 | 5.9e-16 | 5.8e-16 |
| BL, Triton gain loop and mean pass | 7.3e-16 | 6.9e-16 | 5.8e-16 |
| BL, Triton gain loop, WY mean pass | 6.6e-16 | 6.6e-16 | 5.8e-16 |
| token-by-token filter, batched | 6.3e-16 | 7.2e-16 | 8.7e-16 |

- **Gradients (part P2).** Autograd through the four PyTorch forms matches autograd through the batched filter, for every input, to ≤ 9.1e-16 on three configurations. They are scalar or per-channel gates with stationary noise, and free diagonal noise with M_0 and P_init = SSᵀ as inputs. On the ill-conditioned one (gates 0.999–1, β = 50, ridge 1e-2) the gap is up to 1.6e-13, as in claim 3.
- **Float32 (part P3; T = 2048, d = e = C = 64, 2 heads).** On the three well-conditioned configurations every form is within 2.0–5.5e-7 of the float64 filter; the float32 token-by-token filter gives 3.2–5.3e-7. With strong writes at gates 0.999–1, the chunk forms give 6.4–7.5e-6, BL's forms 4.8–5.5e-6 and the token-by-token filter 2.6e-6.
- **Summation order (part P4; T = 4096, 2 heads).** Three float32 token-by-token filters differ only in how they sum: `kalman_memory`'s single-head reference, and the batched filter with kᵀPk as an elementwise sum or as a matrix product. On one head they disagree by up to 1.3× on the two configurations with forgetting, 1.7× without forgetting and with strong writes, and 5.1× without forgetting and with a decaying key spectrum (3.9e-6 against 2.0e-5). On that last configuration the chunk form gives 2.5–2.7e-4 and BL with the WY mean 1.2–2.1e-4, far outside the spread. Ratios between methods smaller than the spread, though, can come from the summation order alone, and the GPU sums in yet other orders.

### GPU benchmark (E3)

`gpu_bench.py` ran on the reference GPU, an RTX 5060 Ti 16 GB, with PyTorch 2.11.0 + CUDA 12.8, Triton 3.6.0 and flash-linear-attention 0.6.0 (`logs/gpu_probe.log` and the first line of each log). Every method gets the same float32 inputs, with TF32 off:
- T = 2048 tokens and N = 16 or 64 independent heads;
- d = e = 64 or 128;
- unit keys, ridge 1, β log-uniform on 0.5–2;
- scalar or per-channel gates, uniform on 0.9–1.

Times are medians of 10 calls after 3 warmup calls, or of 3 calls when one takes over 0.5 s. Each includes the method's own chunk-local precomputation. Forward passes are timed eager under `no_grad`, and as CUDA-graph replays, which remove the CPU's kernel-launch overhead.

The runs:
- `logs/gpu_bench_v2.log`: parts A and B (`results/gpu_bench_20261030_021340.json`);
- `logs/gpu_bench_rerun_n16.log` and `logs/gpu_bench_rerun_n64.log`: part B again, in fresh processes (`results/gpu_bench_20261030_023500.json`, `results/gpu_bench_20261030_025256.json`);
- `logs/gpu_bench_cd.log`: parts C and D (`results/gpu_bench_20261030_021947.json`);
- `logs/gpu_bench_graph_bwd_n16.log` and `logs/gpu_bench_graph_bwd_n64.log`: part B for the chunk forms, BL with the WY mean and the token-by-token filter, with forward + backward also as CUDA graphs (`results/gpu_bench_20261101_102135.json`, `results/gpu_bench_20261101_103540.json`);
- `logs/gpu_bench_drift_bl.log`: part D's no-forgetting configuration with BL's token-by-token PyTorch mean added (`results/gpu_bench_20261101_103901.json`).

Every ratio and range below comes from `aggregate_gpu_bench.py` (`logs/aggregate_gpu_bench_v5.log`). NOTEBOOK.md records two runs that are not quoted. One is a discarded first run (`logs/gpu_bench.log`): after one failed CUDA-graph capture, PyTorch's allocator held on to memory until later methods ran out of it. The other is an interrupted graph run (`logs/gpu_bench_graph_bwd.log`), used only as a stability check.

- **Exactness on the GPU (part A).** In float64, every Kalman method, the Triton kernels compiled for this GPU included, is within 1.1e-15 of the extended-precision token-by-token filter over five cases. With MAGMA preferred, the Cholesky gives bitwise the same results as cuSOLVER in all five cases, so MAGMA is not timed (Method, Implementation).

**Forward pass against BL, each method at its fastest chunk size.** Times in ms, with that chunk size in brackets. The ratio is BL's time over the chunk form's, so above 1 the chunk form is faster. The fastest BL variant is always the Triton gain loop with the Triton mean pass. The fastest chunk-form variant is the explicit L⁻¹ in 14 of the 16 entries.

| N | d | gates | chunk form, eager | BL, eager | ratio | ratio, rerun | chunk form, graph | BL, graph | ratio | ratio, rerun |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64 | scalar | 6.64 (128) | 5.52 (128) | 0.83 | 0.85 | 4.63 (64) | 4.17 (64) | 0.90 | 0.83 |
| 16 | 64 | per-channel | 6.36 (128) | 5.67 (128) | 0.89 | 0.89 | 5.12 (128) | 4.41 (64) | 0.86 | 0.86 |
| 16 | 128 | scalar | 8.20 (128) | 8.90 (64) | 1.08 | 1.11 | 6.27 (64) | 6.24 (32) | 0.99 | 0.99 |
| 16 | 128 | per-channel | 8.51 (128) | 8.82 (64) | 1.04 | 1.05 | 6.89 (64) | 6.70 (32) | 0.97 | 0.97 |
| 64 | 64 | scalar | 13.33 (64) | 8.84 (64) | 0.66 | 0.68 | 10.51 (32) | 7.16 (32) | 0.68 | 0.68 |
| 64 | 64 | per-channel | 14.00 (64) | 9.52 (64) | 0.68 | 0.67 | 12.28 (64) | 8.51 (32) | 0.69 | 0.69 |
| 64 | 128 | scalar | 20.36 (64) | 16.61 (64) | 0.82 | 0.84 | 18.09 (32) | 15.01 (32) | 0.83 | 0.83 |
| 64 | 128 | per-channel | 23.24 (64) | 19.72 (64) | 0.85 | 0.83 | 21.54 (64) | 18.06 (32) | 0.84 | 0.84 |

The times are from the first run; "rerun" is the same ratio in the second run. At each method's fastest chunk size, the chunk form is ahead only eager, at N = 16 and d = 128, by 4–11%. That is beyond the 7.3% spread between runs with scalar gates (1.08, then 1.11), and within it with per-channel gates (1.04, then 1.05). Its fastest eager chunk size there, and in all four N = 16 settings, is 128, the largest tried. As CUDA graphs, BL is as fast or faster in all eight settings.

- **Backlog I-016's hypothesis cells (d = 128, C = 32–64).** Best BL / best chunk form is 0.73–0.92 eager (0.73–0.99 in the rerun) and 0.79–1.09 as CUDA graphs (0.78–1.08). The decision rule was written down before the run (NOTEBOOK.md); it takes the spread as the largest change of this ratio between the runs over all cells, 7.3% eager and 15.2% as graphs. By that rule:
  - BL is faster beyond the spread in 7 of 8 cells eager and 4 of 8 as graphs;
  - the other cells show no measurable difference, including the two (N = 16, C = 64) where the chunk form is ahead by 4–9% as graphs;
  - the hypothesis holds in no cell, so it is refuted at these sizes, for the plain-PyTorch chunk form against BL's Triton loops.
- **All 32 cells.** Best BL / best chunk form is 0.63–1.69 eager and 0.59–1.91 as graphs. The chunk form is faster in 4 cells eager and 7 as graphs, 9 of these 11 at C = 128. There, at d = 128, BL's Triton form slows down: 13.86 ms at C = 128 against 8.90 ms at C = 64, for N = 16 with scalar gates. Eager, the chunk form's time at C = 16 hardly depends on N and d (41.9–45.0 ms), because it is bound by kernel launches; as graphs it takes 7.70–32.27 ms there.
- **Where the time goes (part C).** Measured at C = 64 with per-channel gates, eager, with CUDA events between stages. The chunk form's 14.7–25.6 ms split into:
  - chunk-local precomputation, 6–36%;
  - the products that form 𝒦, X, R and Z, 17–32%;
  - the Cholesky, 8–15%;
  - the triangular solve, 13–22%;
  - the updates of O, M and P, 15–34%.

  BL's Triton form's 10.2–19.4 ms split into precomputation 4–19%, the entry products Π_tk_t 7–21%, the gain loop 31–55%, the exit covariance 10–19% and the mean pass 10–32%. No single stage dominates the chunk form, and each of its stages is one or more separate kernels per chunk. BL's gain loop is one kernel per chunk, and claim 6 counts it as BL's largest order-dependent work. These shares are eager, so they include the gaps between kernel launches. In the same setting (N = 16, d = 64, C = 64, per-channel gates), part B times the chunk form at 10.80 ms eager and 5.17 ms as a CUDA graph. A fused kernel (backlog I-019) should therefore be compared stage by stage as graphs, or with a profiler.
- **Forward + backward through autograd (claim 14).** BL's Triton kernels have no backward, and BL's own fused backward is not public, so training cost is compared only between autograd paths through PyTorch forms. Eager, BL's loops are bound by kernel launches, as in the forward, so the graph runs also time forward + backward as CUDA graphs. As a graph, BL's WY form takes 0.08–0.65× its eager time, and the chunk form 0.32–0.98×. Time over the chunk form's:

  | | eager | CUDA graphs |
  |---|---|---|
  | BL's PyTorch form, per cell | 2.93–47.38× | 1.83–6.02× (28 cells) |
  | BL's PyTorch form, each at its fastest chunk size | 9.01–47.38× | 2.26–6.02× (the 6 of 8 settings timed at every C) |
  | token-by-token filter, per cell | 11.85–83.84× | 8.69–27.49× (8 cells, N = 16, d = 64) |

  The chunk form is faster in every cell where both ran. The graph runs leave out BL's token-by-token PyTorch form, which is slower than its WY form in all 24 cells of eager forward + backward and all 32 of graphed forward. They skip the captures that need over 5.6 GiB: the WY form at N = 64, d = 128, C = 64–128, and the filter at (N, d) = (16, 128) and (64, 64). Autograd keeps intermediate results for every token: peak memory above the inputs is 0.23–3.13 GiB for the chunk form, 0.42–10.11 GiB for BL's loops and 3.08–12.44 GiB for the filter. BL's loop mean and the filter were not run at N = 64, d = 128, where autograd would keep 25–48 GiB.
- **What BL's fused backward could change.** Suppose it costs k times BL's Triton forward. Let R be the chunk form's fastest autograd forward + backward over BL's fastest Triton forward. R is 4.64–12.18 eager and 5.17–14.61 as graphs, per (N, d, gates). So BL's training step beats the chunk form's autograd step in all 8 settings for any k < 3.64 eager and k < 4.17 as graphs. At k = 2, the chunk form's step takes 1.55–4.06× and 1.72–4.87× BL's. Fused kernels in this benchmark fall on both sides of that threshold: k = 2.7–3.1 for fla's KDA, 3.1–3.6 for its MesaNet and 12.7–24.8 for its Gated DeltaNet. These multiples are forward + backward over a `no_grad` forward, whose path can differ from the one used in training. The chunk form's closed-form backward (Proposition 6 and a chain rule not yet written out) was not built either, so neither side is at its best here, and training speed against BL is open.
- **References from other models (claim 15).** Forward, as a multiple of the chunk form's time:
  - study 0001's ungated carried-inverse layer, timed eager only (Method): 0.55–1.04×;
  - 0002's plain-PyTorch MesaNet with 30 CG steps: 0.71–11.23×, so the chunk form is faster in 14 of 16 cells;
  - 0002's exact solver: 1.74–67.86×;
  - flash-linear-attention's fused Triton kernels: 0.02–0.62× (Gated DeltaNet), 0.02–1.40× (KDA) and 0.04–0.44× (MesaNet with 30 CG steps).

  0002's two take scalar gates only, and its exact solver ran out of memory at N = 64, C = 128. fla's MesaNet kernel runs at d = 64 only: at d = 128 it needs 116,236 bytes of shared memory per block, and this GPU allows 101,376. For forward + backward, against the chunk form through autograd, 0001's closed-form backward takes 0.15–0.28×, fla's Gated DeltaNet 0.10–1.18×, KDA 0.02–0.79× and MesaNet with 30 + 30 CG steps 0.04–0.25×. fla's kernels are tuned for bfloat16 and TF32. Here they run in float32 with `TRITON_F32_DEFAULT=ieee`, like every other method, except that fla 0.6.0 itself runs the intra-chunk triangular solves of Gated DeltaNet and KDA in TF32 on this GPU. The second part of backlog I-016's hypothesis, faster than 30-step CG, therefore holds only against CG in plain PyTorch, and only in the forward pass, which is all that 0002's benchmark times.
- **Float32 accuracy (part B).** Against the float64 token-by-token filter on the GPU, the output errors are:
  - chunk form: 2.8e-7–8.2e-7, largest at C = 128;
  - BL's Triton form: 1.7e-7–3.0e-7;
  - BL with the WY mean: 2.7e-7–7.5e-7;
  - the float32 token-by-token filter: 2.4e-7–4.0e-7.
- **Stability.** Part B ran twice, in separate processes. Per method and kind of timing, the median of the rerun's time over the first run's is 0.936–1.011. Single times moved by 0.772–1.291×, the widest for CUDA graphs of the loops over positions (BL's PyTorch forms and the token-by-token filter). Best BL / best chunk form moved by 0.955–1.073× per cell eager, 0.848–1.032× as graphs and 0.841–1.016× for forward + backward. No cell changed which family is faster. At each method's fastest chunk size, the rerun gives 0.67–1.11 eager, 0.68–0.99 as graphs and 8.08–46.91 for forward + backward. Forward + backward as graphs has no full rerun, but an earlier graph run finished 22 cells before its session ended. Against those cells the graph times have a median ratio of 1.000 for every method, single times moved by 0.947–1.074×, and BL's WY form over the chunk form moved by 0.932–1.042×. Eager times vary more across sessions: the graph runs' eager forward + backward, two days after the first run, is 0.933–1.245× the first run's for BL's WY form (median 1.076) and 0.950–1.136× for the chunk forms.

**Float32 drift on the GPU (part D).** The configurations of `check_float32.py`, 4 heads each, d = e = C = 64, against the float64 token-by-token filter on the GPU. The table gives the worst error over the eight segments of the sequence; in the first no-forgetting row, the first segment's error is in brackets.

| Configuration | T | cond(𝒦) max | chunk, triangular solves | chunk, explicit L⁻¹ | BL, Triton | BL, WY mean | float32 token-by-token |
|---|---|---|---|---|---|---|---|
| scalar gates 0.99–1 | 65,536 | 5.97 | 5.9e-7 | 4.7e-7 | 7.1e-7 | 5.1e-7 | 1.0e-6 |
| scalar gates 0.9–1 | 65,536 | 4.32 | 5.0e-7 | 5.0e-7 | 3.2e-7 | 4.6e-7 | 4.1e-7 |
| per-channel gates 0.95–1 | 16,384 | 5.18 | 3.9e-7 | 3.8e-7 | 3.2e-7 | 3.6e-7 | 3.7e-7 |
| per-channel gates 0.1–0.3 | 4,096 | 3.17 | non-finite | non-finite | 2.4e-7 | non-finite | 2.3e-7 |
| no forgetting (gates = 1), β 0.5–2, ridge 1 | 65,536 | 6.72 | 7.9e-7 (3.8e-7) | 7.8e-7 (4.4e-7) | 4.8e-6 (4.7e-7) | 7.7e-7 (3.6e-7) | 6.3e-6 (5.6e-7) |
| strong writes: β 50, ridge 1e-2, gates 0.999–1 | 16,384 | 430 | 7.3e-6 | 8.4e-6 | 1.1e-5 | 1.1e-5 | 4.1e-6 |
| strong writes, gates 0.9999–1 | 65,536 | 2.03e3 | 4.7e-5 | 4.9e-5 | 9.9e-5 | 9.8e-5 | 1.9e-5 |
| decaying key spectrum, β 10, ridge 1e-2, gates 0.99–1 | 16,384 | 2.71e3 | 1.4e-5 | 1.4e-5 | 3.0e-6 | 3.3e-6 | 8.9e-7 |
| no forgetting, strong writes | 65,536 | 1.29e4 | 2.3e-4 | 4.2e-4 | 1.6e-4 | 1.5e-4 | 7.3e-5 |
| no forgetting, decaying key spectrum | 65,536 | 1.35e4 | 2.5e-4 | 3.5e-4 | 2.8e-4 | 1.5e-4 | 8.5e-6 |

- **Without forgetting**, the token-by-token filter drifts from 5.6e-7 to 6.3e-6 over 65,536 tokens, as on the CPU, and the chunk form does not (claim 9). BL's Triton form drifts as well, from 4.7e-7 to 4.8e-6, although its final covariance error (1.7e-6) equals the chunk form's, against 1.9e-5 for the filter. BL with the WY mean, which updates the mean once per chunk, does not drift. `logs/gpu_bench_drift_bl.log` isolates the mean pass, because there BL's token-by-token and WY means use the same PyTorch gains. The first drifts from 4.4e-7 to 5.1e-6, the second stays ≤ 7.7e-7, and both have a final covariance error of 1.6e-6. So carrying the mean token by token without decay accumulates rounding error on its own, whatever happens to the covariance. That run also reproduces the row's other columns to the printed digits, two days after part D.
- **Ill-conditioned rows (cond(𝒦) 430–1.35e4).** The chunk form's worst error is 1.77–30.01× the token-by-token filter's (2.03–41.38× with the explicit L⁻¹). BL's chunk algorithm is about as inaccurate: 2.18–33.27× with the Triton loops and 2.07–17.13× with the WY mean. Neither form is uniformly worse. With a decaying key spectrum at gates 0.99–1, the chunk form gives 1.4e-5 against 3.0e-6 for BL's Triton form. With strong writes at gates 0.9999–1, it gives 4.7e-5 against BL's 9.9e-5.
- **Positive definiteness.** At every chunk boundary of every run with finite output, the smallest-to-largest eigenvalue ratio of P is at least 6.0e-5, and it is the same for every form.
- **Overflow.** With per-channel gates of 0.1–0.3 the chunk forms, and BL's WY mean, which also divides by cumulative decays, give non-finite outputs, as claim 10 predicts. BL's Triton loops do not.

### FLOP model (E1)

The FLOP model gives the following, per token and head, forward, with e = d (`logs/flop_model_v3.log`).

**Against BL's chunk algorithm (same model).** The chunk-parallel form is shown with L⁻¹ formed explicitly. Order-dependent work is split into the part that scales with d or e and the part on C×C matrices only:

| d | C | Chunk-parallel, all FLOPs | ratio to BL v5 | Chunk-parallel, order-dependent (C×C only) | BL v5, order-dependent (scales with d, e) | BL v5 gains + v1 WY mean, order-dependent (scales with d + C×C) | ratio to chunk-parallel |
|---|---|---|---|---|---|---|---|
| 64 | 32 | 81k | 1.64× | 0.68k | 32.8k | 4.1k + 0.34k | 6.50× |
| 64 | 64 | 117k | 2.21× | 2.7k | 36.9k | 8.2k + 1.4k | 3.50× |
| 64 | 128 | 208k | 3.38× | 10.9k | 45.1k | 16.4k + 5.5k | 2.00× |
| 128 | 32 | 257k | 1.36× | 0.68k | 122.9k | 8.2k + 0.34k | 12.50× |
| 128 | 64 | 322k | 1.64× | 2.7k | 131.1k | 16.4k + 1.4k | 6.50× |
| 128 | 128 | 470k | 2.21× | 10.9k | 147.5k | 32.8k + 5.5k | 3.50× |
| 256 | 32 | 904k | 1.23× | 0.68k | 475.1k | 16.4k + 0.34k | 24.50× |
| 256 | 64 | 1027k | 1.36× | 2.7k | 491.5k | 32.8k + 1.4k | 12.50× |
| 256 | 128 | 1289k | 1.64× | 10.9k | 524.3k | 65.5k + 5.5k | 6.50× |

The chunk-parallel form pays 1.23–3.38× BL v5's FLOPs. In exchange, its order-dependent work is 2.0–24.5× smaller than the best BL combination's, and none of it scales with d or e. The ratio is 3d/C + 1/2, so the trade is best for large heads and short chunks, and weakest at d = 64 with C = 128. With triangular solves instead of an explicit inverse, the chunk-parallel form needs fewer FLOPs, 1.20–2.76× BL v5's (the version 2 table, repeated in `logs/flop_model_v3.log`), but its solves loop over positions with e + C + d right-hand sides.

**Against other layers** (triangular-solve form; ratios are each method's FLOPs over the chunk-parallel form's):

| d | C | Chunk-parallel Kalman | MesaNet, 30 CG steps | MesaNet, 10 CG steps | 0001 carried inverse (no gates) | Gated linear attention |
|---|---|---|---|---|---|---|
| 64 | 64 | 104k | 545k (5.26×) | 204k (1.96×) | 63k (0.61×) | 25k (0.24×) |
| 128 | 64 | 300k | 1631k (5.43×) | 620k (2.07×) | 198k (0.66×) | 82k (0.27×) |
| 256 | 64 | 989k | 5426k (5.49×) | 2093k (2.12×) | 689k (0.70×) | 295k (0.30×) |

Over all nine settings, 30-step CG costs 4.72–5.57× the chunk-parallel form, 10-step CG 1.74–2.16×, and the chunk-parallel form costs 1.39–1.85× the ungated carried inverse and 3.18–5.17× gated linear attention. For decoding, 9 CG steps cost 2.03–2.12× the exact Kalman recursion and 30 steps 5.60–5.90× (same log).

FLOPs are not time. 0002's exact solver had a competitive FLOP count, but its eigendecomposition ran far below the GPU's matrix-product speed (0002, claim 9, E3). The chunk-parallel form's work is matrix products plus one small Cholesky and triangular solves per chunk, the structure of 0001's carried inverse. On this GPU, BL's loops as this study's Triton kernels beat the plain-PyTorch chunk form in the forward pass anyway (GPU benchmark, above).

## Novelty check

Searched on 2026-10-29 by two `literature-scout` agents, one on the ML literature and code, one on statistics, time series, Gaussian processes, Kalman filtering, adaptive filtering and econometrics. Sources: web search (about 50 queries in all), the arXiv API, Semantic Scholar citations (of Kalman Linear Attention, Gated KalmaNet and Kalman Delta Networks; the API often returned HTTP 429), OpenAlex (including the citing works of subsampled-updating RLS and of the kernel RLS tracker), the OpenReview search API, the flash-linear-attention repository, publisher and repository pages, and lecture notes. NOTEBOOK.md has the query log. Every work in the table was opened in this session to the extent stated.

| Work | What it does | Difference |
|---|---|---|
| Bayesian Layer, "Memory by Design" [1] (v5 PDF pp. 1–20; v1 PDF pp. 1 and 14–20) | The same model: a Kalman filter on a matrix memory, with a dense covariance shared across value columns, diagonal or rotational transitions and free isotropic process noise. Proposition 2: resetting the covariance gives DeltaNet, Gated DeltaNet, KDA and Longhorn. Exact chunk computation, with the exit covariance as one matrix product. In both versions the gains come from a sequential loop over the chunk's positions. v5 also carries the mean token by token; v1 computed it in parallel with DeltaNet's WY form once the gains were known. Custom backward in Triton | No chunk-parallel form for the gains: that loop is what Proposition 2 removes (and v5's mean loop with it). No noise tied to the gates |
| The Kanerva Machine [12] (pp. 3–4) | A matrix memory M ~ MN(R, U, I) whose K×K row covariance U is shared by its columns. An episode is written in one Bayesian update through the T×T observation covariance Σ_z = WUWᵀ + Σ_ξ: R ← R + Σ_cᵀΣ_z⁻¹Δ, U ← U − Σ_cᵀΣ_z⁻¹Σ_c (eqs. 9–11). The paper notes this "is equivalent to performing the one-sample/on-line update iteratively" | Proposition 3's block update without dynamics or process noise (𝒜 = I, Ψ = 0), years before BL's eq. 27. The memory is read after the episode is written, not at every step |
| Sakov & Bocquet [13] (abstract only; the paper returned 403) | Asynchronous EnKF with additive model error: observations at several times in a window enter one analysis, generalizing a rule that "matches the optimal analysis in the linear perfect model case" | An ensemble method with one analysis per window, as described in the abstract; no per-time filtered outputs from one factorization. Not opened beyond the abstract |
| Kalman Delta Networks [2] (abstract; Sections 4 and 6) | The same exact model as its target, with learned process noise; the delta rule as its fixed-gain special case. Section 4: the exact recursion "is not directly compatible with fully parallel linear attention due to its Riccati covariance recursion". Trains diagonal and isotropic relaxations with associative scans | Approximate covariance. Proposition 2 is an exact chunk-parallel form of the dense filter |
| Voltic [3] (abstract) | Kalman-style write gains with an anisotropic covariance. Abstract: "A dense covariance would have to be propagated token by token, ruling out the parallel training these models depend on." Uses diagonal and quasi-diagonal approximations | Approximate; the same difference |
| Gated KalmaNet [4] (Sections 6.3–6.4) | DeltaNet and Gated DeltaNet as Kalman filters with a fixed identity error covariance (Eqs. 16–19). Calls the general Kalman recursion "inherently sequential". Its own layer is a static-state ridge regression solved by Chebyshev iteration | The earliest isotropic-covariance reduction found; its own layer is the MesaNet-type model, not the state-space one |
| Kalman Linear Attention [5] (abstract) | Exact Bayesian filtering in information form as an associative scan, time-parallel | A per-channel (diagonal) belief over a vector state, not a key–value matrix memory; the diagonal covariance is from the agent's reading of its Section 4 and from study 0001's search |
| Bogunovic, Scarlett & Cevher [6] (pp. 1–4) | Time-varying GP bandits: f_{t+1} = √(1−ε) f_t + √ε g_{t+1}, so f_t ~ GP(0, k) at every t; posterior with the Hadamard product K_t ∘ D_t, D_t = [(1−ε)^{|i−j|/2}] (Eqs. 4, 7–8) | The same stationary prior and kernel identity for a constant gate: with a linear kernel it is Proposition 2's 𝒦 with scalar gates, at the prior. A full t×t solve per step; no chunks, carried posterior, time-varying or per-channel gates |
| Särkkä & García-Fernández [7] (pp. 1–4) | Exact parallel-in-time Kalman filtering and smoothing by an associative scan, also with block-level elements | Each combine inverts a matrix of the state dimension, here d×d |
| MesaNet (through study 0002) | Exponentially weighted ridge regression with a fixed ridge, solved by CG | The contrast model. Its per-token ridge (0002, claim 1) is why it lacks the prefix structure |
| Causal Wiener filter by triangular factorization: Kailath, Sayed & Hassibi, *Linear Estimation*, Section 4.1.2, as given in Jiao's lecture notes [8] (pp. 1–4) | With R_Y = LDL*, the causal estimator is K_f = [R_XYL^{−*}D^{−1}]_L L^{−1}; the notes call it "not only conceptually important but also numerically efficient" | The general identity behind Proposition 2: no blocks, no carried posterior, no model |
| Slock & Maouche, subsampled-updating RLS [9] (preprint pp. 1–12) | Exponentially weighted RLS with a fading prior (eq. 2). R⁻¹ carried across blocks of L samples through one L×L "likelihood variable" matrix (eqs. 28–33). Every in-block RLS error recovered exactly from its UDL factorization (eqs. 41, 47). The authors note it "does not represent much computational gain" without their FFT-based fast version | The same per-block mechanism, without process noise: errors at the regressors rather than readouts at queries, scalar weighting only. It also predates study 0001's identity (see the erratum there) |
| Van Vaerenbergh, Santamaría & Lázaro-Gredilla [10] (pp. 1–4) | Kernel RLS tracker with "back-to-the-prior" forgetting, μ ← √λμ and Σ ← λΣ + (1−λ)K (eq. 4). Shown exactly equivalent to a GP with covariance λ^{|t−t′|/2}k(x, x′) (eqs. 5–6). Used with a linear kernel; the batch matrix Λ∘K + σ²I appears in the marginal likelihood (eq. 7) | The stationary scalar model, and 𝒦 for a first chunk at a constant gate, used only for fitting hyperparameters. Computation is sequential; no chunks, time-varying or per-channel gates |
| Titsias et al. [11] (pp. 3–4) | Kalman filter on linear predictor weights with p(w_i \| w_{i−1}) = N(γ_iw_{i−1}, (1 − γ_i²)σ²I) (eqs. 2–4), γ learned by SGD; K output columns share one covariance (eqs. 7–8) | The stationary scalar model with a shared covariance, computed sequentially, in continual learning rather than as a sequence layer |

**Conclusion.** Already published: the model (Bayesian Layer; the target model of Kalman Delta Networks), its Kalman recursion and exact recurrent decoding, and the isotropic-covariance reduction to DeltaNet, Gated DeltaNet and KDA (Gated KalmaNet, Bayesian Layer, Kalman Delta Networks). Also published: the block update of a matrix memory with a shared covariance (the Kanerva Machine, without dynamics), the chunked exit covariance as a matrix product (Bayesian Layer), and the stationary AR(1) prior with its Hadamard-product kernel (for a scalar gate: Van Vaerenbergh et al., Bogunovic et al., Titsias et al.; for a diagonal state: Kalman Linear Attention). Classical as well: the causal estimator from one triangular factorization (Kailath, Sayed & Hassibi), and block-exact RLS with a carried inverse (Slock & Maouche). No prior publication was found, as of 2026-10-29, of their combination: a chunk-parallel exact Kalman memory with a carried dense posterior, per-channel decay and readouts at queries, in which one C×C Cholesky serves every in-chunk readout. The contrast with MesaNet's forgetting is not claimed as new: it restates the known obstacle that a fixed regularizer under exponential forgetting is a full-rank change at every step (Lai & Bernstein, as cited by study 0002). So this is a classical result applied in a new setting. Three papers from 2025–2026 say that exact dense-covariance filtering is sequential or cannot be trained in parallel. They keep the memory static (Gated KalmaNet) or approximate the covariance (Kalman Delta Networks, Voltic) instead. If Proposition 2 holds at E2, the only order-dependent step left inside a chunk is a C×C factorization, as in DeltaNet's chunkwise algorithm. A web search cannot rule out unpublished or concurrent work.

## Limitations

- **One GPU, and plain PyTorch against Triton.** The timings come from one RTX 5060 Ti (36 SMs, 101,376 bytes of shared memory per block) in strict float32. The chunk form runs as plain PyTorch, with separate kernels for each stage of each chunk, while BL's two loops run as fused Triton kernels written in this study. No fused kernel was written for the chunk form, and CUDA graphs remove only the launch overhead, not each kernel's minimum time. So the forward comparison is between a plain-PyTorch form and a Triton form, at T = 2048, N = 16 and 64, and d = 64 and 128. Other GPUs, bfloat16 or TF32, longer sequences and larger heads were not tested. flash-linear-attention's kernels ran in float32, which they are not tuned for. The exception is Gated DeltaNet's and KDA's intra-chunk triangular solves, which fla 0.6.0 runs in TF32 on this GPU. Its MesaNet kernel does not fit this GPU at d = 128. The chunk form's fastest eager chunk size is 128, the largest tried, in all four N = 16 settings; larger chunks were not tried. The earlier scratch prototypes (NOTEBOOK.md) are superseded by `check_chunk.py` and `check_float32.py`.
- **Accuracy on ill-conditioned chunks.** The chunk form is exact, but its rounding error is about cond(𝒦)·eps. The token-by-token filter is usually more accurate: 2.1–195× in the float64 sweep, depending on the keys, and 2.3–56× in the float32 rows with cond(𝒦) of 430–1.35e4, where the chunk form reached 3.7e-4. Slow or no forgetting with strong writes or anisotropic keys is the bad regime; without forgetting, a weak prior pushes cond(𝒦) past 1e8. Forming 𝒦 in float64 cut the float32 error 2–10× but left one configuration 11× above the token-by-token filter. A square-root form of P_0's term, iterative refinement and a per-head fallback were not tried. The exception runs the other way: without forgetting, the float32 token-by-token filter drifted (to 6.8e-6 over 65,536 tokens) and the chunk form did not.
- **The comparison with BL rests on reading its paper.** BL's FLOP count comes from its Appendix C as read here and is a lower bound for its loop. The combination of its v5 gain loop with its v1 mean pass is this study's construction, and its total FLOPs were not modeled. BL's Triton kernel was not run, and BL's code is not public. The benchmark's BL is this study's implementation of the same reading: the gain loop and mean pass in PyTorch and as Triton kernels, forward only, and a WY mean pass derived here. The session that wrote it could not re-open BL's paper (arXiv was not reachable), and the reading leaves open whether BL transports its update directions at every step, as done here. BL's custom backward was not built, so forward + backward is compared only through autograd on the PyTorch forms.
- **FLOPs, not time.** The chunk-parallel form needs 1.23–3.38× more FLOPs than BL v5's form (1.20–2.76× with triangular solves). On this GPU its parallelism did not pay off in plain PyTorch (claim 13). The order-dependent C×C Cholesky and triangular solve took 8–15% and 13–22% of its forward time, and the rest went to products and updates spread over many small kernels (part C). With per-channel gates the decayed Gram matrices need GLA-style stabilization (secondary chunking or log-space products), which adds overhead the FLOP model ignores.
- **Rotational transitions not worked out.** BL also uses blocks of damped rotations. Proposition 2's formulas as written assume diagonal transitions; the extension through complex cumulative products (Remark) was not checked.
- **Per-channel gates in float32.** The matrix-product form divides by cumulative decays within a chunk. With gates of 0.1–0.3 it overflowed and its Cholesky failed (claim 10), so fast per-channel forgetting needs GLA-style secondary chunking or log-space products (Proposition 4a). The per-channel GPU timings (claims 13–14, gates 0.9–1) use this matrix-product form. It is valid only while a chunk's log decay stays above about −88, a mean gate above about 0.25 at C = 64 and 0.50 at C = 128. A form that is safe for any gates would cost more, and was not timed. The log-space form used here builds a C×C×d tensor per chunk, which is fine for a check but not for a kernel.
- **Backward only in part.** Proposition 6 gives the gradients of the prefix readout with respect to 𝒦, X and R, and autograd through the chunk form matches the token-by-token filter's gradients (claim 3). The closed-form chain rule from there to the layer's inputs was not written out, and the training-pass cost was not modeled. Claim 14 times autograd only, and only estimates the effect of BL's fused backward, so it is not a measurement of training speed against BL.
- **Float32 sizes.** The float32 runs used d = e = C = 64, sequences up to 65,536 tokens, stationary noise, and ten configurations. The covariance did not drift there (claim 9), but no square-root variant was derived, and other sizes, free noise and float16 or bfloat16 were not tested.
- **Model quality is not addressed.** The chunk-parallel form computes exactly what BL's filter computes, so it changes cost, not quality. The stationary parametrization does change the model; whether it trains as well as BL's free noise, or as MesaNet, is untested.

## Next steps

- **Accuracy on ill-conditioned chunks (backlog I-018).** Form 𝒦 and the other chunk matrices (X, Z, R) at higher precision; forming 𝒦 alone in float64 already cut the float32 error 2–10×. Alternatively, form 𝒦's boundary term through a Cholesky factor of P_0 (square-root form). Iterative refinement can help only if its residual uses a more accurate 𝒦. A per-head fallback to the token-by-token filter chosen by cond(𝒦) would miss the cases where the prior-to-posterior downdate dominates. Measure with `check_float32.py`'s five ill-conditioned configurations and `check_chunk.py`'s sweep.
- **A fused kernel for the chunk form (backlog I-019).** Write one Triton kernel per chunk and head for the state-dependent products, the C×C Cholesky, the triangular solve and the updates, the way BL's loops are fused here. Time it against `bl_triton` at C = 32–128 with `gpu_bench.py`. Part C shows what it has to beat: no stage of the plain-PyTorch form dominates. Also time bfloat16 or TF32 for every method, and BL's own kernel if its code is published.
- **E4 (backlog I-017).** Stationary against free process noise, against MesaNet and Gated DeltaNet on MQAR, 3 seeds.
- **The full closed-form backward**: the chain rule from Proposition 6's gradients to the keys, queries, values, β, gates and noise, with the gate gradient as a reverse cumulative sum as in GLA and 0002. Claim 14 compares training cost through autograd only, so the comparison with BL's own fused backward is open.

## References

All were opened in this session, to the extent given in the Novelty check.

1. M. Dowling, H. Jeon, C. Savin, I. M. Park, "Memory by Design: Bayesian Layers for Sequence Modeling", arXiv 2605.31163, v5 (2026-10-14); v1 (2026-05-29) was titled "Memory by Design: Probabilistic Sequence Layers". https://arxiv.org/abs/2605.31163
2. N. Bui, T. Huang, R. Ying, "Kalman Delta Networks: Uncertainty-aware Associative Memory", arXiv 2609.07816, v2 (2026-09-29). https://arxiv.org/abs/2609.07816
3. P. Hejabi, M. Dehghani, P. Piray, "Voltic: Distinguishing Volatility from Stochasticity in Recurrent Memory", arXiv 2610.05700 (2026-10-05). https://arxiv.org/abs/2610.05700
4. L. Peng, A. Chattopadhyay, L. Zancato, E. Nunez, W. Xia, S. Soatto, "Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression", arXiv 2511.21016 (2025). https://arxiv.org/abs/2511.21016
5. V. Shaj, C. Barker, A. Scannell, A. Szecsenyi, E. J. Crowley, A. Storkey, "Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking", ICML 2026, arXiv 2602.10743. https://arxiv.org/abs/2602.10743
6. I. Bogunovic, J. Scarlett, V. Cevher, "Time-Varying Gaussian Process Bandit Optimization", AISTATS 2016, arXiv 1601.06650. https://arxiv.org/abs/1601.06650
7. S. Särkkä, Á. F. García-Fernández, "Temporal Parallelization of Bayesian Smoothers", IEEE Transactions on Automatic Control, arXiv 1905.13002. https://arxiv.org/abs/1905.13002
8. J. Jiao (lecturer), S. Ibraheem (scribe), "Lecture 7: Wiener Filter for Vectors and the Prediction Problem", EECS 225A, UC Berkeley, 2020; follows T. Kailath, A. H. Sayed, B. Hassibi, *Linear Estimation*, Prentice Hall 2000, Section 4.1.2 (the book itself was not opened). https://people.eecs.berkeley.edu/~jiantao/225a2020spring/scribe/EECS225A_Lecture_7.pdf
9. D. T. M. Slock, K. Maouche, "The Fast Subsampled-Updating Recursive Least-Squares (FSU RLS) Algorithm for Adaptive Filtering Based on Displacement Structure and the FFT", Signal Processing 40(1):5–20, 1994, doi:10.1016/0165-1684(94)90018-3. Preprint: https://www.eurecom.fr/publication/99/download/cm-slocdi-941214.pdf
10. S. Van Vaerenbergh, I. Santamaría, M. Lázaro-Gredilla, "Estimation of the Forgetting Factor in Kernel Recursive Least Squares", IEEE MLSP 2012, doi:10.1109/MLSP.2012.6349749. https://gtas.unican.es/files/pub/stkrls_mlsp2012.pdf
11. M. K. Titsias, A. Galashov, A. Rannen-Triki, R. Pascanu, Y. W. Teh, J. Bornschein, "Kalman Filter for Online Classification of Non-Stationary Data", ICLR 2024, arXiv 2306.08448. https://arxiv.org/abs/2306.08448
12. Y. Wu, G. Wayne, A. Graves, T. Lillicrap, "The Kanerva Machine: A Generative Distributed Memory", ICLR 2018, arXiv 1804.01756. https://arxiv.org/abs/1804.01756
13. P. Sakov, M. Bocquet, "Asynchronous data assimilation with the EnKF in presence of additive model error", Tellus A 70(1):1414545, 2018, doi:10.1080/16000870.2017.1414545 (abstract only, via https://researchportal.ip-paris.fr/en/publications/asynchronous-data-assimilation-with-the-enkf-in-presence-of-addit/)

Works of this lab: study 0001, Prefix-Cholesky (`studies/0001-prefix-cholesky/README.md`), and study 0002, Exact chunkwise MesaNet (`studies/0002-gated-mesa-exact/README.md`). MesaNet, flash-linear-attention, GLA and Lai & Bernstein's exponential-resetting RLS are cited through study 0002, which opened them.

## Reproduce

Run from this folder, writing a fresh log to `/tmp` and comparing it with the recorded one. Never `tee` into `logs/`: those files are the immutable record. Each run also adds a timestamped JSON to `results/`; delete those copies, never the recorded ones. `flop_model.py` needs only the standard library, so `python3` works from a clean checkout. The checks need NumPy and PyTorch on the CPU, plus the repository's `lab` package for saving results: use `../../.venv/bin/python` after `scripts/setup_env.sh`, or `--cpu` for the CPU build. Seeds are fixed.

| Command | Compare with | What it checks | Time on the reference machine |
|---|---|---|---|
| `../../.venv/bin/python -u check_chunk.py 2>&1 \| tee /tmp/check_chunk.log` | `logs/check_chunk_v5.log` | Float64 exactness, conditioning sweep, special cases, bounds, gradients (claims 1–5, 7, 8) | ~5 s |
| `../../.venv/bin/python -u check_float32.py 2>&1 \| tee /tmp/check_float32.log` | `logs/check_float32_v2.log` | Float32 over long sequences, overflow and cancellation (claims 9–10) | ~4 min |
| `python3 -u flop_model.py 2>&1 \| tee /tmp/flop_model.log` | `logs/flop_model_v3.log` | The FLOP model (claim 6) | <1 s |
| `TRITON_INTERPRET=1 ../../.venv/bin/python -u check_batched.py 2>&1 \| tee /tmp/check_batched.log` | `logs/check_batched.log` | The batched forms and Triton kernels that the GPU benchmark times, in float64 and float32 (claims 11–12) | ~22 min on a 4-thread cloud CPU, mostly the Triton interpreter |
| `bash run_gpu.sh` (on an NVIDIA GPU, with flash-linear-attention installed as the script's header says) | `logs/gpu_bench_v2.log`, `logs/gpu_bench_cd.log`, `logs/gpu_bench_rerun_n16.log`, `logs/gpu_bench_rerun_n64.log`, `logs/gpu_bench_graph_bwd_n16.log`, `logs/gpu_bench_graph_bwd_n64.log`, `logs/gpu_bench_drift_bl.log`, `logs/aggregate_gpu_bench_v5.log` | The GPU benchmark: claims 13–16 and claim 11's GPU check. It runs `scripts/gpu_probe.py`, the `gpu_bench.py --quick` smoke test, five benchmark runs (parts A–B for N = 16, part B for N = 64, parts C–D, then part B again for each N) and `aggregate_gpu_bench.py`. It writes new numbered logs and JSONs rather than overwriting any; delete those copies. The script's runs use the current `gpu_bench.py`. Its part B times forward + backward as CUDA graphs itself, BL's token-by-token form included, and its part D includes that form too. The recorded runs did these in separate runs. `gpu_bench.py --parts A` alone is a one-minute GPU correctness test | about 2 h for the recorded runs: parts A–B in one process 62 min (the script splits them by N), parts C–D 6 min, the rerun 15 + 18 min, forward + backward as graphs 11 + 14 min, the extra part D case 1 min, a smoke test 1 min. Expect the script to take somewhat longer |

Only the final "saved" line should differ, in `check_float32.py` also the per-configuration timing lines, and in `check_batched.py` the closing timing lines. `--quick` runs a subset of any check as a smoke test. `check_batched.py` needs `TRITON_INTERPRET=1` to run the Triton kernels on the CPU, and skips them without it. In the GPU benchmark every time differs between runs (Results, GPU benchmark, Stability) and between GPUs, so compare its ratios, and its float64 and float32 errors, which match up to summation order.

## Files

| File | Contents |
|---|---|
| `kalman_memory.py` | The token-by-token Kalman filter (torch, and NumPy at any precision), the chunk step (Propositions 2–3) with triangular solves or an explicit L⁻¹ and log-space or matrix-product Grams, the per-token prefix-solve form, and the Mesa readout |
| `check_chunk.py` | The float64 check, parts A, A2, B, C and D |
| `check_float32.py` | The float32 check, parts E1 and E2 |
| `flop_model.py` | The FLOP model |
| `kalman_batched.py` | The chunk form, BL's chunk algorithm (gain loop; v5's token-by-token or v1's WY mean pass) and the token-by-token filter, batched over heads, on any device |
| `bl_triton.py` | Triton kernels for BL's gain loop and mean pass, forward only |
| `check_batched.py` | The check of the batched forms, parts P1–P4 |
| `gpu_bench.py`, `aggregate_gpu_bench.py`, `run_gpu.sh` | The GPU benchmark (parts A–D), its tables and ratios, and the script that runs both |
| `logs/check_chunk_v5.log`, `results/check_chunk_20261029_221909.json` | The float64 check as quoted. v1–v4 (`logs/check_chunk.log`, `_v2`, `_v3`, `_v4` and their JSONs) are earlier runs of the same seeds. Their parts A, B and C are identical to v5. v1 has no part A2 and v2–v4 have a smaller one, and v1–v4 lack the last two gradient configurations (see NOTEBOOK.md) |
| `logs/check_float32_v2.log`, `results/check_float32_20261029_222118.json` | The float32 check as quoted. `logs/check_float32.log` (`results/check_float32_20261029_215528.json`) is version 1: the same six first configurations, which v2 reproduces exactly, without the four hard ones and without the variant that forms 𝒦 in float64 |
| `logs/flop_model_v3.log`, `results/flop_model_20261029_211403.json` | The FLOP model as quoted. `logs/flop_model.log` and `_v2` hold subsets of the same tables |
| `logs/check_batched.log`, `results/check_batched_20261029_225300.json` | The check of the batched forms as quoted (claims 11–12) |
| `logs/gpu_probe.log` | The GPU probe of 2026-10-30, matching docs/COMPUTE.md |
| `logs/gpu_bench_quick.log`, `logs/gpu_bench_quick_v2.log`, `logs/gpu_bench_quick_v3.log`, `logs/gpu_bench_quick_v4.log` | Smoke tests of `gpu_bench.py`: before the changes of 2026-10-30, after them, after the memory fix, and of the forward + backward capture (NOTEBOOK.md). Their JSONs are `results/gpu_bench_quick_20261030_003709.json`, `results/gpu_bench_quick_20261030_004440.json`, `results/gpu_bench_quick_20261030_011137.json` and `results/gpu_bench_quick_20261030_040237.json` |
| `logs/gpu_bench.log` | The first full run of parts A–B, stopped and discarded: after a failed CUDA-graph capture, PyTorch's allocator held on to memory until later methods ran out (NOTEBOOK.md). It saved no JSON |
| `logs/gpu_bench_v2.log`, `results/gpu_bench_20261030_021340.json` | Parts A and B as quoted |
| `logs/gpu_bench_cd.log`, `results/gpu_bench_20261030_021947.json` | Parts C and D as quoted |
| `logs/gpu_bench_rerun_n16.log`, `logs/gpu_bench_rerun_n64.log`, `results/gpu_bench_20261030_023500.json`, `results/gpu_bench_20261030_025256.json` | Part B again, in fresh processes, for the stability of the times |
| `logs/gpu_bench_graph_bwd_n16.log`, `logs/gpu_bench_graph_bwd_n64.log`, `results/gpu_bench_20261101_102135.json`, `results/gpu_bench_20261101_103540.json` | Part B for the chunk forms, BL with the WY mean and the token-by-token filter, with forward + backward also as CUDA graphs, as quoted. `logs/gpu_bench_graph_bwd.log` is an earlier attempt, killed when its session ended after 22 of 32 cells. It saved no JSON, and its printed cells serve as a stability check |
| `logs/gpu_bench_drift_bl.log`, `results/gpu_bench_20261101_103901.json` | Part D's no-forgetting configuration, with BL's token-by-token PyTorch mean added |
| `logs/aggregate_gpu_bench_v5.log` | Every table, ratio and range quoted from the GPU benchmark. `logs/aggregate_gpu_bench.log`, `logs/aggregate_gpu_bench_v2.log`, `logs/aggregate_gpu_bench_v3.log` and `logs/aggregate_gpu_bench_v4.log` are earlier versions with the same first-run tables. Each version adds lines: the rerun's summary (v2), the reference methods' forward + backward ratios and the counts of cells (v3), the decision rule and the graph forward + backward tables (v4), and the like-for-like fastest-C range and the fused kernels' k (v5) |
