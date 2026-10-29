# Idea backlog

A ranked queue of testable ideas. `/research` with no argument takes the highest-priority `ready` item that this machine can run. When a study ends, add its follow-ups here; when an item is taken, set its status to the study id. Ideas are unvetted until a novelty check has been run on them.

**Runs on** says what an item needs: `phone` (no code: reading, web search, writing; any Claude Code session works), `CPU` (Python without a GPU; cloud sessions count), `GPU` (an NVIDIA GPU).

**Status** values: `ready` (can start now), `blocked` (needs another item first), `taken: NNNN`, `done: NNNN`, `dropped` (with the reason).

## I-001 · Fix the text-only review issues of study 0001
- **Runs on** phone · **Track** A, B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** review issues 1, 3, 6, 10, 11 and 12 in `studies/0001-prefix-cholesky/REVIEW.md` need no new runs. Refresh the stale Summary from the existing logs, soften "no float32 drift", discuss DeltaNet's chunkwise UT/WY transform and block recursive least squares in the novelty check, update Limitations, say which timing run is quoted, and add a References section listing only works actually opened.
- **Done when:** those issues are marked fixed, each naming the file that shows it.
- **Note (2026-10-29):** the closest block-RLS precedent is now known: Slock and Maouche's subsampled-updating RLS (1994), which gives every in-block RLS output exactly from one L×L factorization per block. See the erratum at the top of 0001's README and study 0003's Novelty check. The novelty wording must change accordingly.

## I-008 · Rerun study 0001's CPU exactness checks with saved logs
- **Runs on** CPU · **Track** B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** copy the NumPy/SciPy checks out of `archive/exact_mesa_code.zip` (`mesa_exact.py`, `mesa_layers.py`, `pdn_check.py`) into a `cpu_checks/` folder in the study, run them, and save their output in `logs/`, so the paper's 3.1e-15, 8.8e-7 and 2.8e-14 figures become traceable (review issue 9).
- **Compute:** minutes on a CPU.

## I-009 · Fair CG baseline, seeds and the 128-pair failure in study 0001
- **Runs on** GPU · **Track** A, B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** review issues 2, 4, 5, 7 and 8. Give the CG baseline an implicit backward (one extra CG solve), run 3+ seeds at 64 pairs with a tuned DeltaNet, sweep learning rate and ridge at 128 pairs to tell a failure to train from a capacity limit, and time the layers at head dim 32 and at small batch.
- **Compute:** about 2 GPU hours.

## I-015 · Float64 check of study 0003's chunkwise Kalman memory
- **Runs on** CPU · **Track** A, B · **Priority** 1 · **Status** done: 0003 (claims 1–5 and 7–10 at E2; `logs/check_chunk_v4.log`, `logs/check_float32.log`) · **Builds on** 0003 (E1: Propositions 1–6)
- **What:** `check_chunk.py` in study 0003. Compare Propositions 2–3 (chunk readout and boundary update) with the token-by-token Kalman recursion of Proposition 1(b). Use random inputs, scalar and per-channel gates in [0.01, 1] and [0.999, 1], C = 1, sequence lengths that are not a multiple of C, d below and above C, tiny β and badly conditioned keys. Check the gate-free limit against 0001's Mesa readout. Check Proposition 6's adjoint, then full gradients for q, k, v, β, gates and Λ against autograd in float64. Then float32 against float64 over long sequences, to see whether the covariance update drifts (Proposition 4c).
- **Done when:** errors at float64 round-off with saved logs, or the identity is refuted.
- **Compute:** minutes on a CPU.

## I-018 · Accuracy of study 0003's chunk form on ill-conditioned chunks
- **Runs on** CPU · **Track** B · **Priority** 2 · **Status** ready · **Builds on** 0003 (E2: claims 8–9)
- **What:** the chunk form is exact, but its rounding error is about cond(𝒦)·eps. The token-by-token filter is usually more accurate: 2.1–195× in float64, depending on the keys, and 2.3–56× in float32 at cond(𝒦) 430–1.35e4, where the chunk form reached 3.7e-4. Running only the C×C stage in float64 barely helps. Forming 𝒦 itself in float64 cut the float32 error 2–10× but left one configuration 11× above the filter, so the other chunk matrices (X, Z, R) and products matter too. Try:
  - forming 𝒦, X, Z and R at higher precision, or through a Cholesky factor of P_0 (square-root form);
  - iterative refinement, which helps only if its residual uses a more accurate 𝒦;
  - a per-head fallback to the token-by-token filter. Chosen by cond(𝒦), it would miss the cases where the prior-to-posterior downdate dominates. Note too that without forgetting the float32 token-by-token filter drifts and the chunk form does not.
- **Decisive test:** `check_float32.py`'s five ill-conditioned configurations and `check_chunk.py`'s part A2. Success means errors within 2× of the token-by-token filter at a small extra cost.

## I-002 · Exact forgetting with the carried inverse
- **Runs on** phone for the derivation (E1), CPU for the float64 check (E2), GPU for the long float32 drift test · **Track** A, B · **Priority** 2 · **Status** ready · **Builds on** 0001 (E2: carried-inverse identity)
- **What:** add MesaNet's per-token forget gates to the carried inverse. The gates enter as key scalings inside a chunk and as a 1/a_C rescaling of P at its end; resetting the ridge to λ (Mesa-P) costs one d×d inverse per reset.
- **Hypothesis:** exact in float64, and in float32 more accurate than 30 CG steps at lower cost, for gates from 0.9 to 0.999.
- **Decisive test:** a float64 check against a token-by-token gated recurrence, then float32 drift over 65k tokens for several gate values.

## I-011 · GPU benchmark of study 0002's exact gated MesaNet solver
- **Runs on** GPU · **Track** A, B · **Priority** 2 · **Status** ready · **Builds on** 0002 (E2: exact chunk solve and closed-form backward)
- **What:** implement 0002's solver in PyTorch:
  - boundary states by the chunk recurrence;
  - one batched `torch.linalg.eigh` per chunk boundary;
  - batched capacitance Cholesky padded to the largest size (two anchors, ρ ≤ 10), or the d×d form when m ≥ d;
  - the closed-form backward of 0002's Proposition 3.
  Time forward and forward + backward against flash-linear-attention's MesaNet kernels (fixed 30-step CG in both passes) and against CG with MesaNet's diagonal start or Jacobi preconditioning, at equal output error, for gates near 1, 0.9–1 and 0.5–1, d = 64 and 128, chunks 32–128. Where possible, use key statistics taken from a trained MesaNet, because Jacobi-type preconditioners depend on how anisotropic the keys are along the axes. Also measure float32 error on the GPU.
- **Hypothesis:** at d = 128 the exact training pass is faster than 30 + 30-step CG. 0002's FLOP model (claim 8, E1) puts a training pass at 7.7–15.9 CG-step equivalents against 60, and the forward solve at 3.8–19.0.
- **Decisive test:** wall-clock time per training pass at matched error, and the share of time spent in `eigh`.
- **Compute:** about 1–2 GPU hours.

## I-016 · GPU benchmark of study 0003's Kalman memory
- **Runs on** GPU · **Track** A, B · **Priority** 2 · **Status** ready (I-015 done) · **Builds on** 0003 (E2)
- **What:** a PyTorch chunk implementation (a batched C×C Cholesky inside the loop over chunks). The main baseline is the Bayesian Layer's published chunk algorithm (Dowling et al. 2026, Appendix C.2, Eqs. 23–27: sequential gain loop and mean pass), written with the same care, since its code is not public. Also time 0001's carried inverse (`lab.layers.mesa`), 0002's exact solver, and flash-linear-attention's MesaNet (CG), Gated DeltaNet and KDA kernels. Measure forward and forward + backward at d = 64 and 128, chunks 16–128, with scalar and per-channel gates. Also measure float32 drift of the covariance over 65k tokens.
- **Hypothesis:** the chunk-parallel form is faster than the sequential Bayesian Layer algorithm at d = 128 and chunks of 32–64, despite 1.3–1.5× more FLOPs, because its order-dependent work is 6–13× smaller (0003, claim 6, E1). It is also faster than 30 + 30-step CG.
- **Compute:** about 1–2 GPU hours.

## I-017 · Stationary against free process noise in the Kalman memory
- **Runs on** GPU · **Track** A · **Priority** 3 · **Status** ready (I-015 done) · **Builds on** 0003 (E2)
- **What:** the Bayesian Layer (Dowling et al. 2026) learns its process noise freely. Study 0003's stationary choice ties it to the gates, so the prior stays N(0, Λ⁻¹), MesaNet's fixed ridge, with one fewer learned quantity per token. Both forget like Gated DeltaNet, while MesaNet's forgetting saturates (0003, Proposition 1(d)). Train MQAR and selective-copying models, 3 seeds, with both parametrizations (scalar and per-channel gates), against MesaNet (CG) and Gated DeltaNet, at equal parameters and an equal tuning budget. The chunk-parallel form computes either one exactly, so this tests the parametrization, not the algorithm.
- **Decisive test:** recall and loss against the seed spread.
- **Compute:** about 2 GPU hours.

## I-005 · Shared tiny-LM baseline (milestone M2, infrastructure)
- **Runs on** GPU · **Track** all · **Priority** 2 · **Status** ready
- **What:** a Llama-style Transformer, a data pipeline (for example a FineWeb-Edu or TinyStories sample in `data/`), a training loop and evaluation in `lab/`, with a tuned learning rate at 10–30M parameters and 3 seeds. Every E5 claim compares against it.
- **Done when:** the final validation loss reproduces within the seed spread, at under 1 hour per run.
- **Compute:** about 3 GPU hours including tuning.

## I-010 · Recall capacity of fixed-size memories
- **Runs on** phone for the theory (E1), CPU for a simulation check (E2) · **Track** E · **Priority** 3 · **Status** ready · **Builds on** 0001
- **What:** derive the expected recall of linear attention, the delta rule and least-squares (Mesa) memory when N random key–value pairs are written into one head of dimension d, as a function of N/d and the ridge. Compare with study 0001's MQAR results (d = 32, N = 64 and 128), and say whether its 128-pair failure can be a capacity limit at all. Associative-memory capacity is well studied, so start with the novelty check.

## I-003 · Per-head solver policy for MesaNet as published
- **Runs on** GPU · **Track** B · **Priority** 3 · **Status** blocked (needs I-009's fair CG baseline) · **Builds on** 0001 (E3: exact-chunk preconditioner)
- **What:** use the exact-chunk preconditioner for slowly forgetting heads and plain CG for fast-forgetting ones, chosen per head by the condition-number bound.
- **Decisive test:** total solve time and error against all-CG, on gate distributions like MesaNet's.
- **Compute:** about 1 GPU hour.
- **Note (2026-10-29):** study 0002 gives an exact solver whose cost doesn't depend on the gates (E2). That makes it a candidate for the fast-forgetting heads where this preconditioner's bound fails. The policy question becomes exact solver against preconditioned CG, decided per head; I-011 measures the inputs to that choice.

## I-004 · Fused Triton kernel for the carried-inverse layer
- **Runs on** GPU · **Track** B · **Priority** 4 · **Status** ready · **Builds on** 0001
- **What:** one program per head that keeps P and the value memory on chip, replacing the many small PyTorch kernels per chunk.
- **Decisive test:** forward and backward time against the PyTorch version and against flash-linear-attention's Gated DeltaNet kernels at equal sizes.
- **Compute:** mostly development time, little GPU time.

## I-012 · Iterative refinement for ill-conditioned heads in study 0002
- **Runs on** CPU · **Track** B · **Priority** 3 · **Status** ready · **Builds on** 0002 (E2)
- **What:** 0002's solver loses accuracy as conditioning worsens. Its backward error rises to about 1.2e3·eps at condition 1e6, against ≤ 0.5·eps for dense LU. In float32 its error is up to 103× the dense solve's at condition 1e3–1e4 and up to 2000× at 1e4–1e5. Add one refinement step: form the residual q_t − (H_t + Λ)x_t with the chunk product (about one CG step), then re-solve with the same eigendecompositions and capacitance factors.
- **Decisive test:** forward error against the refined reference at condition numbers 1e2–1e8, in float64 and float32, compared with the dense solve.

## I-013 · Exact chunkwise least-squares memory with per-channel gates
- **Runs on** phone for the theory, CPU for the check · **Track** A, E · **Priority** 4 · **Status** ready · **Builds on** 0002
- **What:** with diagonal (per-channel) gates, as in GLA or KDA, 0002's per-token ridge becomes a diagonal matrix that varies per token. The boundary eigenbasis doesn't diagonalize it. Find an exact chunk method costing less than a dense solve per token, or show that none exists of the form "one factorization per chunk plus low-rank work per token".
- **Note (2026-10-29):** study 0003 (E1) gives an exact chunk method for per-channel gates in a closely related model, the Kalman-filter memory of the Bayesian Layer (Dowling et al. 2026). There, forgetting is decay plus process noise on the memory instead of MesaNet's down-weighting of old pairs, and one C×C Cholesky per chunk suffices. MesaNet's own model with per-channel gates remains open.

## I-014 · Capacity-optimal associative memory against least squares
- **Runs on** phone, CPU · **Track** E · **Priority** 4 · **Status** ready (no novelty check yet)
- **What:** least-squares (pseudo-inverse) memory stores d random pairs perfectly. Cover's and Gardner's capacity bounds allow about 2d pairs with binary values for a linear readout. Quantify the gap for MQAR-style retrieval (argmax over a vocabulary). Test whether a cheap recurrent rule, for example a few reweighted least-squares steps on the stored pairs, closes part of it. Related to I-010.

## I-007 · Open tracks C–E
- **Runs on** phone · **Priority** 4 · **Status** ready
- **What:** for each of tracks C, D and E in `ROADMAP.md`, map the literature with `/novelty-check` and add three concrete, testable ideas to this backlog, each marked with what it runs on.

## I-006 · Least-squares memory in a tiny LM
- **Runs on** GPU · **Track** A · **Priority** 5 · **Status** blocked (needs I-005, and I-002 for forgetting) · **Builds on** 0001
- **What:** replace attention in some or all layers with `lab.layers.mesa`, and compare with Gated DeltaNet and the Transformer baseline at equal parameters and tokens.
- **Note (2026-10-29):** study 0003's Kalman memory is another route to forgetting (exact chunk form verified at E2 in float64; quality untested, see I-017).
