# Idea backlog

A ranked queue of testable ideas. `/research` with no argument takes the highest-priority `ready` item that this machine can run. When a study ends, add its follow-ups here; when an item is taken, set its status to the study id. Ideas are unvetted until a novelty check has been run on them.

**Runs on** says what an item needs: `phone` (no code: reading, web search, writing; any Claude Code session works), `CPU` (Python without a GPU; cloud sessions count), `GPU` (an NVIDIA GPU).

**Status** values: `ready` (can start now), `blocked` (needs another item first), `taken: NNNN`, `done: NNNN`, `dropped` (with the reason).

## I-001 · Fix the text-only review issues of study 0001
- **Runs on** phone · **Track** A, B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** review issues 1, 3, 6, 10, 11 and 12 in `studies/0001-prefix-cholesky/REVIEW.md` need no new runs. Refresh the stale Summary from the existing logs, soften "no float32 drift", discuss DeltaNet's chunkwise UT/WY transform and block recursive least squares in the novelty check, update Limitations, say which timing run is quoted, and add a References section listing only works actually opened.
- **Done when:** those issues are marked fixed, each naming the file that shows it.

## I-008 · Rerun study 0001's CPU exactness checks with saved logs
- **Runs on** CPU · **Track** B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** copy the NumPy/SciPy checks out of `archive/exact_mesa_code.zip` (`mesa_exact.py`, `mesa_layers.py`, `pdn_check.py`) into a `cpu_checks/` folder in the study, run them, and save their output in `logs/`, so the paper's 3.1e-15, 8.8e-7 and 2.8e-14 figures become traceable (review issue 9).
- **Compute:** minutes on a CPU.

## I-009 · Fair CG baseline, seeds and the 128-pair failure in study 0001
- **Runs on** GPU · **Track** A, B · **Priority** 1 · **Status** ready · **Builds on** 0001
- **What:** review issues 2, 4, 5, 7 and 8. Give the CG baseline an implicit backward (one extra CG solve), run 3+ seeds at 64 pairs with a tuned DeltaNet, sweep learning rate and ridge at 128 pairs to tell a failure to train from a capacity limit, and time the layers at head dim 32 and at small batch.
- **Compute:** about 2 GPU hours.

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

## I-014 · Capacity-optimal associative memory against least squares
- **Runs on** phone, CPU · **Track** E · **Priority** 4 · **Status** ready (no novelty check yet)
- **What:** least-squares (pseudo-inverse) memory stores d random pairs perfectly. Cover's and Gardner's capacity bounds allow about 2d pairs with binary values for a linear readout. Quantify the gap for MQAR-style retrieval (argmax over a vocabulary). Test whether a cheap recurrent rule, for example a few reweighted least-squares steps on the stored pairs, closes part of it. Related to I-010.

## I-007 · Open tracks C–E
- **Runs on** phone · **Priority** 4 · **Status** ready
- **What:** for each of tracks C, D and E in `ROADMAP.md`, map the literature with `/novelty-check` and add three concrete, testable ideas to this backlog, each marked with what it runs on.

## I-006 · Least-squares memory in a tiny LM
- **Runs on** GPU · **Track** A · **Priority** 5 · **Status** blocked (needs I-005, and I-002 for forgetting) · **Builds on** 0001
- **What:** replace attention in some or all layers with `lab.layers.mesa`, and compare with Gated DeltaNet and the Transformer baseline at equal parameters and tokens.
