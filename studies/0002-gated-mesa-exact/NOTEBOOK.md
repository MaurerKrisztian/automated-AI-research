# 0002 notebook

Dated, append-only log of what was run, why, and what happened, including dead ends and discarded runs. Newest entry last. Don't rewrite old entries; add a new one that corrects them.

## 2026-10-29

Session: cloud container, no GPU, Python 3.13 with NumPy 2.5.3 and SciPy 1.18.1 (SciPy installed this session; the study code needs only NumPy). Run by Claude Opus 5.5 for @MaurerKrisztian, who asked for an ambitious theory-and-math study at E2.

### Choosing the idea

Candidates considered, all in track A/B because the lab's machinery is there:

1. **Exact chunkwise MesaNet with forget gates and a fixed ridge** (chosen). Study 0001 is exact only without gates or with Mesa-P's decaying ridge, and its preconditioner fails for fast-forgetting heads.
2. A cheaper, approximate variant: expand the per-token shifts 1/(s_t + h_m) in a low-rank Cauchy series. Not exact; its cost depends on the gate spread. Kept as a possible follow-up.
3. Self-tuning ridge by closed-form leave-one-out error per token. Needs (1)'s machinery first.
4. Recall capacity of linear attention, the delta rule and least-squares memory (backlog I-010). Largely classical (pseudo-inverse rule capacity).
5. A max-margin associative memory reaching Cover/Gardner capacity (2× least squares for binary values). Would need E4 to matter.
6. Exactly chunkable forgetting by low-rank downdates (multi-window least-squares memory). A new layer design, not MesaNet itself; needs E4.
7. Muon via QDWH or Zolotarev polar iterations. Likely published already (PolarGrad and others); dropped without a search.

Rough FLOP estimates before starting: an exact gated solve needs about m²d per token for a capacitance of size m, plus one d×d eigendecomposition per chunk. That lands near 5–10 CG steps, against 20–30 used in practice. A dense Cholesky per token (d³/3) costs about the same at d = 64 but needs a d×d matrix per token. So the expected gain is real but moderate. The study is worth doing for exactness, predictable cost and exact gradients, not for a large speedup.

**Hypothesis.** For MesaNet with scalar forget gates and a fixed diagonal ridge:
1. The forget gates act as a per-token ridge Λ/a_t on nested data, so one eigendecomposition per chunk boundary plus a per-token Woodbury solve gives every token's exact readout. Target: float64 error ≤ 1e-12 against a token-by-token reference on well-conditioned inputs, across gate regimes, sizes and edge cases.
2. A closed-form backward, including the gate gradient, matches complex-step derivatives to ≤ 1e-12.
3. Its FLOP cost is a small multiple of one CG step.

Refuted if the identity fails, or if it holds only with unbounded error growth in some gate regime. Target level: E2 (no GPU in this session).

### Runs

- Prototype in the session scratchpad (not kept). The start-anchor identity matched the recurrence to 1.6e-15 in three settings. Downdating only from the chunk end lost accuracy at gate 0.5 (3.5e-7). Because that run left no log, the number is not used in the README; `check_downdate.py` below measures the same effect with a log.
- Wrote `gated_mesa.py`: reference, chunk solver with log-space gates, closed-form backward, FLOP model.
- Smoke test (terminal only, no files). The forward pass matched to ≤ 1.5e-14 with policies start/two, and 1.1e-12 with end. The closed-form gradients matched complex-step to ≤ 1e-14.
- `check_forward.py`, first version, run twice without tee (`results/check_forward_20261029_004312.json`, `results/check_forward_20261029_004346.json`). **Superseded.** CG had no breakdown guard and returned NaN at 100 steps in float32. The float64 reference was a plain dense solve, which can't judge ill-conditioned cases.
- `check_forward.py` with the extended-precision refined reference, condition numbers and backward errors (`logs/check_forward.log`, `results/check_forward_20261029_004601.json`). **Superseded.** It showed backward errors up to 3.5e-10 when the capacitance had more rows than d (d = 8, C = 64), which is Woodbury used in the wrong regime. Fix: use the d×d system in the eigenbasis whenever m_t ≥ d.
- `check_forward.py` with that switch (`logs/check_forward_v2.log`, `results/check_forward_20261029_004818.json`). **Current.**
  - Well-conditioned (129 cases): output ≤ 4.5e-14, backward error ≤ 1.3e-14.
  - Ill-conditioned (condition numbers 4.3e4–1.4e6): forward error ≤ 1.1e-9 (≤ 4.6 cond·eps) against ≤ 1.7e-11 for a dense LU solve; backward error ≤ 2.6e-13.
  - Float32: chunk solver 0.6–3.8e-6, dense 0.2–0.75e-6, CG 30 8e-7–1.6e-4.
- `check_backward.py` (`logs/check_backward.log`, `results/check_backward_20261029_004856.json`). All six gradients matched complex-step derivatives to ≤ 1.4e-14 on well-conditioned inputs and 4.8e-12 with ridge 1e-4–1e-3. That includes the gate identity dℓ/dlogγ_j = Σ_{t≥j}(δ·o − g·q + xᵀΛg − β ∂ℓ/∂β).
- `cost_and_cg.py` (`logs/cost_and_cg.log`, `results/cost_and_cg_20261029_005043.json`).
  - Plain CG from zero needs 20–30 steps for 1e-4 at condition numbers 16–54, and 50 to over 100 at ~1e3 with flat key spectra. With keys decaying as i⁻², it needs only 20–30 steps even at ~1e3.
  - FLOP model: the exact solver equals 4–10 CG steps at its best chunk size.
- `check_downdate.py`, first version (`logs/check_downdate.log`, no results file). **Superseded.** It crashed: with gates 0.5–1 and every token downdating, a float32 capacitance lost positive definiteness and Cholesky raised an error. That failure is a result, so the script now records it.
- `check_downdate.py`, recording Cholesky failures (`logs/check_downdate_v2.log`, `results/check_downdate_20261029_005445.json`). **Current.** With ρ ≤ 10: float64 ≤ 1.4e-14, float32 ≤ 2.2e-6, and the mean capacitance falls from 31.5 to 16–25. Unrestricted, with gates 0.5–1: 2.6e-3 in float32.

### Novelty search

Two `literature-scout` agents were launched at 2026-10-29 ~00:40: one for ML literature (MesaNet and its follow-ups, flash-linear-attention), one for numerical linear algebra and adaptive filtering (regularized RLS with forgetting, multi-shift systems). Their results are recorded in the next entry.

### Novelty search results (2026-10-29, about 01:00)

Both agents finished; together they ran about 260 tool calls.

**Numerical linear algebra and adaptive filtering.** The same system is known as leaky RLS (Horita et al. 2004), exponential-resetting RLS (Lai & Bernstein 2023), regularized exponential forgetting (Kulhavý; Schier 1996), RLS with non-fading regularization (Tsakiris et al. 2010) and the discounted aggregating algorithm for regression (Chernov & Zhdanov 2010).
- Exact methods: O(d³) per step, or per-step eigen-tracking.
- O(d²) methods: approximate the regularizer (Gay 1996's cyclic rank-one refresh; Schier's block-accumulated regularization) or solve iteratively (dichotomous coordinate descent).
- No exact block method was found.
- Queries ran over web search and the OpenAlex API (Semantic Scholar returned 429). The full query list is in the agent's report; it covers Gay, Horita, Tsakiris, Kulhavý, van Waterschoot, Lai–Bernstein, Bunch–Nielsen–Sorensen, Golub–Heath–Wahba, shifted systems with Woodbury, GCV paths and parallel-scan RLS.
- Unresolved: APSIPA 2019, "An Alternative Solution to the Dynamically Regularized RLS Algorithm" (closed access; probably specific to tapped-delay-line inputs).

**ML literature.**
- MesaNet: static diagonal Λ; CG with 30 steps in training and tolerance 1e-4 (about 9 steps) at inference; App. D has dℓ/dq and dℓ/dΛ (the same as here) and a state-based dℓ/dγ; no eigendecomposition or Woodbury.
- flash-linear-attention: fixed 30-step CG from zero in both passes, no preconditioner. Issue #1187 reports NaNs in fp16. PR #1410 reports adjoint-CG gradient errors.
- Gated KalmaNet: Chebyshev iteration, ridge scaled by ‖H_t‖_F.
- Memory by Design: exact chunked Kalman memory, a different model, sequential.
- GLA: the reverse-cumulative-sum log-decay gradient.
- Queries ran over web search (27 queries), the arXiv API, Semantic Scholar citations of MesaNet (58), Gated KalmaNet (12) and Preconditioned DeltaNet (9), the OpenReview search API, and the fla repository's issues and PRs.
- The agent reported an ICLR 2026 review comment asking the MesaNet authors about a Woodbury-based exact chunkwise inverse, and that the authors had not investigated it. That came from search-API excerpts. I could not open the forum (it needs a sign-in check), so it is not used in the README.

**Verified myself.**
- MesaNet (arXiv HTML, Sections 2, 4, 5 and Appendices C–E): Λ static and diagonal; 30 CG steps in training; tolerance 1e-4 about 9 steps; CG started from diag(H_t+Λ)⁻¹q_t; App. D gradient formulas as above.
- fla `chunk_cg_solver_fwd.py`: max_CG_iteration = 30, x₀ = 0, no preconditioner, 1e-5 in the denominators, chunk 64.
- Lai & Bernstein 2023, PDF pp. 1–3: ER-RLS recursion; the statement that the matrix inversion lemma doesn't help; O(max{n³, pn²}) per step.
- Gated KalmaNet abstract.
- Golub–Heath–Wahba DOI via Crossref.

**Consequences for the write-up.**
- The q and Λ gradients are no longer claimed as new; the gate gradient is presented as a state-free form (GLA's trick plus a Λ term).
- Added that, at inference (about 9 steps), the FLOP gain is small; the gain is mainly in training.
- Status set to `review`.

### Review and fixes (2026-10-29, morning)

- `/review-study`: the `reviewer` agent returned **revise**, with 4 major and 5 minor issues; see REVIEW.md. It reproduced all four logs bit-for-bit (93 s, NumPy only) and confirmed the math by hand. I confirmed the major issues myself: the start-anchor CG-equivalents (6.8–19.0) recomputed with `gated_mesa.flops_*`; per-row float32 ratios to the dense solve of up to 10.5× in `logs/check_forward_v2.log` part B; MesaNet's diagonal CG start (App. C, read earlier). The reviewer's extra runs went to /tmp and are not cited as evidence. The study's own reruns below replace them.
- Code changes in `gated_mesa.py`:
  - `cg_solve` gained `start="diag"` (MesaNet's x₀) and `jacobi=True` (Jacobi PCG). The defaults are unchanged, so earlier results reproduce.
  - `make_inputs` gained `rotate=True`, which rotates the key spectrum off the coordinate axes; its draws come after the default draws.
  - `policy_sizes` and `solver_flops` replace the idealized FLOP model: they apply the ρ cap and model padding and the backward solve.
- `cost_and_cg.py` version 2 (`logs/cost_and_cg_v2.log`, `results/cost_and_cg_v2_20261029_093825.json`; run in the background).
  - Its "zero" rows reproduce version 1. Version 1's log `logs/cost_and_cg.log` is **superseded for part A** and kept for its absolute FLOP table (part B), which v2 does not repeat.
  - Diag start: 5–15 steps on axis-aligned i⁻² keys, 15–30 otherwise.
  - Jacobi: 5 on axis-aligned keys, 5–10 or 15–50 on rotated keys.
  - All variants: 50 to >100 steps on flat keys with ridge 1e-3–1e-2.
  - Exact forward: 3.8–19.0 CG-step equivalents (CG at chunk 64); training pass 7.7–15.9 against 60.
- New `check_conditioning.py` (`logs/check_conditioning.log`, `results/check_conditioning_20261029_093722.json`; run in the background).
  - Float64: the solve's backward error grows with conditioning (≤ 15·eps at condition ≤ 1e2, up to 172·eps at 1e4–1e6), while dense LU stays ≤ 0.5·eps. This is why "backward stable" was dropped.
  - Float32: the exact solver is within 5× of dense up to condition 1e2, but up to 103× worse at 1e3–1e4 and up to 2000× at 1e4–1e5. It beats 30-step CG of every variant once the condition number passes ~1e2. 100-step CG or Jacobi PCG is sometimes more accurate at ≥ 1e4.
  - The two-anchor policy was less accurate than start-only on 33 of 45 float32 inputs.
- README: Summary, Verdict, Claims 2–8, Results (conditioning tables, CG variants, FLOP equivalents), Limitations and Next steps rewritten against these logs. Preconditioned DeltaNet added to the novelty table (only what its abstract says). The O(d³) statement hedged. Reproduce commands now tee to /tmp. Script docstrings updated the same way; no computation changed.
- The study is still uncommitted: commit attempts were declined in this session. The results JSONs therefore record `git_dirty: true`, and the code version is not pinned by a commit.
