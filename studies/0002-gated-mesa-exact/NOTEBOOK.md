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

### GPU micro-benchmark, forward solve (2026-10-29, evening)

Purpose: first GPU numbers (a small version of backlog I-011). Forward solve only, float32, plain PyTorch for every method. RTX 5060 Ti 16 GB, PyTorch 2.11.0 + CUDA 12.8, TF32 off.

- New `gpu_bench.py`:
  - Ports the **start-anchor** solver to batched PyTorch. Capacitances are padded to C: rows i > t are zero, so A_t is padded with I. It uses one `torch.linalg.eigh` over all chunk-start states and one batched C×C Cholesky per token.
  - The two-anchor policy was not ported, because it was less accurate in float32 (claim 4).
  - Baselines: chunked CG as in fla (x₀ = 0, no preconditioner) and Jacobi PCG from MesaNet's diagonal start, at 10, 30 and 100 steps, both in plain PyTorch. The fla Triton kernels are not installed and were not run.
  - Every timing includes the chunk-start states. The readout is not timed.
- Smoke tests: `logs/gpu_bench_quick.log` (`results/gpu_bench_quick_20261029_173434.json`) is **superseded and invalid for plain-CG timings**. A parsing bug (`int("cg10"[3:])` = 0) ran plain CG for 0 steps in the timed closure; its errors and every other number are unaffected. Fixed, and rerun as `logs/gpu_bench_quick_v2.log` (`results/gpu_bench_quick_20261029_173536.json`). The condition-number estimate was also changed to every 8th token for speed.
- Side check (scratch, not saved, not cited as evidence): `torch.linalg.eigh` on CUDA batches only for n ≤ 32. At n = 64 and 128 it took about 0.6 and 1.5 ms per matrix, slower than the CPU. This is why the full run was expected to be eigh-bound.
- Full run: `../../.venv/bin/python -u gpu_bench.py 2>&1 | tee logs/gpu_bench.log`, about 23 minutes in the background, peak memory 2.24 GB. Results: `results/gpu_bench_20261029_175912.json`.
  - Settings: N = 16 heads, T = 2048, d_v = 64, d ∈ {64, 128}, C ∈ {32, 64}, gates 0.99–1, 0.9–1 and 0.5–1, ridges 0.05–1 (condition 31–37) and 1e-3–1e-2 (1.2e3–1.5e3), flat keys, 3 input seeds for errors.
  - Part A: the port matches `gated_mesa.chunk_solve` in float64 to ≤ 8.7e-15 and a dense solve to ≤ 1.5e-14.
  - **Speed:** the exact forward solve took 299–2019 ms, against 17.0–17.6 ms (d = 64) and 43.4–45.1 ms (d = 128) for 30-step CG. That is 16–45× slower.
  - `eigh` took 89–99.8% of the exact solver's time (265–2004 ms). The capacitance stage alone took 10.8–11.0 ms (d = 64, C = 32), 32.0–32.2 (64, 64), 13.6 (128, 32) and 35.8–35.9 (128, 64).
  - **Accuracy, float32:**
    - At condition about 30, exact reached 5.5e-6–1.9e-5, CG30 1.6e-6–9.7e-6 and Jacobi30 5.7e-7–2.4e-6. CG30 is the more accurate there.
    - At condition about 1e3, exact reached 5.4e-5–1.9e-4 and CG30 0.085–0.29. CG100 reached 2.3e-5–1.9e-3, at 53–145 ms.
  - **Anomaly:** Jacobi PCG at 100 steps gave 20.8 on d = 64, C = 64, gates 0.5–1, ridge 0.05–1, while 30 steps on the same inputs gave 1.2e-6. This looks like a float32 breakdown after convergence that my guard (positivity only) does not catch. It is a defect in this baseline implementation; that cell is not used in any claim.
- Conclusion: refuted *for this implementation*. With `torch.linalg.eigh`, the exact forward solve is 16–45× slower than 30-step CG on this GPU, even when CG needs 100 steps to match its accuracy. Without eigh, the capacitance stage costs 0.30–1.8× a 30-step CG solve, so the method's GPU case depends entirely on a fast batched eigendecomposition (or avoiding it). The hypothesis of I-011 (training pass faster than 30 + 30 CG at d = 128) was not tested directly, since the backward was not ported, but the forward result alone makes it fail with this eigh.

### Review of the GPU benchmark, and version 2 (2026-10-29, evening)

- `/review-study`: the `reviewer` agent returned **revise** (2 major, 5 minor; REVIEW.md). I confirmed both majors myself:
  - The ratios were wrong. Recomputed per cell from `results/gpu_bench_20261029_175912.json`: 15.6–51.3× against CG30 and 4.8–16.5× against CG100, not 16–45× and 5–14×. I had taken ratios of range ends. The "16–45×", "5–14×" and "89–99.8%" in the previous entry are wrong; the corrected values are below.
  - A faster `eigh` was available but not tried. In a scratch check (not saved), 512 matrices took 330 ms (cuSOLVER) vs 69 ms (MAGMA) at n = 64, and 785 vs 212 ms at n = 128.
- Changes to `gpu_bench.py` (version 2):
  - The exact solver runs with three `eigh` backends: default (cuSOLVER), MAGMA, and CPU.
  - CG gets a residual stop (`--cg-tol 1e-6`, relative to the initial residual), which freezes converged tokens. Every step is still computed, so times are unchanged.
  - CG also runs at 200 steps.
  - Part C times `eigh` alone, so the scaling statement now has a log.
  - The timed closures take each method as a function, removing the name parsing behind the v1 smoke-test bug.
- New `aggregate_gpu_bench.py` computes every ratio, the `eigh` share (over the stage sum) and the CG steps needed to match the exact error, per cell, from the results JSON.
- Smoke test: `logs/gpu_bench_quick_v3.log` (`results/gpu_bench_quick_20261029_180916.json`).
- Full run: `../../.venv/bin/python -u gpu_bench.py 2>&1 | tee logs/gpu_bench_v2.log`, about 32 minutes in the background. Results: `results/gpu_bench_20261029_184103.json`.
- Aggregation: `logs/aggregate_gpu_bench_v2.log`, then `logs/aggregate_gpu_bench_v3.log`. These are the same data; v3 adds the matched-error ratios, so it supersedes v2.
- Outcome:
  - Part A is unchanged. Version 1's default-`eigh` exact times reproduce to within 1.6%.
  - `eigh` alone: the default takes 0.64 / 1.5 ms per matrix at n = 64 / 128, MAGMA and CPU 0.13–0.15 / 0.43–0.48. The default batches only at n = 32.
  - Exact against CG30: 15.5–50.8× slower with the default `eigh`, 4.4–11.8× with MAGMA or CPU, where `eigh` still takes 61–97%.
  - At matched error: the exact solver is 1.5–6.4× slower than plain CG at condition ~30. At condition ~1e3 it takes 0.84–2.62× plain CG's time and 0.61–2.43× Jacobi's, where CG needs 100–200 steps.
  - MAGMA and CPU `eigh` also lower the exact solver's float32 error, to 1.3e-6–2.7e-6 at condition ~30.
  - No Jacobi breakdown with the residual stop.
  - `logs/gpu_bench.log` (v1) is **superseded** by `logs/gpu_bench_v2.log`.
- Front-matter evidence goes back to E2 (review issue 7): the headline claims are E2, and claim 9 alone is E3.

### Batched Jacobi eigendecomposition (2026-10-29, night)

Purpose: remove the `eigh` bottleneck found by the benchmark (claim 9).

- New `jacobi_eigh.py`: a Triton kernel with one program per matrix. It runs cyclic two-sided Jacobi in round-robin order, with the matrix and eigenvectors in global-memory scratch, and CTA barriers between the row, column and eigenvector phases.
- New `check_jacobi.py`: compares against `torch.linalg.eigh` in float64 and float32. Inputs are random symmetric matrices, SPD matrices with condition 1e3, and whitened chunk-start states. It measures reconstruction, orthogonality and the sorted eigenvalue error.
- Version 1 used fixed sweeps.
  - Logs: `logs/check_jacobi_quick.log` and `logs/check_jacobi.log` (`results/check_jacobi_20261029_184807.json`), now **superseded**.
  - In float64 the kernel was correct (8 sweeps, 1.6e-14 on Gaussian matrices). In float32, 8 sweeps matched torch at n = 64 and 10 at n = 128 on its inputs. Batch 1024 took 28.7 ms (n = 64) and 299 ms (n = 128, 10 sweeps), against 653 / 1553 ms for the default `eigh` and 124 / 428 for the CPU.
  - Scratch: 8 warps were slightly faster than 4 (26.9 vs 28.7 ms; 275 vs 299 ms).
- Problem: in the benchmark smoke test (`logs/gpu_bench_quick_v4.log`), the solver's float32 error with 8-sweep Jacobi was 3–100× worse than with torch `eigh` (up to 6.9e-3 at gates 0.5–1, ridge 1e-3–1e-2).
  - A scratch diagnosis showed 8 sweeps were not converged on those states; 12 sweeps matched torch.
  - Extra sweeps slowly degraded the error (8.8e-5 at 24 sweeps), because in float32, rotations kept firing on round-off after convergence.
- Version 2 of the kernel:
  - It skips a rotation when |a_pq| ≤ eps·√|a_pp a_qq| or ≤ eps·max|A|.
  - It sets a rotated pair's off-diagonal entries to exactly zero.
  - It stops after a sweep with no rotation, capped at 20 sweeps.
  - Two intermediate attempts, a relative threshold alone and then an eps·max|A|/n floor, still ran to the cap on small-ridge states (scratch runs, not saved).
- `logs/check_jacobi_v2.log` (`results/check_jacobi_v2_20261029_185255.json`):
  - Float32 errors are 0.59–1.38× torch's on every input, using 7–15 sweeps, never the cap.
  - Float64 reaches 2.2e-14–7.2e-14, except 6e-10–9e-10 on n = 128 fast-forgetting states at 20 sweeps.
  - Batch 1024: 31 ms (n = 64) and 315 ms (n = 128).
- Smoke test with the Jacobi backend: `logs/gpu_bench_quick_v5.log`. The solver error now matches the default `eigh`'s.
- Full benchmark v3, with four backends: `../../.venv/bin/python -u gpu_bench.py 2>&1 | grep --line-buffered -v "Warning: " | tee logs/gpu_bench_v3.log`, about 45 minutes in the background. Results: `results/gpu_bench_20261029_192644.json`. The grep only drops PyTorch's two MAGMA deprecation warnings.
  - Default-`eigh` times reproduce v2 to within 1.9%.
- Aggregation:
  - `logs/aggregate_gpu_bench_v4.log` is **superseded and wrong for matched error**: it paired the fastest exact variant's time with the most accurate variant's error, which favoured the exact solver (it showed wins at condition ~30).
  - Fixed so each variant is matched at its own error: `logs/aggregate_gpu_bench_v5.log`. Win counts were then added in `logs/aggregate_gpu_bench_v6.log`, the version cited.
- Outcome (v6):
  - Jacobi `eigh` takes 0.031 / 0.31 ms per matrix at n = 64 / 128 (default 0.64 / 1.5; CPU 0.13 / 0.47).
  - Exact with Jacobi against CG30: 2.1–8.8× slower (default 15.5–50.3×; CPU 4.5–10.7×); `eigh` takes 27–96% of its time.
  - At matched error, condition ~30: 1.36–3.14× slower than plain CG, in all 12 cells.
  - At matched error, condition ~1e3: 0.35–2.28× plain CG's time (faster in 9 of 12) and 0.33–1.16× Jacobi PCG's (faster in 10 of 12).
- README Summary, Verdict, claim 9, GPU results, Limitations, Next steps, Reproduce and Files are updated. Claim 9 changed after the review in REVIEW.md, so it needs another review.
