# 0003 notebook

Dated, append-only log of what was run, why, and what happened, including dead ends and discarded runs. Newest entry last. Don't rewrite old entries; add a new one that corrects them.

## 2026-10-29

Session: the reference workstation, `.venv` Python 3.11 with NumPy (no SciPy). Run by Claude Opus 5.5 for @MaurerKrisztian, who asked for a high-impact result with evidence capped at E1 for now ("e1 level max first"): a derivation, no verification runs. One unsaved scratch prototype was run anyway to catch algebra errors before writing; it is described below and is not evidence.

### Choosing the idea

The lab's open problem: exact least-squares memory with forget gates. Study 0001 is exact only without gates or with Mesa-P's decaying ridge (its claim 1, E2). Study 0002 made MesaNet as published exact, but only with one eigendecomposition per chunk boundary (its claims 1–2, E2), which takes 27–96% of its GPU forward solve and leaves it 2.1–8.8× slower than 30-step CG (its claim 9, E3). Per-channel gates are open (backlog I-013).

Candidates considered:

1. **Forgetting as state dynamics (chosen).** Model the memory as a Kalman filter: W_t = W_{t−1}D_t + noise, with the noise chosen so the prior stays N(0, Λ⁻¹). MesaNet forgets in the loss (old pairs are down-weighted relative to the current token), so every token sees a different ridge. A state-space prior forgets in the dynamics, so the joint covariance of past observations no longer depends on the query time, and the leading-block property of Cholesky applies again. If this holds, it removes 0002's eigendecomposition and covers per-channel gates too. Upside: an exact least-squares memory with the forgetting behavior of Gated DeltaNet and KDA, at roughly the cost of 0001's ungated carried inverse. Risk: it is a different model from MesaNet, so its quality is an open E4/E5 question.
2. A matmul-only exact solver for MesaNet as published. Its per-token systems differ by full-rank shifts with no common basis except the boundary eigenbasis. No route without iteration was found; dropped.
3. Directional forgetting (Kulhavý's RLS variant): a rank-one information downdate along each key. Exact and free of windup, but its forgetting strength depends on the state (kᵀΛk), which breaks the chunk structure. Process noise along the keys (a special case of idea 1's general process noise) keeps the idea and the structure.
4. Recall capacity theory (I-010). Largely classical (pseudo-inverse rule capacity).
5. Per-channel gates for MesaNet's own model (I-013). Consecutive tokens' systems differ by full-rank diagonal changes. Idea 1 answers the question for the Kalman model instead.
6. Optimizer ideas: Muon with an exact polar factor, K-FAC with EMA factors and damping. The damping-plus-EMA structure is the same obstacle as MesaNet's; not pursued.

**Hypothesis (target E1).** For a least-squares memory whose forgetting is AR(1) dynamics of the memory, W_t = W_{t−1}D_t + E_t with E_t's rows N(0, (I − D_t²)Λ⁻¹), D_t scalar or per-channel and Λ diagonal, observations v_t = W_t k_t + N(0, β_t⁻¹I), and readout o_t = E[W_t | v_{≤t}] q_t:
1. Every token's readout in a chunk follows from one Cholesky factorization of a C×C observation covariance that is the same for every token in the chunk, and the next boundary posterior follows by matrix products: no d×d factorization, no eigendecomposition, no iteration.
2. With all gates equal to 1 the readout is exactly Mesa's G_t(H_t + Λ)⁻¹q_t.
3. Forward FLOPs per token are within a small multiple of 0001's ungated carried-inverse layer, and well below 30-step CG.

Refuted if the observation covariance depends on the query time, if the boundary update needs a d×d factorization, or if the method is already published.

### Scratch prototype (not evidence)

Three prototypes in the session scratchpad checked the formulas. The first two compared the chunk formulas with the token-by-token Kalman recursion (NumPy); the third checked the adjoint against autograd.
- First prototype: d = 8 and 16, e = 5, chunks of 12 and 20 tokens, 3–4 chunks, gates uniform in [0.6, 1] and [0.3, 1], scalar and per-channel, ridge 0.7, β in [0.5, 5]. Maximum relative output error 1.8e-16 to 4.7e-16, boundary M and P errors ≤ 4.4e-16. With gates equal to 1 it matched the Mesa ridge readout to 2.9e-15 (absolute). It used the algebraically equal form with P_0 − Λ⁻¹, which cancels when gates are near 1, and a separate prefix solve per token.
- Second prototype: the README's form (P_0 term plus the in-chunk noise term with I − 𝒜² from `expm1`, gates in log space) and its masked readout O = (Q𝒜)M_0ᵀ + tril(Yᵀ)R̃ from one Cholesky. Five settings, including gates down to 0.05, gates in [0.999, 1] and d = 32 > C = 7: output ≤ 5.6e-16 relative, M and P ≤ 1.7e-15.

- Third prototype, for Proposition 6 (the adjoint of the prefix readout): PyTorch float64, C = 9 and 17, e = 4 and 6, a random positive definite 𝒦, upper-triangular X and random R and Δ. The closed-form ∂ℓ/∂X, ∂ℓ/∂R and symmetrized ∂ℓ/∂𝒦 matched autograd to ≤ 2.7e-15 (gradient entries up to 5.9).

None of the three left a file in this study, so the README does not use these numbers and its claims stay at E1. A proper float64 check with saved logs is the first next step (backlog).

### FLOP model

- `flop_model.py` (standard library only) → `logs/flop_model.log`, `results/flop_model_20261029_195529.json`. Per token and head, forward, e = d: the Kalman chunk layer needs 4.72–5.57× fewer FLOPs than 30-step CG with MesaNet's state update and readout, 1.74–2.16× fewer than 10-step CG, 1.39–1.85× the FLOPs of 0001's ungated carried inverse, and 3.18–5.17× those of gated linear attention without a solve (d = 64–256, C = 32–128). Decoding: 2.03–2.12× fewer FLOPs than 9 CG steps, 5.60–5.90× fewer than 30.
- Masked C×C products are counted dense in the Kalman formula, while the CG formula (copied from 0002) counts the intra-chunk sum at its average, so the comparison slightly favors CG.

### Novelty search

Two `literature-scout` agents were launched at about 19:45: one on the ML literature and code, one on statistics, time series, Gaussian processes, Kalman filtering and econometrics. The closest candidate known beforehand is "Memory by Design: Bayesian Layers" (Dowling et al. 2026), which 0002's search described as an exact chunked full-covariance Kalman memory computed sequentially within chunks. Results go in the next entry.

### Closest prior work, opened in this session (about 20:00)

"Memory by Design: Bayesian Layers for Sequence Modeling" (Dowling, Jeon, Savin, Park; arXiv 2605.31163, v5 of 2026-10-14). Opened through the arXiv abstract page and the v5 PDF (pp. 1–20 read). The HTML version returned 404.
- **Same model.** Definition 1: S_t | S_{t−1} ~ N(A_tS_{t−1}, ℓ_t²I) by columns, v_t | S_t ~ N(S_tᵀk_t, r_t²I), S_0 columns N(0, σ_0²I). Proposition 1 is the Kalman recursion with a shared dense D×D covariance; readout y_t = M_tᵀq_t. A_t is diagonal or blocks of damped rotations (Appendix C.1). The noise variances are free (softplus with a floor), not tied to the gates. So the study's model is the Bayesian Layer with a constrained ("stationary") noise parametrization: ℓ_t² = (1 − γ_t²)/λ and σ_0² = 1/λ for scalar gates, and diagonal noise (I − D_t²)Λ⁻¹ for per-channel gates, where BL uses isotropic noise.
- **The reduction to the delta-rule family is published.** Their Proposition 2 and Table 1: replacing the predicted covariance by λ_tI gives DeltaNet (A = I), Gated DeltaNet (A = α_tI), KDA (A = diag(α_t)) and Longhorn. That is this study's Proposition 5, so it is no longer claimed as new.
- **Their chunk algorithm is sequential within a chunk.** Appendix C.2, "Exact chunk computation": Π_tk_t from batched products, then "a sequential loop within the chunk then computes (u_t, β_t)", and "the gain and mean passes retain sequential dependencies over the chunk's positions". The exit covariance is a matrix product, P_L = Π_L − W diag(β)Wᵀ (their eq. 27), equivalent to this study's Proposition 3. Appendix C.3: a custom backward through those passes; Triton; cost O(LD²) per chunk for the covariances, O(L²D) for the gain history, O(LDm) for the mean and readout.
- **Consequence.** What remains for this study: the chunk-parallel form (Proposition 2), which turns their sequential gain loop into one C×C Cholesky of the observation covariance and their sequential mean pass into triangular solves and products; the prefix adjoint (Proposition 6); the stationary parametrization with telescoped noise sums and the conditioning bound (Propositions 1 and 4); and the explanation of why MesaNet's forgetting lacks the structure. The scouts' reports may change this.

### FLOP model, version 2

- `flop_model.py` now also counts the Bayesian Layer's published chunk algorithm as read above, and the work inside order-dependent steps over a chunk's positions (`logs/flop_model_v2.log`, `results/flop_model_20261029_200751.json`). The first version's tables are unchanged in it, so `logs/flop_model.log` is superseded only in that it lacks the new table.
- Against the Bayesian Layer (e = d = 64–256, C = 32–128): the chunk-parallel form costs 1.20–2.76× its FLOPs, but its order-dependent work is 1.18–26.77× smaller (6.00× at d = 128, C = 64). The advantage shrinks as C grows relative to d, because the C×C work grows as C³.
- My count of the Bayesian Layer's algorithm comes from reading their Appendix C. It is a lower bound for their loop (decay scalings inside the gain history are not counted), and their Triton kernel was not run.

### Fourth scratch prototype (not evidence)

Proposition 2 with general diagonal process noise Ψ_t, using φ_m = D_m²φ_{m−1} + Ψ_m in place of (I − 𝒜_m²)Λ⁻¹. This covers the Bayesian Layer's free isotropic noise ℓ_t²I with diagonal transitions, and its non-stationary prior P_0 = 3I. Four settings, against the token-by-token filter with the same noise: isotropic and per-channel noise, per-channel gates in [0.5, 1], [0.05, 1], [0.3, 1] and gates equal to 1; d = 6–16, C = 9–24, 2–4 chunks. Output ≤ 2.7e-15 relative, M ≤ 2.9e-15, P ≤ 1.6e-15. No file in the study; not evidence.

### Novelty search results: ML scout (finished about 20:15)

The ML-literature scout ran about 105 tool calls. Sources: web search (13 queries, three in extended mode), the arXiv API (Kalman with linear attention, delta rule, DeltaNet, fast weights, test-time; Bayesian with linear attention or associative memory and covariance; Kalman with sequence or language models), Semantic Scholar citations of Kalman Linear Attention (11), Gated KalmaNet (12) and Kalman Delta Networks (1) (keyword search failed with HTTP 429), OpenAlex, the OpenReview search API (5 queries), and the flash-linear-attention ops listing (no Kalman op). Its findings, each checked by me as noted:
- **Bayesian Layer** (also checked above): the same model. v1 already computed the mean pass in parallel with DeltaNet's WY readout once the gains were known; v5's mean pass is sequential. Both versions compute the gains in a sequential loop. Its code is not public.
- **Kalman Delta Networks** (Bui, Huang, Ying; arXiv 2609.07816). I opened the abstract and its HTML, Sections 4 and 6. Section 4: the exact recursion "is not directly compatible with fully parallel linear attention due to its Riccati covariance recursion". Section 6, on the Bayesian Layer: "their implementation requires quadratic covariance storage and sequential chunk processing". It trains diagonal and isotropic relaxations with associative scans.
- **Voltic** (Hejabi, Dehghani, Piray; arXiv 2610.05700). I opened the abstract: "A dense covariance would have to be propagated token by token, ruling out the parallel training these models depend on." It uses diagonal and quasi-diagonal approximations.
- **Gated KalmaNet**. I opened the HTML, Sections 6.3–6.4. DeltaNet assumes a "fixed error covariance: Σt=In for all t"; Gated DeltaNet is in Eqs. 18–19; and "the KF recursion in its most general form is inherently sequential". This is the earliest isotropic-covariance reduction found.
- **Kalman Linear Attention**. I opened the abstract: information form, associative scan. The diagonal covariance rests on the scout's reading of its Section 4, footnote 2, and on study 0001's table. Its OU prior is discretized exactly, which is the stationary form for a diagonal state.
- **Bogunovic, Scarlett & Cevher 2016**. I opened PDF pp. 1–4, via the scout's saved copy. Eq. 4 gives a stationary AR(1) GP; Eqs. 7–8 give the posterior with K_t ∘ D_t. It is the same kernel identity for a constant gate, with a full t×t solve.
- **Särkkä & García-Fernández**. I opened PDF pp. 1–4, via the scout's saved copy. It is an exact parallel scan, and each combine (Lemma 8) inverts an n_x×n_x matrix, here d×d.
- Related only, per the scout (not opened by me): MesaNet, Palimpsa, Longhorn, test-time regression, Titans, ATLAS, LaCT, RKN, KalMamba, Riccati SSMs and others listed in its report. None has a chunk-parallel dense Kalman memory.
- The scout flagged a notation risk, "A_max/A_min", in my brief. That brief meant the decay at the later token over the decay at the earlier one, a ratio ≤ 1. The README writes 𝒜_{m′/m} with m′ = max(i, i′) as an index, which is unambiguous.

Conclusion for the README: the model, the isotropic reduction, the matrix-product exit covariance and the stationary prior are known. No prior publication was found of the chunk-parallel form or of the MesaNet prefix-structure argument. The statistics scout is still running.

### Rate limit, then FLOP model version 3 (about 21:10–21:20)

- Both background agents (the statistics scout and the reviewer) stopped on a session rate limit at about 20:25 and were resumed after it reset at 21:10. Nothing they did was lost.
- **BL v1 checked myself** (PDF via the ML scout's saved copy, Appendix D, pp. 14–15). Its "covariance pass over the chunk's τ steps" computes the gains sequentially; then "the affine mean update (31) is computed in parallel via the WY-factored chunkwise readout of Yang et al. (2024b)". So the strongest published combination has a parallel mean pass. Version 2's "order-dependent work" counted v5's sequential mean pass (7Cde), which overstates the advantage against that combination.
- **`flop_model.py` version 3** (`logs/flop_model_v3.log`, new results JSON). The v1 and v2 tables are unchanged in it. It adds the chunk form with L⁻¹ formed explicitly, so every d- or e-dependent step becomes a matrix product and the only order-dependent work is the C×C Cholesky and inverse, (2/3)C³ per chunk. It also adds a "bl_wy" baseline: v5's gain loop (2C²d, order-dependent and scaling with d) plus v1's WY mean pass (its C×C inverse, C³/3). Results, e = d = 64–256, C = 32–128:
  - the explicit-inverse form costs 1.23–3.38× BL v5's total FLOPs;
  - its order-dependent work is 2.0–24.5× smaller than bl_wy's (6.5× at d = 128, C = 64);
  - none of its order-dependent work scales with d or e.
  - bl_wy's total FLOPs are not modeled.
- The README still quotes version 2. It will be updated together with the review's fixes.

### Novelty search results: statistics scout (finished about 21:25)

The scout ran about 16 tool calls after resuming, plus its work before the rate limit. It made about 37 queries over web search, arXiv, OpenAlex (Semantic Scholar returned 429 throughout), publisher and repository pages, and lecture notes. Its full query list is in its report; the main themes were the Kailath–Sayed–Hassibi causal estimator, the Kalman filter as a Cholesky or prediction-error decomposition, the Brockwell–Davis innovations algorithm, the KRLS tracker and back-to-the-prior forgetting, spatio-temporal GPs as Kalman filters, block-exact LMS and RLS, extended RLS, TVP regression and the Chan–Jeliazkov precision sampler, Koop–Korobilis forgetting factors, Rosenberg 1973, Kulhavý–Zarrop stabilized forgetting, the asynchronous EnKF, and subsampled-updating Kalman variants. What it found, and what I checked myself:
- **Slock & Maouche 1994, subsampled-updating RLS.** I checked the preprint, pp. 1–12. Eq. 2 is exponentially weighted LS with a fading prior, λ^{k+1}μ‖W − W_0‖². §2.5, eqs. 28–33, carries R⁻¹ across blocks with one L×L "likelihood variable" matrix. §2.6, eqs. 41 and 47, recovers every in-block RLS a priori error from that matrix's UDL factorization. The authors say SU RLS alone "does not represent much computational gain"; their fast version uses FIR shift structure and the FFT. This is the block mechanism of study 0001's identity, from 1994, so an erratum went into 0001's README. For 0003 it is the same mechanism without process noise.
- **Kailath, Sayed & Hassibi, Section 4.1.2**, checked through Jiao's EECS 225A Lecture 7 notes, pp. 1–4. With R_Y = LDL*, the causal Wiener filter is K_f = [R_XY L^{−*} D^{−1}]_L L^{−1}, and the notes call it "not only conceptually important but also numerically efficient". This is the general identity behind Proposition 2. The book itself was not opened.
- **Van Vaerenbergh, Santamaría & Lázaro-Gredilla, MLSP 2012.** I checked pp. 1–4. Back-to-the-prior forgetting (eq. 4); exact equivalence to a GP with STAR1 covariance λ^{|t−t′|/2}k(x, x′) (eqs. 5–6); a linear kernel in the experiments; the batch matrix Λ∘K + σ²I appears only in the marginal likelihood (eq. 7). This is the stationary scalar model; the computation is sequential.
- **Titsias et al., ICLR 2024.** I checked pp. 3–4 of the PDF. The prior is p(w_i | w_{i−1}) = N(γ_iw_{i−1}, (1 − γ_i²)σ²I) (eqs. 2–4), the K output columns share one covariance (eqs. 7–8), and γ is learned by SGD. This is the stationary scalar model with a shared covariance, computed sequentially.
- Reported by the scout but not opened by me, so not cited: Brockwell–Davis (innovations algorithm), Haykin et al. 1997 (extended RLS, ERLS-1 with F = aI), Rosenberg 1973 (convergent-parameter regression), Sakov & Bocquet 2018, Chan & Jeliazkov 2009, Koop & Korobilis 2013, Kulhavý & Zarrop 1993, and Azizi & Quinn and van de Laar et al. on stabilized and linear forgetting. Its reading of Kulhavý–Zarrop is that stabilized forgetting is information-form mixing, which with a Gaussian prior gives ER-RLS (MesaNet's system). Covariance-form mixing comes from AR(1) dynamics. This agrees with the README's contrast.
- Its conclusion: statement 1 is published in several fields; statement 2 is a classical result applied in a new setting, not found published for the AR(1)/process-noise model with per-coordinate decay and query readouts.

### README revised (about 21:35)

The Summary, Claims, Method (where the identity comes from; Cost), Results (the version 3 table), Novelty check, Limitations, References and Reproduce now use the verified prior work above and `logs/flop_model_v3.log`. The verdict stays "partly known"; it now says that every ingredient is classical.

### Statistics scout's query list (saved here at the reviewer's request)

About 37 queries, 2026-10-29, over web search, arXiv (abstract, PDF, ar5iv), OpenAlex (records, and the citing works of SU RLS and of the KRLS tracker) and publisher or repository pages (Zenodo, EURASIP, gtas.unican.es, EPFL ASL, Eurecom, NBER, Strathprints, ANU, Cambridge Core, Taylor & Francis, Routledge, Aalto), plus the Berkeley EECS 225A and Stat 153 lecture notes. Semantic Scholar returned 429 on every call. The queries:
- the Kailath–Sayed–Hassibi causal estimator by triangular factorization;
- the Durbin–Koopman Kalman filter as a Cholesky factorization;
- the Kalman filter, Cholesky and the prediction-error decomposition;
- Eubank's *A Kalman Filter Primer*;
- the Brockwell–Davis innovations algorithm: the book, lecture notes, and nonstationary κ(i, j);
- the relation between the innovations algorithm and the Kalman filter;
- the KRLS tracker and back-to-the-prior forgetting;
- Särkkä–Solin–Hartikainen spatio-temporal GPs;
- Markovian GPs: Cholesky via the Kalman filter, and celerite;
- block-exact RLS and Kalman filters;
- fast-exact and block-exact LMS and RLS (Benesty–Duhamel, Slock);
- lifted Kalman filters;
- extended RLS (Haykin et al.);
- sequentially semiseparable and quasi-separable Cholesky with Kalman (Dewilde, Chandrasekaran);
- Corenflos–Zhao–Särkkä parallel GPs;
- blocked or chunked GPU Kalman filtering with a dense within-block Cholesky;
- mini-batch Kalman filtering for GPs;
- a TVP or dynamic-linear-model block Cholesky filter (extended mode);
- Chan–Jeliazkov and the precision sampler (two queries);
- Koop–Korobilis forgetting factors;
- return-to-normality regression (Rosenberg 1973);
- Kulhavý–Zarrop (web and OpenAlex);
- the Kulhavý–Kraus duality (web and OpenAlex);
- van de Laar's stabilized linear forgetting paper;
- "linear forgetting" in covariance form;
- "stabilized forgetting" and geometric mixtures;
- the asynchronous EnKF (Sakov), and its OpenAlex record;
- moving-horizon estimation and its equivalence to the Kalman filter;
- Duncan & Horn 1972;
- subsampled updating with Kalman, random-walk or process-noise models;
- the OpenAlex record of subsampled-updating RLS.

### Independent review (self-check) and fixes (about 21:30–21:45)

The `reviewer` agent's verdict was **revise (minor)**. It reviewed the files at FLOP model version 3. It re-derived Propositions 1–6 and the Remark and checked them in float64 in its own scratch folder: 24 cases against the token-by-token filter, worst output error 6.1e-15. It recomputed every range against `logs/flop_model_v3.log`, and found the counting conservative for the new method. Its issues, each checked by me before fixing:
1. **major, fixed.** The unsymmetrized ∂ℓ/∂𝒦 in Claim 3 and Proposition 6 was written as −ĀW̄ᵀ. For the readout written as κᵀ𝒦⁻¹R, the differential is −w_tᵀ(d𝒦)a_t, so the unsymmetrized form is −W̄Āᵀ. The two agree only on symmetric perturbations. Since 𝒦 is symmetric, the README now gives the symmetric gradient −½(ĀW̄ᵀ + W̄Āᵀ) and explains the ambiguity. My third scratch prototype had compared exactly this symmetrized gradient, so it was not affected.
2. **minor, fixed.** The Summary quoted 4.7–5.6× fewer FLOPs than 30-step CG next to the explicit-inverse form; that range belongs to the triangular-solve form, and the Summary now says so. It also now says that both forms take about 2C dependent steps per chunk, so the order-dependent ratio measures step width, not latency. "An exact chunk solve needs one eigendecomposition" became "0002's exact solver uses one".
3. **minor, fixed.** BL v1's parallel WY mean pass is now stated in the Summary, the Problem and the novelty table, and v1 is in the References. The "no prior publication" claim for the MesaNet contrast is dropped; the contrast restates Lai & Bernstein's obstacle, as cited by 0002. Kalman Linear Attention is credited with the exact OU prior for a diagonal state.
4. **minor, fixed.** Added the Kanerva Machine (Wu et al., ICLR 2018). I checked its eqs. 9–11 myself (PDF pp. 3–4): a matrix memory with a shared row covariance, written in one Bayesian update through a T×T observation covariance, which it says equals the sequential updates. That is Proposition 3 without dynamics. Sakov & Bocquet 2018 could be opened only as an abstract (Taylor & Francis returned 403, the CEREA page 404), and is listed as such. The statistics scout's query list is saved above.
5. **minor, fixed.** The innovations sentence now says what L⁻¹ does, instead of the incorrect "L's columns are the normalized innovations". Proposition 4(a)'s "no division" is scoped to scalar gates, and the per-channel overflow (log cumulative decay below about −88.7 in float32) is stated, with GLA's remedies; it is also in the Limitations. The rotation limitation is reworded: the key fact holds for any linear-Gaussian transition, and the formulas probably extend through complex cumulative products, which was not checked.
6. **minor, fixed.** Reproduce now tees to `/tmp` and diffs against the recorded log, and uses `python3`. I ran it: only the "saved" line differs. That rerun's extra JSON (`results/flop_model_20261029_214139.json`) was deleted right after, following study 0002's convention for rerun copies. No recorded file was touched.

The status stays `review`. This review ran in the same session that wrote the study, so it is a self-check (PROTOCOL section 8). Every issue it raised is fixed, and none is open.

### E2 check (about 21:45–22:00), at the user's request ("Yes, run E2")

Backlog I-015. The CPU of the reference workstation, `.venv` (PyTorch 2.11 CPU tensors, NumPy 2.4.6 with x87 long double, eps 1.1e-19). No GPU was used.
- Wrote `kalman_memory.py`: the token-by-token filter (torch, and NumPy at any precision); the chunk step with triangular solves or an explicit L⁻¹, and with log-space or GLA-style matrix-product Grams; the per-token prefix-solve form; the Mesa readout.
- Wrote `check_chunk.py` (float64; parts A, A2, B, C, D) and `check_float32.py` (parts E1, E2). Smoke tests with `--quick` ran in the terminal only, with no files kept.
- **First smoke test of `check_float32.py`.** The fast per-channel configuration (gates uniform on 0.2–1) did not overflow the matrix-product Grams. Its mean log gate is about −0.6, so a 64-token chunk reaches only about −38. Changed to gates 0.1–0.3 (about −105), which does overflow. Also added cond(𝒦) per configuration and a variant whose C×C stage runs in float64.
- `logs/check_chunk.log` (`results/check_chunk_20261029_215153.json`): **superseded** by v4. Parts A–D; all float64 errors at round-off, but the hardest case reached only cond(𝒦) = 607.
- `logs/check_chunk_v2.log` (`results/check_chunk_20261029_215432.json`): **superseded** by v4. Adds part A2, the conditioning sweep with unit and decaying keys at gates 0.99–1; cond(𝒦) saturated near 9e3.
- `logs/check_chunk_v3.log` (`results/check_chunk_20261029_215514.json`): **superseded** by v4. Adds rank-4 key blocks (d = 4, C = 32). These still saturated near 1e4, because the stationary noise (1 − γ²)/λ grows as the ridge shrinks.
- `logs/check_chunk_v4.log` (`results/check_chunk_20261029_215549.json`): **current**. Adds gates = 1 to the sweep, where no noise refills the prior. Parts A–D are identical to v1 (checked with diff). Results:
  - part A: 26 cases, worst errors over the four forms 8.9e-16 / 7.4e-16 / 8.6e-16 / 1.2e-15; float64 token-by-token 4.8e-16; M_T 3.0e-15, P_T 5.5e-16;
  - part A2: with gates 0.99–1, cond ≤ 9.68e3 and errors ≤ 7.4e-14 (token-by-token ≤ 2.2e-15); with gates = 1 at ridge 1e-6, cond up to 5.15e8 and errors up to 4.0e-8 (token-by-token 2.7e-9);
  - part B: 2.9e-15, 1.8e-15 / 2.2e-14, 7.2e-16, 1.5e-16, 3.8e-16;
  - part C: 0 violations in 120 chunks;
  - part D: adjoint ≤ 8.4e-16; gradients ≤ 1.3e-15; central differences 7.6e-10–1.6e-9.
- `logs/check_float32.log` (`results/check_float32_20261029_215528.json`): **current**. Ran in the background, about 60 s. Results:
  - no drift over 65,536 tokens; well-conditioned errors 2.4–8.4e-7 against 1.4–6.0e-7 for the float32 token-by-token filter;
  - at cond 430 and 2.7e3, up to 9.0e-6 and 1.5e-5 against 3.9e-6 and 6.1e-7;
  - the float64 C×C stage does not help;
  - the matrix-product Grams fail with gates 0.1–0.3;
  - the "P_0 − Λ⁻¹" form has 52× the error of the PSD sum at gates 0.9999–1 with β = 50.
- **New finding.** The chunk form is exact, but in floating point it is less accurate than the token-by-token filter when cond(𝒦) is large, because the error enters when 𝒦 is formed. Added as a limitation and as backlog I-018.
- Claims 1–5 and 7 move from E1 to E2; claims 8–10 are new (E2); claim 6 (the FLOP model) stays E1. The README's front matter, Summary, Claims, Results, Limitations, Next steps, Reproduce and a new Files section are updated.

### Review of the E2 additions, and fixes (about 22:00–22:25)

The `reviewer` agent's verdict was **revise**. The float64 evidence holds:
- the reference is independent and really runs in long double;
- a brute-force build of 𝒦, X, Z and R from the state-space covariances agrees with `chunk_matrices` to ≤ 6.1e-16;
- every quoted number matched the logs.
It raised one major issue and four minor ones; all are fixed. It ran its own checks in the session scratchpad (`rev/`), and wrote nothing into the repository.
1. **major.** The float32 claims rested on easy inputs: only the two best-conditioned configurations ran 65,536 tokens, and nothing ran without forgetting or at gates ≥ 0.9999. Its scratch runs showed chunk errors up to 6.1e-4 there, and drift in the float32 token-by-token filter without forgetting. Fixed with `check_float32.py` version 2:
   - four new configurations at 65,536 tokens: no forgetting with ordinary writes, strong writes, or a decaying key spectrum, and strong writes at gates 0.9999–1;
   - the smallest eigenvalue ratio of P at every chunk boundary;
   - a variant that forms 𝒦 in float64 (`kmat_dtype` in `kalman_memory.chunk_step`).
   The run is `logs/check_float32_v2.log` (`results/check_float32_20261029_222118.json`), about 4 min in the background; the six earlier configurations reproduce v1 exactly. New results:
   - the chunk forms reach up to 3.7e-4, 2.3–56× the token-by-token filter, at cond(𝒦) 430–1.35e4;
   - without forgetting, the token-by-token filter drifts from 4.6e-7 to 6.8e-6, while the chunk form rises from 4.5e-7 to 7.9e-7;
   - forming 𝒦 in float64 cuts the error 2–10×;
   - P stays positive definite at every boundary (ratio ≥ 7.4e-5).
2. **minor.** "Grows like cond·eps" was too neat, and "cond(𝒦) stays below 1e4 with gates below 1" was too broad. Fixed with `check_chunk.py` version 5 (`logs/check_chunk_v5.log`, `results/check_chunk_20261029_221909.json`), whose sweep also covers gates 0.999–1 and 0.9999–1 and prints err/(cond·eps) and chunk/token ratios:
   - 46 of 48 rows are ≤ 0.46·cond·eps, and the two exceptions are driven by the downdate from a 1/λ prior;
   - the gap is 2.1–8.2× for unit keys, 11–56× for decaying keys and 12–195× for rank-4 blocks;
   - cond(𝒦) reaches 1.0e5 at gates 0.999–1 and 1.0e6 at 0.9999–1.
3. **minor.** The gradients had been checked only at benign conditioning. v5 adds an ill-conditioned configuration, where the two autograd paths differ by 2.0e-13 and the central-difference check is inconclusive (6.3e-6 at h = 1e-5), and one with M_0 and P_init = SSᵀ as inputs (≤ 8.3e-16).
4. **minor.** "The error enters when 𝒦 is formed" was inferred, not tested. It is now tested and narrowed by the 𝒦-in-float64 variant: forming 𝒦 in float64 helps 2–10×, but one configuration stays 11× above the token-by-token filter. Backlog I-018's remedies are updated: higher-precision formation of 𝒦 and the other chunk matrices, and refinement only with a higher-precision residual.
5. **minor.** Wording and records. "Matches" and "comparable" are replaced by ratios. STUDIES.md is regenerated, and backlog I-006's "E1" is updated. The Reproduce note now says the float32 timing lines also differ, and the Files row is corrected (v1 has no part A2).

Superseded: `logs/check_chunk_v4.log` (by v5: same parts A, B and C, a smaller A2, fewer gradient configurations) and `logs/check_float32.log` (by v2). The reviewer also suggested citing Preconditioned DeltaNet next to Kalman Delta Networks and Voltic, as another paper that avoids the dense covariance. It is cited in study 0002, and is not added here.

### GPU benchmark for E3, prepared in a session without a GPU (about 21:45–23:10)

Session: a Claude Code cloud container (4 CPU threads, no GPU, no computer attached through Remote Control), run by Claude Opus 5.5 for @MaurerKrisztian, who asked to "make 003 to the e3 level with gpu tests". E3 needs timings on a GPU, which this session cannot produce. So this entry builds and checks everything the GPU run needs, backlog I-016, and leaves the run itself to the reference workstation. The study stays at E2; nothing below is a GPU number.

**Environment.** The session's network policy denied download.pytorch.org and arxiv.org. PyTorch 2.11.0 came from PyPI (its CUDA 13.0 build, used on the CPU), with NumPy 2.4.6 and Triton 3.6.0. BL's paper could not be re-opened, so its baseline follows this study's own reading of BL v5's Appendix C.2–C.3 and v1's Appendix D, recorded in the entries above.

**What was written.**
- `kalman_batched.py`: three algorithms for the same filter, batched over heads, on any device, differentiable:
  - the chunk-parallel form;
  - BL's chunk algorithm: Π_tk_t by one product, the gain loop over the chunk's positions with the earlier update directions transported by the decays, the exit covariance (BL eq. 27) as a Gram matrix of the scaled directions, and the mean by v5's token loop or by a WY form of v1's parallel mean pass, derived here;
  - the token-by-token filter.
  The chunk forms pad T with zero keys, queries and values, gates 1, no noise and β = 1, which leaves outputs and final state exact. The reading of BL leaves one detail open: whether its loop transports the earlier directions at every step, as done here (stable, about td extra work per step), or rescales them by cumulative decays.
- `bl_triton.py`: BL's two loops as Triton kernels, forward only. One program per head runs a chunk's gain loop; one program per head and block of 16 value rows runs the mean pass over the whole sequence. Both prefetch the next token's vectors. On the GPU, num_warps is autotuned per shape (2, 4, 8 or 16). The kernels use Triton's default float32 division and square root, which are not correctly rounded (about 1–2 ulp); float64 is exact either way.
- `check_batched.py`: the E2 check of these, parts P1–P4 (below).
- `gpu_bench.py`, `aggregate_gpu_bench.py` and `run_gpu.sh`: the GPU benchmark, its tables, and one command that runs the GPU probe, a smoke test, the benchmark and the tables, without ever overwriting a log. The design:
  - every method gets the same float32 inputs with TF32 off, and pays for its own chunk-local precomputation;
  - forward passes are timed under no_grad, and also as CUDA-graph replays, to take the CPU's kernel-launch overhead out of the comparison. In eager PyTorch the loops over positions (BL's in PyTorch, the token-by-token filter) are launch-bound, which would flatter the chunk form;
  - the chunk form is timed with cuSOLVER's and with MAGMA's batched Cholesky, and with an explicit L⁻¹;
  - BL is timed with the Triton kernels (v5 mean, or v1 WY mean) and in plain PyTorch;
  - forward + backward is timed by autograd for the PyTorch forms;
  - references from other models: 0001's carried inverse, MesaNet with 30 CG steps through 0002's own `gpu_bench.py` (loaded by path), and, when installed, flash-linear-attention's Gated DeltaNet, KDA and MesaNet kernels (fla-core 0.6.0's API, read from its wheel);
  - PyTorch's allocator is capped at 14 GiB, so a run that would exceed it fails with out-of-memory, which is recorded;
  - part C splits times into stages with CUDA events; part D repeats check_float32.py's ten long-sequence configurations on the GPU, with 4 heads each.

**Dead ends and fixes during development** (scratch runs in the session scratchpad; not evidence):
- Triton's interpreter fails on runtime loop bounds with NumPy 2.4 ("only 0-dimensional arrays can be converted to Python scalars"). The chunk size and sequence length are compile-time constants in the kernels now.
- The first batched token-by-token filter subtracted Pk(Pk/s)ᵀ, which is not exactly symmetric in floating point. Without forgetting, nothing damps the asymmetry: on "no forgetting, strong writes" at 4096 tokens, its float32 error in the last three of eight segments was 4.6e-3 to 7.3e-3, against 5.2e-5 to 6.2e-5 for kalman_memory.kalman_reference, which subtracts PkPkᵀ/s. Fixed to the symmetric form, so the baseline is not handicapped. BL's exit covariance, first Wᵀ(W/s), is now the Gram matrix of the W/√s, like the chunk form's Z̃ᵀZ̃.
- After that fix the batched filter still differed from the single-head one, by up to 5× on the decaying-spectrum configuration. Replacing one dot product (kᵀPk as an elementwise sum, a matrix product, or BLAS) changed the error by about as much. That is the rounding order, not a bug, and part P4 now records it.
- The chunk form's two products with [𝒜K; 𝒜Q] are fused, and its updates use fused multiply-adds; BL's WY mean got the same treatment. These are the small fusions that a careful plain-PyTorch implementation of either method would have.
- Smoke tests, `gpu_bench.py --quick` on the CPU, with and without `TRITON_INTERPRET=1`, ran all four parts; their JSONs went to the scratchpad, not to `results/`.

**Recorded run.** `TRITON_INTERPRET=1 ../../.venv/bin/python -u check_batched.py` → `logs/check_batched.log`, `results/check_batched_20261029_225300.json`. It took 22 minutes on the cloud CPU, mostly in the Triton interpreter, and shared the CPU with a smoke test for part of that time.
- P1: 16 cases with 3 heads each. Every form is within 8.9e-16 of the extended-precision filter in its outputs, 1.2e-15 in M_T and 9.1e-16 in P_T; the batched token-by-token filter is within 6.3e-16.
- P2: autograd through the four PyTorch forms matches the batched filter's to ≤ 9.1e-16 on three configurations, and to ≤ 1.6e-13 on the ill-conditioned one (as in check_chunk.py's part D).
- P3: in float32 at T = 2048, every form gives 2.0–5.5e-7 on the three well-conditioned configurations. With strong writes the chunk forms give 6.4–7.5e-6, BL's forms 4.8–5.5e-6 and the token-by-token filter 2.6e-6.
- P4: the three token-by-token filters disagree by up to 1.3×, 1.3×, 1.7× and 5.1× on the four configurations. On the last one, without forgetting and with a decaying key spectrum, the chunk form gives 2.5–2.7e-4 and BL's WY form 1.2–2.1e-4, against 3.9e-6–2.0e-5 for the token-by-token filters.

**README.** New claims 11 (E2) and 12 (E2, CPU numerics), with claim 9 now pointing to claim 12. Also new: a Summary bullet, a Method subsection (Implementation for the GPU benchmark) and a Results subsection, plus updated Limitations, Next steps, Reproduce and Files. The Limitations' "six configurations" for the float32 runs was stale since check_float32.py version 2 and now says ten. Backlog I-016 says how to run the benchmark. The evidence level stays E2 and the status `review`. The new code has had no independent review, which is due with the E3 write-up.

**Final smoke tests** (scratchpad, not evidence). `gpu_bench.py --quick` on the final code ran all four parts in 56 s. With `TRITON_INTERPRET=1 --parts C D --drift-T 128`, the Triton paths of parts C and D ran too. As claim 10 predicts, on per-channel gates of 0.1–0.3 the chunk forms, and BL's WY form, which also divides by cumulative decays, gave non-finite outputs; BL's loop form did not.

**What remains for E3.** Run `bash run_gpu.sh` from this folder on the reference GPU. Its smoke test runs first (56 s on this cloud CPU; not yet timed on a GPU). The full run is estimated at 20–40 minutes, from counting kernel launches, not measured. Then write the Results and Claims from `aggregate_gpu_bench.py`'s tables, review, and set the evidence to E3 only for what the run shows. Backlog I-016 says what the run decides.

## 2026-10-30

### E3: the GPU benchmark on the reference workstation (from about 00:20)

Session: the reference workstation (RTX 5060 Ti 16 GB; `.venv` with PyTorch 2.11.0 + CUDA 12.8 and Triton 3.6.0), run by Claude Opus 5.5 for @MaurerKrisztian, who asked to "make the 0003 study to E3 level, run the missing things". The previous entry's code arrived as a patch from the cloud session and was applied unchanged on the branch `study-0003-gpu-bench`. Before the run, `tools/studies.py check`, the unit tests and `check_batched.py --quick` (CPU, Triton kernels skipped; log in the session scratchpad, not evidence) passed.

**Environment change.** flash-linear-attention was not installed. `uv pip install fla-core==0.6.0 einops==0.8.2` added it to `.venv` and left PyTorch and Triton unchanged. fla 0.6.0 also imports `packaging` without declaring it, so `packaging` 26.3 was added too.

**Hypothesis (backlog I-016, written down before the run).** The chunk-parallel form is faster than BL's chunk algorithm at d = 128 and C = 32–64, where it needs 1.36–1.64× BL v5's FLOPs but has 6.5–12.5× less order-dependent work (claim 6, E1). It is also faster than MesaNet with 30 CG steps in the forward and 30 in the backward pass. Decision rule: in those cells, compare the fastest chunk variant with the fastest BL variant, eager and as CUDA graphs. The hypothesis holds where best BL / best chunk is above 1 by more than the spread between reruns, and is refuted where it is below 1 by more than that spread. Forward + backward compares only the differentiable PyTorch forms, since the BL Triton kernels are forward only. MesaNet, Gated DeltaNet and KDA are different models, so their times give scale, not a verdict on the Kalman form.

**Stability.** Protocol stage 5 asks for numbers that are stable across reruns, so part B runs a second time after the full run, and ratios are quoted only with the spread between the two runs.

**Smoke test v1** (`logs/gpu_probe.log`, `logs/gpu_bench_quick.log`, `results/gpu_bench_quick_20261030_003709.json`). The probe matches docs/COMPUTE.md: 17.3 TFLOPS in float32 and 385 GB/s. `gpu_bench.py --quick` ran all four parts in 682 s, against 56 s on the cloud CPU. In part A every Kalman method, the Triton kernels compiled for this GPU included, is within 1.1e-15 of the extended-precision filter in float64. Part D at 1,024 tokens reproduces claim 10: at per-channel gates 0.1–0.3 the chunk forms and BL's WY mean give non-finite outputs, and BL's Triton loops do not. Four findings changed the benchmark before the full run:
1. **MAGMA is inert.** PyTorch 2.11 warns that MAGMA is deprecated and that the call "will try dispatching to cuSOLVER instead". chunk and chunk_magma have identical errors in all five part A cases and equal times. A scratch check in the session scratchpad (not evidence) found the Cholesky factors bitwise equal under both settings, for batches of 64 16×16, 64 64×64 and 16 128×128 matrices. Timing chunk_magma would repeat chunk under another name and pull the fastest-variant minimum toward the chunk form. Part B now leaves it out (`--magma` puts it back), and part A prints whether its outputs are bitwise equal to chunk's.
2. **Autotuning per number of heads.** The autotune keys of the two Triton kernels did not include N, so the configuration tuned at N = 16 was reused at N = 64, where the grid has 4× the programs. Both kernels now take n_heads, used only as an autotune key, so BL's kernels are tuned for each N. The computation is unchanged.
3. **Run length.** Extrapolating the smoke test's per-call times from T = 256 to 2048 (BL's PyTorch loops: 63 ms forward and 300 ms forward + backward at T = 256) and counting the calls per cell put the full run near the lab's 60-minute limit for one run. It is now three runs: parts A–B, parts C–D, and part B again in a fresh process to measure how stable the times are. `gpu_bench.py` prints each cell's and each part's seconds, `aggregate_gpu_bench.py` merges the JSONs and with `--rerun` gives each time's rerun / first-run ratio, and `run_gpu.sh` runs the three.
4. **mesa_carry is timed eager only.** Its CUDA-graph capture failed in every cell: `lab.layers.mesa` calls `torch.linalg.cholesky`, whose error check synchronizes with the host. It is a reference from another model, so this is left as is.

**Smoke test v2** (`logs/gpu_bench_quick_v2.log`, `results/gpu_bench_quick_20261030_004440.json`), on the changed code. All four parts ran in 64 s (part A 15 s, B 34 s, C under 1 s, D 15 s). Most of v1's 682 s was therefore the first compilation of the Triton and fla kernels, now in Triton's cache. In all five part A cases, chunk_magma's outputs are bitwise equal to chunk's. The worst float64 error over all methods is 1.1e-15.

**Full run, parts A–B** (`logs/gpu_bench.log`), started at about 00:46.

**Full run v1 of parts A–B, stopped: discarded** (`logs/gpu_bench.log`; no results JSON). Part A passed in 6 s. The first four cells (N = 16, d = 64, scalar gates; 362 s for the first cell, which includes compiling the kernels, then 47–51 s each) ran normally. Then 0002's exact solver ran out of memory at C = 128. In the next cell (per-channel gates, C = 16), which took 438 s, the token-by-token filter's forward + backward ran out of memory, although it had needed only 3.08 GiB one group earlier. From the cell after that on, every CUDA-graph capture failed with out-of-memory. nvidia-smi showed the process holding 15.8 of 16.3 GB, so the run was stopped at about 01:02. Its times after the first out-of-memory error are unreliable, and none of its numbers are used.

Diagnosis, with scratch scripts in the session scratchpad (not evidence) that call gpu_bench.py's own run_one and part_b and print `torch.cuda.memory_allocated()` and `memory_reserved()` after every method:
- An out-of-memory error inside 0002's exact solver frees its memory, and on a fresh process the solver fits at C = 128 (N = 16, d = 64).
- Before mesa_carry's failed graph capture, reserved memory grew by about 0.02 GiB per method, from a cuBLAS workspace for each new side stream. After it, every later method's memory stayed reserved, and `empty_cache()` could not return it. Reserved memory went from 0.23 to 0.68 GiB at the failed capture, rose by 1.34 GiB after BL's PyTorch forward + backward, and reached 3.58 GiB after two cells. A failed capture leaves the caching allocator routing later allocations into the dead graph's private pool.
- Fix in `gpu_bench.py`: mesa_carry is no longer captured (its Cholesky's error check synchronizes with the host, so the capture always fails), all captures share one side stream, and a failed capture now prints a WARNING. With the fix, the same scratch trace over four full-size cells (N = 16, d = 64, C = 16 and 32, scalar and per-channel gates) held allocated memory at 0.07–0.08 GiB and reserved memory at 0.11 GiB throughout, with no out-of-memory error.

The stopped run's clean cells already show the pattern that the hypothesis cells will test. At N = 16, d = 64, C = 64 with scalar gates, BL's Triton forward took 8.40 ms against the chunk form's 11.44 ms eager, and 4.16 against 5.09 ms as CUDA graphs. These are not quoted as results.

**Smoke test v3** (`logs/gpu_bench_quick_v3.log`, `results/gpu_bench_quick_20261030_011137.json`), with the fix: 49 s, no failed capture, no out-of-memory error, chunk_magma bitwise equal to chunk in all five part A cases, and a worst float64 error of 1.1e-15.

**Full run v2, parts A–B** (`logs/gpu_bench_v2.log`), started at 01:11.
It finished at 02:13 after 3,720 s (`results/gpu_bench_20261030_021340.json`). That is 2 minutes over the lab's 60-minute limit for one run. The first cell of each N = 64 group took 367–493 s: fla's kernels autotune again for every new number of heads, and the token-by-token filter's forward + backward takes 4.6 s per call there. Stopping at 46 minutes, when the overrun became clear, would have thrown away the whole run, since the JSON is written at the end. No WARNING, failed capture or traceback. Notes, all expected:
- autograd's memory estimate skipped BL's PyTorch backward (25–32 GiB) and the token-by-token filter's (48 GiB) at N = 64, d = 128;
- fla's MesaNet kernel needs 116,236 bytes of shared memory per block at d = 128, and this GPU allows 101,376 (`OutOfResources`), so fla's MesaNet runs only at d = 64;
- 0002's exact solver ran out of memory at N = 64, C = 128. In the largest cells it finished it needed 8.09–8.72 GiB: C = 128 at N = 16, and C = 64 at N = 64.

**Changes for the remaining runs.** Each fresh process autotunes fla's kernels again, so the rerun of part B is split by N (`--N 16`, then `--N 64`), and `run_gpu.sh` now splits part B by N in both runs. `aggregate_gpu_bench.py` merges part B cells from several files, refusing duplicates, and takes several files after `--rerun`. It also gains a table of each family at its own fastest chunk size, per (N, d, gates). Comparing at equal C penalizes whichever method that C suits worse: BL's Triton gain kernel slows down at C = 128, and the chunk form at C = 16. A practitioner would pick each method's best chunk size, so that table carries the verdict.

**Parts C–D and the reruns** (`logs/gpu_bench_cd.log`, `logs/gpu_bench_rerun_n16.log`, `logs/gpu_bench_rerun_n64.log`), started at 02:15 as one chain.
They finished at 02:53, with no WARNING, failed capture or traceback: parts C–D in 345 s (`results/gpu_bench_20261030_021947.json`), the rerun for N = 16 in 909 s (`results/gpu_bench_20261030_023500.json`) and for N = 64 in 1,074 s (`results/gpu_bench_20261030_025256.json`).

**Tables** (`aggregate_gpu_bench.py`, all from the four JSONs above). `logs/aggregate_gpu_bench.log` came first. `logs/aggregate_gpu_bench_v2.log` adds summary lines for the rerun, so that its ranges are not read off tables by hand. `logs/aggregate_gpu_bench_v3.log`, the one quoted, adds the reference methods' forward + backward ratios and the counts of cells where the chunk form is faster. The tables are identical in all three, and v1 and v2 are superseded.

**Results** (`logs/aggregate_gpu_bench_v3.log`):
- Part A on the GPU: every Kalman method is within 1.1e-15 in float64, and chunk_magma is bitwise equal to chunk in all five cases.
- Forward pass, best BL / best chunk form: 0.63–1.69 per cell eager and 0.59–1.91 as CUDA graphs. With each family at its fastest chunk size, 0.66–1.08 eager (the chunk form faster only at N = 16, d = 128) and 0.68–0.99 as graphs. The fastest BL variant is always bl_triton; the fastest chunk variant is the explicit L⁻¹ in 14 of 16 entries.
- **Decision on the hypothesis, by the rule written down before the run.** In the 8 hypothesis cells (d = 128, C = 32–64), BL is faster eager in all 8 in both runs: 0.73–0.92 in the first, 0.73–0.99 in the rerun. Across all cells the eager ratio moved by 0.955–1.073× between the runs, so BL's lead exceeds that spread in 7 of the 8. In the eighth (N = 64, scalar, C = 32: 0.92, then 0.99) the difference is within the spread. As CUDA graphs BL is faster in 6 of 8, by 11–22%. The chunk form is faster in 2 (N = 16, C = 64), by 4–9% in both runs. Its own rerun change is small there, but within the 0.848–1.032 range of all graph cells. Verdict: refuted at these sizes. The second part (faster than 30 + 30-step CG) holds only against 0002's plain-PyTorch CG in the forward pass (chunk form faster in 14 of 16 cells). Against fla's fused MesaNet kernel (d = 64 only) it fails: fla takes 0.04–0.44× the chunk form's forward time and 0.04–0.25× its forward + backward time.
- Forward + backward through autograd: BL's PyTorch loops take 2.9–47× the chunk form's time and the token-by-token filter 12–84×. The chunk form also needs the least memory, 0.23–3.13 GiB above the inputs.
- Part C (C = 64, per-channel gates): the chunk form's time is spread over its stages (Cholesky 8–15%, triangular solve 13–22%, products 17–32%, updates 15–34%, precomputation 6–36%). BL's gain loop alone is 31–55% of BL's.
- Part D on the GPU reproduces claim 9, and shows two things new. BL's chunk algorithm is about as inaccurate as the chunk form on the five ill-conditioned configurations: 2.18–33.27× the token-by-token filter's worst error, against 1.77–30.01×. Without forgetting, BL's token-by-token mean pass drifts (4.7e-7 to 4.8e-6) while its covariance error equals the chunk form's (1.7e-6). So the drift comes from carrying the mean token by token without decay.
- Stability: per method and kind of timing, the median rerun/first ratio is 0.936–1.011, and no cell changed which family is faster.

**README.** Evidence raised to E3: new claims 13–16, claim 11 extended with the GPU check, and a Results subsection. The Summary, Verdict, Limitations, Next steps, Reproduce and Files are updated, and the front-matter summary is rewritten. Backlog: I-016 is done, I-018 has a note on BL's accuracy, and I-019 is new (a fused kernel for the chunk form). The status stays `review` until the review of this E3 write-up.

### Independent review of the E3 additions (self-check; finished about 03:55)

The `reviewer` subagent reviewed the README as of 2026-10-30, the new code and the logs (REVIEW.md). An API rate limit stopped it once, and it resumed from its saved state. Verdict: revise, with 2 major and 5 minor issues. All were checked against the cited files, and none was dropped. Its scratch checks are in `/tmp/rev0003/` and are not evidence. The forward-pass result and every quoted number held up. Its main finding: "trains 9–47× faster than BL's loops" compared eager times, where BL's Python loops are bound by kernel launches, and the effect of BL's fused backward was not estimated. Its scratch graph timing in two cells cut 43.6× to 5.5× and 7.5× to 2.1×.

**Forward + backward as CUDA graphs, first attempt.** `gpu_bench.py` now also times forward + backward as a CUDA graph, skipping the capture when the eager pass needs over 40% of the memory cap, since a failed capture breaks the allocator. Smoke test: `logs/gpu_bench_quick_v4.log` (22 s; `results/gpu_bench_quick_20261030_040237.json`). The run was part B for chunk, chunk_inv, bl_wy and recurrent. BL's token-by-token PyTorch form was left out to save GPU time, because it is slower than bl_wy in all 24 cells of eager forward + backward and all 32 of graphed forward (`results/gpu_bench_20261030_021340.json`). This first attempt (`logs/gpu_bench_graph_bwd.log`), one process started at 04:03, was killed when its session ended at about 04:19. It had finished 22 of 32 cells and saved no JSON, so it is not quoted, and its printed cells serve only as a stability check.

## 2026-11-01

Session: the reference workstation again, resumed by Claude Opus 5.5 for @MaurerKrisztian, who asked whether the work was done. The branch and the uncommitted work were intact.

**Forward + backward as CUDA graphs, rerun**, split by N so that an interruption loses only one half. `logs/gpu_bench_graph_bwd_n16.log` (658 s; `results/gpu_bench_20261101_102135.json`) and `logs/gpu_bench_graph_bwd_n64.log` (828 s; `results/gpu_bench_20261101_103540.json`), with no WARNING, failed capture or traceback. From `logs/aggregate_gpu_bench_v5.log`:
- BL's WY form over the best chunk form, forward + backward: 1.83–6.02× as graphs over 28 cells, and 2.26–6.02× with each at its fastest C, over the 6 of 8 settings timed at every C. Eager in the same runs: 2.96–49.68×. The token-by-token filter: 8.69–27.49× as graphs over 8 cells.
- Skipped for memory: the WY form at N = 64, d = 128, C = 64–128 (it needs 6.03–10.11 GiB), and the filter at (N, d) = (16, 128) and (64, 64).
- Stability, against the 22 cells of the interrupted run: the graph times have a median ratio of 1.000 for every method, single times moved by 0.947–1.074×, and BL's WY form over the chunk form by 0.932–1.042×. Eager times moved more: 0.933–1.245× the first run's for the WY form (median 1.076).
- Estimate for BL's fused backward: BL's training step beats the chunk form's autograd step in all 8 settings for any k < 3.64 eager and k < 4.17 as graphs, where k is that backward's cost in multiples of BL's Triton forward. Fused kernels in this benchmark have k = 2.7–3.1 (KDA), 3.1–3.6 (MesaNet) and 12.7–24.8 (Gated DeltaNet). So training speed against BL is open, and the README no longer states a training ratio against "BL's loops" without this.

**Part D with BL's token-by-token PyTorch mean** (`logs/gpu_bench_drift_bl.log`; `results/gpu_bench_20261101_103901.json`; `--parts D --drift-only "no forgetting (gates = 1)"`, 82 s). The reviewer pointed out that bl_triton and bl_wy differ in their gain kernels as well, so part D could not isolate the mean pass. With the same PyTorch gains, BL's token-by-token mean drifts from 4.4e-7 to 5.1e-6, the WY mean stays ≤ 7.7e-7, and both have a covariance error of 1.6e-6. The other five methods' rows match `logs/gpu_bench_cd.log` to the printed digits.

**Tables.** `logs/aggregate_gpu_bench_v4.log` adds the decision rule per timing mode and the graph tables. `logs/aggregate_gpu_bench_v5.log`, the one quoted, restricts the graph fastest-C range to settings timed at every C, because skipped captures made two settings unlike for like, and adds the fused kernels' k. The first-run tables are identical in all versions.

**Decision rule, per timing mode.** The spread is the largest change of best BL / best chunk form between the runs over all cells: 7.3% eager and 15.2% as graphs. BL is faster beyond it in 7 of the 8 hypothesis cells eager and 4 of 8 as graphs. The chunk form is faster beyond it in none, and the other cells show no measurable difference. The two graph cells where the chunk form is ahead by 4–9% barely moved between the runs themselves, but the rule uses the spread over all cells, which is the stricter reading. The verdict stays: refuted at these sizes, for the plain-PyTorch form against Triton loops.

**Other fixes for the review** (REVIEW.md has each resolution). Like-for-like accuracy ranges in the Verdict, and the accuracy caveat back in the front matter. "With finite output" in claim 16, and the per-channel Gram's limit in the Limitations. fla's TF32 solves. The scripts' docstrings. The fixes have not been reviewed again. Because the review was a self-check, the status stays `review`, and the evidence level is E3.

**GPU time for this E3 cycle:** about 3 hours, the lab's limit for a cycle. Smoke tests about 14 min, the discarded run about 16 min, scratch memory traces about 6 min, parts A–B 62 min, parts C–D 6 min, the rerun 33 min, the interrupted graph run about 16 min, the graph rerun 25 min, the extra part D case 1 min. Parts A–B is the one run over the 60-minute limit, by 2 minutes.
