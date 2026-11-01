---
id: "0004"
title: Does the Kalman memory's forgetting train to better recall?
status: review
evidence: E4
track: A
builds_on: ["0001", "0003"]
tags: [kalman-filter, bayesian-layer, mesanet, gated-deltanet, forgetting, process-noise, mqar, selective-copying, recall]
authors: ["@MaurerKrisztian", "Claude Opus 5.5"]
summary: At 2 layers, d_model 64 and 3000 steps (E4), no Kalman variant beat both MesaNet with 30 CG steps and Gated DeltaNet on MQAR, Volatile MQAR or selective copying once the one flagged cell was rerun on fresh seeds. MesaNet leads MQAR at N = 128 with 81.9% (79.8–83.1) against 72.4% (69.9–75.8) for the best Kalman variant, at lower cost per step (8.1 against 13.6 ms). The Kalman line stops.
updated: 2026-11-01
---

# Does the Kalman memory's forgetting train to better recall?

Nov 1, 2026 · @MaurerKrisztian with Claude Opus 5.5

## Summary

This study tested whether the Kalman memory is worth more work, on model quality instead of solver speed. Five Kalman variants (stationary or free process noise, scalar or per-channel gates, and the Bayesian Layer's published MQAR setting of free noise with no gates) trained on MQAR, Volatile MQAR and selective copying, against MesaNet with 30 CG steps and Gated DeltaNet in flash-linear-attention's kernels. All models share one frame (2 layers, d_model 64, 2 heads of dim 32), the same data, 3 seeds each and an equal tuning budget. The hypothesis and the stopping rule were written down before any training. The rule: if no Kalman variant beats both baselines beyond the seed spread on any task, the result is negative and the line stops.

- **The result is negative (E4; `logs/aggregate_final_v2.log`).** On MQAR and Volatile MQAR no Kalman variant beats both baselines at any size. MesaNet is ahead of every Kalman variant beyond the spread on MQAR at N = 64, 128 and 256: 81.9% (79.8–83.1) at N = 128 against 72.4% (69.9–75.8) for the best variant, the stationary one with scalar gates. Gated DeltaNet there is at 63.1% (55.7–68.5).
- **The one flagged cell did not replicate (`logs/replicate_selcopy.log`).** On selective copying, `kalman_free_channel`'s worst seed (99.658%) was above Gated DeltaNet's best (99.652%), by 0.006 points, with runs stopping early at 99.5%. On 3 fresh seeds, set aside for this before training, Gated DeltaNet scores 99.8 (99.7–99.9) and the Kalman variant 99.4 (99.1–99.8). So the win does not count.
- **Stationary noise does as well as free noise (E4).** Free noise is never ahead of stationary noise with the same gates beyond the spread on MQAR or Volatile MQAR, and stationary noise is ahead in places: at N = 256, 28.3% (26.0–30.9) with scalar gates against 17.4% (12.0–25.0).
- **Forgetting is not what limits these tasks.** The two-write formulas (Proposition 1, E2) predict that the Kalman memory overwrites stale pairs better than MesaNet. Trained models did not show it: stale answers are at most 2.4% of the queries on changed keys in any run, MesaNet gets 97.9% (97.7–98.2) of changed keys right, and the best Kalman variant 92.9% (91.7–94.7).
- **The Kalman variants are the less reliable ones.** Three runs stayed near 1/K recall on Volatile MQAR. These are failures to train, not capacity limits, since the best seed of each of those variants reaches 85.3–97.6% at K = 32. They were diagnosed from the logs and not rerun, by the maintainer's decision.
- **It costs more too.** Per training step the Kalman variants through autograd take 13.2–54.9 ms, MesaNet's fused kernels 6.3–8.1 ms and Gated DeltaNet's 12.8–20.9 ms, and the comparison is at equal steps, not equal compute.
- **Backlog I-018 no longer matters here.** cond(𝒦) reached 4.67e4 on trained inputs, but the chunk form's float32 error stayed ≤ 2.4e-5.
- **Two findings about the baselines.** MesaNet with fla's starting gate of about 0.5 stayed at chance on MQAR and Volatile MQAR at this size (one seed per configuration), and starting at half-lives 32–4096 fixes it. With that start, MesaNet reaches 81.9% at 128 pairs with heads of dim 32, so study 0001's collapse to 1.3% there was not a capacity limit (erratum added to 0001).

**Verdict:** Refuted at small scale: at these sizes and this budget, the Kalman memory's forgetting does not train to better recall than MesaNet with 30 CG steps and Gated DeltaNet on MQAR, Volatile MQAR or selective copying, with stationary noise, free noise or the Bayesian Layer's published setting. This is E4 on synthetic tasks and says nothing about language models. It does not show the memory could never win: budgets beyond 3000 steps, larger widths and the Bayesian Layer's own protocol were not tested. The line stops under the pre-registered rule.

**Not independently reviewed.** The maintainer decided to skip the review of this negative result (REVIEW.md), so the status stays `review`. The numbers were checked by the author against the logs and results, not by a referee.

## Claims

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | No Kalman variant beats both MesaNet with 30 CG steps and Gated DeltaNet beyond the seed spread on any task, once the one flagged cell is rerun on fresh seeds. Hypothesis H1 is refuted at these sizes and this budget | E4 | `logs/aggregate_final_v2.log`, `logs/replicate_selcopy.log` |
| 2 | Trained on N ≤ 64, MesaNet is ahead of every Kalman variant beyond the spread at N = 64 (99.8%, 99.7–99.8), N = 128 (81.9%, 79.8–83.1) and N = 256 (37.2%, 34.6–39.0). The best Kalman variant, stationary with scalar gates, gives 99.0%, 72.4% and 28.3% | E4 | `logs/aggregate_final_v2.log` |
| 3 | On Volatile MQAR, MesaNet gives 98.9% (98.7–99.0) at K = 32. The Kalman variants' means are 43.2–96.0%; two seeds of `kalman_free_scalar` reach 99.4–99.5%, but its third gives 44.0%. Stale answers are ≤ 2.4% of queries on changed keys in every run | E4 | `logs/aggregate_final_v2.log` |
| 4 | On selective copying, the rule's only flagged cell (`kalman_free_channel` against Gated DeltaNet, 99.658% against 99.652% at early stopping) does not replicate: 99.4% (99.1–99.8) against 99.8% (99.7–99.9) on fresh seeds. MesaNet is behind there: 86.1% (84.7–87.2), and 90.0% (82.0–94.6) on fresh seeds | E4 | `logs/aggregate_final_v2.log`, `logs/replicate_selcopy.log` |
| 5 | Free noise is not ahead of stationary noise with the same gates, beyond the spread, on MQAR or Volatile MQAR. Stationary is ahead with scalar gates at N = 256 and with per-channel gates at every MQAR size (each model at its own chosen configuration). H2 holds; its prediction that free noise would win on rewrites does not | E4 | `logs/aggregate_final_v2.log` |
| 6 | For one key written twice, the stale-over-new ratio and retained strength are as in Proposition 1, to 5.8e-16 over 50 draws. At equal stale ratio and equal strengths, the stationary Kalman memory retains a pair written once at least as strongly as MesaNet | E2 (the inequality E1) | Method, Proposition 1; `logs/check_layers_v2.log` part D |
| 7 | The layer form (whitened, log-space Grams) matches 0003's extended-precision filter to 2.3e-15 over 20 cases, and its gradients match the filter's to 1.1e-15. It stays finite in float32 with per-channel gates down to 1e-6 (chunk log decay −701.8), with error ≤ 5.9e-7, where 0003's matmul Grams give NaN | E2 | `logs/check_layers_v2.log` parts A–C |
| 8 | Three of the 63 final runs did not train: `volatilemix` seed 0 of `kalman_stat_channel`, `kalman_free_channel` and `kalman_free_identity`, with recall about 1/K at K = 16, 32 and 64. The other seeds of those variants reach 78.0–97.6% at K = 32, except one at 29.7%. Not rerun | E4 (diagnosis from logs) | `logs/final_volatilemix.log`, `logs/aggregate_final_v2.log` |
| 9 | On trained inputs, cond(𝒦) reaches 4.67e4 with free noise and 1.39e3 with stationary noise, against ≤ 58.3 with scalar gates and stationary noise. The chunk form's float32 error is ≤ 2.4e-5, and at most 50× the float32 filter's (1.8e-5 against 5.6e-7), the one case above H3's 10× line. No Cholesky failed and no gradient was non-finite | E4 | `logs/aggregate_final_v2.log` |
| 10 | On trained inputs, fla's MesaNet with 30 CG steps differs from a dense float64 solve by 6.6e-3 on MQAR, 2.4e-2 on Volatile MQAR and 3.6e-1 on selective copying (cond(H + Λ) up to 448) | E4 | `logs/aggregate_final_v2.log` |
| 11 | MesaNet from fla's starting gate of about 0.5 stays at chance on MQAR-64, MQAR-128 and Volatile MQAR at K = 32, in same-size training (0.16–0.21%), and starting at half-lives 32–4096 trains | Below E4: one seed per configuration | `logs/scout_baselines.log`, `logs/scout2.log`, `logs/aggregate_pilot2.log` |

## Problem

The lab's three studies so far are about solver speed in one family of least-squares memories. Study 0001 found that exactness added nothing over 5 CG steps on its recall task (its claim 7, E4 with 2 seeds). Study 0002's exact solver is slower than 30-step CG (its claim 9, E3). Study 0003's chunk form is mostly not faster than the Bayesian Layer's Triton loops (its claim 13, E3), but it trains through autograd without a custom kernel (its claim 14, E3), which makes a quality test cheap. The lab's goal is a model that beats a tuned baseline at equal compute. So this study asks whether the Kalman line deserves more work on the only ground left: model quality.

Two questions:

1. Does the Kalman memory's forgetting train to better recall than MesaNet and Gated DeltaNet (GDN)?
2. Does study 0003's stationary process noise, which keeps the prior at MesaNet's fixed ridge with one fewer learned quantity per token, do as well as the Bayesian Layer's (BL) free noise?

## Method

### Claims used

From study 0003, at E2: the chunk form computes the Kalman memory exactly (claims 1–2), its autograd gradients match the filter's (claim 3), its matmul Grams overflow in float32 once a chunk's log decay falls below about −88 (claim 10), and the one-pair formulas of its Proposition 1(d) hold (claim 4). At E3: it trains through autograd without a custom kernel (claim 14). From study 0001, at E4 with 2 seeds: the MQAR recall of Mesa and DeltaNet at 64 pairs (claim 7), as a sanity check.

### The layer

`kalman_layer.py` is study 0003's chunk form (its Propositions 2–3) with two changes that leave the function unchanged. `check_layers.py` verifies them against 0003's token-by-token filter in extended precision.

- **Whitened observations.** v_t = W_tk_t + noise of precision β_t is the same observation as √β_t v_t = W_t(√β_t k_t) + unit noise. So K and V are scaled by √β, and the C×C matrix factored per chunk is 𝒦̃ = I + (noiseless part). Its eigenvalues are at least 1, a token that writes nothing needs no 1/β, and the bound of 0003's Proposition 4(b) becomes cond(𝒦̃) ≤ 1 + Σ_i β_i k_iᵀP k_i for the entry covariance P.
- **Safe Grams.** Relative decays exp(log 𝒜_t − log 𝒜_i) are formed in log space before anything is multiplied, for scalar and per-channel gates and both kinds of noise, so no learned gate can overflow them (0003, claim 10). Per-channel gates then need a C×C×d tensor per chunk. The chunk size does not change the function, so each model uses its fastest of 16, 32 and 64 (`logs/bench_chunk.log`): C = 32 with per-channel gates, C = 64 otherwise.

### The models

All models share study 0001's MQAR frame, so only the memory differs: embedding, 2 blocks of RMSNorm + mixer + RMSNorm + 4× MLP, RMSNorm, a linear head; d_model 64, 2 heads of dim 32. Each mixer makes q, k, v with one linear projection and a length-four causal convolution; q and k pass through SiLU and are l2-normalized; each head's output passes through RMSNorm; a linear layer mixes the heads. Everything runs in float32 with TF32 off, and the parameter counts agree within 4% at vocabulary 1024 (232,072–240,780) and within 8.4% at vocabulary 16 (103,048–111,756; `logs/aggregate_final_v2.log`).

| Model | Forgetting | Gate | Write strength | Other |
|---|---|---|---|---|
| `gdn` | Gated DeltaNet, fla's fused kernels | log γ_t = −exp(A)·softplus(a(x) + b_dt), a without bias, A uniform on 0–16, dt log-uniform on 0.001–0.1 | β = sigmoid(b(x)), without bias | Convolution and SiLU on v |
| `mesanet` | MesaNet, 30 CG steps, fla's fused kernels | log γ_t = logsigmoid(a(x)), starting at half-lives 32–4096 (from fla's start of about 0.5 it stayed at chance; see Results) | β = sigmoid(b(x)) | λ = softplus + 0.25 per channel, starts at 1. No convolution or SiLU on v |
| `kalman_stat_scalar`, `kalman_stat_channel` | Stationary: Ψ_t = (I − D_t²)Λ⁻¹, P_init = Λ⁻¹ | D_t = γ_tI or diag(α_t); logsigmoid(a(x)), starting at half-lives 32–4096 (BL, Appendix F.2) | r² = softplus(b(x)) + 0.01, β = 1/r², starts at r² = 1 (BL) | λ as MesaNet's |
| `kalman_free_scalar`, `kalman_free_channel` | Free: Ψ_t = ψ_tI, P_init = σ₀²I (BL) | as above | as above | ψ_t = softplus(c(x)) + 1e-4, starts at 0.1, 0.25 learning-rate multiplier on c's weights (BL, Appendix F.1); σ₀² learned per head, starts at 1 |
| `kalman_free_identity` | BL's published MQAR setting: D_t = I, free noise | none | as above | as above |

The gates, write strengths and ridge of `gdn` and `mesanet` follow fla's layers (`fla/layers/gated_deltanet.py`, `fla/layers/mesa_net.py` on `main`, read through a subagent on 2026-11-01). The Kalman settings follow BL's Appendices C.1 and F; BL states that its variances have a positive floor but not its value, so the floors above are this study's. Sigmoid write strengths are computed through logsigmoid, so √β stays accurate at small β.

### What the math predicts: one key written twice

Take a unit key k written twice, (k, v₁) with strength β₁, then (k, v₂) with strength β₂ τ tokens later, and read it at k after s more tokens. All other tokens write nothing. With a scalar gate γ and a = γ^τ, each memory answers w₁v₁ + w₂v₂.

**Proposition 1.** The readouts are as follows (checked against each recurrence in float64 over 50 random draws to 5.8e-16; `logs/check_layers_v2.log` part D).

| Memory | Stale over new, w₁/w₂ | Strength of a pair written once, at age τ |
|---|---|---|
| MesaNet | aβ₁/β₂ | aβ/(aβ + λ) |
| Kalman, stationary | (aβ₁/β₂) · λ/(λ + (1 − a²)β₁) | aβ/(β + λ) |
| Kalman, free noise, identity dynamics (Ψ accumulated between the writes, prior σ₀²) | σ₀²β₁ / (β₂(σ₀²Ψβ₁ + σ₀² + Ψ)) | σ₀²β/(σ₀²β + 1) |
| Gated DeltaNet | a(1 − β₂)β₁/β₂ | aβ |

*Proof sketch.* MesaNet: H = (a₁β₁ + a₂β₂)kkᵀ and G = (a₁β₁v₁ + a₂β₂v₂)kᵀ along k, with a₁ = γ^{τ+s}, a₂ = γ^s, so o = G(H + λ)⁻¹k. Kalman: the state along k is scalar, so Gaussian conditioning on two observations gives the weights. With the stationary AR(1) prior, Cov(y₁, y₂) = a/λ and Var(y_i) = 1/λ + 1/β_i. Writing D for the determinant, w₁ ∝ a/(λβ₂D) and w₂ ∝ ((1 − a²)/λ² + 1/(λβ₁))/D. With identity dynamics, Var(x₂) = σ₀² + Ψ and Cov = σ₀². Gated DeltaNet: expand the recurrence twice. ∎

Three consequences, for equal strengths β₁ = β₂ = β and b = β/λ:

- **The Kalman memory overwrites better than MesaNet at the same retention.** MesaNet's ratio is a_M. The stationary one solves ρ = a/(1 + (1 − a²)b). At equal ρ, the retained strengths satisfy R_K ≥ R_M: R_K = ab/(b + 1) ≥ ρb/(ρb + 1) = R_M is equivalent to a − ρ ≥ ρb(1 − a), and a − ρ = ρb(1 − a)(1 + a). So 0003's Proposition 1(d), read alone, is misleading: MesaNet keeps an old pair at higher strength for the same gate product, but it needs a much smaller gate product to overwrite as well. Where this margin matters depends on b.
- **Free noise with identity dynamics decouples the two.** Its ratio goes to zero as Ψ grows while a pair written once keeps full strength. Stationary noise can raise the gain only by fading the mean. So on a task with rewrites, free noise should do better than stationary (H2 in NOTEBOOK.md).
- **Gated DeltaNet overwrites exactly at β₂ = 1 and keeps full strength at a = 1,** for one key or orthogonal keys. Its cost is the overlap between keys, which this table leaves out: it writes along k_t, whatever is stored there (BL, Section 2.3; 0003, Proposition 5). With K keys in a head of dim d near K, this is what the Kalman memory's covariance avoids.

So the math predicts a Kalman advantage on rewrites at K near d, which is what Volatile MQAR at K = 32 and d = 32 tests. These formulas ignore everything training does. The prediction failed (Results, What the trained memories do).

### The tasks

- **MQAR** (study 0001's generator, `tasks.mqar_batch`, identical to it at N = 32, 64, 128; `logs/check_layers_v2.log` part E): N key–value pairs, then N queries; T = 3N, vocabulary 1024. `mqarmix` trains on a mix of N ∈ {8, 16, 32, 64} and is evaluated at N = 32, 64, 128 and 256, so N = 128 and 256 are larger than trained on, as in BL's paper (its Appendix F.1).
- **Volatile MQAR** (Voltic, Hejabi et al. 2026, Section 4.2 and Appendix E.1, at this lab's sizes): three rounds that each visit all K keys in a fresh order, as (key, value) token pairs. Before rounds 2 and 3, each key's value changes with probability 0.30. Then K queries in a fresh order, each followed by a random filler token. T = 8K. The target is the final value, scored at the query's own position. Voltic does not state the order inside a pair or which position in a pair holds the query, so key-then-value and query-then-filler are this study's reading. `volatilemix` trains on K ∈ {8, 16, 32} and is evaluated at K = 16, 32 and 64, at vocabulary 1024 instead of Voltic's 8192. The planned control with no changes was not run, for budget.
- **Selective copying** (the MAD suite, Poli et al. 2024, `mad/data/instances.py`): 16 data tokens, with replacement, from 14 symbols, 223 blanks inserted before them at random, a copy token, 16 blank answer slots. T = 256, vocabulary 16. No teacher forcing. The generators match brute-force replays over 2,000 sequences (`logs/check_layers_v2.log` part E).

## Results

### Protocol

Each model first got the same pilot: learning rates 1e-3, 2e-3 and 4e-3 at its default configuration, plus one alternative chosen before any results, at 2e-3. The alternatives were MesaNet's sigmoid write strength for the Kalman variants (BL's softplus is the default), fla's start of about 0.5 for MesaNet, and fla's output gate for Gated DeltaNet. The pilot used one seed that is not reused, half-length runs (1500 steps) and three tasks, 84 runs in all, chosen by mean recall over the sizes evaluated (`logs/aggregate_pilot2.log`, `results/best_config.json`). The chosen configuration then ran 3000 steps with seeds 0, 1 and 2, on the same training stream per seed and the same evaluation batches, with TF32 off. Learning rate warms up for 10% of the steps and then follows a cosine to zero, with AdamW (0.9, 0.98), weight decay 0.1 and gradient clipping at 1, as in study 0001. Selective copying stops early at 99.5% evaluation recall; the other two tasks never stop early, because their larger sizes keep improving. Every run is reported below.

| Task | `gdn` | `mesanet` | `kalman_stat_scalar` | `kalman_stat_channel` | `kalman_free_scalar` | `kalman_free_channel` | `kalman_free_identity` |
|---|---|---|---|---|---|---|---|
| `mqarmix` | 4e-3 | 4e-3 | 2e-3, sigmoid | 4e-3 | 4e-3 | 2e-3, sigmoid | 2e-3, sigmoid |
| `volatilemix` | 4e-3 | 4e-3 | 2e-3, sigmoid | 4e-3 | 4e-3 | 4e-3 | 4e-3 |
| `selcopy` | 4e-3 | 4e-3 | 2e-3, sigmoid | 4e-3 | 4e-3 | 4e-3 | 2e-3, sigmoid |

Fifteen of the 21 choices are the top of the learning-rate grid and the other six are the alternative, whose learning rate was fixed. Parameters: 232,072–240,780 on MQAR and Volatile MQAR, 103,048–111,756 on selective copying, where the per-channel gates are a larger share (`logs/aggregate_final_v2.log`).

**The two training changes made before the pilot were decided from the baselines alone** (NOTEBOOK.md). First, a pilot at 1000 steps of same-size MQAR-64 stayed on the loss plateau, so it was discarded (`logs/pilot_mqar64.log`), and mixed-size training was added, as BL and Voltic train: `mqarmix` draws N ∈ {8, 16, 32, 64} per step and `volatilemix` K ∈ {8, 16, 32}, with equal queries per step. Second, MesaNet stayed at chance from fla's starting gate, so it starts at half-lives 32–4096, as the Kalman cores do.

### Recall (E4)

Recall in %, mean (min–max) over 3 seeds. Models train on N ≤ 64 or K ≤ 32, so N = 128, N = 256 and K = 64 are larger than trained on. Source: `logs/aggregate_final_v2.log`.

| Task and size | `gdn` | `mesanet` | `kalman_stat_scalar` | `kalman_stat_channel` | `kalman_free_scalar` | `kalman_free_channel` | `kalman_free_identity` |
|---|---|---|---|---|---|---|---|
| MQAR, N = 32 | 99.8 (99.7–99.9) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | 100.0 (99.9–100.0) | 100.0 (99.9–100.0) | 95.2 (86.3–99.8) | 100.0 (100.0–100.0) |
| MQAR, N = 64 | 93.9 (90.7–96.4) | **99.8 (99.7–99.8)** | 99.0 (98.6–99.4) | 92.8 (83.9–98.8) | 93.6 (88.3–99.4) | 67.0 (48.1–77.4) | 96.1 (91.9–99.1) |
| MQAR, N = 128 | 63.1 (55.7–68.5) | **81.9 (79.8–83.1)** | 72.4 (69.9–75.8) | 51.4 (33.0–69.0) | 50.3 (37.4–71.2) | 24.9 (16.7–29.3) | 59.0 (43.3–70.7) |
| MQAR, N = 256 | 27.0 (23.4–29.8) | **37.2 (34.6–39.0)** | 28.3 (26.0–30.9) | 18.7 (11.5–27.1) | 17.4 (12.0–25.0) | 8.4 (6.4–9.7) | 23.0 (14.9–27.9) |
| Volatile, K = 16 | 99.3 (99.0–99.7) | 99.7 (99.5–99.8) | 99.2 (98.8–99.7) | 68.4 (6.1–99.5) | 91.1 (73.6–99.9) | 68.0 (6.5–99.7) | 53.8 (6.0–99.8) |
| Volatile, K = 32 | 95.7 (94.2–96.7) | 98.9 (98.7–99.0) | 96.0 (95.2–97.2) | 65.5 (2.9–97.6) | 80.9 (44.0–99.5) | 55.3 (2.7–85.3) | 43.2 (2.5–97.3) |
| Volatile, K = 64 | 62.7 (58.9–65.2) | 69.2 (59.3–87.4) | 62.1 (51.7–67.8) | 32.1 (1.7–48.8) | 55.0 (19.3–73.6) | 22.6 (1.6–43.1) | 20.1 (1.2–49.5) |
| Selective copying | 92.3 (77.6–99.7) | 86.1 (84.7–87.2) | 90.9 (86.8–96.3) | 99.3 (98.6–99.6) | 95.5 (88.7–98.9) | 99.8 (99.7–99.9) | 88.8 (84.2–93.1) |

Bold marks a model ahead of every Kalman variant beyond the spread. No Kalman cell is bold.

### Against the decision rule

- **MQAR and Volatile MQAR:** no Kalman variant's lowest seed is above both baselines' highest, at any size. At K = 32, `kalman_free_scalar`'s two seeds that trained reach 99.4% and 99.5%, above MesaNet's best, 99.0%. Its third seed gives 44.0% and was still improving when the learning rate reached zero. At K = 64 the same variant is at most 73.6% against MesaNet's 87.4%. That is a hint at best, inside the spread by the rule.
- **Selective copying:** the rule flags `kalman_free_channel` over both baselines, 99.658–99.860% against Gated DeltaNet's 99.554%, 77.612% and 99.652%. The margin over Gated DeltaNet's best seed is 0.006 points. Both stopped at the first evaluation above the 99.5% target, so these numbers are stopping points, not final recalls. As written before training, any apparent win had to hold on 3 fresh seeds. It does not (`logs/replicate_selcopy.log`):

| Seeds 10, 11, 12 | Recall % | Steps run |
|---|---|---|
| `kalman_free_channel` | 99.316, 99.799, 99.109 | 3000, 1500, 3000 |
| `gdn` | 99.695, 99.817, 99.884 | 1750, 2250, 2000 |
| `mesanet` | 93.433, 81.958, 94.580 | 3000, 3000, 3000 |

Over all six seeds, both reach about 99% or more except one Gated DeltaNet seed (77.6%). So on this task Gated DeltaNet and the per-channel Kalman memory show no measurable difference, and Gated DeltaNet costs 2.7× less per step (about 20 against 54 ms). Per-channel gates help the Kalman memory here: 99.3 and 99.8 against 90.9 and 95.5 with scalar gates, at each model's chosen configuration. That may be what the task rewards rather than the covariance; no per-channel delta rule (KDA) was run to separate them.

### Stationary against free noise

Each pair compares a model's own chosen configuration, and the write-strength parametrization differs in some pairs (table above).

| Task and size | Beyond the spread |
|---|---|
| MQAR, N = 32, 64, 128 | stationary per-channel over free per-channel |
| MQAR, N = 256 | stationary scalar over free scalar; stationary per-channel over free per-channel |
| Volatile, K = 16, 32 | ranges overlap |
| Volatile, K = 64 | stationary scalar over free identity |
| Selective copying | free per-channel over stationary per-channel at early stopping (99.66–99.86 against 98.64–99.59), which the fresh seeds' 99.11–99.80 do not support; stationary per-channel over free identity |

So stationary noise, with one fewer learned quantity per token, does as well as free noise in this frame. BL's published setting, free noise with no gates, is not ahead of `kalman_stat_scalar` anywhere beyond the spread, and is behind at K = 64.

### What the trained memories do

**Forgetting (Volatile MQAR, K = 32, `logs/aggregate_final_v2.log`).** Proposition 1 predicts a Kalman advantage on rewrites. Stale answers are rare everywhere, so the errors on changed keys are mostly not the old value. The gap between changed and unchanged keys is smallest for MesaNet.

| Model | Unchanged keys | Changed keys | Stale answers, of changed keys |
|---|---|---|---|
| `gdn` | 99.5 (99.0–99.7) | 92.1 (89.7–93.8) | 1.3 (0.9–1.6) |
| `mesanet` | 99.9 (99.8–99.9) | 97.9 (97.7–98.2) | 0.8 (0.5–1.2) |
| `kalman_stat_scalar` | 99.3 (98.7–99.7) | 92.9 (91.7–94.7) | 1.3 (0.7–2.2) |
| `kalman_stat_channel` | 67.2 (3.1–99.5) | 63.8 (2.6–95.7) | 0.6 (0.2–1.2) |
| `kalman_free_scalar` | 85.4 (56.4–99.9) | 76.7 (32.0–99.1) | 1.1 (0.4–2.0) |
| `kalman_free_channel` | 61.0 (3.2–93.3) | 49.9 (2.2–77.7) | 1.3 (0.3–2.4) |
| `kalman_free_identity` | 47.2 (3.1–99.3) | 39.4 (2.0–95.4) | 0.7 (0.3–1.3) |

Learned gates stay near 1: at K = 32 the median gate is 0.978–0.997 in every gated model whose gates were recorded (all but Gated DeltaNet), averaged over layers and seeds, and on MQAR 0.984–0.998 (`results/final/`). The formulas compare memories at equal strength and gate product, for one key, with no interference and no learning, while training picks each model's own gates and keys.

**Runs that did not train (`logs/final_volatilemix.log`).** On `volatilemix` seed 0, `kalman_stat_channel`, `kalman_free_channel` and `kalman_free_identity` end at 2.5–2.9% at K = 32, 6.0–6.5% at K = 16 and 1.2–1.7% at K = 64, which is about 1/K. The loss levels off at 4.6–4.8 from about step 2000. So these are consistent with a readout that does not use the query, a poor local solution. Other seeds of the same variants reach 78.0–97.6% at K = 32, except `kalman_free_identity`'s seed 1 at 29.7%, whose loss was still falling (with `kalman_free_scalar`'s seed 0 at 44.0%, likewise) when the learning rate reached zero. The same seed trains Gated DeltaNet, MesaNet and the stationary scalar variant normally. All three failed runs used a learning rate of 4e-3, the top of the grid, so a lower one might have helped. These runs were not repeated: after the rule gave a negative result, the maintainer decided that no more GPU time goes to this line.

**Conditioning and float32 (backlog I-018; `logs/aggregate_final_v2.log`).** The largest cond(𝒦), at the end of training, over layers and seeds: 18–58 for stationary scalar, 204–1.39e3 for stationary per-channel, 536–4.67e4 for free scalar, 14–5.12e3 for free per-channel, 48–6.09e3 for free identity. On the trained inputs, relative to the largest output, the chunk form's float32 error is at most 2.4e-5, and the float32 token-by-token filter's at most 2.2e-5. The ratio between them is at most 6.25, except on `volatilemix` with `kalman_free_identity`, where it reaches 50.36 (1.8e-5 against 5.6e-7). That meets the 10× line written before the runs, but at an absolute size far below anything that shows in recall. No Cholesky failed and no gradient was non-finite in any of the 63 + 9 runs. The chunk form in float64 stays within 4.3e-14 of the float64 filter. So the ill-conditioned regime of 0003's claim 9 does occur in training, with free noise, and does no visible harm.

**MesaNet's CG on trained inputs.** Fla's kernel against a dense float64 solve per token: 6.6e-3 on MQAR, 2.4e-2 on Volatile MQAR and 3.6e-1 on selective copying, relative to the largest output (cond(H + Λ) up to 21.6, 87.8 and 448). So 30 CG steps are far from exact on selective copying, where MesaNet is also weakest. An exact MesaNet was not tried. On the other tasks, 0001's finding that the model tolerates inexact solves holds.

**Cost (`logs/final_volatilemix.log`, `logs/final_mqarmix.log`, `logs/final_selcopy.log`, `logs/replicate_selcopy.log`).** Per training step, in float32 on the RTX 5060 Ti, over all 72 runs: MesaNet 6.3–8.1 ms, Gated DeltaNet 12.8–20.9 ms, the Kalman variants through autograd 13.2–54.9 ms, with per-channel gates the slowest. These are fused kernels against plain PyTorch, so the ratio says what it costs today, not what a fused kernel would cost.

### Against the published numbers

BL's Table 4 at width 64 puts its layer above MesaNet and MesaNet above Gated DeltaNet, with bands that overlap. Here the order is MesaNet first, and BL's setting (`kalman_free_identity`) is not ahead of Gated DeltaNet beyond the spread.

| (K, L) in BL; N here | BL's table: BL, MesaNet, GDN | Here: `kalman_free_identity`, `mesanet`, `gdn` |
|---|---|---|
| 64 pairs | 97.19 ± 0.57, 89.97 ± 12.90, 76.85 ± 23.94 | 96.1 (91.9–99.1), 99.8 (99.7–99.8), 93.9 (90.7–96.4) |
| 128 pairs | 51.07 ± 2.81, 45.29 ± 22.45, 33.68 ± 20.53 | 59.0 (43.3–70.7), 81.9 (79.8–83.1), 63.1 (55.7–68.5) |
| 256 pairs | 12.14 ± 1.11, 11.73 ± 6.73, 8.93 ± 6.98 | 23.0 (14.9–27.9), 37.2 (34.6–39.0), 27.0 (23.4–29.8) |

The protocols differ, so this does not contradict BL: its vocabulary is 8192 against 1024 here, it has no MLP and ties embeddings, its sequences have L = 4K against T = 3N here, its budget is up to about 22,500 steps at batch 256 (32 epochs of 180,000 examples, my arithmetic) against 3000 here, and its learning rates come from a pilot on {3e-4, 6e-4, 1e-3}. BL's sections read here do not state MesaNet's starting gate. If it used fla's start of about 0.5, the large spread of its MesaNet row could come partly from that; this was not tested.

## Novelty check

Searched on 2026-11-01 by a `literature-scout` agent: arXiv API, web search, GitHub READMEs of flash-linear-attention, MAD-Lab and Mamba. Semantic Scholar answered HTTP 429 at every try and OpenReview a browser check, so neither was searched. NOTEBOOK.md has the query log. I read the Bayesian Layer's PDF myself. A `web-fetch` agent read Voltic's and MAD's tasks and fla's layers verbatim. The other rows are as far as the scout read them.

| Work | What it does | Difference |
|---|---|---|
| Bayesian Layer, Dowling et al. 2026 [1] (read: all of v5's Sections 1–4, Appendices C, E.3–E.4, F) | The same layer. MQAR at width 64–256, trained on L ≤ 256, tested up to L = 1024: Table 4, against MesaNet, GDN and others | Free noise only, and identity dynamics on MQAR, where stationary noise would vanish. No rewrites. Parameter counts not equalized |
| Voltic, Hejabi et al. 2026 [2] (read through an agent: Section 4.2, Appendix E) | A Kalman-style memory with free input-dependent noise. Volatile MQAR: Voltic-Q 99.48, GDN-2 70.25, KDA 48.76 | Diagonal or short-block covariance; no MesaNet, no dense filter, no stationary arm |
| Kalman Linear Attention, Shaj et al. 2026 [3] (the scout: pp. 3–13 and HTML appendices) | A diagonal filter with OU-tied noise and an ablation of it, on Selective Copy | Time-invariant noise, one task, figure only |
| KATA, Ghriss 2026 [4] (the scout: pp. 8–13, partial HTML) | Each key written twice: DeltaNet and GDN recall against K | No Kalman memory, no MesaNet; protocol not found |
| Soft Reset, Galashov et al. 2024 [5] (the scout: pp. 4–5, 20) | The same math on network weights: tied against fixed variance | SGD on a network, not a memory; learned variance untested |
| KRLS-T, Van Vaerenbergh et al. 2012 [6] (the scout: pp. 1315–1325) | Back-to-prior against uncertainty injection after a channel switch | Kernel regression with a hand-set forgetting factor |
| Palimpsa, Bonnet et al. 2026 [7] (the scout: pp. 3–7, HTML) | Precision relaxing to a prior; MQAR against GDN | Diagonal, precision space; no free-noise arm |
| Titsias et al. 2024 [8] (the scout: HTML; also opened in 0003) | Variance-preserving filter, learned γ | A classifier head; no free arm |
| Kalman Delta Networks, Bui et al. 2026 [9] (the scout: HTML, Table 5) | Learned against constant noise: "neither constraint consistently improves" | Approximate covariance; not tied to the gates |
| MAD, Poli et al. 2024 [10]; MesaNet, von Oswald et al. [11] (MAD read through an agent; MesaNet by the scout) | The selective-copying task; MesaNet reports 99.2 on it | The task, not a Kalman comparison |

**Conclusion.** The layer is published (BL). Volatile MQAR is published (Voltic), without MesaNet or a dense filter. The stationary prior is classical (0003, Novelty check). No prior publication was found, as of 2026-11-01, of a direct test of stationary against free process noise in a learned Kalman memory, or of MesaNet on a task with rewrites. A web search cannot rule out unpublished or concurrent work.

## Limitations

- **One small frame and a short budget.** d_model 64, 2 heads of dim 32, 2 layers, vocabulary 1024 (16 on selective copying), 3000 steps. Larger widths, BL's vocabulary of 8192 and its budget of up to about 22,500 steps were not tested. At equal steps the Kalman variants still train more slowly on some seeds, so a longer budget could narrow the gaps. Models were compared at equal steps and data, not equal compute, which favors the slower Kalman variants.
- **Tuning.** Fifteen of the 21 pilot choices sit at the top of the learning-rate grid, for baselines and Kalman variants alike, so higher rates were not explored. The pilot ran half-length, which favors high rates. Three failures to train came at that top rate, and they were not repeated at a lower one. Each model's alternative was a single configuration, and floors such as r² ≥ 0.01 and ψ ≥ 1e-4 are this study's, since BL does not state its values.
- **Same-size training** at N = 64 and K = 32 ran for the baselines only (`logs/scout2.log`). There, Gated DeltaNet does somewhat worse at N = 64 (60.3%) and better at K = 32 (90.1%) than with mixed sizes, on one seed. Study 0001's N = 128 same-size training was not repeated.
- **The baselines.** Gated DeltaNet runs without fla's default output gate, and that alternative lost in the pilot on all three tasks. fla's MesaNet passes q and k through float16 in its kernel. No per-channel delta rule (KDA) or GDN-2 was run, so the per-channel gain on selective copying cannot be attributed to the covariance. The starting gates differ by design: Gated DeltaNet keeps fla's, and MesaNet starts like the Kalman cores, which is a change from fla's made because fla's did not train here.
- **Stationary against free** compares each model's own chosen configuration, and in some pairs those differ in the write-strength parametrization, so H2 is a comparison of tuned models, not of the noise alone.
- **Selective copying** stops early at 99.5%, so above that level recall is a stopping point and does not rank models. On this task the parameter counts differ by up to 8%.
- **Volatile MQAR** is at vocabulary 1024 with this study's reading of the order inside a pair and the query's position, which Voltic does not state, and only at K ≤ 64. Its numbers are not comparable with Voltic's budget of 10,000 updates at batch 16.
- **MQAR at N = 128** is evaluated larger than trained on, unlike 0001's training at that size.
- **Evidence.** E4, synthetic tasks. It says nothing about language models. 3 seeds per configuration, and one replication.

## Next steps

The Kalman line stops. By the pre-registered rule and the maintainer's decision, no further GPU time goes to it: backlog I-018 and I-019 are dropped, and I-017 is done. What would reopen it is outside evidence, such as a result at equal gate starts and budgets on a language model, not more runs here.

Follow-ups that do not depend on the Kalman memory:

- **Does an exact solve help MesaNet on selective copying?** 30 CG steps leave a 3.6e-1 error on trained inputs there, the task where it is weakest (claim 10). Backlog I-020.
- **Use MesaNet's starting gate near 1 in any later comparison** (claim 11), and mixed-size training for recall tasks. A note is added to backlog I-009, whose 128-pair question this partly answers.

## References

Only works opened in this session, to the extent given in the Novelty check.

1. M. Dowling, H. Jeon, C. Savin, I. M. Park, "Memory by Design: Bayesian Layers for Sequence Modeling", arXiv 2605.31163, v5 (2026-10-14). https://arxiv.org/abs/2605.31163
2. P. Hejabi, M. Dehghani, P. Piray, "Voltic: Distinguishing Volatility from Stochasticity in Recurrent Memory", arXiv 2610.05700. https://arxiv.org/abs/2610.05700
3. V. Shaj et al., "Kalman Linear Attention", ICML 2026, arXiv 2602.10743. https://arxiv.org/abs/2602.10743
4. Ghriss, KATA, arXiv 2607.17419 (2026). https://arxiv.org/abs/2607.17419
5. A. Galashov et al., "Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset", arXiv 2411.04034 (2024). https://arxiv.org/abs/2411.04034
6. S. Van Vaerenbergh et al., KRLS-T, IEEE TNNLS 2012. Cited by the scout's reading.
7. D. Bonnet et al., "Learning to remember, learn, and forget in attention-based models" (Palimpsa), arXiv 2602.09075, ICML 2026. https://arxiv.org/abs/2602.09075
8. M. K. Titsias et al., "Kalman Filter for Online Classification of Non-Stationary Data", ICLR 2024, arXiv 2306.08448. https://arxiv.org/abs/2306.08448
9. N. Bui, T. Huang, R. Ying, "Kalman Delta Networks", arXiv 2609.07816. https://arxiv.org/abs/2609.07816
10. M. Poli et al., "Mechanistic Design and Scaling of Hybrid Architectures", arXiv 2403.17844; code https://github.com/athms/mad-lab. https://arxiv.org/abs/2403.17844
11. J. von Oswald et al., "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training", arXiv 2506.05233. https://arxiv.org/abs/2506.05233
12. flash-linear-attention, https://github.com/fla-org/flash-linear-attention (fla-core 0.6.0 installed; layer sources read from `main`).

## Reproduce

Run from this folder with the repository's virtual environment (`scripts/setup_env.sh`), with flash-linear-attention installed, on an NVIDIA GPU. Never `tee` into `logs/` or write into recorded `results/`: `SUFFIX=_rerun` makes the grids write new folders and logs. `check_layers.py` parts A–E need only a CPU, and part F needs the GPU. The seeds are fixed. Recalls match only up to GPU nondeterminism and `TRITON_CACHE_AUTOTUNING`, because fla's kernels are autotuned, and the recorded grids ran under a checkout before `tasks.py` lost four never-run task entries and `run_grid.sh` gained `SUFFIX`.

| Command | Compare with | What it checks | Time on the reference machine |
|---|---|---|---|
| `../../.venv/bin/python -u check_layers.py 2>&1 \| tee /tmp/check_layers.log` | `logs/check_layers_v2.log` | Claims 6 and 7, fla's kernels, the generators | ~2 min |
| `../../.venv/bin/python -u bench_chunk.py 2>&1 \| tee /tmp/bench_chunk.log` | `logs/bench_chunk.log` | The chunk sizes | ~2 min |
| `SUFFIX=_rerun ./run_grid.sh pilot` | `logs/pilot2_mqarmix.log`, `logs/pilot2_volatilemix.log`, `logs/pilot2_selcopy.log`, `logs/aggregate_pilot2.log` | The pilot, 84 runs | ~55 min |
| `SUFFIX=_rerun ./run_grid.sh final volatilemix mqarmix selcopy` | `logs/final_volatilemix.log`, `logs/final_mqarmix.log`, `logs/final_selcopy.log` | The final grid, 63 runs | ~77 min |
| `SUFFIX=_rerun ./run_grid.sh replicate` | `logs/replicate_selcopy.log` | The replication, 9 runs | ~11 min |
| `../../.venv/bin/python -u aggregate.py --final results/final --replicate results/replicate` | `logs/aggregate_final_v2.log` | Every table and the decision rule | seconds |

Total GPU time for this study, all recorded and discarded runs included: about 2.9 hours (NOTEBOOK.md).

## Files

| File | Contents |
|---|---|
| `kalman_layer.py` | The chunk form with whitened observations and log-space Grams, and the token-by-token reference |
| `models.py` | The shared frame and the Gated DeltaNet, MesaNet and Kalman cores, with each model's alternative configuration |
| `tasks.py` | MQAR, Volatile MQAR and selective copying, and mixed-size training |
| `train.py`, `diagnostics.py` | Training with one JSON per run; cond(𝒦), gates, and float32 against float64 on trained inputs |
| `aggregate.py`, `run_grid.sh`, `bench_chunk.py` | The tables and the decision rule; the grids; the chunk-size timing |
| `check_layers.py` | The checks, parts A–F |
| `logs/check_layers_v2.log` | The recorded check of the frozen code. `logs/check_layers.log` is an earlier run, before MesaNet's default start and the chunk adjustment changed |
| `logs/final_volatilemix.log`, `logs/final_mqarmix.log`, `logs/final_selcopy.log`, `results/final/` | The final grid, as quoted |
| `logs/replicate_selcopy.log`, `results/replicate/` | The replication |
| `logs/aggregate_final_v2.log` | Every table quoted from the final grid and the replication. `logs/aggregate_final.log` is the same without the replication |
| `logs/pilot2_mqarmix.log`, `logs/pilot2_volatilemix.log`, `logs/pilot2_selcopy.log`, `logs/aggregate_pilot2.log`, `results/pilot2/`, `results/best_config.json` | The pilot and its choices |
| `logs/scout_baselines.log`, `logs/scout2.log`, `logs/scout2_short.log` and their results | Baseline-only runs that set the budget, the batch sizes and MesaNet's start. In `logs/scout_baselines.log` the batch is 128 at N = 64, and MesaNet starts from fla's gate |
| `logs/pilot_mqar64.log`, `logs/pilot_volatile32.log`, `logs/pilot_selcopy.log`, `results/pilot/` | The first pilot at 1000 steps, stopped and discarded during MQAR-64 (NOTEBOOK.md). The other two logs are empty: it never reached those tasks |
| `logs/bench_chunk.log`, `results/bench_chunk/` | The chunk-size timing |
| `results/smoke/` | The 40-step smoke test; its log was not kept. Not quoted |
