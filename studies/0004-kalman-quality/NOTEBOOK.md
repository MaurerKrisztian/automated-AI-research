# 0004 notebook

Dated, append-only log of what was run, why, and what happened, including dead ends and discarded runs. Newest entry last. Don't rewrite old entries; add a new one that corrects them.

## 2026-11-01

Session: the reference workstation (RTX 5060 Ti), Claude Opus 5.5 for @MaurerKrisztian, started from another session with backlog I-017 scoped as a model-quality test. Branch `study-0004-kalman-quality`, off `main` at `e3a8603`. Nothing is committed unless the maintainer asks.

### Relied on, with evidence levels

- 0003, claims 1–2 (E2): the chunk form computes the Kalman memory exactly. Claim 11 (E2): the batched code matches the filter. Claim 3 (E2): autograd through the chunk form matches autograd through the filter. Claim 10 (E2): the matmul Grams overflow in float32 once a chunk's log decay falls below about −88. Claim 4 and Proposition 1(d) (E2): a stored pair fades by the gate product in the Kalman memory, and MesaNet keeps it near full strength while aβ‖k‖² ≫ λ. Claim 14 (E3): the chunk form trains through autograd without a custom kernel.
- 0001, claim 7 (E4, 2 seeds only): MQAR at 64 pairs, 99.2–99.4% for Mesa, 97.7% for linear attention, 39.8% (2.1–77.6) for DeltaNet. Its 128-pair collapse to about 1% is not diagnosed (backlog I-009).

### Novelty check

Run before any training, by a `literature-scout` subagent (web search and the arXiv API; Semantic Scholar answered 429 and OpenReview a browser check), plus my own reading of the Bayesian Layer's PDF (pp. 1–12 as page images, and the extracted text of Appendices C, E.3–E.4 and F) and of fla's layer sources through a `web-fetch` subagent. Query log, from the scout: arXiv API on Kalman + linear attention/DeltaNet, associative recall, overwrite/stale/repeated-key, process noise + attention, OU + online, variance-preserving, MesaNet/test-time regression/selective copying, Bayesian/RLS + delta rule. Web search on MesaNet MQAR, back-to-the-prior, OU against random walk, tied process noise, MQAR overwrite, entity tracking, forgetting against process noise, regularized forgetting, TVP AR(1), selective copying. GitHub READMEs of flash-linear-attention, MAD-Lab and Mamba. The scout's table is reproduced in the README's Novelty check. What changes the design:

- **The Bayesian Layer's MQAR setting has no gates.** BL Appendix F.1: "identity state dynamics with learned, input-dependent process and observation noise", process variance initialized to 0.1, with a 0.25 learning-rate multiplier on the noise projection. Its stationary analogue would have zero noise, so "stationary against free" cannot be read off BL's table. I add `kalman_free_identity`, BL's published setting, to the four planned variants.
- **BL's gain is mostly inside its own spread.** BL Table 4, width 64: (K, L) = (64, 256): BL 97.19 ± 0.57, MesaNet 89.97 ± 12.90, GDN 76.85 ± 23.94. (128, 512): 51.07 ± 2.81, 45.29 ± 22.45, 33.68 ± 20.53. Width 128: (128, 512): 99.52 ± 0.20, 97.34 ± 2.63, 88.14 ± 8.93. (256, 1024): 69.09 ± 3.30, 59.66 ± 13.79, 41.86 ± 20.13. Its models train on L ≤ 256 and are evaluated on longer sequences. So I also evaluate each MQAR model at larger N than it trained on, at no extra training cost.
- **The rewrite task is published.** Volatile MQAR (Voltic, Hejabi et al. 2026, Section 4.2 and Appendix E): three rounds over K keys, values change with p = 0.30 before rounds 2 and 3, latest wins. Its Table 1: Voltic-Q 99.48, GDN-2 70.25, KDA 48.76, with no MesaNet and no dense Kalman filter. I use its definition instead of the rewrite task I had first drafted, and add MesaNet and the dense filter. KATA (Ghriss 2026) writes each key twice; the scout could not find its protocol.
- **Selective copying has a concrete generator** in the MAD suite (Poli et al. 2024, `mad/data/instances.py`), and MesaNet's paper reports 99.2 on it, so I use that generator's definition: vocabulary 16, length 256, 16 tokens.
- **Stationary against free noise:** no direct test in a learned Kalman memory found as of 2026-11-01. Nearest: Kalman Linear Attention's OU ablation on Selective Copy (diagonal filter, figure only), Soft Reset's Appendix B (on network weights), KRLS-T's Fig. 3 (hand-set forgetting), Kalman Delta Networks' Table 5 (learned against constant noise: no consistent improvement).

**Reading fla's layers** (fetched from `main`): MesaNet: logsigmoid(a(x)) with bias, sigmoid(b(x)) with bias, no convolution or SiLU on v, λ = softplus + 0.25 per channel starting at 1, 30 CG steps. Gated DeltaNet: a and b without bias, log gate −exp(A_log)·softplus(a(x) + dt_bias) with A uniform on 0–16 and dt log-uniform on 0.001–0.1, sigmoid(b(x)), SiLU and convolution on v too, optional output gate (default on).

### Hypothesis, written before any training

**H1 (the primary claim).** At least one Kalman variant (`kalman_stat_scalar`, `kalman_stat_channel`, `kalman_free_scalar`, `kalman_free_channel`, `kalman_free_identity`) trains to better recall than both MesaNet with 30 CG steps and Gated DeltaNet on at least one task, beyond the seed spread: over 3 seeds at the configuration chosen by the pilot, its lowest seed's recall is above the highest seed's recall of each baseline. Aim: E4.

**Stopping rule, as the maintainer set it.** If no Kalman variant beats both baselines beyond the seed spread on any task, the result is negative and the Kalman line stops.

**Guard against a false positive, added here.** With 5 variants and about 9 task and size cells, a chance win is possible: three seeds beat six by chance 1 time in 84, per comparison. So any apparent win is rerun with 3 fresh seeds, which were not used for anything else, and counts only if it holds there too.

**H2.** Stationary noise does as well as free noise: for each gate type on each task, recall ranges overlap. Refuted if a free variant is above the stationary variant with the same gates, beyond the spread, on some task. Prediction from the two-write formulas (README, Proposition 1): on Volatile MQAR, free noise should win, because it can raise the gain without fading the mean, while stationary noise can raise the gain only by fading every pair.

**H3, for backlog I-018.** On the trained models' own inputs, cond(𝒦) stays where the chunk form's float32 error matches the token-by-token filter's. I-018 matters for training if, on a trained model, the chunk form's float32 output error exceeds 1e-3 or 10× the float32 filter's.

**Tasks and sizes.** MQAR at N = 32, 64 and 128 (0001's generator, vocabulary 1024, T = 3N, trained at that N), also evaluated at larger N than trained on. Volatile MQAR at K = 16 and 32 (T = 8K), with K = 32 at p = 0 as a control. Selective copying (MAD). d_model 64, 2 heads of dim 32, 2 layers, as in 0001, unless a baseline does not train at that size, which would be decided from the baselines alone.

**Equal budget.** Every model gets the same pilot: 3 learning rates at its default configuration and 1 alternative configuration, chosen before any results (`models.ALT`: for Kalman variants MesaNet's sigmoid write strength instead of BL's softplus; for MesaNet gates starting at half-lives 32–4096 instead of 0.5; for Gated DeltaNet fla's output gate). One tuning seed that is not reused. Then 3 fresh seeds at the chosen configuration. Same data stream and evaluation batches for all models.

**What would not count.** A difference inside the seed spread is "no measurable difference". A task on which nothing trains, or on which everything saturates above 99%, decides nothing, and is reported as such. A failure to train is diagnosed before it is called a capacity limit.

### Checks before training

**`check_layers.py`** (`logs/check_layers.log`). The first run of part A failed: the scalar-gate form was off by orders of magnitude while the per-channel form passed. Cause: `tril()` on a 1×C tensor in the noise-Gram factor, a bug in my code, fixed by expanding first. Part B then failed only on P_init (relative gap 1.0–1.2) because P_init's gradient was not symmetrized; only symmetric perturbations of P exist, so the check now compares the symmetric part. Neither of those runs left a log. The recorded run: part A 20 cases, worst 2.1e-15 against the extended-precision filter. Part B, gradients ≤ 1.1e-15. Part C, per-channel gates down to 1e-6 at C = 64 in float32 (chunk log decay down to −701.8): 0003's matmul Grams give NaN from gates 0.1–0.3 on, this form stays ≤ 5.9e-7 with finite gradients. Part D, the two-write formulas, 5.8e-16. Part E, the generators against brute force. Part F, fla's kernels against float64 references: Gated DeltaNet forward 1.3e-4, backward ≤ 1.4e-4; MesaNet with 30 CG steps forward 1.5e-3, backward ≤ 1.3e-3 (cond(H + Λ) up to 2.83, q and k pass through float16 in the kernel). The log predates two later edits: MesaNet's default start (below) and the chunk size used in part F's last table. Neither changes the layer.

### Sizing the budget, from the baselines alone

**`logs/scout_baselines.log`** (`results/scout_baselines/`, seed 100, lr 2e-3, 3000 steps, batch 128 at N = 64 and 64 otherwise; 455 s). Gated DeltaNet: MQAR-64 92.65%, MQAR-128 33.68%, Volatile K = 32 91.59%, selective copying 99.80% by step 2000. **MesaNet from fla's start of γ ≈ 0.5: MQAR-64 0.17%, MQAR-128 0.21%, Volatile K = 32 0.16%, all at chance, with the loss flat at 6.24 and the first layer's gates still 0.49–0.52**; selective copying 80.3%. So at 0.5 per token a pair is gone within about 10 tokens and no gradient reaches the gate. This is a failure to train, not a capacity limit: BL's MesaNet reaches 90–100% at these sizes (its Table 4), but BL trains on a mix that includes short sequences, where a gate of 0.5 still leaves a signal.

**Decision, from baselines only:** MesaNet's default start becomes half-lives 32–4096, the same as the Kalman cores, and fla's start becomes its alternative configuration. Kalman results played no part: none had been run past the 40-step smoke test.

**`logs/bench_chunk.log`** (`results/bench_chunk/bench_chunk_20261101_120001.json`): per-channel gates are fastest at C = 32 and the other Kalman cores at C = 64. At MQAR-64, batch 64: Gated DeltaNet 16.7 ms/step, MesaNet 7.1, stationary scalar 16.6, stationary per-channel 38.6, free scalar 28.8, free per-channel 41.3, free identity 23.0.

**Batch sizes**, to fit the budget, with the same queries per step across MQAR sizes (B·N = 4096): N = 32: 128; N = 64: 64; N = 128: 32. Volatile: K = 16: 64; K = 32: 32. Selective copying: 64. Steps: 3000, as in 0001, stopping early at 99.5% evaluation recall. Pilot: 1000 steps (1/3 of a final run), on MQAR-64, Volatile K = 32 and selective copying, seed 100, lr ∈ {1e-3, 2e-3, 4e-3}, plus the alternative at 2e-3; the choice on MQAR-64 serves all MQAR sizes, and the choice on Volatile K = 32 all Volatile tasks. Final seeds 0, 1, 2; replication seeds 10, 11, 12. Estimated GPU time: pilot 35 min, final about 2 h.

**GPU time so far:** about 15 min (checks, smoke test, scouting, chunk timing).

### The first pilot, stopped: discarded

`./run_grid.sh pilot` at 1000 steps, batch 64 (`logs/pilot_mqar64.log`, `results/pilot/`, 22 runs). Every run that finished on MQAR-64 was still on the loss plateau: recall 0.20–1.27%, against 0.195% for chance. Both baselines at chance too. A pilot in which nothing has left the plateau cannot choose a configuration, so I stopped it after about 9 minutes and use none of it. The pattern of those numbers should not be read as a ranking: the highest learning rate simply left the plateau a little earlier. Stopping it, my `pkill -f` pattern also matched the shell running the command, which killed that shell before the rest of the command ran, and left one training process alive; I killed that process by id.

**Why the plateau, and the change.** 0001's same-size training is the hard way to start: at N = 64 a batch has little signal until the embeddings align. BL trains on a pool dominated by short sequences (its Appendix F.1), and Voltic alternates K = 16 and K = 64 (its Appendix E). So I add mixed-size training, in which each step draws a size and takes a batch of equal queries: `mqarmix` (N ∈ {8, 16, 32, 64}, evaluated at N = 32, 64, 128 and 256) and `volatilemix` (K ∈ {8, 16, 32}, evaluated at K = 16, 32 and 64, with changes). One training run then gives every size, which also saves GPU time. Rescouting with the baselines alone (`logs/scout2.log`, `logs/scout2_short.log`) decides whether a 1000-step pilot on these tasks has signal.

**Rescouting, baselines only** (`logs/scout2.log`, `results/scout2/`; `logs/scout2_short.log`, `results/scout2_short/`; seed 100, lr 2e-3, 450 s). One seed, so not results. Gated DeltaNet and MesaNet (now starting near gate 1):

| Task | Gated DeltaNet | MesaNet |
|---|---|---|
| `mqarmix`, N = 32 / 64 / 128 / 256 | 96.3 / 72.8 / 34.7 / 13.3 | 100.0 / 99.7 / 79.8 / 35.6 |
| `mqar64` (same-size), N = 64 / 128 / 256 | 60.3 / 27.1 / 11.1 | 96.6 / 60.5 / 20.2 |
| `volatilemix`, K = 16 / 32 / 64 | 97.3 / 81.8 / 35.0 | 99.6 / 98.5 / 87.9 |
| `volatile32` (same-size), K = 32 / 64 | 90.1 / 52.6 | 99.0 / 81.7 |
| `selcopy` | 99.9 (stopped at step 2500) | 87.4 |

Mixed-size training leaves the plateau around step 700 on MQAR, against 1100 at the same size, and gives larger-size recall at no extra training. 1000 steps is still too short for a pilot (Gated DeltaNet 1.4%, MesaNet 54.9% on `mqarmix`). So the pilot runs 1500 steps (half of a final run). **Grid decision, from baselines only:** pilot and final on `mqarmix`, `volatilemix` and `selcopy`. A same-size run or the p = 0 control only if budget remains. MesaNet at N = 128 reaching about 80% when trained on N ≤ 64 also says 0001's collapse to 1.3% at 128 pairs was not a capacity limit of a head of dim 32, as far as one seed can.

**Pilot's selection metric, set before the pilot:** mean recall over the sizes evaluated, rounded to 0.1 point, then lowest loss. The chunk size in `kalman_layer.kalman_chunk` now adjusts to the sequence, with the same number of chunks and the least padding (T = 24 at C = 64 no longer pads to 64); parts A and B of the check were rerun after this edit, and a final recorded run of `check_layers.py` follows.

**GPU time so far:** about 35 min.

### The pilot

`./run_grid.sh pilot` (`logs/pilot2_mqarmix.log`, `logs/pilot2_volatilemix.log`, `logs/pilot2_selcopy.log`; `results/pilot2/`, 84 runs, 3291 s; choices in `logs/aggregate_pilot2.log` and `results/best_config.json`). Seed 100, 1500 steps. No Cholesky failure and no non-finite gradient in any run. Largest cond(𝒦) seen at the end of a pilot run: 1.57e4 (`kalman_free_identity` on selective copying at lr 4e-3). Most choices sit at the top of the learning-rate grid: 15 of 21 picked 4e-3, and the other 6 picked the alternative at 2e-3, whose own learning rate was fixed in advance. So higher rates might do better at this budget, for every model alike; a limitation, not fixed, because a wider grid does not fit the budget.

The choices hold for the final runs, unchanged. Mean recall over sizes, as chosen: on `mqarmix` Gated DeltaNet 44.0, MesaNet 76.1, stationary scalar 68.6 (sigmoid), stationary per-channel 44.2, free scalar 59.1, free per-channel 9.2 (sigmoid), free identity 14.5 (sigmoid). These are half-length runs on the tuning seed and are not results.

**GPU time so far:** about 95 min. The final grid (63 runs) should take about 80 min, leaving about 10–15 min for any replication. Code frozen here; `logs/check_layers_v2.log` is the recorded check of the frozen code, and it supersedes `logs/check_layers.log`.

### Final runs

`./run_grid.sh final volatilemix mqarmix selcopy`: seeds 0, 1, 2, 3000 steps, at each model's chosen configuration. Results in `results/final/`, one JSON per run; logs `logs/final_<task>.log`.

It finished with no failed run and no traceback: 63 runs, 4598 s (`results/final/`; tables in `logs/aggregate_final.log`). The hypothesis named same-size MQAR and K = 16 and 32 as separate tasks, with a p = 0 control. After the plateau, the grid became the three mixed-size or selective-copying tasks above, decided from baselines only. The same-size runs and the control were not run, for budget.

**Decision rule.** On MQAR and Volatile MQAR, no Kalman variant beats both baselines at any size; MesaNet is ahead of all of them beyond the spread at N = 64, 128 and 256. The script flagged one win: selective copying, `kalman_free_channel` 99.658–99.860% against Gated DeltaNet's 99.554, 77.612 and 99.652. The margin is 0.006 points, between runs that all stopped at their first evaluation above 99.5%, so I did not believe it, but the rule was written down before training and says fresh seeds decide.

**Replication** (`logs/replicate_selcopy.log`, `results/replicate/`, seeds 10, 11, 12, 641 s): Gated DeltaNet 99.695, 99.817, 99.884; `kalman_free_channel` 99.316, 99.799, 99.109; MesaNet 93.433, 81.958, 94.580. Not a win. `logs/aggregate_final_v2.log` is the table with this added.

**Result: H1 is refuted at this size and budget. By the stopping rule the Kalman line stops.** H2 holds and its prediction fails: free noise is nowhere ahead of stationary noise on MQAR or Volatile MQAR. H3: cond(𝒦) reaches 4.67e4; the chunk form's float32 error stays ≤ 2.4e-5; the 10× line is crossed once (50×, 1.8e-5 against 5.6e-7, `kalman_free_identity` on `volatilemix`), at an absolute size that does not matter here.

**The maintainer's decision, during the final grid:** "if it not significantly better just stop this line dont waste time on it". So the seeds that did not train are diagnosed from the logs and not rerun, and no diagnostic GPU run follows. The earlier plan to rerun them at twice the steps is dropped. I had told the maintainer the replication would be 8 minutes and it took about 11.

**Failed seeds, from the logs.** `volatilemix` seed 0 of `kalman_stat_channel`, `kalman_free_channel` and `kalman_free_identity`: recall about 1/K at each K (2.5–2.9% at 32, 6.0–6.5% at 16, 1.2–1.7% at 64), consistent with a readout that does not use the query. That is my reading, not a test. Two more runs are slow rather than stuck: `kalman_free_identity` seed 1 (29.7%) and `kalman_free_scalar` seed 0 (44.0%), both still improving when the learning rate reached zero.

**Tidying after the runs, without changing behavior:** `tasks.py` lost entries for tasks never run (`mqar32`, `volatile16`, `standard32`, `standardmix`); `run_grid.sh` gained `SUFFIX` and a `replicate` case, so a rerun cannot append to a recorded log.

**Errata and backlog.** 0001's README gets a dated erratum: its 128-pair collapse is not a capacity limit (MesaNet here: 81.9% at N = 128 with heads of dim 32). Backlog: I-017 done, I-018 and I-019 dropped, note on I-009, new I-020.

**GPU time for this cycle:** about 2.9 hours. Checks, smoke test, first scouting and chunk timing about 15 min; the discarded pilot about 9 min; second scouting 8 min; pilot 55 min; recorded check 2 min; final grid 77 min; replication 11 min.

**No independent review, by the maintainer's decision.** I started the `reviewer` subagent on the written-up study, and the maintainer then said "dont run independent review". I stopped it before it had reported anything (it had only announced that it would start reading), and used nothing from it. REVIEW.md records this. The status stays `review`, and `tools/studies.py check` keeps its warning. The backlog item is marked done on the result, with the study unreviewed.

**Correction to the entry above:** `tools/studies.py check` does not keep its warning. It stops warning once REVIEW.md has content, even though that content says no review took place, so a clean check says nothing about review here. REVIEW.md and the `review` status are the record.
