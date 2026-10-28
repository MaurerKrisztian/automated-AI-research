# Review of 0001

Reviewer: Claude Opus 5.5 in a Claude Code session (not the `reviewer` subagent) · Date: 2026-10-28, issues 9–12 added 2026-10-29 · Reviewed: the note as written on 2026-10-26

## Verdict

**Revise.** The central claims hold: the chunk identity is exact, and in plain PyTorch the exact solver is much faster and more accurate than MesaNet-style CG. But the Summary is out of date, the headline training speedup uses a CG baseline without its implicit backward, and several claims are stronger than the logs show.

## Issues

| # | Severity | Issue | Evidence | Resolution |
|---|---|---|---|---|
| 1 | major | The Summary describes the first version (71 ms per step, "2–3.7× faster"). The improved layer trains at 37.2 ms per step: 7.8× faster than 30 CG steps and 1.75× faster than 5. | `logs/mqar_full.log` (71.1 ms); `logs/mqar_fast.log` (mesa_fast 37.2, cg5 65.1, cg30 290.6 ms) | open |
| 2 | major | The 29× training-pass speedup compares against CG differentiated by autograd through every step. MesaNet's implicit backward costs about one more CG solve, so a fair ratio is likely nearer 17× (estimate: 2 × 182.9 ms ≈ 366 ms vs 21.5 ms). The note's own Method section mentions the implicit backward. | `logs/carry_closed_form.log` | open |
| 3 | major | "No float32 drift" is overstated. With badly conditioned keys the carried inverse's error grows from 1.1e-5 to 4.2e-4 over 1024 chunks, about 40×. It drifts far less than the alternatives, but it drifts, and every drift test is without forget gates. | `logs/carry_drift_alpha4.log` | open |
| 4 | major | "Mesa collapsed above capacity" at 128 pairs looks like a failure to train. The loss stays near chance (6.142 → 6.079 over 2000 steps) with about 1% recall for every Mesa variant, while linear attention reaches 60.5% with the same head size. Softmax attention also sat at chance (loss 6.221) at step 500 before reaching 100% by step 1000, so Mesa may need more steps or a different learning rate or ridge. | `logs/mqar_fast.log`, `logs/mqar_full.log` | open |
| 5 | major | Only 2 seeds per configuration, with very wide spreads: DeltaNet 2.1% vs 77.6%, and mesa_cg5 at learning rate 1e-3 1.9% vs 69.9%. The DeltaNet baseline looks undertuned, which weakens the 39.8% comparison. | `results/seeds/summary.md` | open |
| 6 | major | The novelty discussion skips the closest standard techniques: DeltaNet's chunkwise UT/WY transform (a C×C triangular solve per chunk, which `mqar_recall.py` itself implements in `deltanet_chunk`), and classical block RLS and sequential Kalman measurement updates, which the carried inverse is an instance of. | `mqar_recall.py`; README Novelty check | open |
| 7 | minor | Speed is benchmarked at head dim 128, recall at head dim 32, so the speedups may not carry over between them. | `logs/bench_gpu_fp32.log`, `logs/mqar_fast.log` | open |
| 8 | minor | The carried inverse processes chunks one after another, giving up the first version's chunk parallelism. Fine with 64 sequence-heads; untested at small batch with long sequences. | `bench_carry.py` | open |
| 9 | minor | The CPU float64 checks (3.1e-15, 8.8e-7, 2.8e-14 and the variant checks) have code in `archive/exact_mesa_code.zip` but no saved logs, and they need SciPy, which the virtual environment lacks. | `archive/exact_mesa_code.zip` | open |
| 10 | minor | Limitations still says gradients come from autograd; the Improvements section replaced that with the closed-form backward. | README Limitations vs Improvements | open |
| 11 | minor | Two timing runs at 128 pairs disagree by about 10% (15 CG steps: 140.0 vs 153.4 ms; 30 steps: 261.5 vs 290.6 ms), and the note quotes one without saying which. | `logs/mqar_full.log`, `logs/mqar_fast.log` | open |
| 12 | minor | There is no References section: the Novelty check names works (MesaNet, Preconditioned DeltaNet, Gated KalmaNet and others) without citations or links, so readers can't check them. | README Novelty check | open |

## What holds up

- Exactness: the chunked layer matches the token-by-token recurrence to 6.3e-16 and the carried-inverse layer to 6.2e-16, with gradients to 1.3e-15 (`logs/mqar_fast.log`). The closed-form backward matches autograd with dK error 1.6e-15 and dQ error 5.1e-16 (`logs/carry_closed_form.log`). On 2026-10-29 `tests/test_mesa.py` confirmed this again for the copy promoted to `lab/layers/mesa.py`.
- Every number in the Improvements speed table matches `logs/carry_closed_form.log`, and the 3.2× (68.7 / 21.5) and 29× (621.4 / 21.5) ratios are correct arithmetic.
- The first version's solver speed (16.8 vs 178.8 ms, 10.6×) and the preconditioner results (4.4× and 3.0×) match `logs/bench_gpu_fp32.log`.
- Recall of the improved layer (99.0%, 99.6%) matches `logs/mqar_fast.log`, and the first version's (98.9%, 99.5%) matches `results/seeds/summary.md`, which `aggregate_mqar.py` reproduces exactly.
