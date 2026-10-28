# Exact chunkwise Mesa — GPU micro-benchmark

Tests the core claim of the research draft on a real GPU: one Cholesky factorization per
chunk gives every token's exact least-squares memory readout, faster than MesaNet's
iterative conjugate-gradient (CG) solver.

## Setup (once)

```bash
./setup_env.sh        # creates .venv with CUDA PyTorch (~3 GB download)
```

## Run

```bash
.venv/bin/python bench_exact_chunk.py                 # full run, ~a few minutes
.venv/bin/python bench_exact_chunk.py --tf32          # same with TF32 matmuls (faster, less accurate)
python3 bench_exact_chunk.py --device cpu --batch 1 --heads 2 --seq 256 --reps 2 --warmup 1   # CPU smoke test
```

Defaults model one layer's worth of work: batch 8 × 8 heads × 2048 tokens, head dim 128,
chunk sizes 16 and 64, ridge 0.25 (MesaNet's floor). Results are also saved to `results/`.

## What the output means

- **Section A — exact one pass.** Times the exact chunk solve against CG with 5–30
  iterations on the same systems (the case where the trick is exact: no forgetting or
  scale-locked ridge). "max rel err" is measured against a float64 per-token reference.
- **Section B — MesaNet's own model** (fixed ridge + forget gates). Compares MesaNet-style CG
  with CG preconditioned by the exact chunk solver, for a long-memory gate (0.999) and a
  fast-forgetting gate (0.9). The printed bound is the theoretical condition number of the
  preconditioned system.

Plain PyTorch issues many small kernels, so absolute times are pessimistic for every
method; the ratios between methods are the interesting part. A fused Triton kernel would
be needed for a fair speed claim.

## Improved version: carried inverse + closed-form backward

```bash
python3 carry_check.py                                    # numpy check of the carried-inverse update (CPU)
.venv/bin/python bench_carry.py                           # speed vs per-chunk Cholesky and CG, plus float32 drift test
.venv/bin/python bench_carry.py --tf32                    # same with TF32 matmuls
.venv/bin/python bench_carry.py --skip-speed --drift-alpha 4   # drift test with badly conditioned keys
./run_fast.sh                                             # MQAR: improved layer (mesa_fast) speed and recall
```

- **Carried inverse:** keeps P = (ridge + H)^-1 as the state; per chunk only a C x C Cholesky,
  then P <- P - U U^T. No d x d factorization.
- **Closed-form backward:** dq_t = P_t dx_t, dk_i = -sum_{t>=i} (g_t x_t^T + x_t g_t^T) k_i,
  dridge = -sum_t g_t . x_t, computed for all chunks at once (no differentiation through Cholesky).
- Results land in `results_carry/` and `results_fast/`; logs in `carry_*.log` and `mqar_fast.log`.

## Recall test (MQAR)

```bash
.venv/bin/python mqar_recall.py --quick                   # smoke test
./run_seeds.sh                                            # 2 learning rates x 2 seeds at 64 pairs + summary
```
