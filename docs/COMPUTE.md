# Compute budget

What this machine can run, so experiments are sized before they start. The measured numbers come from [gpu_probe.log](gpu_probe.log), written by `scripts/gpu_probe.py` on 2026-10-29. Rerun the probe after a driver, PyTorch or hardware change.

## Machine

| Part | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB (15.5 GiB usable by PyTorch), 36 SMs, driver 580.178.04 |
| Software | Python 3.11, PyTorch 2.11.0 + CUDA 12.8, Triton 3.6.0 (in `.venv`) |
| CPU and RAM | 12 threads, 46 GB RAM |
| Disk | external drive, about 519 GB free on 2026-10-29 |

## Measured throughput

| Operation | Throughput |
|---|---|
| Dense matmul 8192², float32 (TF32 off) | 17.3 TFLOPS |
| Same, TF32 | 25.3 TFLOPS |
| Same, bfloat16 | 50.6 TFLOPS |
| Same, float16 | 49.0 TFLOPS |
| Device memory copy | 386 GB/s (read + write) |

These are peak numbers for large matmuls. Small batched operations (Cholesky, triangular solves, many small kernels) run far below them, which is why plain-PyTorch timings understate what fused kernels could do.

## What fits

Training a dense model costs about 6 FLOPs per parameter per token. At 20–40% of the measured bfloat16 peak, that gives these rough rates. They are estimates, not measurements; measure your own model before planning long runs.

| Model size | Tokens per second | Tokens per hour | 20 tokens per parameter takes |
|---|---|---|---|
| 10M | 170k–340k | 0.6–1.2B | 10–20 min |
| 30M | 56k–110k | 0.2–0.4B | 1.5–3 h |
| 125M | 13k–27k | 49–97M | 26–51 h |
| 350M | 4.8k–9.6k | 17–35M | 8–17 days |

Memory: AdamW with mixed precision needs about 16 bytes per parameter (weights, gradients, two moments) plus activations, so memory is rarely the limit below 300M parameters. Time is.

## Scales

Pick the smallest scale that can decide the hypothesis (see the evidence levels in [PROTOCOL.md](PROTOCOL.md)).

| Scale | Typical run | Evidence |
|---|---|---|
| Numerics | float64 checks on CPU, seconds | E2 |
| Micro-benchmark | one layer on the GPU, minutes | E3 |
| Synthetic tasks | MQAR, copying, selective copy, modular arithmetic; 2–4 layer models, 5–30 min per run, 3+ seeds | E4 |
| Tiny LM | 10–30M parameters, 100–600M tokens, 10 min–3 h per run | E5 |
| Small LM | 100M+ parameters, many hours; only for ideas that already passed the tiny-LM test, with the user's approval | E5 |

## Other machines

The numbers above come from the reference machine. Contributors run on other hardware:

| Machine | What it can do | Notes |
|---|---|---|
| Cloud session (Claude Code on the web or in the Claude app) | E1 and E2: derivations, literature, float64 checks | No GPU. Install packages with pip, and keep CPU runs to a few minutes |
| A computer without an NVIDIA GPU | E2 | `scripts/setup_env.sh --cpu`. On Apple silicon, float64 only runs on the CPU |
| Another NVIDIA GPU | E3 and up | Run `scripts/gpu_probe.py` first and use its numbers with the tables above. Keep peak memory under 90% of the card |

Speed results are only comparable between runs on the same machine, so a study that reports timings names its GPU (`lab.save_results` records it). A larger GPU fits bigger models from the "What fits" table, but the time limits per run and per cycle stay the same unless the person running the session approves more.

## Rules for GPU runs

- Check the GPU is free with `nvidia-smi` and run one GPU job at a time.
- Smoke-test at a small size (under a minute) before any full run.
- Keep peak memory under 90% of the card (14 GB here); measure it with `lab.peak_memory_gb`.
- A single run should take under 60 minutes and a research cycle about 3 GPU hours, unless the person running the session approved more.
- Turn TF32 off when measuring accuracy (`torch.backends.cuda.matmul.allow_tf32 = False`), and state the precision of every timing.
- Datasets go in `data/` (not tracked by git). Record each dataset's source, version and preprocessing in the study.
