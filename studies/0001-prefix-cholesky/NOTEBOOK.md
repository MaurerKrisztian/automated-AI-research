# 0001 notebook

Dated, append-only log of what was run, why, and what happened. Entries before 2026-10-28 were reconstructed from file timestamps and log contents when the study moved into this repository; the original session notes were not kept.

## 2026-10-25

- 23:55 · CPU sanity checks in NumPy/SciPy: the nested-inverse identity, the Mesa-P and sliding-window variants, exactly preconditioned DeltaNet, preconditioned CG for MesaNet's gated model, and a cost model. Snapshot: `archive/exact_mesa_code.zip`. Their outputs were not saved as logs.

## 2026-10-26

- 00:10–00:13 · GPU micro-benchmark of the first version (per-chunk d×d Cholesky) against CG: a CPU smoke run (`results/bench/bench_x86_64_20261026_001042.json`), then the RTX 5060 Ti run (`logs/bench_gpu_fp32.log`, `results/bench/bench_NVIDIA_GeForce_RTX_5060_Ti_20261026_001350.json`).
- 00:22–00:39 · MQAR recall: smoke test (`logs/mqar_quick.log`), pilot (`logs/mqar_pilot.log`, `results/pilot/`), full grid at 32, 64 and 128 pairs (`logs/mqar_full.log`, `results/mqar/`).
- 00:44–01:00 · Seeds at 64 pairs: 2 learning rates × 2 seeds (`run_seeds.sh`, `logs/mqar_seeds.log`, `results/seeds/`).
- 01:22–01:26 · Carried inverse: smoke run (`results/smoke/`), float32 speed and drift (`logs/carry_fp32.log`), TF32 (`logs/carry_tf32.log`), drift with key-spectrum alpha 2 and 4 (`logs/carry_drift_alpha2.log`, `logs/carry_drift_alpha4.log`).
- 01:33–01:34 · Closed-form backward: speed run (`logs/carry_closed_form.log`, `results/carry/carry_fp32_speed_20261026_013432.json`).
- 01:38–01:40 · Improved layer (`mesa_fast`) in MQAR (`run_fast.sh`, `logs/mqar_fast.log`, `results/fast/`).
- The write-up was drafted as a standalone document, now `archive/note-original.txt`.

## 2026-10-28

- Reviewed the write-up against the logs: issues 1–8 in `REVIEW.md`.

## 2026-10-29

- Moved the study into this repository. Logs went into `logs/`. Results folders were renamed: `results/` became `results/bench/` and `results/mqar/`, `results_carry` became `results/carry`, `results_fast` became `results/fast`, `results_pilot` became `results/pilot`, `results_seeds` became `results/seeds`, `results_smoke` became `results/smoke`. The "saved …" lines inside the logs still show the old paths.
- Updated the scripts' default output folders, the run scripts' virtual-environment path and `aggregate_mqar.py`'s input path to match. No other code changed.
- `aggregate_mqar.py` reproduces `results/seeds/summary.md` exactly from the moved results.
- The note became `README.md`: formatting changes only, plus the parts marked *added* (Claims, sources, Reproduce, Files).
- Promoted the carried-inverse layer (`_CarrySolve`, `mesa_fast`, `mesa_reference` from `mqar_recall.py`) to `lab/layers/mesa.py`. `tests/test_mesa.py` passes: forward output and all five gradients match the token-by-token reference in float64, and float32 on the GPU stays within 1e-4.
- Added review issues 9–12.
