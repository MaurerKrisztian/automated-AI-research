---
name: run-experiment
description: Conventions for writing and running an experiment in a study on this lab's GPU (smoke test, logs, results files, timing, seeds, background runs, notebook entries). Use whenever you write an experiment script or start a GPU or long CPU run in studies/.
---

# Running an experiment

## Before the run
- Run `nvidia-smi`. Wait if another job is using the GPU; run one GPU job at a time.
- Estimate memory and time with `docs/COMPUTE.md`. Stay under 90% of the GPU's memory (14 GB on the reference GPU) and 60 minutes per run unless the user approved more. On a machine other than the reference one, run `scripts/gpu_probe.py` once and compare timings only within this machine.
- Smoke-test with `--quick` (tiny sizes, under a minute) before the full run.

## Script conventions
Scripts live at the top level of the study folder, take every setting as an argument, and save results next to themselves:

```python
#!/usr/bin/env python3
"""What this experiment tests. Usage:
    ../../.venv/bin/python -u exp_speed.py 2>&1 | tee logs/exp_speed.log
"""
import argparse
from pathlib import Path

import torch

from lab import bench, max_rel_err, peak_memory_gb, reset_peak_memory, save_results, seed_everything

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test at tiny sizes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False  # accuracy runs: TF32 off, and say so
    rows = []
    # ... run each method on identical inputs, append a dict per measurement ...
    # ... print a readable table ...
    path = save_results({"args": vars(args), "rows": rows}, HERE / "results" / "exp_speed", "exp_speed")
    print(f"saved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
```

- Time with `lab.bench` (warmup, CUDA sync, median); report precision and sizes. Measure memory with `reset_peak_memory` and `peak_memory_gb`.
- Measure accuracy against a float64 reference with `max_rel_err` or `rel_err`.
- Give every method identical inputs, sizes, precision and tuning budget.
- For a grid (several seeds, sizes or learning rates), write `run_<name>.sh` so the whole grid can be rerun, and `aggregate_<name>.py` to turn the results into a table.

## Running
- From the study folder: `../../.venv/bin/python -u <script>.py ... 2>&1 | tee logs/<descriptive-name>.log`.
- Run anything over about 2 minutes in the background and wait for its completion notification instead of polling.
- Never edit or delete files in `logs/` or `results/`. A failed or discarded run stays; NOTEBOOK.md says why.

## After the run
- Read the whole log: errors, NaN or inf, warnings, suspicious timings.
- Add a NOTEBOOK.md entry: date, purpose, exact command, outcome with key numbers, and the log and results files.
- If the result changes a claim, update the README's Summary and Claims in the same edit.
- On out-of-memory errors, shrink the batch or sequence for every method alike, never only for one.
