#!/usr/bin/env python3
"""Study 0004: which chunk size trains fastest, per model. The chunk size does not change the function (check_layers.py),
so this only sets the cost of the grid. Times a few training steps on MQAR at 64 pairs, median, with synchronization.

    ../../.venv/bin/python -u bench_chunk.py 2>&1 | tee logs/bench_chunk.log
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lab import peak_memory_gb, reset_peak_memory, save_results
from models import MODELS, Net
from tasks import TASKS

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[64])
    p.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--tasks", nargs="+", default=["mqar64", "mqar128"])
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for B in args.batches:
            for model, (kind, opts) in MODELS.items():
                for C in (args.chunks if kind == "kalman" else [None]):
                    kw = dict(opts)
                    if kind == "kalman":
                        kw["chunk"] = C
                    torch.manual_seed(0)
                    gen = torch.Generator(device=device).manual_seed(1)
                    m = Net(task.vocab, 64, 2, 2, kind, **kw).to(device)
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                    reset_peak_memory(device)
                    times = []
                    for _ in range(args.steps):
                        x, y, _ = task.batch(B, device, gen)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten(), ignore_index=-100)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    ms = statistics.median(times[5:]) * 1e3
                    rows.append(dict(task=task_name, batch=B, model=model, chunk=C, ms=ms, peak_gb=peak_memory_gb(device)))
                    print(f"{task_name:>9} B {B:>4} {model:<22} C {str(C):>4}: {ms:6.1f} ms/step, {peak_memory_gb(device):.2f} GB", flush=True)
                    del m, opt
                    torch.cuda.empty_cache()
    path = save_results({"args": vars(args), "rows": rows}, HERE / "results" / "bench_chunk", "bench_chunk")
    print(f"saved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
