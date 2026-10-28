"""Timing, memory and accuracy helpers. Timings synchronize the GPU, so they cover the queued work."""
import random
import statistics
import time

import torch


def sync(device):
    """Wait for all queued work on `device` (no-op on CPU)."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup=3, reps=10):
    """Median wall-clock time of fn() in milliseconds, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def max_rel_err(x, ref):
    """Largest row-wise relative error ||x - ref|| / ||ref|| over the last dimension, in float64."""
    num = (x.double() - ref.double()).norm(dim=-1)
    den = ref.double().norm(dim=-1).clamp_min(1e-30)
    return (num / den).max().item()


def rel_err(x, ref):
    """Relative error of the whole tensor, ||x - ref|| / ||ref|| (Frobenius norm), in float64."""
    return ((x.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device):
    """Peak memory PyTorch allocated on `device` since the last reset, in GiB (nan on CPU)."""
    if torch.device(device).type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / 2**30


def seed_everything(seed):
    """Seed Python, PyTorch (CPU and every GPU) and NumPy when it is installed."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
