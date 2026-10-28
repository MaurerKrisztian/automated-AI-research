#!/usr/bin/env python3
"""Measure this machine's GPU: memory, dense matmul throughput per precision, and copy bandwidth.
docs/COMPUTE.md quotes its output; rerun it after a driver, PyTorch or hardware change:

    .venv/bin/python scripts/gpu_probe.py 2>&1 | tee docs/gpu_probe.log
"""
import torch

from lab import bench, env_info


def main():
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    env = env_info()
    free, total = torch.cuda.mem_get_info()
    print(f"{props.name} | {total / 2**30:.1f} GiB ({free / 2**30:.1f} free now) | {props.multi_processor_count} SMs"
          f" | torch {env['torch']} CUDA {env['cuda']} | {env['time']}")
    n = 8192
    for name, dtype, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True),
                              ("bf16", torch.bfloat16, False), ("fp16", torch.float16, False)]:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a, b = (torch.randn(n, n, device=dev, dtype=dtype) for _ in range(2))
        ms = bench(lambda: a @ b, dev, warmup=3, reps=10)
        print(f"matmul {n}x{n} {name}: {ms:7.2f} ms  {2 * n**3 / ms / 1e9:6.1f} TFLOPS")
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.empty(2**28, device=dev, dtype=torch.float32)  # 1 GiB
    ms = bench(lambda: x.clone(), dev, warmup=3, reps=10)
    print(f"copy 1 GiB on device: {ms:.2f} ms  {2 * x.numel() * 4 / ms / 1e6:.0f} GB/s (read + write)")


if __name__ == "__main__":
    main()
