#!/usr/bin/env python3
"""Aggregate MQAR runs at the 64-pair level (d_model 64): recall per memory layer,
learning rate and seed, and the best learning rate per layer (mean over seeds)."""
import collections
import glob
import json
import statistics

runs = collections.defaultdict(list)  # (kind, lr) -> [(seed, steps, recall)]
for path in sorted(glob.glob("results/seeds/*/mqar_*.json")):
    R = json.load(open(path))
    a = R["args"]
    if a["d"] != 64 or a.get("quick") or a["steps"] != 3000:
        continue
    for r in R["runs"]:
        if r["pairs"] == 64 and r["kind"] != "attn":
            runs[(r["kind"], a["lr"])].append((a["seed"], a["steps"], r["recall"]))

print("| Memory layer | Learning rate | Runs (seed: recall) | Mean recall |")
print("|---|---|---|---|")
best = {}
for (kind, lr), rs in sorted(runs.items()):
    mean = statistics.mean(r[2] for r in rs)
    detail = ", ".join(f"s{s}: {acc * 100:.1f}%" for s, _, acc in rs)
    print(f"| {kind} | {lr:g} | {detail} | {mean * 100:.1f}% |")
    if kind not in best or mean > best[kind][1]:
        best[kind] = (lr, mean, [r[2] for r in rs])

print("\nBest learning rate per layer (mean over seeds):")
for kind, (lr, mean, accs) in sorted(best.items(), key=lambda kv: -kv[1][1]):
    print(f"  {kind:>10}: {mean * 100:5.1f}%  (lr {lr:g}, range {min(accs) * 100:.1f}-{max(accs) * 100:.1f}%)")
