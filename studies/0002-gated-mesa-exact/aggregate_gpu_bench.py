#!/usr/bin/env python3
"""Per-cell ratios and summary ranges from a gpu_bench results JSON, so no ratio in the write-up is computed by hand.
Usage:
    python3 aggregate_gpu_bench.py results/gpu_bench_<timestamp>.json 2>&1 | tee logs/aggregate_gpu_bench.log
"""
import json
import sys


def rng(xs, fmt):
    return f"{format(min(xs), fmt)}–{format(max(xs), fmt)}"


def main(path):
    data = json.load(open(path))
    rows = data["part_b"]
    backends = data["args"]["eigh"]
    steps = data["args"]["cg_steps"]
    print(f"source: {path}\n")
    print("Per cell: exact time / CG time (per eigh backend), eigh share of the exact solver's stage sum,")
    print("and, at matched error, each exact variant's time over CG's at the fewest steps reaching that variant's error;")
    print("the best variant is shown (plain CG / Jacobi PCG).\n")
    for r in rows:
        st = r["exact_stage_ms"]
        line = f"d={r['d']:3d} C={r['C']:3d} gates {r['gates'][0]}-{r['gates'][1]} ridge {r['ridge'][0]:g}  cond {r['cond_max']:.1e}"
        print(line)
        for b in backends:
            t = r["ms"][f"exact_{b}"]
            share = st[f"eigh_{b}"] / (st["states"] + st["capacitance"] + st[f"eigh_{b}"])
            ratios = "  ".join(f"/cg{s} {t / r['ms'][f'cg{s}']:5.2f}" for s in steps)
            print(f"    exact_{b:8s} {t:8.2f} ms  err {r['err'][f'exact_{b}']:.2e}  eigh share {share:.3f}  {ratios}")
        # each exact variant is matched against CG at its own error; the best (smallest) time ratio is reported
        res = {}
        for m in ("cg", "jac"):
            best = None
            for b in backends:
                e, t = r["err"][f"exact_{b}"], r["ms"][f"exact_{b}"]
                s_ = next((s for s in steps if r["err"][f"{m}{s}"] <= e), None)
                if s_ is not None and (best is None or t / r["ms"][f"{m}{s_}"] < best[0]):
                    best = (t / r["ms"][f"{m}{s_}"], b, s_)
            res[m] = best
        fmt = lambda x: f"{x[0]:.2f} ({x[1]}, {x[2]} steps)" if x else f"n/a (no match within {steps[-1]} steps)"
        print(f"    capacitance stage / cg30 {st['capacitance'] / r['ms']['cg30']:.2f};  "
              f"best exact / CG at matched error: cg {fmt(res['cg'])}, jacobi {fmt(res['jac'])}")
        r["_matched"] = (r["cond_max"], res["cg"][0] if res["cg"] else None, res["jac"][0] if res["jac"] else None)
    print("\nRanges over all cells:")
    for b in backends:
        for s in steps:
            print(f"  exact_{b} / cg{s}: {rng([r['ms'][f'exact_{b}'] / r['ms'][f'cg{s}'] for r in rows], '.1f')}x")
        sh = [r["exact_stage_ms"][f"eigh_{b}"] / (r["exact_stage_ms"]["states"] + r["exact_stage_ms"]["capacitance"]
                                                 + r["exact_stage_ms"][f"eigh_{b}"]) for r in rows]
        print(f"  eigh_{b} share of stage sum: {rng([100 * x for x in sh], '.1f')}%   "
              f"exact_{b} time {rng([r['ms'][f'exact_{b}'] for r in rows], '.0f')} ms   "
              f"peak {rng([r['peak_gb'][f'exact_{b}'] for r in rows], '.2f')} GB")
    print(f"  capacitance stage / cg30: {rng([r['exact_stage_ms']['capacitance'] / r['ms']['cg30'] for r in rows], '.2f')}")
    for m in ["cg", "jac"]:
        for s in steps:
            print(f"  {m}{s}: time {rng([r['ms'][f'{m}{s}'] for r in rows], '.1f')} ms   "
                  f"peak {rng([r['peak_gb'][f'{m}{s}'] for r in rows], '.2f')} GB")
    for lo, hi, label in ((0, 100, "cond <= 1e2"), (100, 1e9, "cond > 1e2")):
        sel = [r for r in rows if lo < r["cond_max"] <= hi]
        if not sel:
            continue
        print(f"\n  errors, {label} ({len(sel)} cells, cond {rng([r['cond_max'] for r in sel], '.1e')}):")
        for name in sel[0]["err"]:
            print(f"    {name:14s} {rng([r['err'][name] for r in sel], '.1e')}")
    for lo, hi, label in ((0, 100, "cond <= 1e2"), (100, 1e9, "cond > 1e2")):
        m = [r["_matched"] for r in rows if lo < r["cond_max"] <= hi]
        for i, name in ((1, "cg"), (2, "jacobi")):
            vals = [x[i] for x in m if x[i] is not None]
            print(f"  best exact / {name} at matched error, {label}: {rng(vals, '.2f')}x "
                  f"({len(vals)} of {len(m)} cells matched within {steps[-1]} steps; exact faster in "
                  f"{sum(v < 1 for v in vals)})")
    if "part_c" in data:
        print("\nPart C (eigh alone, ms per matrix):")
        for r in data["part_c"]:
            print(f"  n={r['n']:3d} batch={r['batch']:5d}  " + "  ".join(f"{b} {t / r['batch']:.3f}" for b, t in r["ms"].items()))


if __name__ == "__main__":
    main(sys.argv[1])
