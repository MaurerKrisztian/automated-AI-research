#!/usr/bin/env python3
"""Tables, ratios and ranges from a gpu_bench.py results JSON, so that no number in the write-up is computed by hand.

Per cell of part B: forward times of every method, the time of the best Bayesian Layer variant over the best variant
of the chunk-parallel form (above 1: the chunk form is faster), the same for CUDA-graph replays and for forward +
backward, and the float32 errors. Then summary ranges, the cells of backlog I-016's hypothesis (d = 128, C = 32-64),
the stage shares of part C and the drift table of part D. "Best" is the fastest variant of a family in that cell:
chunk = {chunk, chunk_magma, chunk_inv}; BL = {bl_triton, bl_triton_wy, bl, bl_wy}, forward + backward only over the
differentiable ones. The parts may come from several runs, and part B's cells may be split over runs by N
(run_gpu.sh). With --rerun, a second run of part B in fresh processes, it also gives each time's rerun / first-run
ratio and the verdict ratios of both runs.
With --graph-bwd, part B runs whose forward + backward was also timed as CUDA graphs, it gives those tables and the
estimate for BL's fused backward; --graph-bwd-log compares them with the cells printed in the log of an earlier such run
that died before saving its JSON. run_gpu.sh writes the recorded table log; by hand, write it to /tmp:
    python3 aggregate_gpu_bench.py results/gpu_bench_<A, B for N = 16>.json results/gpu_bench_<B for N = 64>.json \
        results/gpu_bench_<C, D>.json --rerun results/gpu_bench_<B again, N = 16>.json \
        results/gpu_bench_<B again, N = 64>.json 2>&1 | tee /tmp/aggregate_gpu_bench.log
"""
import argparse
import json
import math
import re
import statistics

CHUNK =["chunk", "chunk_magma", "chunk_inv"]
BL = ["bl_triton", "bl_triton_wy", "bl", "bl_wy"]
REFS = ["recurrent", "mesa_carry", "mesanet_cg30", "mesanet_exact", "fla_gdn", "fla_kda", "fla_mesanet"]


def best(times, names):
    xs = [(times[n], n) for n in names if n in times and times[n] is not None and math.isfinite(times[n])]
    return min(xs) if xs else (None, None)


def family_ratio(times, num, den):
    """Fastest of family `num` over fastest of family `den` among one cell's times; None if either is missing."""
    a, b = best(times, num)[0], best(times, den)[0]
    return a / b if a and b else None


def fastest(rows, kind, fam):
    """Per (N, d, gates): the fastest `kind` time of family `fam` over all C, as (ms, variant, C)."""
    out = {}
    for key in sorted({(r["N"], r["d"], r["gates"]) for r in rows}):
        sub = [r for r in rows if (r["N"], r["d"], r["gates"]) == key]
        t = min(((t, n, r["C"]) for r in sub for t, n in [best(r.get(kind, {}), fam)] if t), default=None)
        if t:
            out[key] = t
    return out


def best_over_c(rows, kind, num, den):
    """Each family at its own fastest chunk size: per (N, d, gates), the fastest `kind` time of family `num` over all C,
    over the same for family `den`. Returns {(N, d, gates): (ratio, (ms, variant, C) of num, (ms, variant, C) of den)}."""
    a, b = fastest(rows, kind, num), fastest(rows, kind, den)
    return {key: (a[key][0] / b[key][0], a[key], b[key]) for key in a if key in b}


# The comparisons quoted in the write-up: best BL over best chunk form, forward eager, as CUDA graphs, and forward +
# backward (where only BL's PyTorch forms are differentiable).
VERDICTS = (("fwd", BL), ("graph", BL), ("fwd_bwd", ["bl", "bl_wy"]))


def rng(xs, fmt):
    xs = [x for x in xs if x is not None]
    return f"{format(min(xs), fmt)}–{format(max(xs), fmt)}" if xs else "n/a"


def fmt_ms(x):
    return "–" if x is None else f"{x:.2f}" if x < 100 else f"{x:.0f}"


def part_b(rows):
    names = [n for n in CHUNK + BL + REFS if any(n in r["fwd"] for r in rows)]
    print("## Part B: forward times (ms), float32, TF32 off\n")
    print("ratio = best BL / best chunk form (above 1: the chunk form is faster); graph = the same for CUDA-graph replays.\n")
    print("| N | d | gates | C | " + " | ".join(names) + " | ratio | graph ratio |")
    print("|---" * (6 + len(names)) + "|")
    for r in rows:
        bc, bb = best(r["fwd"], CHUNK), best(r["fwd"], BL)
        gc, gb = best(r.get("graph", {}), CHUNK), best(r.get("graph", {}), BL)
        r["_ratio"] = bb[0] / bc[0] if bc[0] and bb[0] else None
        r["_gratio"] = gb[0] / gc[0] if gc[0] and gb[0] else None
        r["_best"] = (bc, bb)
        cells = [fmt_ms(r["fwd"].get(n)) for n in names]
        ratio = f"{r['_ratio']:.2f} ({bb[1]} / {bc[1]})" if r["_ratio"] else "–"
        gratio = f"{r['_gratio']:.2f}" if r["_gratio"] else "–"
        print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | " + " | ".join(cells) + f" | {ratio} | {gratio} |")
    print()
    if any(r.get("graph") for r in rows):
        gnames = [n for n in CHUNK + BL + REFS if any(n in r.get("graph", {}) for r in rows)]
        print("## Part B: CUDA-graph forward replays (ms)\n")
        print("| N | d | gates | C | " + " | ".join(gnames) + " |")
        print("|---" * (4 + len(gnames)) + "|")
        for r in rows:
            print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | "
                  + " | ".join(fmt_ms(r["graph"].get(n)) for n in gnames) + " |")
        print()
    bnames = [n for n in CHUNK + BL + REFS if any(n in r["fwd_bwd"] for r in rows)]
    if bnames:
        print("## Part B: forward + backward (ms; autograd, except mesa_carry's closed form and fla's kernels)\n")
        print("ratio = best differentiable BL (bl, bl_wy) / best chunk form; rec = recurrent / best chunk form.\n")
        print("| N | d | gates | C | " + " | ".join(bnames) + " | ratio | rec |")
        print("|---" * (6 + len(bnames)) + "|")
        for r in rows:
            bc, bb = best(r["fwd_bwd"], CHUNK), best(r["fwd_bwd"], ["bl", "bl_wy"])
            rec = r["fwd_bwd"].get("recurrent")
            r["_bratio"] = bb[0] / bc[0] if bc[0] and bb[0] else None
            r["_brec"] = rec / bc[0] if bc[0] and rec else None
            print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | "
                  + " | ".join(fmt_ms(r["fwd_bwd"].get(n)) for n in bnames)
                  + (f" | {r['_bratio']:.2f}" if r["_bratio"] else " | –") + (f" | {r['_brec']:.2f} |" if r["_brec"] else " | – |"))
        print()
    enames = [n for n in CHUNK + BL + ["recurrent"] if any(n in r["err"] for r in rows)]
    print("## Part B: float32 output error against the float64 token-by-token filter\n")
    print("| N | d | gates | C | " + " | ".join(enames) + " |")
    print("|---" * (4 + len(enames)) + "|")
    for r in rows:
        print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | "
              + " | ".join(f"{r['err'][n]:.1e}" if n in r["err"] else "–" for n in enames) + " |")
    print()
    mnames = [n for n in CHUNK + BL + REFS if any(n in r["mem"] for r in rows)]
    print("## Part B: peak memory above the inputs (GiB), forward / forward + backward\n")
    print("| N | d | gates | C | " + " | ".join(mnames) + " |")
    print("|---" * (4 + len(mnames)) + "|")
    for r in rows:
        cells = []
        for n in mnames:
            a, b = r["mem"].get(n), r["mem_bwd"].get(n)
            cells.append(("–" if a is None else f"{a:.2f}") + ("" if b is None else f" / {b:.2f}"))
        print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | " + " | ".join(cells) + " |")
    print()
    notes = [(r, n, t) for r in rows for n, t in r["notes"].items()]
    if notes:
        print("Notes (skipped or failed runs):")
        for r, n, t in notes:
            print(f"  N={r['N']} d={r['d']} {r['gates']} C={r['C']} {n}: {t}")
        print()

    print("## Part B: ranges\n")
    print(f"- best BL / best chunk form, forward: {rng([r['_ratio'] for r in rows], '.2f')} over {len(rows)} cells;"
          f" chunk form faster in {sum(1 for r in rows if r['_ratio'] and r['_ratio'] > 1)} of"
          f" {sum(1 for r in rows if r['_ratio'])}")
    if any(r["_gratio"] for r in rows):
        print(f"- the same with CUDA graphs: {rng([r['_gratio'] for r in rows], '.2f')}; chunk form faster in"
              f" {sum(1 for r in rows if r['_gratio'] and r['_gratio'] > 1)} of {sum(1 for r in rows if r['_gratio'])}")
    for n in BL + REFS:
        xs = [r["fwd"][n] / r["_best"][0][0] for r in rows if n in r["fwd"] and r["_best"][0][0]]
        if xs:
            print(f"- {n} / best chunk form, forward: {rng(xs, '.2f')}; above 1 (chunk form faster) in"
                  f" {sum(1 for x in xs if x > 1)} of {len(xs)} cells")
    if bnames:
        print(f"- best differentiable BL / best chunk form, forward + backward: {rng([r['_bratio'] for r in rows], '.2f')}")
        print(f"- recurrent / best chunk form, forward + backward: {rng([r['_brec'] for r in rows], '.2f')}")
        for n in REFS[1:]:
            xs = [r["fwd_bwd"][n] / best(r["fwd_bwd"], CHUNK)[0] for r in rows
                  if r["fwd_bwd"].get(n) and best(r["fwd_bwd"], CHUNK)[0]]
            if xs:
                print(f"- {n} / best chunk form, forward + backward: {rng(xs, '.2f')}; above 1 (chunk form faster)"
                      f" in {sum(1 for x in xs if x > 1)} of {len(xs)} cells")
    for n in CHUNK + BL + ["recurrent"]:
        xs = [r["err"][n] for r in rows if n in r["err"]]
        if xs:
            print(f"- {n} float32 error: {rng(xs, '.1e')}")
    for n in CHUNK + BL + REFS:
        a = [r["mem"][n] for r in rows if r["mem"].get(n) is not None]
        b = [r["mem_bwd"][n] for r in rows if r["mem_bwd"].get(n) is not None]
        if a:
            print(f"- {n} peak memory above the inputs (GiB): forward {rng(a, '.2f')}"
                  + (f", forward + backward {rng(b, '.2f')}" if b else ""))
    for N in sorted({r["N"] for r in rows}):
        for d in sorted({r["d"] for r in rows}):
            sub = [r for r in rows if r["N"] == N and r["d"] == d and r["_best"][0][0]]
            if sub:
                fastest = min(sub, key=lambda r: r["_best"][0][0])
                print(f"- N={N} d={d}: fastest chunk-form cell C={fastest['C']} ({fastest['gates']}),"
                      f" {fastest['_best'][0][0]:.2f} ms ({fastest['_best'][0][1]})")
    hyp = [r for r in rows if r["d"] == 128 and r["C"] in (32, 64) and r["_ratio"]]
    if hyp:
        print(f"- backlog I-016's hypothesis cells (d = 128, C = 32-64): best BL / best chunk {rng([r['_ratio'] for r in hyp], '.2f')}"
              f" eager" + (f", {rng([r['_gratio'] for r in hyp], '.2f')} with graphs" if any(r['_gratio'] for r in hyp) else "")
              + f"; chunk form faster in {sum(1 for r in hyp if r['_ratio'] > 1)} of {len(hyp)} eager cells"
              + f" and {sum(1 for r in hyp if r['_gratio'] and r['_gratio'] > 1)} of"
              f" {sum(1 for r in hyp if r['_gratio'])} with graphs")
    print()
    print("## Part B: each family at its own fastest chunk size\n")
    print("Per (N, d, gates): the fastest time of each family over all C (variant, C); ratio = best BL / best chunk form.\n")
    print("| N | d | gates | timing | best chunk form | best BL | ratio |")
    print("|---|---|---|---|---|---|---|")
    for kind, num in VERDICTS:
        for key, (ratio, a, b) in best_over_c(rows, kind, num, CHUNK).items():
            print(f"| {key[0]} | {key[1]} | {key[2]} | {kind} | {b[0]:.2f} ({b[1]}, C = {b[2]}) |"
                  f" {a[0]:.2f} ({a[1]}, C = {a[2]}) | {ratio:.2f} |")
    print()
    for kind, num in VERDICTS:
        xs = [v[0] for v in best_over_c(rows, kind, num, CHUNK).values()]
        if xs:
            print(f"- {kind}, each family at its fastest C: best BL / best chunk form {rng(xs, '.2f')} over {len(xs)}"
                  f" (N, d, gates); chunk form faster in {sum(1 for x in xs if x > 1)}")
    print()


def cell_key(r):
    return r["N"], r["d"], r["gates"], r["C"]


def stability(rows, rerun):
    """Part B run twice, in separate processes: each rerun time over the first run's, and the verdict ratios of both."""
    first = {cell_key(r): r for r in rows}
    pairs = [(first[cell_key(r)], r) for r in rerun if cell_key(r) in first]
    print(f"## Part B rerun: stability ({len(pairs)} cells in both runs)\n")
    print("Each median time of the rerun over the same median in the first run.\n")
    print("| timing | method | cells | rerun / first: range | median |")
    print("|---|---|---|---|---|")
    medians, singles = [], []
    for kind in ("fwd", "graph", "fwd_bwd"):
        for n in CHUNK + BL + REFS:
            xs = [b[kind][n] / a[kind][n] for a, b in pairs if a.get(kind, {}).get(n) and b.get(kind, {}).get(n)]
            if xs:
                print(f"| {kind} | {n} | {len(xs)} | {rng(xs, '.3f')} | {statistics.median(xs):.3f} |")
                medians.append(statistics.median(xs))
                singles += xs
    print()
    print(f"- the medians above: {rng(medians, '.3f')}; every single time: {rng(singles, '.3f')}")
    print()
    print("Best BL / best chunk form in both runs (forward + backward: over bl and bl_wy only):\n")
    print("| N | d | gates | C | fwd | fwd, rerun | graph | graph, rerun | fwd+bwd | fwd+bwd, rerun |")
    print("|---" * 10 + "|")
    moves, flips = {"fwd": [], "graph": [], "fwd_bwd": []}, []
    for a, b in pairs:
        cells = []
        for kind, num in (("fwd", BL), ("graph", BL), ("fwd_bwd", ["bl", "bl_wy"])):
            x, y = family_ratio(a.get(kind, {}), num, CHUNK), family_ratio(b.get(kind, {}), num, CHUNK)
            cells += [f"{x:.2f}" if x else "–", f"{y:.2f}" if y else "–"]
            if x and y:
                moves[kind].append(y / x)
                if (x > 1) != (y > 1):
                    flips.append(f"N={a['N']} d={a['d']} {a['gates']} C={a['C']} {kind}: {x:.2f} -> {y:.2f}")
        print(f"| {a['N']} | {a['d']} | {a['gates']} | {a['C']} | " + " | ".join(cells) + " |")
    print()
    for kind, xs in moves.items():
        if xs:
            print(f"- {kind}: best BL / best chunk form, rerun over first run: {rng(xs, '.3f')} over {len(xs)} cells")
    print("- cells where the faster family changed between the runs: " + ("; ".join(flips) if flips else "none"))
    print()
    print("Each family at its own fastest chunk size, in both runs:\n")
    print("| N | d | gates | timing | ratio | ratio, rerun |")
    print("|---|---|---|---|---|---|")
    for kind, num in VERDICTS:
        x, y = best_over_c(rows, kind, num, CHUNK), best_over_c(rerun, kind, num, CHUNK)
        for key in x:
            if key in y:
                print(f"| {key[0]} | {key[1]} | {key[2]} | {kind} | {x[key][0]:.2f} | {y[key][0]:.2f} |")
    print()
    for kind, num in VERDICTS:
        for label, rs in (("first run", rows), ("rerun", rerun)):
            xs = [v[0] for v in best_over_c(rs, kind, num, CHUNK).values()]
            if xs:
                print(f"- {kind}, each family at its fastest C, {label}: {rng(xs, '.2f')};"
                      f" chunk form faster in {sum(1 for x in xs if x > 1)} of {len(xs)}")
    for kind, num in VERDICTS[:2]:
        for label, rs in (("first run", rows), ("rerun", rerun)):
            xs = [family_ratio(r.get(kind, {}), num, CHUNK) for r in rs if r["d"] == 128 and r["C"] in (32, 64)]
            xs = [x for x in xs if x]
            if xs:
                print(f"- {kind}, backlog I-016's hypothesis cells (d = 128, C = 32-64), {label}: {rng(xs, '.2f')};"
                      f" chunk form faster in {sum(1 for x in xs if x > 1)} of {len(xs)}")
    print()
    print("Decision rule (NOTEBOOK.md, 2026-10-30) in the hypothesis cells. The spread s is the largest relative change of"
          " best BL / best chunk form between the runs over all cells of that timing; a family is faster beyond the"
          " spread where the ratio is below 1 - s (BL) or above 1 + s (chunk form) in both runs.\n")
    for kind, num in VERDICTS[:2]:
        s = max(abs(x - 1) for x in moves[kind])
        cells = [(family_ratio(a.get(kind, {}), num, CHUNK), family_ratio(b.get(kind, {}), num, CHUNK)) for a, b in pairs
                 if a["d"] == 128 and a["C"] in (32, 64)]
        cells = [(x, y) for x, y in cells if x and y]
        bl = sum(1 for x, y in cells if x < 1 - s and y < 1 - s)
        ch = sum(1 for x, y in cells if x > 1 + s and y > 1 + s)
        print(f"- {kind}: s = {s:.3f}; BL faster beyond the spread in {bl} of {len(cells)} cells, the chunk form in {ch},"
              f" no measurable difference in {len(cells) - bl - ch}")
    print()


def log_cells(path):
    """The forward + backward times printed in a gpu_bench.py log, for a run that died before saving its JSON:
    {cell key: {"fwd_bwd": {method: ms}, "graph_bwd": {method: ms}}}."""
    cells, key = {}, None
    for line in open(path):
        m = re.match(r"\s+N=(\d+) d=(\d+) C=(\d+) (\w+) gates", line)
        if m:
            key = (int(m[1]), int(m[2]), m[4], int(m[3]))
            cells[key] = dict(fwd_bwd={}, graph_bwd={})
            continue
        m = re.match(r"\s+(\S+)\s+fwd\s.*?fwd\+bwd\s+([\d.]+) ms(?:\s+graph fwd\+bwd\s+([\d.]+))?", line)
        if m and key:
            cells[key]["fwd_bwd"][m[1]] = float(m[2])
            if m[3]:
                cells[key]["graph_bwd"][m[1]] = float(m[3])
    return cells


def graph_backward(main_rows, gb_rows, earlier_log=None):
    """Forward + backward timed eager and as CUDA graphs in the same run (gb_rows: the methods that can be captured),
    how its times compare with the first run's and with an earlier run's log, and the estimate for BL's fused backward."""
    bls = ["bl", "bl_wy"]
    names = [n for n in CHUNK + bls + ["recurrent"] if any(n in r.get("graph_bwd", {}) for r in gb_rows)]
    forms = ", ".join(n for n in bls if n in names)
    print("## Part B: forward + backward as CUDA graphs (ms; autograd)\n")
    print(f"Eager and graph times from the same run. ratio = BL's PyTorch form ({forms}) / best chunk form; rec ="
          " recurrent / best chunk form.\n")
    print("| N | d | gates | C | " + " | ".join(f"{n} eager | {n} graph" for n in names)
          + " | ratio eager | ratio graph | rec eager | rec graph |")
    print("|---" * (8 + 2 * len(names)) + "|")
    for r in gb_rows:
        cells = []
        for n in names:
            cells += [fmt_ms(r["fwd_bwd"].get(n)), fmt_ms(r.get("graph_bwd", {}).get(n))]
        ratios = [family_ratio(r["fwd_bwd"], bls, CHUNK), family_ratio(r.get("graph_bwd", {}), bls, CHUNK),
                  family_ratio(r["fwd_bwd"], ["recurrent"], CHUNK), family_ratio(r.get("graph_bwd", {}), ["recurrent"], CHUNK)]
        print(f"| {r['N']} | {r['d']} | {r['gates']} | {r['C']} | " + " | ".join(cells) + " | "
              + " | ".join("–" if x is None else f"{x:.2f}" for x in ratios) + " |")
    print()
    for label, kind in (("eager", "fwd_bwd"), ("CUDA graphs", "graph_bwd")):
        for fam, name in ((bls, f"BL's PyTorch form ({forms})"), (["recurrent"], "recurrent")):
            xs = [family_ratio(r.get(kind, {}), fam, CHUNK) for r in gb_rows]
            xs = [x for x in xs if x]
            if xs:
                print(f"- {label}: {name} / best chunk form, forward + backward: {rng(xs, '.2f')} over {len(xs)} cells;"
                      f" chunk form faster in {sum(1 for x in xs if x > 1)}")
        complete = [r for r in gb_rows if all(family_ratio(s.get(kind, {}), bls, CHUNK) for s in gb_rows
                                              if (s["N"], s["d"], s["gates"]) == (r["N"], r["d"], r["gates"]))]
        xs = [v[0] for v in best_over_c(complete, kind, bls, CHUNK).values()]
        if xs:
            print(f"- {label}, each family at its fastest C: BL's PyTorch form ({forms}) / best chunk form {rng(xs, '.2f')}"
                  f" over the {len(xs)} of {len({(r['N'], r['d'], r['gates']) for r in gb_rows})} (N, d, gates) timed at"
                  " every C")
    for n in names:
        xs = [r.get("graph_bwd", {}).get(n) / r["fwd_bwd"][n] for r in gb_rows
              if r.get("graph_bwd", {}).get(n) and r["fwd_bwd"].get(n)]
        if xs:
            print(f"- {n}: graph / eager forward + backward {rng(xs, '.3f')}")
    first = {cell_key(r): r for r in main_rows}
    for n in names:
        xs = [r["fwd_bwd"][n] / first[cell_key(r)]["fwd_bwd"][n] for r in gb_rows
              if cell_key(r) in first and r["fwd_bwd"].get(n) and first[cell_key(r)]["fwd_bwd"].get(n)]
        if xs:
            print(f"- {n}: eager forward + backward in this run over the first run's {rng(xs, '.3f')}"
                  f" (median {statistics.median(xs):.3f})")
    for n in names:
        xs = [r for r in gb_rows if "graph forward + backward skipped" in r["notes"].get(n, "")]
        if xs:
            print(f"- {n}: graph forward + backward skipped for memory in {len(xs)} of {len(gb_rows)} cells:"
                  f" N, d = {sorted({(r['N'], r['d']) for r in xs})}")
    for r in gb_rows:
        for n, t in r["notes"].items():
            if "graph" in t and "skipped" not in t:
                print(f"- note: N={r['N']} d={r['d']} {r['gates']} C={r['C']} {n}: {t}")
    print()
    if earlier_log:
        earlier = log_cells(earlier_log)
        print(f"Against {earlier_log}, whose run died before saving a JSON ({len(earlier)} cells finished), this run over"
              " that one:\n")
        for kind, label in (("graph_bwd", "CUDA graphs"), ("fwd_bwd", "eager")):
            for n in names:
                xs = [r[kind][n] / earlier[cell_key(r)][kind][n] for r in gb_rows if cell_key(r) in earlier
                      and r.get(kind, {}).get(n) and earlier[cell_key(r)][kind].get(n)]
                if xs:
                    print(f"- {label}, {n}: {rng(xs, '.3f')} (median {statistics.median(xs):.3f}) over {len(xs)} cells")
            pairs = [(family_ratio(r.get(kind, {}), bls, CHUNK), family_ratio(earlier[cell_key(r)][kind], bls, CHUNK))
                     for r in gb_rows if cell_key(r) in earlier]
            xs = [x / y for x, y in pairs if x and y]
            if xs:
                print(f"- {label}, BL's PyTorch form ({forms}) / best chunk form: {rng(xs, '.3f')} over {len(xs)} cells")
        print()
    print("BL's own backward is a fused kernel that this study does not have. If it costs k times BL's Triton forward,"
          " BL's training step beats the chunk form's autograd step wherever k < R - 1, with R = the chunk form's"
          " fastest forward + backward over bl_triton's fastest forward (each over C), per (N, d, gates):\n")
    for label, rows_fb, kind_fb, kind_f in (("eager", main_rows, "fwd_bwd", "fwd"),
                                             ("CUDA graphs", gb_rows, "graph_bwd", "graph")):
        a, b = fastest(rows_fb, kind_fb, CHUNK), fastest(main_rows, kind_f, ["bl_triton"])
        xs = [a[k][0] / b[k][0] for k in a if k in b]
        if xs:
            print(f"- {label}: R = {rng(xs, '.2f')} over {len(xs)} (N, d, gates), so BL's step is faster in all of them"
                  f" for any k < {min(xs) - 1:.2f}; at k = 2 the chunk form's step takes {rng([x / 3 for x in xs], '.2f')}×"
                  " BL's")
    for n in REFS[-3:]:
        xs = [r["fwd_bwd"][n] / r["fwd"][n] - 1 for r in main_rows if r["fwd_bwd"].get(n) and r["fwd"].get(n)]
        if xs:
            print(f"- for scale, a fused kernel in this benchmark, {n}: forward + backward over forward, minus 1:"
                  f" k = {rng(xs, '.1f')}")
    print()


def part_c(rows):
    print("## Part C: stage times (ms; median of 5 runs)\n")
    for r in rows:
        tot = sum(r["stage_ms"].values())
        print(f"- {r['method']} N={r['N']} d={r['d']} C={r['C']} T={r['T']}: total {tot:.2f} ms; "
              + ", ".join(f"{k} {v:.2f} ({100 * v / tot:.0f}%)" for k, v in r["stage_ms"].items()))
    print()


def part_d(rows):
    names = list(dict.fromkeys(n for r in rows for n in r["methods"]))
    print("## Part D: float32 drift (worst error over the eight segments; first -> last segment in brackets)\n")
    print("| configuration | T | cond(K) max | " + " | ".join(names) + " |")
    print("|---" * (3 + len(names)) + "|")
    for r in rows:
        cells = []
        for n in names:
            m = r["methods"].get(n)
            if not m or not m["seg"]:
                cells.append(m["status"] if m else "–")
            else:
                cells.append(f"{max(m['seg']):.1e} ({m['seg'][0]:.1e} -> {m['seg'][-1]:.1e})")
        print(f"| {r['config']} | {r['T']} | {r['cond_max']:.2e} | " + " | ".join(cells) + " |")
    print()
    print("Final covariance error and the smallest eigenvalue ratio of P over every chunk boundary:\n")
    print("| configuration | " + " | ".join(names) + " |")
    print("|---" * (1 + len(names)) + "|")
    for r in rows:
        cells = []
        for n in names:
            m = r["methods"].get(n)
            cells.append("–" if not m else f"{m['P_err']:.1e}, {m['P_eig_ratio_min_over_boundaries']:.1e}")
        print(f"| {r['config']} | " + " | ".join(cells) + " |")
    print()
    for n in names:
        if n == "recurrent":
            continue
        for label, keep in (("all", lambda r: True), ("cond(K) <= 100", lambda r: r["cond_max"] <= 100),
                            ("cond(K) > 100", lambda r: r["cond_max"] > 100)):
            xs = [max(r["methods"][n]["seg"]) / max(r["methods"]["recurrent"]["seg"]) for r in rows
                  if keep(r) and r["methods"].get(n, {}).get("seg") and r["methods"].get("recurrent", {}).get("seg")]
            if xs:
                print(f"- {n} / recurrent, worst segment error, {label}: {rng(xs, '.2f')} over {len(xs)} configurations")
    print()


def load(paths):
    """The parts of several gpu_bench.py results JSONs, merged. Part B's cells may be split over files (by N, say),
    but no cell may appear twice."""
    data = {}
    for path in paths:
        d = json.load(open(path))
        parts = [k for k in d if k.startswith("part_")]
        print(f"source: {path} (parts {', '.join(k[-1].upper() for k in parts)}"
              + (f"; N = {', '.join(str(n) for n in sorted({r['N'] for r in d['part_b']}))}" if d.get("part_b") else "")
              + f"; {sum(d.get('seconds', {}).values()) / 60:.0f} min)")
        print("environment: " + ", ".join(f"{k} {v}" for k, v in d.get("info", {}).items()))
        for k in parts:
            data.setdefault(k, []).extend(d[k])
    keys = [cell_key(r) for r in data.get("part_b", [])]
    if len(keys) != len(set(keys)):
        raise SystemExit("a part B cell is in more than one file; give a second run of part B with --rerun")
    print()
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+", help="gpu_bench.py results")
    p.add_argument("--rerun", nargs="+", default=[], help="a second run of part B (may be split over files)")
    p.add_argument("--graph-bwd", nargs="+", default=[], help="part B runs with forward + backward timed as CUDA graphs;"
                   " without it, the main part B's rows are used if they have such times")
    p.add_argument("--graph-bwd-log", help="the log of an earlier such run that died before saving its JSON")
    args = p.parse_args()
    data = load(args.json)
    if "part_a" in data:
        w = {}
        for r in data["part_a"]:
            for n, x in r["err"].items():
                w[n] = max(w.get(n, 0.0), *(v for v in x.values() if isinstance(v, float))) if "O" in x else float("nan")
        print("Part A, worst float64 error per method: " + ", ".join(f"{n} {v:.1e}" for n, v in w.items()))
        eq = [r["magma_bitwise_equal_chunk"] for r in data["part_a"] if "magma_bitwise_equal_chunk" in r]
        if eq:
            print(f"Part A, chunk_magma bitwise equal to chunk in {sum(eq)} of {len(eq)} cases")
        print()
    if data.get("part_b"):
        part_b(data["part_b"])
        if args.rerun:
            print("rerun of part B:")
            stability(data["part_b"], load(args.rerun)["part_b"])
        if args.graph_bwd:
            print("forward + backward as CUDA graphs:")
            graph_backward(data["part_b"], load(args.graph_bwd)["part_b"], args.graph_bwd_log)
        elif any(r.get("graph_bwd") for r in data["part_b"]):
            graph_backward(data["part_b"], data["part_b"])
    if data.get("part_c"):
        part_c(data["part_c"])
    if data.get("part_d"):
        part_d(data["part_d"])


if __name__ == "__main__":
    main()
