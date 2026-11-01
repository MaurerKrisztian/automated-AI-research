#!/usr/bin/env python3
"""Study 0004: tables and decisions from the per-run JSONs in results/<grid>/. Reads, never writes, those files.

    ../../.venv/bin/python -u aggregate.py --pilot results/pilot --write-best results/best_config.json
    ../../.venv/bin/python -u aggregate.py --final results/final --replicate results/replicate 2>&1 | tee logs/aggregate.log

Pilot: per model and task, each configuration's recall, then the choice (highest recall, then lowest loss).
Final: recall as mean and min-max over seeds, per task and size, including sizes larger than trained on; the decision rule
of NOTEBOOK.md (a Kalman variant's lowest seed above each baseline's highest seed); stationary against free noise with
the same gates; cond(K), gates, write strengths; float32 against float64 on trained inputs.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

KALMAN = ["kalman_stat_scalar", "kalman_stat_channel", "kalman_free_scalar", "kalman_free_channel", "kalman_free_identity"]
BASE = ["gdn", "mesanet"]
ORDER = BASE + KALMAN
PAIRS = [("kalman_free_scalar", "kalman_stat_scalar"), ("kalman_free_channel", "kalman_stat_channel"),
         ("kalman_free_identity", "kalman_stat_scalar"), ("kalman_free_identity", "kalman_stat_channel")]


def load(folder):
    runs = []
    for path in sorted(Path(folder).glob("*.json")):
        r = json.loads(path.read_text())["run"]
        r["file"] = path.name
        runs.append(r)
    return runs


def fmt(values):
    """Mean (min-max) in percent. None marks a run that failed."""
    vals = [v * 100 for v in values if v is not None]
    if not vals:
        return "failed"
    s = f"{statistics.mean(vals):.1f} ({min(vals):.1f}–{max(vals):.1f})"
    if len(vals) < len(values):
        s += f" [{len(values) - len(vals)} failed]"
    return s


def pilot(args):
    runs = load(args.pilot)
    best, table = {}, defaultdict(list)
    for r in runs:
        if r.get("recall") is not None:
            table[(r["task"], r["model"])].append(r)
    print("Pilot: mean recall over the sizes evaluated, by configuration (tuning seed, not reused). 'alt' is the model's "
          "alternative configuration.")
    def score(r):
        """Mean recall over the sizes evaluated, so the choice also reflects the sizes where models differ."""
        vals = [r["recall"]] + [v for v in (r.get("extra_recall") or {}).values() if v is not None]
        return statistics.mean(vals)
    for (task, model), rs in sorted(table.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1]))):
        rs.sort(key=lambda r: (r["alt"], r["lr"]))
        pick = max(rs, key=lambda r: (round(score(r), 3), -r["final_loss"]))
        row = "  ".join(f"{'alt ' if r['alt'] else ''}{r['lr']:g}: {score(r) * 100:.1f}%" for r in rs)
        print(f"{task:<12} {model:<22} {row}  -> {'alt ' if pick['alt'] else ''}{pick['lr']:g}")
        best[f"{model}/{task}"] = dict(lr=pick["lr"], alt=pick["alt"], score=score(pick))
    if args.write_best:
        Path(args.write_best).write_text(json.dumps(best, indent=1))
        print(f"wrote {args.write_best}")
    return best


def by_cell(runs):
    cells = defaultdict(list)
    for r in runs:
        cells[(r["task"], r["model"])].append(r)
    return cells


def recall_values(rs, eval_label=None):
    if eval_label is None:
        return [r.get("recall") for r in rs]
    return [(r.get("extra_recall") or {}).get(eval_label) for r in rs]


def beats(a_vals, b_vals):
    a, b = [v for v in a_vals if v is not None], [v for v in b_vals if v is not None]
    return bool(a) and (not b or min(a) > max(b))


def final(args):
    runs = load(args.final)
    cells = by_cell(runs)
    tasks = sorted({t for t, _ in cells})
    print("Final runs: recall (%) as mean (min–max) over seeds. Every run is listed; none is dropped.\n")
    print("| Task | " + " | ".join(ORDER) + " |")
    print("|---" * (len(ORDER) + 1) + "|")
    labels = []
    for t in tasks:
        extra = sorted({k for m in ORDER for r in cells.get((t, m), []) for k in (r.get("extra_recall") or {})},
                       key=lambda s: int("".join(c for c in s if c.isdigit())))
        labels.append((t, None))
        labels.extend((t, e) for e in extra)
    for t, e in labels:
        name = t if e is None else f"{t} → {e}"
        vals = [fmt(recall_values(cells.get((t, m), []), e)) if cells.get((t, m)) else "–" for m in ORDER]
        print(f"| {name} | " + " | ".join(vals) + " |")

    print("\nDecision rule (NOTEBOOK.md): a Kalman variant's lowest seed above the highest seed of both baselines.")
    wins = []
    for t, e in labels:
        base = [v for m in BASE for v in recall_values(cells.get((t, m), []), e)]
        for m in KALMAN:
            vals = recall_values(cells.get((t, m), []), e)
            if vals and beats(vals, base):
                wins.append((t, e, m))
                print(f"  WIN {t if e is None else t + ' → ' + e}: {m} {fmt(vals)}; baselines "
                      + ", ".join(f"{b} {fmt(recall_values(cells.get((t, b), []), e))}" for b in BASE))
    if not wins:
        print("  none.")
    print("\nThe reverse: a baseline's lowest seed above every Kalman variant's highest seed.")
    rev = 0
    for t, e in labels:
        kal = [v for m in KALMAN for v in recall_values(cells.get((t, m), []), e)]
        for b in BASE:
            vals = recall_values(cells.get((t, b), []), e)
            if vals and beats(vals, kal):
                rev += 1
                print(f"  {t if e is None else t + ' → ' + e}: {b} {fmt(vals)}")
    if not rev:
        print("  none.")

    print("\nWhere each Kalman variant stands against each baseline, separately (lowest seed above highest seed):")
    for t, e in labels:
        row = []
        for m in KALMAN:
            for b in BASE:
                va, vb = recall_values(cells.get((t, m), []), e), recall_values(cells.get((t, b), []), e)
                if va and vb and beats(va, vb):
                    row.append(f"{m} > {b}")
                elif va and vb and beats(vb, va):
                    row.append(f"{b} > {m}")
        print(f"  {t if e is None else t + ' → ' + e}: " + ("; ".join(row) if row else "no separation"))

    print("\nStationary against free noise (H2), beyond the spread:")
    for t, e in labels:
        row = []
        for free, stat in PAIRS:
            va, vb = recall_values(cells.get((t, free), []), e), recall_values(cells.get((t, stat), []), e)
            if va and vb and beats(va, vb):
                row.append(f"{free} > {stat}")
            elif va and vb and beats(vb, va):
                row.append(f"{stat} > {free}")
        print(f"  {t if e is None else t + ' → ' + e}: " + ("; ".join(row) if row else "ranges overlap"))

    if any(r.get("recall_changed") is not None for r in runs):
        print("\nVolatile MQAR: recall of changed and unchanged keys, and stale answers among changed keys (%).")
        print("| Task | Model | unchanged keys | changed keys | stale rate |\n|---|---|---|---|---|")
        for t in tasks:
            for m in ORDER:
                rs = [r for r in cells.get((t, m), []) if r.get("recall_changed") is not None]
                if rs:
                    print(f"| {t} | {m} | {fmt([r['recall_same'] for r in rs])} | {fmt([r['recall_changed'] for r in rs])} "
                          f"| {fmt([r['stale_rate'] for r in rs])} |")

    print("\nConfiguration, parameters, cost: learning rate, alternative or default, parameters, steps run, ms per step, peak GB.")
    print("| Task | Model | lr | alt | params | steps run | ms/step | peak GB | non-finite steps | Cholesky failures |\n|---|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for m in ORDER:
            rs = cells.get((t, m), [])
            ok = [r for r in rs if r.get("params")]
            if ok:
                print(f"| {t} | {m} | {ok[0]['lr']:g} | {'yes' if ok[0]['alt'] else 'no'} | {ok[0]['params']} | "
                      f"{','.join(str(r['steps_run']) for r in ok)} | {statistics.mean(r['ms_per_step'] for r in ok):.1f} | "
                      f"{max(r['peak_gb'] for r in ok):.2f} | {sum(r['nonfinite_steps'] for r in ok)} | "
                      f"{sum(r['cholesky_failures'] for r in ok)} |")

    print("\nLearned memories at the end of training, worst over layers and seeds (Kalman: cond(K) of the matrix factored; "
          "gate, beta, noise as 5%/50%/95% over tokens).")
    print("| Task | Model | cond(K) max | min chunk log decay | gate 5/50/95 | beta 5/50/95 | psi 50 | lambda 50 |\n|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for m in ORDER[1:]:
            rs = cells.get((t, m), [])
            cond, decay, gates, betas, psi, lam = [], [], [], [], [], []
            for r in rs:
                last = (r.get("curve") or [{}])[-1]
                for L in ("L0", "L1"):
                    info = last.get(L)
                    if not info:
                        continue
                    if "cond" in info:
                        cond.append(info["cond"][-1])
                    if "min_chunk_logdecay" in info:
                        decay.append(info["min_chunk_logdecay"])
                    if "gate" in info:
                        gates.append(info["gate"][1:4])
                    if "beta" in info:
                        betas.append(info["beta"][1:4])
                    if "psi" in info:
                        psi.append(info["psi"][2])
                    if "lam" in info:
                        lam.append(info["lam"][2])
            if not rs or not (gates or betas):
                continue
            g = "/".join(f"{statistics.mean(x[i] for x in gates):.3f}" for i in range(3)) if gates else "–"
            b = "/".join(f"{statistics.mean(x[i] for x in betas):.3g}" for i in range(3)) if betas else "–"
            print(f"| {t} | {m} | {max(cond) if cond else float('nan'):.3g} | {min(decay) if decay else float('nan'):.1f} | {g} | {b} | "
                  f"{statistics.mean(psi) if psi else float('nan'):.3g} | {statistics.mean(lam) if lam else float('nan'):.3g} |")

    print("\nFloat32 on trained inputs (backlog I-018), worst over layers and seeds, relative to max |output|:")
    print("| Task | Model | chunk f32 | filter f32 | ratio | chunk f64 | MesaNet CG30 vs exact | cond(H+Λ) |\n|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for m in ORDER[1:]:
            rs = cells.get((t, m), [])
            vals = defaultdict(list)
            for r in rs:
                for L, info in (r.get("numerics") or {}).items():
                    for k, v in info.items():
                        vals[k].append(v)
            if not vals:
                continue
            if "cg30_vs_exact" in vals:
                print(f"| {t} | {m} | – | – | – | – | {max(vals['cg30_vs_exact']):.1e} | {max(vals['cond_H_max']):.3g} |")
            else:
                ratio = max(a / max(b, 1e-30) for a, b in zip(vals["chunk_f32_vs_filter_f64"], vals["filter_f32_vs_filter_f64"]))
                print(f"| {t} | {m} | {max(vals['chunk_f32_vs_filter_f64']):.1e} | {max(vals['filter_f32_vs_filter_f64']):.1e} | "
                      f"{ratio:.2f} | {max(vals['chunk_f64_vs_filter_f64']):.1e} | – | – |")

    if args.replicate:
        rep = by_cell(load(args.replicate))
        print("\nReplication on fresh seeds (3 other seeds, same configuration):")
        for (t, m), rs in sorted(rep.items()):
            print(f"  {t} {m}: {fmt(recall_values(rs))}")
            for e in sorted({k for r in rs for k in (r.get('extra_recall') or {})}):
                print(f"     → {e}: {fmt(recall_values(rs, e))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot")
    p.add_argument("--write-best")
    p.add_argument("--final")
    p.add_argument("--replicate")
    args = p.parse_args()
    if args.pilot:
        pilot(args)
    if args.final:
        final(args)


if __name__ == "__main__":
    main()
