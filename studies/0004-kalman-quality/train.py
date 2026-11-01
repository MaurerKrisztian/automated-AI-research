#!/usr/bin/env python3
"""Study 0004: train models on the tasks and save one JSON per run, right after it finishes, so a run that dies loses
only itself. A run whose tag already has a results file is skipped, so the same command resumes a grid.

    ../../.venv/bin/python -u train.py --grid pilot --models gdn mesanet --tasks mqar64 --lrs 1e-3 2e-3 4e-3 --seeds 100 \
        --steps-scale 0.5 2>&1 | tee logs/<name>.log

Every model sees the same training stream for a given seed and task, and the same evaluation batches. Runs stop early
once the evaluation recall reaches --target. Recorded per evaluation: step, loss, recall, the task's breakdown, and for
Kalman cores cond(K) with the learned gates, write strengths, noise and ridge. Recorded at the end: float32 against
float64 errors of the memory on the trained model's inputs, and recall at sizes larger than trained on.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import diagnostics
from lab import peak_memory_gb, reset_peak_memory, save_results
from models import ALT, MODELS, Net
from tasks import TASKS

HERE = Path(__file__).resolve().parent


def param_groups(model, lr, weight_decay):
    """AdamW groups: parameters marked _no_weight_decay (ridge, prior, fla's A_log and dt_bias) skip weight decay, as in
    fla's layers, and BL's noise-projection weights take a learning-rate multiplier."""
    groups = {}
    for p in model.parameters():
        key = (getattr(p, "_lr_mult", 1.0), getattr(p, "_no_weight_decay", False))
        groups.setdefault(key, []).append(p)
    return [dict(params=ps, lr=lr * mult, weight_decay=0.0 if nd else weight_decay) for (mult, nd), ps in groups.items()]


def query_stats(pred, y, info):
    """Recall, and on Volatile MQAR recall of keys whose value did or did not change, and the rate of stale answers."""
    m = y != -100
    out = dict(correct=(pred[m] == y[m]).sum().item(), total=m.sum().item())
    if "changed" in info:
        B = y.shape[0]
        p, t = pred[m].reshape(B, -1), y[m].reshape(B, -1)
        hit, changed = p == t, info["changed"]
        stale = (info["older"] == p[:, :, None]).any(-1) & changed & ~hit
        out.update(same_correct=(hit & ~changed).sum().item(), same_total=(~changed).sum().item(),
                   changed_correct=(hit & changed).sum().item(), changed_total=changed.sum().item(),
                   stale=stale.sum().item())
    return out


@torch.no_grad()
def evaluate(model, make, B, batches, device):
    """Recall on a fixed evaluation stream, the same for every model."""
    gen = torch.Generator(device=device).manual_seed(12345)
    total = {}
    was_training = model.training
    model.eval()
    for _ in range(batches):
        x, y, info = make(B, device, gen)
        pred = model(x).argmax(-1)
        for k, v in query_stats(pred, y, info).items():
            total[k] = total.get(k, 0) + v
    model.train(was_training)
    out = dict(recall=total["correct"] / total["total"])
    if "same_total" in total:
        out.update(recall_same=total["same_correct"] / max(total["same_total"], 1),
                   recall_changed=total["changed_correct"] / max(total["changed_total"], 1),
                   stale_rate=total["stale"] / max(total["changed_total"], 1))
    return out


def train_one(model_name, task_name, lr, seed, opts, args, device):
    task = TASKS[task_name]
    kind, base = MODELS[model_name]
    kw = {**base, **opts}
    if kind == "kalman":
        kw["chunk"] = args.chunk_channel if kw.get("gates") == "channel" else args.chunk
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    model = Net(task.vocab, args.d, args.heads, args.layers, kind, **kw).to(device)
    steps = max(1, int(task.steps * args.steps_scale))
    opt = torch.optim.AdamW(param_groups(model, lr, 0.1), betas=(0.9, 0.98))
    warm = max(1, steps // 10)                                         # study 0001's schedule
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    params = sum(p.numel() for p in model.parameters())
    reset_peak_memory(device)
    make = task.batch
    curve, losses, times, nonfinite, stopped_at = [], [], [], 0, None
    for step in range(1, steps + 1):
        x, y, _ = task.train_batch(task.batch_size, device, gen)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        else:
            nonfinite += 1                                             # a skipped step, counted; never silent
            opt.zero_grad(set_to_none=True)
        sched.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if nonfinite >= 50:
            stopped_at = step
            print(f"  {model_name} {task_name}: 50 non-finite gradients, stopped at step {step}", flush=True)
            break
        if step % args.eval_every == 0 or step == steps:
            ev = evaluate(model, make, task.batch_size, args.eval_batches, device)
            row = dict(step=step, loss=statistics.mean(losses[-50:]), grad_norm=gn.item(), **ev)
            row.update(diagnostics.snapshot(model, make, device))
            curve.append(row)
            print(f"  {model_name:>21} {task_name:>9} lr {lr:g} s{seed} step {step:5d} loss {row['loss']:.3f} "
                  f"recall {ev['recall'] * 100:6.2f}% {statistics.median(times[5:] or times) * 1e3:6.1f} ms/step"
                  + diagnostics.short(row), flush=True)
            if ev["recall"] >= min(args.target, task.target):
                stopped_at = step
                break
    final = curve[-1] if curve else {}
    extras = {}
    if args.extra_evals:
        for label, mk, B in task.extra_evals:
            try:
                extras[label] = evaluate(model, mk, B, 4, device)["recall"]
            except (torch.cuda.OutOfMemoryError, MemoryError):
                extras[label] = None
                torch.cuda.empty_cache()
    numerics = diagnostics.numerics(model, make, device) if not args.no_numerics else {}
    fails = sum(int(c.chol_fail) for c in model.cores() if hasattr(c, "chol_fail"))
    return dict(model=model_name, task=task_name, lr=lr, seed=seed, opts=opts, core_options=kw, d=args.d, heads=args.heads,
                layers=args.layers, params=params, steps_limit=steps, steps_run=step, stopped_early_at=stopped_at,
                batch=task.batch_size, recall=final.get("recall"), recall_same=final.get("recall_same"),
                recall_changed=final.get("recall_changed"), stale_rate=final.get("stale_rate"),
                final_loss=final.get("loss"), ms_per_step=statistics.median(times[5:] or times) * 1e3,
                peak_gb=peak_memory_gb(device), nonfinite_steps=nonfinite, cholesky_failures=fails,
                extra_recall=extras, numerics=numerics, curve=curve)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="smoke", help="results go to results/<grid>/")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--tasks", nargs="+", default=["mqar64"])
    p.add_argument("--lrs", type=float, nargs="+", default=[2e-3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--alt", action="store_true", help="use each model's alternative configuration (models.ALT)")
    p.add_argument("--best-from", default=None, help="a JSON made by aggregate.py --write-best: model/task -> lr, alt")
    p.add_argument("--opts", default="{}", help="extra core options as JSON")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--chunk", type=int, default=64, help="chunk size of Kalman cores with scalar or identity gates")
    p.add_argument("--chunk-channel", type=int, default=32, help="chunk size of Kalman cores with per-channel gates")
    p.add_argument("--steps-scale", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--target", type=float, default=2.0, help="stop early at this evaluation recall, if the task's own target is not lower")
    p.add_argument("--extra-evals", action="store_true", help="recall at sizes larger than trained on, at the end")
    p.add_argument("--no-numerics", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="smoke test: 40 steps")
    args = p.parse_args()
    if args.quick:
        args.steps_scale, args.eval_every, args.eval_batches = 40 / 3000, 20, 2
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = HERE / "results" / args.grid
    out.mkdir(parents=True, exist_ok=True)
    best = json.loads(Path(args.best_from).read_text()) if args.best_from else None
    extra = json.loads(args.opts)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'} | "
          f"grid {args.grid} | d {args.d}, {args.heads} heads, {args.layers} layers | TF32 off", flush=True)
    t_all = time.time()
    for task in args.tasks:
        for model in args.models:
            for seed in args.seeds:
                for lr in ([None] if best else args.lrs):
                    opts = dict(extra)
                    if best:
                        choice = best[f"{model}/{task}"]
                        lr, use_alt = choice["lr"], choice["alt"]
                    else:
                        use_alt = args.alt
                    if use_alt:
                        opts.update(ALT[model])
                    tag = f"{task}_{model}_lr{lr:g}_s{seed}" + ("_alt" if use_alt else "")
                    if any(out.glob(tag + "_2*.json")):
                        print(f"skip {tag}: results exist", flush=True)
                        continue
                    t0 = time.time()
                    try:
                        res = train_one(model, task, lr, seed, opts, args, device)
                    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                        print(f"OOM {tag}: {str(exc)[:200]}", flush=True)
                        torch.cuda.empty_cache()
                        res = dict(model=model, task=task, lr=lr, seed=seed, opts=opts, failed="out of memory")
                    res["alt"] = use_alt
                    res["seconds"] = time.time() - t0
                    path = save_results({"args": vars(args), "run": res}, out, tag)
                    print(f"{tag}: recall {res.get('recall')} in {res['seconds']:.0f} s, {res.get('steps_run')} steps, "
                          f"{res.get('ms_per_step') or 0:.1f} ms/step, {res.get('params')} params, "
                          f"{res.get('peak_gb') or 0:.2f} GB; saved {path.relative_to(HERE)}", flush=True)
                    torch.cuda.empty_cache()
    print(f"grid {args.grid}: finished in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
