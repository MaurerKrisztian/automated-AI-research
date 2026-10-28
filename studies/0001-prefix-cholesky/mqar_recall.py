#!/usr/bin/env python3
"""
MQAR recall test (multi-query associative recall).

Each sequence lists N random key-value pairs, then asks for N of the keys again;
the model must output each key's value. Tiny 2-layer models with different memory
layers are trained from scratch on the GPU:

  attn        softmax attention (upper bound; its memory grows with the sequence)
  linear      linear attention          (fixed-size memory, additive writes)
  deltanet    DeltaNet                  (fixed-size memory, one gradient step per token)
  mesa_exact  exact least-squares memory, computed with the exact chunk solver
  mesa_fast   the same memory: carried inverse + closed-form backward (the improved version)
  mesa_cgK    the same memory computed with K conjugate-gradient rounds (MesaNet style)

Reports recall accuracy and training time per step.

  python mqar_recall.py --quick          # smoke test, ~1 minute
  python mqar_recall.py                  # full grid, see --pairs / --kinds
"""
import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== memory kernels
def linear_attn(q, k, v, beta):
    T = q.shape[2]
    A = (q @ k.transpose(-1, -2)) * beta.unsqueeze(-2)
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    return A.masked_fill(~mask, 0.0) @ v


def deltanet_chunk(q, k, v, beta, C):
    """Exact chunkwise DeltaNet (UT transform, Yang et al. 2024)."""
    B, H, T, dk = k.shape
    M = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for s in range(0, T, C):
        Q, K, V, b = q[:, :, s:s + C], k[:, :, s:s + C], v[:, :, s:s + C], beta[:, :, s:s + C]
        c = K.shape[2]
        A = torch.eye(c, device=q.device, dtype=q.dtype) + torch.tril(
            b.unsqueeze(-1) * (K @ K.transpose(-1, -2)), diagonal=-1)
        Tm = torch.linalg.solve_triangular(A, torch.diag_embed(b), upper=False, unitriangular=True)
        Un = Tm @ V - (Tm @ K) @ M
        outs.append(Q @ M + torch.tril(Q @ K.transpose(-1, -2)) @ Un)
        M = M + K.transpose(-1, -2) @ Un
    return torch.cat(outs, dim=2)


def nested_solve(Bm, K, Q):
    """x_tau = (Bm + K[:tau]^T K[:tau])^{-1} q_tau for all tau: one Cholesky per chunk."""
    c = K.shape[-2]
    LB = torch.linalg.cholesky(Bm)
    Sol = torch.cholesky_solve(torch.cat([K, Q], dim=-2).transpose(-1, -2), LB)
    BiKt, BiQt = Sol[..., :c], Sol[..., c:]
    A = K @ BiKt + torch.eye(c, device=K.device, dtype=K.dtype)
    L = torch.linalg.cholesky(A)
    W = torch.triu(torch.linalg.solve_triangular(L, K @ BiQt, upper=False))
    Z = torch.linalg.solve_triangular(L.transpose(-1, -2), W, upper=True)
    return (BiQt - BiKt @ Z).transpose(-1, -2)


def cg_nested(Bm, K, Q, iters):
    """Same systems solved MesaNet-style: Jacobi start + `iters` CG rounds."""
    def mv(P):
        S = torch.triu(K @ P.transpose(-1, -2))
        return P @ Bm + S.transpose(-1, -2) @ K
    diag = torch.diagonal(Bm, dim1=-2, dim2=-1).unsqueeze(-2) + torch.cumsum(K * K, dim=-2)
    X = Q / diag
    R = Q - mv(X)
    P = R
    rr = (R * R).sum(-1, keepdim=True)
    for _ in range(iters):
        AP = mv(P)
        alpha = rr / (P * AP).sum(-1, keepdim=True).clamp_min(1e-20)
        X = X + alpha * P
        R = R - alpha * AP
        rr_new = (R * R).sum(-1, keepdim=True)
        P = R + (rr_new / rr.clamp_min(1e-20)) * P
        rr = rr_new
    return X


def mesa(q, k, v, beta, lam, C, cg_iters=None):
    """Mesa layer without forgetting: o_t = G_t (lam I + H_t)^{-1} q_t."""
    B, H, T, dk = k.shape
    Hs = lam.view(1, H, 1, 1) * torch.eye(dk, device=q.device, dtype=q.dtype).expand(B, H, dk, dk)
    G = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for s in range(0, T, C):
        Q, K, V, b = q[:, :, s:s + C], k[:, :, s:s + C], v[:, :, s:s + C], beta[:, :, s:s + C]
        Kh = K * b.unsqueeze(-1).sqrt()
        X = nested_solve(Hs, Kh, Q) if cg_iters is None else cg_nested(Hs, Kh, Q, cg_iters)
        S = torch.tril((X @ K.transpose(-1, -2)) * b.unsqueeze(-2))   # [tau, i] = (x_tau . k_i) beta_i
        outs.append(X @ G + S @ V)
        Hs = Hs + Kh.transpose(-1, -2) @ Kh
        G = G + (K * b.unsqueeze(-1)).transpose(-1, -2) @ V
    return torch.cat(outs, dim=2)


class _CarrySolve(torch.autograd.Function):
    """x_t = (lam I + sum_{i<=t} kh_i kh_i^T)^{-1} q_t for every t, chunked inputs (B, H, N, C, d), lam (H,).

    Forward carries the inverse P across chunks (one C x C Cholesky per chunk, no d x d factorization).
    Backward is closed-form and runs on all chunks at once:
        dq_t = g_t = P_t dx_t,   dk_i = -sum_{t>=i} (g_t x_t^T + x_t g_t^T) k_i,   dlam = -sum_t g_t . x_t
    """

    @staticmethod
    def forward(ctx, Kh, Q, lam):
        B, H, N, C, d = Kh.shape
        eye = torch.eye(d, device=Kh.device, dtype=Kh.dtype)
        eyeC = torch.eye(C, device=Kh.device, dtype=Kh.dtype)
        P = (eye / lam.view(1, H, 1, 1)).expand(B, H, d, d).contiguous()
        Ps, Us, Xs = [], [], []
        for n in range(N):
            Kn, Qn = Kh[:, :, n], Q[:, :, n]
            KP = Kn @ P
            L = torch.linalg.cholesky(KP @ Kn.transpose(-1, -2) + eyeC)
            Ut = torch.linalg.solve_triangular(L, KP, upper=False)
            Xs.append(Qn @ P - torch.tril(Qn @ Ut.transpose(-1, -2)) @ Ut)
            Ps.append(P)
            Us.append(Ut)
            P = P - Ut.transpose(-1, -2) @ Ut
        X = torch.stack(Xs, dim=2)
        ctx.save_for_backward(Kh, torch.stack(Ps, dim=2), torch.stack(Us, dim=2), X)
        return X

    @staticmethod
    def backward(ctx, dX):
        Kh, Ps, Us, X = ctx.saved_tensors
        G = dX @ Ps - torch.tril(dX @ Us.transpose(-1, -2)) @ Us
        T = G.transpose(-1, -2) @ X
        T = T + T.transpose(-1, -2)
        S_next = torch.flip(torch.cumsum(torch.flip(T, [2]), 2), [2]) - T
        dK = (-(Kh @ S_next) - torch.triu(Kh @ X.transpose(-1, -2)) @ G
              - torch.triu(Kh @ G.transpose(-1, -2)) @ X)
        return dK, G, -(G * X).sum(dim=(0, 2, 3, 4))


def mesa_fast(q, k, v, beta, lam, C):
    """Same layer as mesa(): carried inverse + closed-form backward; value path for all chunks at once."""
    B, H, T, dk = k.shape
    pad = (-T) % C
    if pad:  # zero keys/values/queries after the end change nothing for earlier tokens
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
    N = (T + pad) // C
    Kc, Vc, Qc = (t.reshape(B, H, N, C, t.shape[-1]) for t in (k, v, q))
    b = beta.reshape(B, H, N, C)
    X = _CarrySolve.apply(Kc * b.unsqueeze(-1).sqrt(), Qc, lam)
    Gc = (Kc * b.unsqueeze(-1)).transpose(-1, -2) @ Vc        # each chunk's sum of beta k v^T
    G0 = torch.cumsum(Gc, dim=2) - Gc                          # memory at each chunk start
    S = torch.tril((X @ Kc.transpose(-1, -2)) * b.unsqueeze(-2))
    return (X @ G0 + S @ Vc).reshape(B, H, N * C, -1)[:, :, :T]


# ------------------------------------------------------------ token-level references
def deltanet_reference(q, k, v, beta):
    B, H, T, dk = k.shape
    M = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        pred = (kt.unsqueeze(-2) @ M).squeeze(-2)
        M = M + kt.unsqueeze(-1) * (bt * (vt - pred)).unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ M).squeeze(-2))
    return torch.stack(outs, dim=2)


def mesa_reference(q, k, v, beta, lam):
    B, H, T, dk = k.shape
    Hs = lam.view(1, H, 1, 1) * torch.eye(dk, device=q.device, dtype=q.dtype).expand(B, H, dk, dk)
    G = q.new_zeros(B, H, dk, v.shape[-1])
    outs = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None, None]
        Hs = Hs + bt * kt.unsqueeze(-1) * kt.unsqueeze(-2)
        G = G + bt * kt.unsqueeze(-1) * vt.unsqueeze(-2)
        x = torch.linalg.solve(Hs, q[:, :, t])
        outs.append((x.unsqueeze(-2) @ G).squeeze(-2))
    return torch.stack(outs, dim=2)


def self_test(device):
    g = torch.Generator(device=device).manual_seed(0)
    B, H, T, d, C = 2, 2, 50, 16, 16
    q, k, v = (torch.randn(B, H, T, d, generator=g, device=device, dtype=torch.float64) for _ in range(3))
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    beta = torch.rand(B, H, T, generator=g, device=device, dtype=torch.float64)
    lam = torch.tensor([0.5, 1.0], device=device, dtype=torch.float64)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    e1 = rel(deltanet_chunk(q, k, v, beta, C), deltanet_reference(q, k, v, beta))
    ref = mesa_reference(q, k, v, beta, lam)
    e2 = rel(mesa(q, k, v, beta, lam, C), ref)
    e3 = rel(mesa(q, k, v, beta, lam, C, cg_iters=60), ref)
    e4 = rel(mesa_fast(q, k, v, beta, lam, C), ref)
    # closed-form backward of mesa_fast vs autograd through mesa (all inputs, including the ridge)
    R = torch.randn(B, H, T, d, generator=g, device=device, dtype=torch.float64)
    grads = []
    for f in (lambda *a: mesa(*a, C), lambda *a: mesa_fast(*a, C)):
        ins = [t.clone().requires_grad_(True) for t in (q, k, v, beta, lam)]
        (f(*ins) * R).sum().backward()
        grads.append([t.grad for t in ins])
    e5 = max(rel(a, b) for a, b in zip(grads[1], grads[0]))
    print(f"self-test (fp64): DeltaNet chunk vs token-by-token {e1:.1e} | "
          f"Mesa exact chunk vs token-by-token {e2:.1e} | Mesa CG-60 {e3:.1e} | "
          f"Mesa fast {e4:.1e}, its gradients {e5:.1e}", flush=True)
    assert e1 < 1e-10 and e2 < 1e-10 and e3 < 1e-6 and e4 < 1e-10 and e5 < 1e-8, "self-test failed"


# =============================================================== model
class Mixer(nn.Module):
    def __init__(self, d, h, kind, chunk):
        super().__init__()
        self.h, self.dh, self.kind, self.chunk = h, d // h, kind, chunk
        self.cg_iters = int(kind[len("mesa_cg"):]) if kind.startswith("mesa_cg") else None
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, padding=3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if kind != "attn":
            self.beta = nn.Linear(d, h)
            self.onorm = nn.RMSNorm(self.dh)
        if kind.startswith("mesa"):
            self.lam_raw = nn.Parameter(torch.full((h,), 0.5413))  # ridge = 0.25 + softplus(.) ~ 1.25

    def forward(self, x):
        Bsz, T, D = x.shape
        z = self.conv(self.qkv(x).transpose(1, 2))[..., :T].transpose(1, 2)  # causal short conv
        q, k, v = (t.reshape(Bsz, T, self.h, self.dh).transpose(1, 2) for t in z.split(D, dim=-1))
        if self.kind == "attn":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            q = F.normalize(F.silu(q), dim=-1)
            k = F.normalize(F.silu(k), dim=-1)
            beta = torch.sigmoid(self.beta(x)).transpose(1, 2)
            if self.kind == "linear":
                o = linear_attn(q, k, v, beta)
            elif self.kind == "deltanet":
                o = deltanet_chunk(q, k, v, beta, self.chunk)
            else:
                lam = 0.25 + F.softplus(self.lam_raw)
                if self.kind == "mesa_fast":
                    o = mesa_fast(q, k, v, beta, lam, self.chunk)
                else:
                    o = mesa(q, k, v, beta, lam, self.chunk, self.cg_iters)
            o = self.onorm(o)
        return self.out(o.transpose(1, 2).reshape(Bsz, T, D))


class Block(nn.Module):
    def __init__(self, d, h, kind, chunk):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = Mixer(d, h, kind, chunk)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class Net(nn.Module):
    def __init__(self, vocab, d, h, layers, kind, chunk):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, kind, chunk) for _ in range(layers)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


# =============================================================== data
def mqar_batch(B, n_pairs, vocab, device, gen):
    half = vocab // 2
    keys = torch.argsort(torch.rand(B, half - 1, device=device, generator=gen), dim=1)[:, :n_pairs] + 1
    vals = torch.randint(half, vocab, (B, n_pairs), device=device, generator=gen)
    kv = torch.stack([keys, vals], dim=2).reshape(B, 2 * n_pairs)
    idx = torch.randint(0, n_pairs, (B, n_pairs), device=device, generator=gen)
    x = torch.cat([kv, torch.gather(keys, 1, idx)], dim=1)
    y = torch.full_like(x, -100)
    y[:, 2 * n_pairs:] = torch.gather(vals, 1, idx)
    return x, y


@torch.no_grad()
def evaluate(model, n_pairs, args, device):
    gen = torch.Generator(device=device).manual_seed(12345)
    correct = total = 0
    model.eval()
    for _ in range(args.eval_batches):
        x, y = mqar_batch(args.batch, n_pairs, args.vocab, device, gen)
        pred = model(x).argmax(-1)
        m = y != -100
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


# =============================================================== training
def train_one(kind, n_pairs, args, device):
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    model = Net(args.vocab, args.d, args.heads, args.layers, kind, args.chunk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.98))
    warm = max(1, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    times, losses = [], []
    for step in range(1, args.steps + 1):
        x, y = mqar_batch(args.batch, n_pairs, args.vocab, device, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps:
            acc = evaluate(model, n_pairs, args, device)
            print(f"  {kind:>10} N={n_pairs:3d} step {step:5d}  loss {statistics.mean(losses[-50:]):.3f}"
                  f"  recall {acc * 100:5.1f}%  {statistics.median(times[5:]) * 1e3:6.1f} ms/step", flush=True)
    return {"kind": kind, "pairs": n_pairs, "seq_len": 3 * n_pairs, "recall": acc,
            "final_loss": statistics.mean(losses[-50:]),
            "ms_per_step": statistics.median(times[5:]) * 1e3,
            "params": sum(p.numel() for p in model.parameters())}


def speed_probe(kind, n_pairs, args, device, steps=25):
    """Time per training step only (no convergence), for expensive CG settings."""
    model = Net(args.vocab, args.d, args.heads, args.layers, kind, args.chunk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(0)
    times = []
    for _ in range(steps):
        x, y = mqar_batch(args.batch, n_pairs, args.vocab, device, gen)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - t0)
    return statistics.median(times[5:]) * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kinds", nargs="+", default=["attn", "linear", "deltanet", "mesa_exact", "mesa_cg5"])
    p.add_argument("--pairs", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--speed-kinds", nargs="+", default=["mesa_exact", "mesa_cg15", "mesa_cg30"])
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/mqar")
    args = p.parse_args()
    if args.quick:
        args.steps, args.log_every, args.pairs, args.eval_batches = 60, 30, [32], 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | {name} | d_model {args.d}, {args.heads} heads "
          f"(head dim {args.d // args.heads}), {args.layers} layers, chunk {args.chunk}", flush=True)
    self_test(device)

    results = {"args": vars(args), "device": name, "runs": [], "speed": []}
    for n in args.pairs:
        for kind in args.kinds:
            results["runs"].append(train_one(kind, n, args, device))
    n_speed = max(args.pairs)
    for kind in args.speed_kinds:
        ms = speed_probe(kind, n_speed, args, device)
        results["speed"].append({"kind": kind, "pairs": n_speed, "ms_per_step": ms})
        print(f"speed probe {kind:>10} N={n_speed}: {ms:6.1f} ms per training step", flush=True)

    # summary table
    lines = ["| Memory layer | " + " | ".join(f"N={n} recall" for n in args.pairs) + " | ms/step (largest N) |",
             "|---" * (len(args.pairs) + 2) + "|"]
    for kind in args.kinds:
        row = [r for r in results["runs"] if r["kind"] == kind]
        accs = " | ".join(f"{r['recall'] * 100:.1f}%" for r in row)
        lines.append(f"| {kind} | {accs} | {row[-1]['ms_per_step']:.1f} |")
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.out, f"mqar_{stamp}.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(args.out, f"mqar_{stamp}.md"), "w") as f:
        f.write(summary + "\n")
    print(f"saved {args.out}/mqar_{stamp}.json", flush=True)


if __name__ == "__main__":
    main()
