"""Study 0004: what the trained memories look like, without changing training.

snapshot(): for Kalman cores, on a small fixed batch, cond(K) per chunk (float64 eigenvalues of the whitened C x C
matrix, the one that is factored) and the learned gates, write strengths, process noise and ridge. numerics(): float32
against float64 on the trained model's own inputs: for Kalman cores the chunk form and the float32 token-by-token filter
against the float64 filter; for MesaNet fla's CG kernel against a dense float64 solve. Together they decide whether
backlog I-018 (accuracy on ill-conditioned chunks) matters for trained models.
"""
import torch
import torch.nn.functional as F

from kalman_layer import kalman_chunk, kalman_filter

N_SEQ = 8     # sequences per diagnostic batch


def _quant(t):
    t = t.flatten().float()
    if t.numel() > 200000:
        t = t[torch.randperm(t.numel(), device=t.device)[:200000]]
    q = torch.quantile(t, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=t.device))
    return [round(x, 6) for x in q.tolist()]


@torch.no_grad()
def _capture(model, make, device):
    """Run a fixed small batch with each core recording its inputs. Returns one dict per layer and the Kmats."""
    gen = torch.Generator(device=device).manual_seed(777)
    x, _, _ = make(N_SEQ, device, gen)
    cores = model.cores()
    Ks = [[] for _ in cores]
    was_training = model.training
    model.eval()
    for c, store in zip(cores, Ks):
        if hasattr(c, "capture"):
            c.capture = {}
        if hasattr(c, "on_chunk"):
            c.on_chunk = lambda Kmat, info, s=store: s.append((Kmat.double(), info))
    model(x)
    caps = [getattr(c, "capture", None) for c in cores]
    for c in cores:
        if hasattr(c, "capture"):
            c.capture = None
        if hasattr(c, "on_chunk"):
            c.on_chunk = None
    model.train(was_training)
    return caps, Ks


@torch.no_grad()
def snapshot(model, make, device):
    kind = model.blocks[0].mix.kind
    if kind == "gdn":
        return {}
    caps, Ks = _capture(model, make, device)
    out = {}
    for i, (cap, store) in enumerate(zip(caps, Ks)):
        if kind == "mesanet":
            out[f"L{i}"] = dict(gate=_quant(torch.exp(cap["g"])), beta=_quant(cap["beta"]), lam=_quant(cap["lamb"]))
            continue
        ev = torch.cat([torch.linalg.eigvalsh(K) for K, _ in store])  # (chunks * heads, C)
        cond = ev[:, -1] / ev[:, 0]
        info = dict(cond=_quant(cond), lam_min=ev[:, 0].min().item(), cholesky_info=sum(int((i2 != 0).sum()) for _, i2 in store))
        s2 = (cap["K"].norm(dim=-1)) ** 2                               # beta_t ||k_t||^2 = beta_t for unit keys
        info["beta"] = _quant(s2)
        if cap["logD"] is not None:
            info["gate"] = _quant(torch.exp(cap["logD"]))
            C = model.cores()[i].chunk
            info["min_chunk_logdecay"] = min(cap["logD"][:, s:s + C].sum(1).min().item()
                                             for s in range(0, cap["logD"].shape[1], C))
        if "lam" in cap and cap["lam"] is not None:
            info["lam"] = _quant(cap["lam"])
        if "psi" in cap and cap["psi"] is not None:
            info["psi"] = _quant(cap["psi"])
            info["prior"] = _quant(torch.diagonal(cap["P_init"], dim1=-2, dim2=-1))
        out[f"L{i}"] = info
    return out


def short(row):
    """A short text for the training log."""
    bits = []
    for name in ("L0", "L1"):
        info = row.get(name)
        if not info:
            continue
        if "cond" in info:
            bits.append(f"{name} cond max {info['cond'][-1]:.3g}")
        if "gate" in info:
            bits.append(f"gate med {info['gate'][2]:.3f}")
    fails = [row[n].get("cholesky_info", 0) for n in ("L0", "L1") if n in row]
    if any(fails):
        bits.append(f"CHOLESKY FAILURES {sum(fails)}")
    return (" | " + ", ".join(bits)) if bits else ""


def _rel(a, b):
    """max |a - b| / max |b|, the yardstick of study 0003."""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300)).item()


def _mesa_exact(cap):
    """MesaNet's exact readout by a dense float64 solve per token: H_t = g_t H_{t-1} + beta_t k k^T, likewise G_t."""
    q, k, v = (F.normalize(cap[n].double(), dim=-1) if n != "v" else cap[n].double() for n in ("q", "k", "v"))
    g, beta, lamb = cap["g"].double(), cap["beta"].double(), cap["lamb"].double()
    B, T, H, d = q.shape
    Hs, G = torch.zeros(B, H, d, d, dtype=torch.float64, device=q.device), torch.zeros(B, H, d, d, dtype=torch.float64, device=q.device)
    lamm = torch.diag_embed(lamb)[None]
    cond, out = 0.0, []
    for t in range(T):
        gt = torch.exp(g[:, t])[..., None, None]
        kb = k[:, t] * beta[:, t, :, None]
        Hs = gt * Hs + kb[..., None] * k[:, t][..., None, :]
        G = gt * G + kb[..., None] * v[:, t][..., None, :]
        x = torch.linalg.solve(Hs + lamm, q[:, t][..., None])
        out.append((x.mT @ G)[..., 0, :])
        if t % 16 == 15 or t == T - 1:
            ev = torch.linalg.eigvalsh(Hs + lamm)
            cond = max(cond, (ev[..., -1] / ev[..., 0]).max().item())
    return torch.stack(out, 1), cond


@torch.no_grad()
def numerics(model, make, device):
    kind = model.blocks[0].mix.kind
    if kind == "gdn":
        return {}
    caps, _ = _capture(model, make, device)
    out = {}
    for i, cap in enumerate(caps):
        if kind == "mesanet":
            exact, cond = _mesa_exact(cap)
            out[f"L{i}"] = dict(cg30_vs_exact=_rel(cap["o"], exact), cond_H_max=cond)
            continue
        double = {n: (None if cap.get(n) is None else cap[n].double()) for n in ("K", "V", "Q", "logD", "lam", "psi", "P_init")}
        ref, _, _ = kalman_filter(**double)
        single = {n: cap.get(n) for n in ("K", "V", "Q", "logD", "lam", "psi", "P_init")}
        O32, _, _ = kalman_filter(**single)
        c = model.cores()[i].chunk
        chunk64, _, _ = kalman_chunk(C=c, **double)
        out[f"L{i}"] = dict(chunk_f32_vs_filter_f64=_rel(cap["O"], ref), filter_f32_vs_filter_f64=_rel(O32, ref),
                            chunk_f64_vs_filter_f64=_rel(chunk64, ref))
    return out
