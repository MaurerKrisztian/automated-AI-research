"""Study 0004: one model frame with different memory cores, so that only the memory differs.

The frame is study 0001's MQAR model: embedding, blocks of RMSNorm + mixer + RMSNorm + 4x MLP, RMSNorm, linear head. In
each mixer, linear projections and a length-four causal convolution make q, k, v per head; q and k pass through SiLU and
are l2-normalized; the memory reads them; each head's output goes through RMSNorm and a linear output projection.

Cores, each with its own published gate and write parametrization:
- gdn: Gated DeltaNet, flash-linear-attention's Triton kernels with fla's layer settings: SiLU on v too, a_proj and
  b_proj without bias, log gate -exp(A_log) softplus(a(x) + dt_bias), beta = sigmoid(b(x)). The option output_gate=True
  adds fla's default output gate.
- mesanet: MesaNet with 30 CG steps, fla's kernels with fla's layer settings: no convolution or SiLU on v, a_proj and
  b_proj with bias, log gate logsigmoid(a(x)), beta = sigmoid(b(x)), lambda = softplus(.) + 0.25 per channel, starting
  at 1. Its gates start at half-lives 32-4096, as the Kalman cores' do, because from fla's start of about 0.5 it stayed at
  chance on MQAR and Volatile MQAR (NOTEBOOK.md, 2026-11-01); gate_init="fla" is that start.
- kalman: the Kalman memory by kalman_layer.kalman_chunk through autograd, with SiLU on v as in BL's GDN-like mixer.
  gates: "scalar" (one per head and token), "channel" (one per head, channel and token), or "identity". log gate
  logsigmoid(a(x)), starting at half-lives 32-4096 (BL, Appendix F.2). noise: "stationary" (Lambda as MesaNet's, tied
  noise) or "free" (psi = softplus(c(x)) + floor per head and token, starting at 0.1, with a 0.25 learning-rate
  multiplier on c's weights, and a learned prior variance per head starting at 1; BL, Appendix F.1). Write strength:
  beta_param="softplus" (BL: r^2 = softplus(b(x)) + floor, beta = 1/r^2, starting at r^2 = 1) or "sigmoid" (MesaNet's).

fla's kernels need a GPU. Everything is float32 with TF32 off.
"""
import math
import os

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # float32 tl.dot without TF32 in fla's kernels

import torch
import torch.nn as nn
import torch.nn.functional as F

from kalman_layer import kalman_chunk

R2_FLOOR = 1e-2     # floor of the observation variance r^2 (BL states a positive floor, not its value)
PSI_FLOOR = 1e-4    # floor of the process variance
LAMBDA_FLOOR = 0.25 # MesaNet's lower bound on the ridge


def inv_softplus(y):
    return math.log(math.expm1(y))


def halflife_logits(n):
    """Logits of gates whose half-lives are spaced log-uniformly over 32-4096 tokens (BL, Appendix F.2)."""
    h = torch.logspace(math.log10(32.0), math.log10(4096.0), n, dtype=torch.float64) if n > 1 else torch.tensor([256.0], dtype=torch.float64)
    return torch.logit(2.0 ** (-1.0 / h)).float()


def small_linear(d, n, bias=True, bias_value=None):
    """A gate or noise projection: weights N(0, 0.02^2), fla's initializer_range, and a given starting bias."""
    m = nn.Linear(d, n, bias=bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        if bias_value is None:
            nn.init.zeros_(m.bias)
        else:
            with torch.no_grad():
                m.bias.copy_(torch.as_tensor(bias_value, dtype=m.bias.dtype).expand_as(m.bias))
    return m


def no_decay(p):
    p._no_weight_decay = True
    return p


class GDNCore(nn.Module):
    v_activation = True

    def __init__(self, d, h, dh, output_gate=False, **_):
        super().__init__()
        self.a, self.b = small_linear(d, h, bias=False), small_linear(d, h, bias=False)
        A = torch.empty(h).uniform_(0, 16)
        self.A_log = no_decay(nn.Parameter(torch.log(A)))
        dt = torch.exp(torch.rand(h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = no_decay(nn.Parameter(dt + torch.log(-torch.expm1(-dt))))
        self.g_proj = nn.Linear(d, h * dh, bias=False) if output_gate else None

    def forward(self, x, q, k, v):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        o, _ = chunk_gated_delta_rule(q, k, v, self.a(x), self.b(x), use_qk_l2norm_in_kernel=True,
                                      use_beta_sigmoid_in_kernel=True, use_gate_in_kernel=True,
                                      A_log=self.A_log, dt_bias=self.dt_bias)
        return o


class MesaCore(nn.Module):
    v_activation = False

    def __init__(self, d, h, dh, gate_init="halflife", **_):
        super().__init__()
        self.h = h
        self.a = small_linear(d, h, bias_value=halflife_logits(h) if gate_init == "halflife" else None)
        self.b = small_linear(d, h)
        self.lam_raw = no_decay(nn.Parameter(torch.full((h * dh,), inv_softplus(1.0 - LAMBDA_FLOOR))))
        self.capture = None

    def forward(self, x, q, k, v):
        from fla.ops.mesa_net import chunk_mesa_net
        g = F.logsigmoid(self.a(x))
        beta = torch.sigmoid(self.b(x))
        lamb = (F.softplus(self.lam_raw) + LAMBDA_FLOOR).reshape(self.h, -1)
        o, _, _ = chunk_mesa_net(q, k, v, g, beta, lamb, max_CG_iteration=30, use_qk_l2norm_in_kernel=True)
        if self.capture is not None:
            self.capture.update(q=q, k=k, v=v, g=g, beta=beta, lamb=lamb, o=o)
        return o


class KalmanCore(nn.Module):
    v_activation = True

    def __init__(self, d, h, dh, gates="scalar", noise="stationary", beta_param="softplus", chunk=16, **_):
        super().__init__()
        self.h, self.dh, self.gates, self.noise, self.beta_param, self.chunk = h, dh, gates, noise, beta_param, chunk
        if gates == "scalar":
            self.a = small_linear(d, h, bias_value=halflife_logits(h))
        elif gates == "channel":
            self.a = small_linear(d, h * dh, bias_value=halflife_logits(h * dh))
        elif gates != "identity":
            raise ValueError(gates)
        self.b = small_linear(d, h, bias_value=inv_softplus(1.0 - R2_FLOOR) if beta_param == "softplus" else None)
        if noise == "free":
            self.c = small_linear(d, h, bias_value=inv_softplus(0.1 - PSI_FLOOR))
            self.c.weight._lr_mult = 0.25
            self.prior_raw = no_decay(nn.Parameter(torch.full((h,), inv_softplus(1.0))))
        elif noise == "stationary":
            self.lam_raw = no_decay(nn.Parameter(torch.full((h * dh,), inv_softplus(1.0 - LAMBDA_FLOOR))))
        else:
            raise ValueError(noise)
        self.on_chunk, self.capture = None, None
        self.register_buffer("chol_fail", torch.zeros((), dtype=torch.long), persistent=False)

    def _count(self, Kmat, info):
        self.chol_fail += (info != 0).sum()
        if self.on_chunk:
            self.on_chunk(Kmat, info)

    def inputs(self, x, q, k, v):
        """Whitened keys and values, log gates and noise, heads flattened into the batch: (B H, T, ...)."""
        B, T, H, dh = q.shape
        flat = lambda t: t.transpose(1, 2).reshape(B * H, T, *t.shape[3:])
        if self.beta_param == "softplus":
            s = (F.softplus(self.b(x)) + R2_FLOOR).pow(-0.5)
        else:
            s = torch.exp(0.5 * F.logsigmoid(self.b(x)))
        s = flat(s)
        K = F.normalize(flat(k), dim=-1) * s[..., None]
        V = flat(v) * s[..., None]
        Q = F.normalize(flat(q), dim=-1)
        logD = None
        if self.gates == "scalar":
            logD = flat(F.logsigmoid(self.a(x))).unsqueeze(-1)
        elif self.gates == "channel":
            logD = flat(F.logsigmoid(self.a(x)).reshape(B, T, H, dh))
        kw = dict(K=K, V=V, Q=Q, logD=logD)
        if self.noise == "stationary":
            kw["lam"] = (F.softplus(self.lam_raw) + LAMBDA_FLOOR).reshape(H, dh).repeat(B, 1)
        else:
            kw["psi"] = flat(F.softplus(self.c(x)) + PSI_FLOOR)
            prior = F.softplus(self.prior_raw).repeat(B)
            kw["P_init"] = prior[:, None, None] * torch.eye(dh, dtype=K.dtype, device=K.device)
        return kw

    def forward(self, x, q, k, v):
        B, T, H, dh = q.shape
        kw = self.inputs(x, q, k, v)
        O, _, _ = kalman_chunk(C=self.chunk, on_chunk=self._count, **kw)
        if self.capture is not None:
            self.capture.update(kw, O=O)
        return O.reshape(B, H, T, -1).transpose(1, 2)


CORES = dict(gdn=GDNCore, mesanet=MesaCore, kalman=KalmanCore)


class Mixer(nn.Module):
    def __init__(self, d, h, kind, **opts):
        super().__init__()
        self.h, self.dh, self.kind = h, d // h, kind
        self.core = CORES[kind](d, h, self.dh, **opts)
        self.conv_v = self.core.v_activation                          # fla's MesaNet has no convolution or SiLU on v
        channels = 3 * d if self.conv_v else 2 * d
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, groups=channels, padding=3, bias=False)
        self.onorm = nn.RMSNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        z = self.qkv(x)
        if self.conv_v:
            z = self.conv(z.transpose(1, 2))[..., :T].transpose(1, 2)
            q, k, v = z.split(D, dim=-1)
            v = F.silu(v)
        else:
            q, k, v = z.split(D, dim=-1)
            q, k = self.conv(torch.cat([q, k], -1).transpose(1, 2))[..., :T].transpose(1, 2).split(D, dim=-1)
        q, k = F.silu(q), F.silu(k)
        q, k, v = (t.reshape(B, T, self.h, self.dh) for t in (q, k, v))
        o = self.onorm(self.core(x, q, k, v))                          # (B, T, H, dh)
        if self.core.__class__ is GDNCore and self.core.g_proj is not None:
            o = o * F.silu(self.core.g_proj(x)).reshape(B, T, self.h, self.dh)  # fla's gated output norm, unfused
        return self.out(o.reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, h, kind, **opts):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = Mixer(d, h, kind, **opts)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class Net(nn.Module):
    def __init__(self, vocab, d, h, layers, kind, **opts):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, kind, **opts) for _ in range(layers)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    def cores(self):
        return [b.mix.core for b in self.blocks]


# The models of the study: name -> (kind, options)
MODELS = {
    "gdn": ("gdn", {}),
    "mesanet": ("mesanet", {}),
    "kalman_stat_scalar": ("kalman", dict(gates="scalar", noise="stationary")),
    "kalman_stat_channel": ("kalman", dict(gates="channel", noise="stationary")),
    "kalman_free_scalar": ("kalman", dict(gates="scalar", noise="free")),
    "kalman_free_channel": ("kalman", dict(gates="channel", noise="free")),
    "kalman_free_identity": ("kalman", dict(gates="identity", noise="free")),
}
# The alternative configuration each model may pick in the pilot, instead of its default: an equal tuning budget.
ALT = {
    "gdn": dict(output_gate=True),
    "mesanet": dict(gate_init="fla"),
    "kalman_stat_scalar": dict(beta_param="sigmoid"),
    "kalman_stat_channel": dict(beta_param="sigmoid"),
    "kalman_free_scalar": dict(beta_param="sigmoid"),
    "kalman_free_channel": dict(beta_param="sigmoid"),
    "kalman_free_identity": dict(beta_param="sigmoid"),
}
