#!/usr/bin/env python3
"""GPU benchmark (E3) of the chunk-parallel Kalman memory against the Bayesian Layer's chunk algorithm (backlog I-016).

Every Kalman method computes the same filter (stationary noise, unit keys, ridge 1, beta 0.5-2, gates 0.9-1, scalar or
per-channel) on the same float32 inputs with TF32 off, for N independent heads (batch x heads):
- chunk, chunk_magma, chunk_inv: this study's chunk-parallel form (kalman_batched.kalman_chunkwise): one batched C x C
  Cholesky per chunk (cuSOLVER, which PyTorch picks by default, or MAGMA) and triangular solves, or an explicit L^-1.
  PyTorch 2.11 has deprecated MAGMA and sends cholesky_ex to cuSOLVER even when MAGMA is preferred, so chunk_magma
  repeats chunk: part A prints whether their outputs are bitwise equal, and part B times chunk_magma only with --magma.
- bl_triton: the Bayesian Layer's chunk algorithm with its two sequential loops as Triton kernels (bl_triton.py: the
  gain loop per chunk, the v5 mean pass over the sequence), the strongest version of BL built here, since BL's own
  Triton code is not public. bl_triton_wy: the same gain kernel with v1's WY mean pass. Forward only.
- bl, bl_wy: the same algorithm in plain PyTorch (differentiable, so also timed forward + backward).
- recurrent: the token-by-token filter in plain PyTorch (one step per token).
References, different models at the same sizes: mesa_carry (study 0001's ungated carried-inverse Mesa layer with its
closed-form backward, lab.layers.mesa), mesanet_cg30 and mesanet_exact (MesaNet with 30 CG steps, and study 0002's exact
chunk solver with its Triton Jacobi eigh, both from 0002's plain-PyTorch gpu_bench.py; forward only, scalar gates), and
when flash-linear-attention is installed its Triton kernels: Gated DeltaNet (scalar gates), KDA (per-channel) and
MesaNet (30 CG steps, scalar).

Part A: float64 on the device, every Kalman method against the extended-precision token-by-token filter (CPU),
  including the Triton kernels compiled for this GPU.
Part B: the timing grid. Forward (no_grad) and forward + backward (autograd.grad of a random projection of O) medians
  with lab.bench, peak memory above the inputs, and the float32 output error against the float64 token-by-token
  filter on the device. With --graphs, the forward, and the forward + backward of the differentiable methods, are also
  timed as CUDA graph replays, which removes the CPU's kernel-launch overhead from every method. Methods whose cost
  does not depend on C are timed once per (N, d, gates).
Part C: where the time goes. CUDA events between the stages of the chunk form and of BL (Triton).
Part D: float32 drift on the GPU over long sequences: the ten configurations of check_float32.py, 4 heads each,
  d = e = C = 64, against the float64 token-by-token filter on the device: error per eighth of the sequence, the
  final covariance's error, and P's smallest-to-largest eigenvalue ratio over every chunk boundary.

Usage, from this folder on an NVIDIA GPU. run_gpu.sh runs the whole benchmark, writing fresh log names in logs/, and
then the tables. By hand, write the log to /tmp, never over a recorded log in logs/:
    ../../.venv/bin/python -u gpu_bench.py --quick 2>&1 | tee /tmp/gpu_bench_quick.log            # smoke test
    ../../.venv/bin/python -u gpu_bench.py --parts A B --N 16 2>&1 | tee /tmp/gpu_bench_n16.log   # GPU check, grid
    ../../.venv/bin/python -u gpu_bench.py --parts B --N 64 2>&1 | tee /tmp/gpu_bench_n64.log     # rest of the grid
    ../../.venv/bin/python -u gpu_bench.py --parts C D 2>&1 | tee /tmp/gpu_bench_cd.log           # stages, drift
The full benchmark is split into runs so that no single run is long, and part B runs twice, in separate processes,
to measure how stable the times are. Each run also saves a timestamped JSON in --results-dir (default results/).
--quick also runs on a CPU (tiny sizes, Triton methods only with TRITON_INTERPRET=1), as a test of the script.
"""
import argparse
import gc
import importlib.util
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # float32 tl.dot without TF32 in fla's kernels; TF32 is off here

import torch

import check_batched
import check_float32
import kalman_batched as kb
from kalman_memory import max_cond, rel, stationary_noise
from lab import bench, save_results, seed_everything, sync

HERE = Path(__file__).resolve().parent
OOM = (torch.cuda.OutOfMemoryError, MemoryError)
# Methods timed as CUDA graphs too. MAGMA synchronizes with the host, so it cannot be captured, and fla's kernels
# are a few fused launches, which a graph would not change much. mesa_carry is left out because lab.layers.mesa calls
# torch.linalg.cholesky, whose error check synchronizes with the host. Never attempt a capture that will fail: in
# PyTorch 2.11 a failed capture leaves the caching allocator routing later allocations into the dead graph's private
# pool, which empty_cache() cannot free, so reserved memory grows until every method runs out of it.
GRAPHABLE = {"chunk", "chunk_inv", "bl_triton", "bl_triton_wy", "bl", "bl_wy", "recurrent", "mesanet_cg30"}
_CAPTURE_STREAM = []    # one side stream for every capture: each new stream would get its own cuBLAS workspace


# --------------------------------------------------------------------------- inputs
def make_inputs(seed, N, T, d, e, channel, gates=(0.9, 1.0), beta=(0.5, 2.0), lam=1.0, device="cpu",
                dtype=torch.float32):
    """Unit keys, queries N(0, I/d), values N(0, I), log-uniform beta, uniform gates, ridge lam; drawn in float64 on
    the CPU, so every device and precision sees the same numbers."""
    g = torch.Generator().manual_seed(seed)
    f64 = dict(dtype=torch.float64)
    K = torch.randn(N, T, d, generator=g, **f64)
    K = K / K.norm(dim=-1, keepdim=True)
    Q = torch.randn(N, T, d, generator=g, **f64) / math.sqrt(d)
    V = torch.randn(N, T, e, generator=g, **f64)
    logD = torch.empty(N, T, d if channel else 1, **f64).uniform_(*gates, generator=g).log()
    b = torch.empty(N, T, **f64).uniform_(math.log(beta[0]), math.log(beta[1]), generator=g).exp()
    x = dict(K=K, V=V, Q=Q, logD=logD, beta=b, lam=torch.full((N, d), lam, **f64))
    return {k: v.to(device=device, dtype=dtype) for k, v in x.items()}


def err(o, ref):
    """max|o - ref| / max|ref| (the study's yardstick), in float64 on the device."""
    o, ref = o.double(), ref.double()
    return ((o - ref).abs().max() / ref.abs().max()).item()


# --------------------------------------------------------------------------- methods
def triton_ok(device):
    return torch.device(device).type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"


def with_linalg(backend, fn):
    """Run fn with torch.backends.cuda.preferred_linalg_library(backend), restoring the previous choice."""
    def run(*a, **k):
        prev = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend)
        try:
            return fn(*a, **k)
        finally:
            torch.backends.cuda.preferred_linalg_library(prev)
    return run


def kalman_methods(device):
    """name -> (fn(x, C) -> (O, M_T, P_T), differentiable, depends on C). Same model, so errors are comparable."""
    m = {"chunk": (lambda x, C: kb.kalman_chunkwise(C=C, **x), True, True)}
    if torch.device(device).type == "cuda":
        m["chunk_magma"] = (with_linalg("magma", lambda x, C: kb.kalman_chunkwise(C=C, **x)), True, True)
    m["chunk_inv"] = (lambda x, C: kb.kalman_chunkwise(C=C, inverse="explicit", **x), True, True)
    if triton_ok(device):
        m["bl_triton"] = (lambda x, C: kb.bl_chunkwise(C=C, gains="triton", mean="triton", **x), False, True)
        m["bl_triton_wy"] = (lambda x, C: kb.bl_chunkwise(C=C, gains="triton", mean="wy", **x), False, True)
    m["bl"] = (lambda x, C: kb.bl_chunkwise(C=C, **x), True, True)
    m["bl_wy"] = (lambda x, C: kb.bl_chunkwise(C=C, mean="wy", **x), True, True)
    m["recurrent"] = (lambda x, C: kb.kalman_recurrent(**x), True, False)
    return m


def load_0002():
    """Study 0002's plain-PyTorch MesaNet CG (its gpu_bench.py), loaded under another module name."""
    path = HERE.parent / "0002-gated-mesa-exact"
    if str(path) not in sys.path:
        sys.path.append(str(path))
    spec = importlib.util.spec_from_file_location("gpu_bench_0002", path / "gpu_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_fla():
    try:
        import fla
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from fla.ops.kda import chunk_kda
        from fla.ops.mesa_net import chunk_mesa_net
    except Exception as exc:  # not installed, or no GPU
        return dict(error=repr(exc)[:200])
    return dict(version=getattr(fla, "__version__", "unknown"), gdn=chunk_gated_delta_rule, kda=chunk_kda,
                mesa=chunk_mesa_net)


def reference_methods(device, fla, g2):
    """name -> (prepare(x) -> inputs, fn(inputs, C) -> output, differentiable, depends on C, gate types it takes).
    Different models from the Kalman memory, timed at the same sizes; prepare puts the inputs in each method's own
    layout, outside the timed region."""
    m = {}
    from lab.layers.mesa import mesa_carry

    def prep_carry(x):
        N = x["K"].shape[0]
        return dict(q=x["Q"][None], k=x["K"][None], v=x["V"][None], beta=x["beta"][None],
                    lam=torch.ones(N, dtype=x["K"].dtype, device=x["K"].device))
    m["mesa_carry"] = (prep_carry, lambda z, C: mesa_carry(z["q"], z["k"], z["v"], z["beta"], z["lam"], C), True, True,
                       ("scalar", "channel"))
    if g2 is not None:
        def prep_cg(x):
            return dict(q=x["Q"], k=x["K"], v=x["V"], beta=x["beta"], g=x["logD"][..., 0],
                        lam=torch.ones(x["K"].shape[-1], dtype=x["K"].dtype, device=x["K"].device))

        def cg30(z, C):
            H0 = g2.start_states(z["k"], z["beta"], z["g"], C)
            X = g2.cg_solve(z["q"], H0, z["k"], z["beta"], z["g"], z["lam"], C, 30)
            return g2.readout(X, z["k"], z["v"], z["beta"], z["g"], C)

        eigh = "jacobi" if torch.device(device).type == "cuda" else "default"   # 0002's fastest backend on the GPU

        def exact(z, C):
            H0 = g2.start_states(z["k"], z["beta"], z["g"], C)
            X = g2.exact_solve(z["q"], z["k"], z["beta"], z["g"], z["lam"], C, *g2.exact_eig(H0, z["lam"], eigh))
            return g2.readout(X, z["k"], z["v"], z["beta"], z["g"], C)
        m["mesanet_cg30"] = (prep_cg, cg30, False, True, ("scalar",))
        m["mesanet_exact"] = (prep_cg, exact, False, True, ("scalar",))
    if "error" not in fla and torch.device(device).type == "cuda":
        def bt(t):
            return t.transpose(0, 1).unsqueeze(0).contiguous()   # (N, T, ...) -> (1, T, N, ...)

        def prep_fla(x):
            N, T, d = x["K"].shape
            sig = torch.sigmoid(torch.randn(N, T, generator=torch.Generator().manual_seed(7))).to(x["K"])
            return dict(q=bt(x["Q"]), k=bt(x["K"]), v=bt(x["V"]), g1=bt(x["logD"][..., 0]),
                        gd=bt(x["logD"].expand(N, T, d)), b_delta=bt(sig), b_mesa=bt(x["beta"]),
                        lamb=torch.ones(N, d, dtype=x["K"].dtype, device=x["K"].device))
        m["fla_gdn"] = (prep_fla, lambda z, C: fla["gdn"](z["q"], z["k"], z["v"], z["g1"], z["b_delta"])[0], True,
                        False, ("scalar",))
        m["fla_kda"] = (prep_fla, lambda z, C: fla["kda"](z["q"], z["k"], z["v"], z["gd"], z["b_delta"])[0], True,
                        False, ("channel",))
        m["fla_mesanet"] = (prep_fla, lambda z, C: fla["mesa"](z["q"], z["k"], z["v"], z["g1"], z["b_mesa"], z["lamb"],
                                                              max_CG_iteration=30)[0], True, False, ("scalar",))
    return m


# --------------------------------------------------------------------------- timing helpers
def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timeit(fn, device, warmup, reps):
    """lab.bench median in ms. Two untimed calls first (compilation, caches); a method slower than 0.5 s per call
    gets 3 repeats instead of `reps`."""
    fn()
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    if time.perf_counter() - t0 > 0.5:
        return bench(fn, device, warmup=0, reps=3)
    return bench(fn, device, warmup=warmup, reps=reps)


def peak_extra_gb(fn, device):
    """Peak memory allocated during one call of fn, above what was allocated before it, in GiB (nan on CPU)."""
    if torch.device(device).type != "cuda":
        fn()
        return float("nan")
    sync(device)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync(device)
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def graph_ms(fn, device, warmup, reps):
    """Median replay time of fn captured as a CUDA graph (static inputs), in ms."""
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(torch.cuda.Stream())
    s = _CAPTURE_STREAM[0]
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = bench(g.replay, device, warmup=warmup, reps=reps)
    del g
    return t


def first(out):
    """The readouts O of a method that returns (O, M_T, P_T) or O alone."""
    return out[0] if isinstance(out, tuple) else out


def grad_inputs(x):
    """Leaves for the backward pass: every input of the method, the ridge and the gates included."""
    return {k: v.detach().clone().requires_grad_() for k, v in x.items()}


def fwd_bwd(fn, leaves, w):
    """One training step's worth: forward, then the gradients of <O, w> with respect to every leaf."""
    def step():
        O = fn()
        torch.autograd.grad((O * w).sum(), list(leaves.values()), allow_unused=True)
    return step


class Marks:
    """Records a CUDA event (or a host time on the CPU) at each stage boundary; totals() sums ms per stage."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.names, self.stamps = [], []
        self("start")

    def __call__(self, name):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.stamps.append(ev)
        else:
            self.stamps.append(time.perf_counter())
        self.names.append(name)

    def totals(self):
        if self.cuda:
            torch.cuda.synchronize()
            dt = [a.elapsed_time(b) for a, b in zip(self.stamps[:-1], self.stamps[1:])]
        else:
            dt = [(b - a) * 1e3 for a, b in zip(self.stamps[:-1], self.stamps[1:])]
        out = defaultdict(float)
        for name, t in zip(self.names[1:], dt):
            out[name] += t
        return dict(out)


# --------------------------------------------------------------------------- part A
def part_a(device):
    print(f"Part A. float64 on {device}: every Kalman method against the extended-precision token-by-token filter (CPU).")
    print("Errors are max|x - x_ref| / max|x_ref| over all heads: outputs O, then the final M_T and P_T.")
    cases = [("scalar gates 0.9-1, T = 300, d = 32, C = 64", dict(seed=1, T=300, d=32, e=16, gates=(0.9, 1.0)), 64),
             ("per-channel gates 0.5-1, T = 200, d = 32, C = 32", dict(seed=2, T=200, d=32, e=16, per_channel=True), 32),
             ("per-channel gates 0.9-1, d = 128 > C = 16", dict(seed=3, T=96, d=128, e=64, N=2, gates=(0.9, 1.0),
                                                                per_channel=True), 16),
             ("scalar gates 0.99-1, T = 260 (padded), C = 128", dict(seed=4, T=260, d=64, e=64, N=2, gates=(0.99, 1.0)),
              128),
             ("free diagonal noise, dense P_init, M_0 != 0", dict(seed=5, T=70, d=16, e=8, noise="diag",
                                                                 per_channel=True, prior_mean=True), 16)]
    rows = []
    for name, kw, C in cases:
        c = check_batched.make_case(**kw)
        o_ref, M_ref, P_ref = check_batched.reference(c)
        x = {k: v.to(device) for k, v in c.items()}
        r = dict(case=name, C=C, err={})
        outs = {}
        for label, (fn, _, _) in kalman_methods(device).items():
            try:
                O, M, P = outs[label] = fn(x, C)
                r["err"][label] = dict(O=rel(O, o_ref), M_T=rel(M, M_ref), P_T=rel(P, P_ref))
            except Exception as exc:
                r["err"][label] = dict(error=repr(exc)[:200])
        if "chunk" in outs and "chunk_magma" in outs:
            r["magma_bitwise_equal_chunk"] = all(torch.equal(a, b) for a, b in zip(outs["chunk"], outs["chunk_magma"]))
        rows.append(r)
        print(f"{name}")
        for label, x_ in r["err"].items():
            print(f"  {label:>13}: " + (f"O {x_['O']:.1e}   M_T {x_['M_T']:.1e}   P_T {x_['P_T']:.1e}"
                                        if "O" in x_ else x_["error"]))
        if "magma_bitwise_equal_chunk" in r:
            print(f"  chunk_magma's O, M_T and P_T bitwise equal to chunk's: {r['magma_bitwise_equal_chunk']}")
    worst = max((v for r in rows for x_ in r["err"].values() for v in x_.values() if isinstance(v, float)), default=0)
    print(f"worst error over all cases and methods: {worst:.1e}"
          + ("" if worst < 1e-12 else "   WARNING: above 1e-12, check the port before trusting the timings"))
    print()
    return rows


# --------------------------------------------------------------------------- part B
def est_backward_gb(name, N, T, d, e, C):
    """Rough memory that autograd keeps for the loop-based methods, to skip runs that cannot fit."""
    f = 4 / 2**30
    if name == "recurrent":
        return N * T * (3 * d * d + 3 * e * d) * f
    if name == "bl":
        return N * T * (3 * e * d + C * d) * f
    if name == "bl_wy":
        return N * T * C * d * f
    return 0.0


def part_b(args, device, fla, g2):
    dev_name = torch.cuda.get_device_name() if torch.device(device).type == "cuda" else device
    print(f"Part B. Timing on {dev_name}: float32, TF32 off, T = {args.T}, e = d; gates uniform on {args.gate_lo}-1;"
          f" unit keys, ridge 1, beta 0.5-2. Medians in ms (lab.bench; warmup {args.warmup}, reps {args.reps}, or 3 reps"
          f" for calls over 0.5 s). err: float32 output error against the float64 token-by-token filter on the device."
          f" mem: peak GiB above the inputs, forward (and forward + backward). Memory cap {args.mem_cap:.1f} GiB.")
    rows = []
    kms = kalman_methods(device)
    refs = reference_methods(device, fla, g2)
    for N in args.N:
        for d in args.d:
            e = d
            for gates in args.gates:
                channel = gates == "channel"
                x64 = make_inputs(100 + N + d, N, args.T, d, e, channel, (args.gate_lo, 1.0), device=device,
                                  dtype=torch.float64)
                with torch.no_grad():
                    o_ref = kb.kalman_recurrent(**x64)[0]
                x = {k: v.float() for k, v in x64.items()}
                del x64
                clean()
                once = {}
                for C in args.C:
                    if args.T % C:
                        continue
                    t_cell = time.time()
                    cell = dict(N=N, T=args.T, d=d, e=e, C=C, gates=gates, fwd={}, fwd_bwd={}, graph={}, graph_bwd={},
                                mem={}, mem_bwd={}, err={}, notes={})
                    jobs = [(n, fn, (lambda z: z), dif, dep, ("scalar", "channel")) for n, (fn, dif, dep) in kms.items()
                            if n != "chunk_magma" or args.magma]
                    jobs += [(n, fn, prep, dif, dep, kinds) for n, (prep, fn, dif, dep, kinds) in refs.items()]
                    for name, fn, prep, dif, dep, kinds in jobs:
                        if (args.methods and name not in args.methods) or gates not in kinds:
                            continue
                        if not dep and name in once:
                            for key, val in once[name].items():
                                cell[key][name] = val
                            continue
                        res, notes = {}, []
                        run_one(res, notes, name, fn, prep, dif, x, o_ref, C, name in kms, args, device)
                        clean()
                        if notes:
                            res["notes"] = "; ".join(notes)
                        for key, val in res.items():
                            cell[key][name] = val
                        if not dep:
                            once[name] = res
                    cell["seconds"] = round(time.time() - t_cell, 1)
                    rows.append(cell)
                    print_cell(cell)
                del x, o_ref
                clean()
    return rows


def run_one(res, notes, name, fn, prep, dif, x, o_ref, C, same_model, args, device):
    """Time one method in one cell, filling res (fwd, graph, fwd_bwd, mem, mem_bwd, err) and notes. Every tensor it
    makes is local, so an out-of-memory error frees it all on the way out."""
    try:
        z = prep(x)
        with torch.no_grad():
            O = first(fn(z, C))
            if same_model:
                res["err"] = err(O, o_ref)
            res["fwd"] = timeit(lambda: fn(z, C), device, args.warmup, args.reps)
            res["mem"] = peak_extra_gb(lambda: fn(z, C), device)
            if args.graphs and torch.device(device).type == "cuda" and name in GRAPHABLE:
                try:
                    res["graph"] = graph_ms(lambda: fn(z, C), device, args.warmup, args.reps)
                except Exception as exc:
                    notes.append(f"graph capture failed: {repr(exc)[:120]}")
                    print(f"WARNING: the CUDA-graph capture of {name} failed; memory and times after this point in this"
                          " process are suspect (see GRAPHABLE)")
                    torch.cuda.synchronize()
        if dif and not args.no_backward:
            N, T, d = x["K"].shape
            est = est_backward_gb(name, N, T, d, x["V"].shape[-1], C)
            if est > 1.5 * args.mem_cap:
                notes.append(f"backward skipped: autograd would keep about {est:.0f} GiB")
            else:
                leaves = grad_inputs(z)
                w = torch.randn(O.shape, generator=torch.Generator().manual_seed(3)).to(O)
                step = fwd_bwd(lambda: first(fn(leaves, C)), leaves, w)
                res["fwd_bwd"] = timeit(step, device, args.warmup, args.reps)
                res["mem_bwd"] = peak_extra_gb(step, device)
                if args.graphs and torch.device(device).type == "cuda" and name in GRAPHABLE:
                    # The loops over positions are launch-bound eager, in the backward as in the forward. A capture
                    # that ran out of memory would leave the allocator broken (see GRAPHABLE), so large ones are
                    # skipped.
                    if res["mem_bwd"] > 0.4 * args.mem_cap:
                        notes.append(f"graph forward + backward skipped: it needs {res['mem_bwd']:.1f} GiB")
                    else:
                        try:
                            res["graph_bwd"] = graph_ms(step, device, args.warmup, args.reps)
                        except Exception as exc:
                            notes.append(f"graph capture of forward + backward failed: {repr(exc)[:120]}")
                            print(f"WARNING: the CUDA-graph capture of {name}'s forward + backward failed; memory and"
                                  " times after this point in this process are suspect")
                            torch.cuda.synchronize()
    except OOM:
        notes.append("out of memory")
    except Exception as exc:
        notes.append(f"error: {repr(exc)[:160]}")


def print_cell(cell):
    print(f"\n  N={cell['N']} d={cell['d']} C={cell['C']} {cell['gates']} gates  ({cell['seconds']:.0f} s)")
    names = list(dict.fromkeys(list(cell["fwd"]) + list(cell["notes"])))
    for n in names:
        f = cell["fwd"].get(n)
        s = f"    {n:>13}  fwd {f:9.2f} ms" if f is not None else f"    {n:>13}  fwd       n/a   "
        if n in cell["graph"]:
            s += f"  graph {cell['graph'][n]:9.2f}"
        if n in cell["fwd_bwd"]:
            s += f"  fwd+bwd {cell['fwd_bwd'][n]:9.2f} ms"
        if n in cell["graph_bwd"]:
            s += f"  graph fwd+bwd {cell['graph_bwd'][n]:9.2f}"
        if n in cell["err"]:
            s += f"  err {cell['err'][n]:.1e}"
        if n in cell["mem"]:
            s += f"  mem {cell['mem'][n]:.2f}" + (f" / {cell['mem_bwd'][n]:.2f}" if n in cell["mem_bwd"] else "")
        if n in cell["notes"]:
            s += f"  [{cell['notes'][n]}]"
        print(s)


# --------------------------------------------------------------------------- part C
def part_c(args, device):
    print("\nPart C. Stage times (CUDA events between stages; medians over 5 runs, ms), float32, per-channel gates.")
    rows = []
    for N in args.N:
        for d in args.d:
            C = 64 if 64 in args.C else args.C[0]
            x = make_inputs(200 + N + d, N, args.T, d, d, True, (args.gate_lo, 1.0), device=device)
            runs = [("chunk", lambda mk: kb.kalman_chunkwise(C=C, mark=mk, **x))]
            if triton_ok(device):
                runs.append(("bl_triton", lambda mk: kb.bl_chunkwise(C=C, gains="triton", mean="triton", mark=mk, **x)))
            for name, fn in runs:
                with torch.no_grad():
                    fn(None)
                    samples = []
                    for _ in range(5):
                        mk = Marks(device)
                        fn(mk)
                        samples.append(mk.totals())
                med = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
                total = sum(med.values())
                rows.append(dict(method=name, N=N, d=d, C=C, T=args.T, stage_ms=med))
                print(f"  {name:>9} N={N} d={d} C={C}: total {total:.2f} ms; "
                      + ", ".join(f"{k} {v:.2f} ({100 * v / total:.0f}%)" for k, v in med.items()))
            del x
            clean()
    return rows


# --------------------------------------------------------------------------- part D
def part_d(args, device):
    Tmax = args.drift_T
    print(f"\nPart D. float32 drift on {device}: the configurations of check_float32.py, {args.drift_heads} heads each,"
          f" d = e = C = 64, sequences up to {Tmax} tokens, against the float64 token-by-token filter on the device.")
    print("Per eighth of the sequence: max|o32 - o64| / max|o64| over the heads. P: final error; smallest/largest"
          " eigenvalue of P at the worst chunk boundary (float64 eigenvalues).")
    C = 64
    rows = []
    methods = [("chunk", lambda x, on: kb.kalman_chunkwise(C=C, on_state=on, **x)),
               ("chunk_inv", lambda x, on: kb.kalman_chunkwise(C=C, inverse="explicit", on_state=on, **x))]
    if triton_ok(device):
        methods.append(("bl_triton", lambda x, on: kb.bl_chunkwise(C=C, gains="triton", mean="triton", on_state=on, **x)))
    # bl and bl_wy share their PyTorch gain loop and differ only in the mean pass, which isolates its share of the drift.
    methods += [("bl", lambda x, on: kb.bl_chunkwise(C=C, on_state=on, **x)),
                ("bl_wy", lambda x, on: kb.bl_chunkwise(C=C, mean="wy", on_state=on, **x)),
                ("recurrent", lambda x, on: kb.kalman_recurrent(on_state=on, state_every=C, **x))]
    for i, (name, T, gates, pc, beta, lam, keys) in enumerate(check_float32.CONFIGS):
        if args.drift_only and args.drift_only not in name:
            continue
        T = min(T, Tmax)
        t0 = time.time()
        x64 = drift_inputs(1000 + i, args.drift_heads, T, gates, pc, beta, lam, keys, device)
        with torch.no_grad():
            o64, _, P64 = kb.kalman_recurrent(**x64)
        cond = max(max_cond(*single_head(x64, n), C) for n in range(min(args.cond_heads, args.drift_heads)))
        x32 = {k: v.float() for k, v in x64.items()}
        res = dict(config=name, T=T, heads=args.drift_heads, cond_max=cond, methods={})
        print(f"{name}  (T = {T}; largest cond(K) over the chunks of {min(args.cond_heads, args.drift_heads)} heads,"
              f" float64: {cond:.2e})")
        for label, fn in methods:
            Ps = []
            try:
                with torch.no_grad():
                    O, _, P = fn(x32, lambda P_: Ps.append(P_.detach().double().cpu()))
                finite = bool(torch.isfinite(O).all())
                seg = check_float32.seg_errors(O.transpose(0, 1).reshape(T, -1), o64.transpose(0, 1).reshape(T, -1)) \
                    if finite else []
                ratios = eig_ratios(Ps)
                r = dict(status="ok" if finite else "non-finite output", seg=seg, P_err=err(P, P64),
                         P_eig_ratio_min_over_boundaries=min(ratios) if ratios else float("nan"))
            except Exception as exc:
                r = dict(status=f"error: {repr(exc)[:100]}", seg=[], P_err=float("nan"),
                         P_eig_ratio_min_over_boundaries=float("nan"))
            res["methods"][label] = r
            s = " ".join(f"{v:.1e}" for v in r["seg"]) if r["seg"] else r["status"]
            print(f"  {label:>10}: {s}   P err {r['P_err']:.1e}, eig min/max (worst boundary)"
                  f" {r['P_eig_ratio_min_over_boundaries']:.1e}")
            del Ps
            clean()
        print(f"  ({time.time() - t0:.1f} s)")
        rows.append(res)
        del x64, x32, o64, P64
        clean()
    return rows


def drift_inputs(seed, H, T, gates, pc, beta, lam, keys, device):
    """H heads drawn as check_float32.make draws one (heads h > 0 from seeds seed + 7919 h). A scalar gate is kept
    as one column, so the chunk form takes the same Hadamard path as in part B."""
    heads = [check_float32.make(seed + 7919 * h, T, gates, pc, beta, lam, keys, torch.float64) for h in range(H)]
    K, V, Q, logD, Psi, b, M0, P0 = (torch.stack([hd[j] for hd in heads]) for j in range(8))
    logD = logD if pc else logD[..., :1]
    return dict(K=K.to(device), V=V.to(device), Q=Q.to(device), logD=logD.to(device), beta=b.to(device),
                lam=(1 / torch.diagonal(P0, dim1=-2, dim2=-1)).to(device))


def single_head(x, n):
    """Arguments of kalman_memory's single-head functions for head n (on the CPU, float64)."""
    K, V, Q, logD, b, lam = (x[k][n].double().cpu() for k in ("K", "V", "Q", "logD", "beta", "lam"))
    logD = logD.expand(K.shape[0], K.shape[1])
    return K, V, Q, logD, stationary_noise(logD, lam), b, torch.zeros(V.shape[1], K.shape[1], dtype=K.dtype), \
        torch.diag(1 / lam)


def eig_ratios(Ps):
    """Smallest over largest eigenvalue of the symmetric part of each boundary covariance (all heads)."""
    if not Ps:
        return []
    P = torch.cat(Ps)
    if not bool(torch.isfinite(P).all()):
        return [float("nan")]
    ev = torch.linalg.eigvalsh(0.5 * (P + P.mT))
    return (ev[:, 0] / ev[:, -1]).tolist()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test at tiny sizes (also on a CPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--N", type=int, nargs="+", default=[16, 64], help="independent heads (batch x heads)")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--d", type=int, nargs="+", default=[64, 128], help="key dimension; e = d")
    p.add_argument("--C", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--gates", nargs="+", default=["scalar", "channel"])
    p.add_argument("--gate-lo", type=float, default=0.9)
    p.add_argument("--methods", nargs="*", default=[], help="restrict part B to these methods")
    p.add_argument("--magma", action="store_true", help="also time chunk_magma in part B (see the module docstring)")
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True, help="CUDA-graph forward timing")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--drift-T", type=int, default=65536)
    p.add_argument("--drift-heads", type=int, default=4)
    p.add_argument("--drift-only", help="run only the part D configurations whose name contains this")
    p.add_argument("--cond-heads", type=int, default=1, help="heads whose cond(K) part D computes (CPU, float64)")
    p.add_argument("--mem-cap", type=float, default=14.0, help="GiB; PyTorch raises out-of-memory above it")
    p.add_argument("--results-dir", default=str(HERE / "results"), help="where the JSON goes (smoke tests: elsewhere)")
    args = p.parse_args()
    if args.quick:
        args.N, args.T, args.d, args.C, args.reps, args.warmup = [2], 256, [32], [16, 32], 3, 1
        args.drift_T, args.drift_heads = 1024, 2
    dev = torch.device(args.device)
    seed_everything(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = dict(torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                triton_f32_default=os.environ.get("TRITON_F32_DEFAULT"))
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap / total))
        info.update(gpu=torch.cuda.get_device_name(), gpu_mem_gib=total, sms=torch.cuda.get_device_properties(dev).multi_processor_count)
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    fla = load_fla() if dev.type == "cuda" else dict(error="no GPU")
    info["fla"] = fla.get("version", fla.get("error"))
    try:
        g2 = load_0002()
    except Exception as exc:
        g2 = None
        info["study_0002"] = f"not loaded: {exc!r}"[:200]
    print("Environment: " + ", ".join(f"{k} {v}" for k, v in info.items()))
    print(f"Kalman methods: {list(kalman_methods(args.device))}; references: {list(reference_methods(args.device, fla, g2))}\n")
    out = dict(args=vars(args), info=info, seconds={})
    t0 = time.time()
    parts = dict(A=lambda: part_a(args.device), B=lambda: part_b(args, args.device, fla, g2),
                 C=lambda: part_c(args, args.device), D=lambda: part_d(args, args.device))
    for part, run in parts.items():
        if part in args.parts:
            t = time.time()
            out[f"part_{part.lower()}"] = run()
            out["seconds"][part] = round(time.time() - t, 1)
            print(f"(part {part}: {out['seconds'][part]:.0f} s)")
    if triton_ok(args.device):
        import bl_triton
        out["triton_autotune"] = bl_triton.tuned_configs()
        if out["triton_autotune"]:
            print("\nTriton autotuning (kernel, shape key: chosen config):")
            for k, v in out["triton_autotune"].items():
                print(f"  {k}: {v}")
    print(f"\ntotal {time.time() - t0:.0f} s")
    path = save_results(out, args.results_dir, "gpu_bench_quick" if args.quick else "gpu_bench")
    print(f"saved {path.relative_to(HERE) if path.resolve().is_relative_to(HERE) else path}")


if __name__ == "__main__":
    main()
