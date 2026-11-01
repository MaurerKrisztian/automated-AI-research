"""Study 0004: the tasks. Each batch is (x, y, info): input tokens (B, T), targets (B, T) with -100 where nothing is
scored, and a dict of tensors the evaluation uses for its breakdowns, each aligned with the scored positions.

- MQAR: study 0001's generator (N key-value pairs, then N queries of those keys; T = 3N).
- Volatile MQAR (Voltic, Hejabi et al. 2026, Section 4.2 and Appendix E.1): three rounds that each visit all K keys in a
  fresh random order, as (key, value) token pairs. Before rounds 2 and 3 each key's value changes with probability p to
  another value, so older pairs are stale. Then K queries in a fresh order, each followed by a random filler token.
  T = 8K. The target is the key's final value, scored at the query's own position, without shifting. p = 0 is Voltic's
  Standard condition, the control.
- Selective copying (the MAD suite, Poli et al. 2024; its generator mad/data/instances.py): 16 data tokens with
  replacement from 14 symbols, blanks inserted before them at random up to length L - n - 1, a copy token, then n blanks
  as answer slots. Vocabulary 16 in all, L = 256, n = 16. The target at the answer slots is the data tokens in order.
"""
import torch


def mqar_batch(B, n_pairs, vocab, device, gen):
    """Study 0001's generator (studies/0001-prefix-cholesky/mqar_recall.py); check_layers.py compares the two."""
    half = vocab // 2
    keys = torch.argsort(torch.rand(B, half - 1, device=device, generator=gen), dim=1)[:, :n_pairs] + 1
    vals = torch.randint(half, vocab, (B, n_pairs), device=device, generator=gen)
    kv = torch.stack([keys, vals], dim=2).reshape(B, 2 * n_pairs)
    idx = torch.randint(0, n_pairs, (B, n_pairs), device=device, generator=gen)
    x = torch.cat([kv, torch.gather(keys, 1, idx)], dim=1)
    y = torch.full_like(x, -100)
    y[:, 2 * n_pairs:] = torch.gather(vals, 1, idx)
    return x, y, {}


def volatile_batch(B, K, p, vocab, device, gen):
    """info: changed (B, K) says whether the queried key's value ever changed; older (B, K, 2) holds the values it
    showed in rounds 1 and 2, the stale answers."""
    half = vocab // 2
    rand = lambda *shape: torch.rand(*shape, device=device, generator=gen)
    keys = torch.argsort(rand(B, half - 1), dim=1)[:, :K] + 1
    vals = torch.argsort(rand(B, vocab - half), dim=1)[:, :K] + half    # initial values, without replacement
    rounds, shown = [], []
    for r in range(3):
        if r:
            change = rand(B, K) < p
            # A change draws uniformly from the other values: an offset in 1 .. vocab - half - 1, modulo the value range.
            off = torch.randint(1, vocab - half, (B, K), device=device, generator=gen)
            vals = torch.where(change, (vals - half + off) % (vocab - half) + half, vals)
        order = torch.argsort(rand(B, K), dim=1)
        rounds.append(torch.stack([torch.gather(keys, 1, order), torch.gather(vals, 1, order)], 2).reshape(B, 2 * K))
        shown.append(vals)
    qorder = torch.argsort(rand(B, K), dim=1)
    filler = torch.randint(0, vocab, (B, K), device=device, generator=gen)
    query = torch.stack([torch.gather(keys, 1, qorder), filler], 2).reshape(B, 2 * K)
    x = torch.cat(rounds + [query], 1)
    y = torch.full_like(x, -100)
    final = torch.gather(shown[2], 1, qorder)
    y[:, 6 * K::2] = final
    older = torch.stack([torch.gather(shown[0], 1, qorder), torch.gather(shown[1], 1, qorder)], 2)
    return x, y, dict(changed=(older != final[:, :, None]).any(-1), older=older)


def selcopy_batch(B, L, n, vocab, device, gen):
    """Token ids as in MAD: 0..vocab-3 are symbols, vocab-2 the blank, vocab-1 the copy token."""
    blank, copy = vocab - 2, vocab - 1
    data = torch.randint(0, vocab - 2, (B, n), device=device, generator=gen)
    m = L - n - 1                                                       # data tokens + blanks before the copy token
    n_blank = m - n
    # Each blank goes before one of the n data tokens, uniformly and independently, as numpy.insert does in MAD.
    gaps = torch.randint(0, n, (B, n_blank), device=device, generator=gen)
    counts = torch.zeros(B, n, dtype=torch.long, device=device).scatter_add_(1, gaps, torch.ones_like(gaps))
    pos = (counts + 1).cumsum(1) - 1                                    # where each data token lands
    x = torch.full((B, L), blank, dtype=torch.long, device=device).scatter(1, pos, data)
    x[:, m] = copy
    y = torch.full_like(x, -100)
    y[:, m + 1:] = data
    return x, y, {}


class Task:
    """A named task: batch(B, device, gen) for evaluation, train_batch likewise for training (it may mix sizes), the
    vocabulary, the evaluation batch size, the step limit, an early-stopping target, and sizes evaluated at the end."""

    def __init__(self, name, make, vocab, batch, steps, extra_evals=(), train=None, target=0.995):
        self.name, self.batch, self.vocab, self.batch_size, self.steps = name, make, vocab, batch, steps
        self.extra_evals, self.train_batch, self.target = extra_evals, train or make, target


VOCAB = 1024                                                            # study 0001's seed runs
SEL_VOCAB, SEL_LEN, SEL_N = 16, 256, 16                                 # MAD's baseline setting
QUERIES = 4096                                                          # scored queries per MQAR training step
VOL_QUERIES = 1024                                                      # scored queries per Volatile MQAR training step


def mqar(n):
    return lambda B, device, gen: mqar_batch(B, n, VOCAB, device, gen)


def volatile(K, p=0.30):
    return lambda B, device, gen: volatile_batch(B, K, p, VOCAB, device, gen)


def selcopy(L=SEL_LEN, n=SEL_N):
    return lambda B, device, gen: selcopy_batch(B, L, n, SEL_VOCAB, device, gen)


def mixed(sizes, queries, make):
    """Training on a mix of sizes, as BL and Voltic train: each step draws a size N, with a batch of queries // N."""
    def train(B, device, gen):
        n = sizes[torch.randint(0, len(sizes), (1,), device=device, generator=gen).item()]
        return make(n)(queries // n, device, gen)
    return train


TASKS = {
    "mqar64": Task("mqar64", mqar(64), VOCAB, 64, 3000, (("mqar128", mqar(128), 32), ("mqar256", mqar(256), 16))),
    "mqar128": Task("mqar128", mqar(128), VOCAB, 32, 3000, (("mqar256", mqar(256), 16),)),
    # Trained on N <= 64 and evaluated up to 256, the protocol of BL's Table 4 at this lab's scale. No early stop, because
    # the larger sizes keep improving after N = 64 saturates.
    "mqarmix": Task("mqarmix", mqar(64), VOCAB, 64, 3000,
                    (("mqar32", mqar(32), 128), ("mqar128", mqar(128), 32), ("mqar256", mqar(256), 16)),
                    train=mixed((8, 16, 32, 64), QUERIES, mqar), target=2.0),
    "volatile32": Task("volatile32", volatile(32), VOCAB, 32, 3000, (("volatile64", volatile(64), 16),)),
    # Trained on K <= 32, as Voltic alternates sizes, and evaluated at K = 16, 32 and 64.
    "volatilemix": Task("volatilemix", volatile(32), VOCAB, 32, 3000,
                        (("volatile16", volatile(16), 64), ("volatile64", volatile(64), 16)),
                        train=mixed((8, 16, 32), VOL_QUERIES, volatile), target=2.0),
    "selcopy": Task("selcopy", selcopy(), SEL_VOCAB, 64, 3000),
}
