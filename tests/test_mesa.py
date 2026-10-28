"""lab.layers.mesa against the token-by-token recurrence (float64, CPU) and on the GPU (float32)."""
import unittest

import torch
import torch.nn.functional as F

from lab import rel_err
from lab.layers.mesa import mesa_carry, mesa_reference


def inputs(device="cpu", dtype=torch.float64, B=2, H=2, T=50, d=16, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q, k, v = (torch.randn(B, H, T, d, generator=g, device=device, dtype=dtype) for _ in range(3))
    beta = torch.rand(B, H, T, generator=g, device=device, dtype=dtype)
    lam = torch.tensor([0.5, 1.0], device=device, dtype=dtype)[:H]
    return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, beta, lam


class TestMesaCarry(unittest.TestCase):
    def test_forward_matches_token_by_token(self):
        args = inputs()
        ref = mesa_reference(*args)
        for chunk in (8, 16, 64):  # 50 tokens: padded last chunk; 64 is a single padded chunk
            with self.subTest(chunk=chunk):
                self.assertLess(rel_err(mesa_carry(*args, chunk), ref), 1e-12)

    def test_closed_form_gradients_match_autograd(self):
        args = inputs(seed=1)
        R = torch.randn(args[0].shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        grads = []
        for f in (mesa_reference, lambda *a: mesa_carry(*a, 16)):
            ins = [t.clone().requires_grad_(True) for t in args]
            (f(*ins) * R).sum().backward()
            grads.append([t.grad for t in ins])
        for name, a, b in zip(("q", "k", "v", "beta", "lam"), grads[1], grads[0]):
            with self.subTest(grad=name):
                self.assertLess(rel_err(a, b), 1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
    def test_float32_on_gpu(self):
        args64 = inputs("cuda", T=256, d=32)
        ref = mesa_reference(*args64)
        out = mesa_carry(*(t.float() for t in args64), 64)
        self.assertLess(rel_err(out, ref), 1e-4)


if __name__ == "__main__":
    unittest.main()
