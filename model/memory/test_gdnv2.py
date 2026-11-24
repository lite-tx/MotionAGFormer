"""CPU reference tests. Run: python -m unittest model.memory.test_gdnv2"""

import unittest

import torch
from torch.nn import functional as F

from .gdnv2 import gdn2_read, gdn2_step, gdn2_update


class GDN2ReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.b, self.c, self.j, self.h, self.dk, self.dv = 2, 5, 17, 2, 3, 4
        self.state = torch.randn(self.b, self.j, self.h, self.dk, self.dv)
        key_shape = (self.b, self.c, self.j, self.h, self.dk)
        value_shape = (self.b, self.c, self.j, self.h, self.dv)
        self.k = torch.randn(key_shape)
        self.v = torch.randn(value_shape)
        self.g = -torch.rand(key_shape)
        self.e = torch.rand(key_shape)
        self.w = torch.rand(value_shape)
        self.inputs = (self.k, self.v, self.g, self.e, self.w)

    def test_recurrence_matches_dense_matrix_formula(self):
        expected = self.state.clone()
        for t in range(self.c):
            k = F.normalize(self.k[:, t], dim=-1, eps=1e-6).unsqueeze(-1)
            e = self.e[:, t].unsqueeze(-1) * k
            z = (self.w[:, t] * self.v[:, t]).unsqueeze(-1)
            d = torch.diag_embed(self.g[:, t].exp())
            expected = (torch.eye(self.dk) - k @ e.transpose(-2, -1)) @ d @ expected
            expected = expected + k @ z.transpose(-2, -1)
        actual = gdn2_update(self.state, *self.inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_read_is_parallel_and_does_not_mutate(self):
        old = self.state.clone()
        q = F.normalize(self.k, dim=-1, eps=1e-6)
        expected = (q.unsqueeze(-1) * old.unsqueeze(1)).sum(-2)
        actual = gdn2_read(self.state, self.k)
        torch.testing.assert_close(actual, expected)
        gdn2_update(self.state, *self.inputs)
        torch.testing.assert_close(self.state, old, rtol=0, atol=0)
        self.assertEqual(actual.shape, (self.b, self.c, self.j, self.h, self.dv))

    def test_step_and_split_chunk_equivalence(self):
        state = self.state
        for t in range(self.c):
            state = gdn2_step(state, *(x[:, t] for x in self.inputs))
        full = gdn2_update(self.state, *self.inputs)
        partial = gdn2_update(self.state, *(x[:, :2] for x in self.inputs))
        split = gdn2_update(partial, *(x[:, 2:] for x in self.inputs))
        torch.testing.assert_close(state, full, rtol=0, atol=0)
        torch.testing.assert_close(split, full, rtol=0, atol=0)

    def test_independent_erase_and_write(self):
        k = torch.zeros_like(self.k[:, 0])
        k[..., 0] = 1
        v = self.v[:, 0]
        g = torch.zeros_like(k)
        e = torch.ones_like(k)
        w = torch.ones_like(v)
        erased = gdn2_step(self.state, k, v, g, e, torch.zeros_like(w))
        torch.testing.assert_close(erased[..., 0, :], torch.zeros_like(v))
        torch.testing.assert_close(erased[..., 1:, :], self.state[..., 1:, :])
        written = gdn2_step(self.state, k, v, g, torch.zeros_like(e), w)
        torch.testing.assert_close(written[..., 0, :], self.state[..., 0, :] + v)
        torch.testing.assert_close(written[..., 1:, :], self.state[..., 1:, :])

    def test_decay_only_and_empty_chunk(self):
        state = gdn2_update(self.state, self.k, self.v, self.g,
                            torch.zeros_like(self.e), torch.zeros_like(self.w))
        expected = self.g.sum(dim=1).exp().unsqueeze(-1) * self.state
        torch.testing.assert_close(state, expected)
        empty = gdn2_update(self.state, *(x[:, :0] for x in self.inputs))
        torch.testing.assert_close(empty, self.state, rtol=0, atol=0)

    def test_padding_is_identity_even_with_nan_features(self):
        mask = torch.tensor([[True, True, False, False, False], [False] * 5])
        bad = tuple(torch.where(mask[:, :, None, None, None], x, float("nan"))
                    for x in self.inputs)
        actual = gdn2_update(self.state, *bad, mask)
        expected = gdn2_update(self.state[:1], *(x[:1, :2] for x in self.inputs))
        torch.testing.assert_close(actual[:1], expected)
        torch.testing.assert_close(actual[1], self.state[1], rtol=0, atol=0)
        read = gdn2_read(self.state, bad[0], mask)
        self.assertTrue(torch.isfinite(read).all())
        self.assertEqual(torch.count_nonzero(read[~mask]).item(), 0)

    def test_batch_joint_and_head_isolation(self):
        altered = self.v.clone()
        altered[0, :, 3, 1, :] += 10
        old = gdn2_update(self.state, *self.inputs)
        new = gdn2_update(self.state, self.k, altered, self.g, self.e, self.w)
        diff = new - old
        self.assertGreater(diff[0, 3, 1].abs().max().item(), 0)
        diff[0, 3, 1] = 0
        torch.testing.assert_close(diff, torch.zeros_like(diff), rtol=0, atol=0)

    def test_backward_through_state_and_all_inputs(self):
        state = self.state.clone().requires_grad_()
        inputs = tuple(x.clone().requires_grad_() for x in self.inputs)
        updated = gdn2_update(state, *inputs)
        out = gdn2_read(updated, inputs[0])
        (out.square().mean() + updated.square().mean()).backward()
        for x in (state, *inputs):
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_masked_nan_backward_is_finite(self):
        mask = torch.tensor([[True, True, False, False, False], [False] * 5])
        inputs = tuple(torch.where(mask[:, :, None, None, None], x, float("nan"))
                       .requires_grad_() for x in self.inputs)
        state = self.state.clone().requires_grad_()
        updated = gdn2_update(state, *inputs, mask)
        out = gdn2_read(updated, inputs[0], mask)
        (updated.square().mean() + out.square().mean()).backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        for x in inputs:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(torch.count_nonzero(x.grad[~mask]).item(), 0)

    def test_fp32_under_autocast_and_shape_validation(self):
        half_inputs = tuple(x.to(torch.bfloat16) for x in self.inputs)
        expected = gdn2_update(self.state, *(x.float() for x in half_inputs))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = gdn2_update(self.state, *half_inputs)
            out = gdn2_read(actual, half_inputs[0])
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(out, gdn2_read(expected, half_inputs[0]), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            gdn2_update(self.state, self.k, self.v, self.g, self.e, self.e)
        with self.assertRaises(ValueError):
            gdn2_read(self.state.half(), self.k)
        with self.assertRaises(ValueError):
            gdn2_read(self.state, self.k, torch.ones(self.b, self.c))


if __name__ == "__main__":
    unittest.main()
