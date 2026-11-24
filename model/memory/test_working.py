"""Run: python -m unittest model.memory.test_working."""

import unittest
from dataclasses import replace, fields

import torch

from .config import MemoryConfig
from .state import ReadStats, init_state
from .working import WorkingMemory


class WorkingMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.cfg = MemoryConfig(window_size=12, chunk_size=4, dim_rep=12,
                                working_capacity=3, write_stride=2,
                                explicit_key_dim=5, explicit_value_dim=7)
        self.model = WorkingMemory(self.cfg)
        self.state = init_state(self.cfg, 2, "cpu").working
        self.z = torch.randn(2, 4, 17, 12)
        self.p = torch.randn(2, 4, 17, 3)
        self.mask = torch.ones(2, 4, dtype=torch.bool)
        self.ids = torch.arange(4).expand(2, -1)

    def stats(self, mask=None, usage=None):
        count = (self.mask if mask is None else mask).sum(1) * 17
        return ReadStats(torch.zeros(2, 3) if usage is None else usage,
                         torch.zeros(2, 16), count)

    def update(self, state=None, ids=None, mask=None, usage=None):
        mask = self.mask if mask is None else mask
        k = self.model.encode_query(self.z, mask)
        v = self.model.encode_value(self.z, self.p, mask)
        return self.model.update(self.state if state is None else state, k, v,
                                 self.ids if ids is None else ids,
                                 self.stats(mask, usage), mask)

    def test_encoders_and_pose_information(self):
        k = self.model.encode_query(self.z)
        v = self.model.encode_value(self.z, self.p)
        self.assertEqual(k.shape, (2, 4, 17, 5))
        self.assertEqual(v.shape, (2, 4, 17, 7))
        torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k[..., 0]))
        self.assertGreater((v - self.model.encode_value(self.z, self.p + 1)).abs().sum().item(), 0)

    def test_absolute_stride_and_feedback(self):
        bank, feedback, evicted = self.update(ids=self.ids + 1)
        self.assertEqual(bank.timestamps.tolist(), [[1, 3, -1], [1, 3, -1]])
        self.assertEqual(feedback.mask.squeeze(-1).tolist(), [[True, False, True, False]] * 2)
        v = self.model.encode_value(self.z, self.p)
        torch.testing.assert_close(bank.values[:, :2], v[:, [0, 2]])
        torch.testing.assert_close(feedback.saved_value[:, [0, 2]], v[:, [0, 2]])
        self.assertEqual(torch.count_nonzero(feedback.saved_value[:, [1, 3]]).item(), 0)
        self.assertEqual(evicted.keys.shape, (2, 0, 17, 5))
        self.assertFalse(self.state.valid.any())

    def test_relevance_ema_and_age_without_renormalization(self):
        bank, _, _ = self.update()
        count = 4 * 17
        usage = torch.tensor([[0.2, 0.1, 0.0], [0.3, 0.1, 0.0]]) * count
        new, _, evicted = self.update(bank, self.ids + 4, usage=usage)
        # One old slot survives; the oldest with lower score is evicted.
        self.assertEqual(new.timestamps.tolist(), [[1, 5, 7], [1, 5, 7]])
        self.assertEqual(evicted.timestamps.tolist(), [[3], [3]])
        torch.testing.assert_close(new.relevance[:, 0], torch.tensor([0.04, 0.06]))
        torch.testing.assert_close(evicted.relevance[:, 0], torch.tensor([0.02, 0.02]))
        self.assertEqual(new.age.tolist(), [[5, 1, 1]] * 2)
        torch.testing.assert_close(evicted.keys[:, 0], bank.keys[:, 1])
        torch.testing.assert_close(evicted.values[:, 0], bank.values[:, 1])

    def test_freshness_and_deterministic_ties(self):
        score = self.model.retention_score(torch.zeros(2), torch.tensor([20, 2]))
        self.assertGreater(score[1].item(), score[0].item())
        bank, _, _ = self.update()
        new, _, evicted = self.update(bank, self.ids + 4)
        self.assertEqual(evicted.timestamps.tolist(), [[1], [1]])
        self.assertEqual(new.timestamps.tolist(), [[3, 5, 7]] * 2)

    def test_variable_eviction_counts_padding_and_no_mutation(self):
        bank, _, _ = self.update()
        old = {f.name: getattr(bank, f.name).clone() for f in fields(bank)}
        mask = torch.tensor([[True] * 4, [False] * 4])
        new, feedback, evicted = self.update(bank, self.ids + 4, mask)
        self.assertEqual(evicted.valid.tolist(), [[True], [False]])
        self.assertEqual(evicted.timestamps.tolist(), [[1], [-1]])
        self.assertFalse(feedback.mask[1].any())
        for name, value in old.items():
            torch.testing.assert_close(getattr(bank, name), value, rtol=0, atol=0)
            torch.testing.assert_close(getattr(new, name)[1], value[1], rtol=0, atol=0)

    def test_gradients_through_retained_and_evicted_kv(self):
        z = self.z.clone().requires_grad_()
        p = self.p.clone().requires_grad_()
        k = self.model.encode_query(z)
        v = self.model.encode_value(z, p)
        bank, _, _ = self.model.update(self.state, k, v, self.ids, self.stats(), self.mask)
        new, feedback, evicted = self.update(bank, self.ids + 4)
        loss = (new.keys.square().mean() + new.values.square().mean()
                + evicted.keys.square().mean() + evicted.values.square().mean()
                + feedback.saved_value.square().mean())
        loss.backward()
        for x in (z, p):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad[:, 1::2].abs().sum().item(), 0)
            self.assertEqual(torch.count_nonzero(x.grad[:, ::2]).item(), 0)
        self.assertFalse(new.relevance.requires_grad)

    def test_nan_padding_and_batch_isolation(self):
        mask = torch.tensor([[True, True, False, False], [False] * 4])
        z = torch.where(mask[:, :, None, None], self.z, float("nan")).requires_grad_()
        p = torch.where(mask[:, :, None, None], self.p, float("nan")).requires_grad_()
        k = self.model.encode_query(z, mask)
        v = self.model.encode_value(z, p, mask)
        bank, feedback, _ = self.model.update(self.state, k, v, self.ids, self.stats(mask), mask)
        self.assertEqual(bank.valid.sum(1).tolist(), [1, 0])
        self.assertTrue(torch.isfinite(bank.values).all())
        (bank.keys.square().sum() + feedback.saved_value.square().sum()).backward()
        for x in (z, p):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(torch.count_nonzero(x.grad[~mask]).item(), 0)

    def test_default_capacity_and_cross_chunk_stride(self):
        cfg = MemoryConfig()
        model = WorkingMemory(cfg)
        z = torch.randn(1, 27, 17, 512, dtype=torch.bfloat16)
        state = init_state(cfg, 1, "cpu", z.dtype).working
        mask = torch.ones(1, 27, dtype=torch.bool)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            k, v = model.encode_query(z), model.encode_value(z, z[..., :3])
        self.assertEqual(k.dtype, z.dtype)
        self.assertEqual(v.dtype, z.dtype)
        for i in range(5):
            ids = torch.arange(27)[None] + i * 27
            stats = ReadStats(torch.zeros(1, 9), torch.zeros(1, 16), torch.tensor([27 * 17]))
            state, feedback, evicted = model.update(state, k, v, ids, stats, mask)
            self.assertEqual(state.valid.sum().item(), min(3 * (i + 1), 9))
            self.assertEqual(feedback.mask.sum().item(), 3)
            self.assertEqual(evicted.valid.sum().item(), 3 if i >= 3 else 0)
        self.assertEqual(state.keys.shape, (1, 9, 17, 64))
        self.assertEqual(state.values.shape, (1, 9, 17, 128))
        self.assertEqual(state.timestamps[0, -1].item(), 134)

    def test_empty_chunk_and_invalid_input(self):
        k = self.model.encode_query(self.z[:, :0])
        v = self.model.encode_value(self.z[:, :0], self.p[:, :0])
        bank, feedback, evicted = self.model.update(self.state, k, v, self.ids[:, :0],
                                                     self.stats(self.mask[:, :0]), self.mask[:, :0])
        self.assertFalse(bank.valid.any())
        self.assertEqual(feedback.mask.shape, (2, 0, 1))
        self.assertEqual(evicted.valid.shape, (2, 0))
        bank, _, _ = self.update()
        with self.assertRaises(ValueError):
            self.update(bank)
        with self.assertRaises(ValueError):
            self.update(mask=torch.tensor([[True, False, True, False], [True] * 4]))
        with self.assertRaises(ValueError):
            replace(self.cfg, working_freshness_weight=float("nan"))


if __name__ == "__main__":
    unittest.main()
