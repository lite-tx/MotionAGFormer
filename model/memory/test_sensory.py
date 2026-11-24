"""Run from repository root: python -m unittest model.memory.test_sensory."""

import unittest
from dataclasses import replace

import torch

from .config import MemoryConfig
from .gdnv2 import gdn2_read, gdn2_update
from .sama import MotionAdaptiveModulator, StructureAwareInteraction
from .sensory import SensoryMemory
from .state import WriteFeedback, init_state


class SensoryMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.cfg = MemoryConfig(window_size=6, chunk_size=3, dim_rep=12,
                                sensory_heads=2, sensory_key_dim=3,
                                sensory_value_dim=5, explicit_value_dim=7)
        self.model = SensoryMemory(self.cfg)
        self.z = torch.randn(2, 3, 17, 12)
        self.u = torch.randn_like(self.z)
        self.state = init_state(self.cfg, 2, "cpu").sensory
        self.mask = torch.ones(2, 3, dtype=torch.bool)
        self.feedback = WriteFeedback(
            torch.tensor([[[False], [True], [False]], [[True], [False], [False]]]),
            torch.randn(2, 3, 17, 7),
        )

    def test_ssi_topology_and_temporary_state(self):
        ssi = self.model.ssi
        a = ssi.adjacency()
        torch.testing.assert_close(a.sum(-1), torch.ones(17))
        self.assertGreater(a[0, 1].item(), a[0, 16].item())
        self.assertGreater(a[0, 7].item(), a[0, 16].item())
        matrix = torch.zeros_like(self.state.matrix)
        matrix[:, 1] = 1
        old = matrix.clone()
        ctx = ssi.state_context(matrix)
        self.assertGreater(ctx[:, 0].mean().item(), ctx[:, 16].mean().item())
        torch.testing.assert_close(matrix, old, rtol=0, atol=0)
        self.assertEqual(ctx.dtype, torch.float32)
        x = torch.zeros_like(self.z)
        x[:, 0, 1] = 1
        y = ssi.feature_fusion(x)
        self.assertGreater(y[:, 0, 0].mean().item(), 0)
        self.assertEqual(torch.count_nonzero(y[:, 1:]).item(), 0)

    def test_empty_read_and_post_prediction_update(self):
        old = self.state.matrix.clone()
        self.assertEqual(torch.count_nonzero(self.model.read(self.z, self.state)).item(), 0)
        new = self.model.update(self.z, self.u, self.state, self.feedback)
        self.assertTrue(new.valid.all())
        self.assertGreater(new.matrix.abs().sum().item(), 0)
        self.assertGreater(self.model.read(self.z, new).abs().sum().item(), 0)
        torch.testing.assert_close(self.state.matrix, old, rtol=0, atol=0)
        torch.testing.assert_close(new.last_feature, self.model.ssi.feature_fusion(self.z)[:, -1])
        self.assertEqual(new.matrix.shape, (2, 17, 2, 3, 5))

    def test_read_matches_gdn2_on_old_context(self):
        state = self.model.update(self.z, self.u, self.state)
        before = state.matrix.clone()
        q = self.model.q_proj(self.model.ssi.feature_fusion(self.z)).unflatten(-1, (2, 3))
        r = gdn2_read(self.model.ssi.state_context(state.matrix), q)
        expected = self.model.out_proj(r.flatten(-2))
        torch.testing.assert_close(self.model.read(self.z, state), expected)
        torch.testing.assert_close(state.matrix, before, rtol=0, atol=0)

    def test_qk_from_z_value_and_base_gates_from_u(self):
        p = self.model.build_update_inputs(self.z, self.u, self.state)
        q = self.model.build_update_inputs(self.z, self.u + 1, self.state)
        torch.testing.assert_close(p.key, q.key, rtol=0, atol=0)
        self.assertGreater((p.value - q.value).abs().sum().item(), 0)
        q = self.model.build_update_inputs(self.z + 1, self.u, self.state)
        torch.testing.assert_close(p.value, q.value, rtol=0, atol=0)
        torch.testing.assert_close(p.erase, q.erase, rtol=0, atol=0)
        torch.testing.assert_close(p.log_decay.exp(), torch.full_like(p.log_decay, 0.95))
        self.assertTrue(((p.erase >= 0) & (p.erase <= 1)).all())
        self.assertTrue(((p.write >= 0) & (p.write <= 1)).all())
        self.assertLessEqual(p.log_decay.max().item(), 0)

    def test_msm_neutral_then_uses_previous_chunk(self):
        msm = self.model.msm
        empty = self.state.last_feature
        g = msm(self.z, empty, self.state.valid, self.mask)
        torch.testing.assert_close(g, torch.ones_like(g))
        with torch.no_grad():
            msm.proj.weight.zero_()
            msm.proj.weight[0, self.cfg.dim_rep:] = 0.2
        last = torch.ones_like(empty)
        g = msm(self.z, last, torch.tensor([True, False]), self.mask)
        self.assertGreater(g[0, 0].min().item(), 1)
        torch.testing.assert_close(g[1, 0], torch.ones_like(g[1, 0]))
        g2 = msm(self.z, -last, torch.tensor([True, False]), self.mask)
        self.assertLess(g2[0, 0].max().item(), 1)
        torch.testing.assert_close(g[:, 1:], g2[:, 1:])

    def test_deep_update_only_saved_frames_and_gates(self):
        normal = self.model.build_update_inputs(self.z, self.u, self.state)
        neutral = self.model.build_update_inputs(self.z, self.u, self.state, self.feedback)
        torch.testing.assert_close(normal.erase, neutral.erase, rtol=0, atol=0)
        torch.testing.assert_close(normal.write, neutral.write, rtol=0, atol=0)
        with torch.no_grad():
            self.model.cooperation[-1].bias[:6].fill_(1)
            self.model.cooperation[-1].bias[6:].fill_(-1)
            self.model.cooperation[-1].weight.fill_(0.01)
        selected = self.feedback.mask.squeeze(-1)
        deep = self.model.build_update_inputs(self.z, self.u, self.state, self.feedback)
        torch.testing.assert_close(deep.erase[~selected], normal.erase[~selected], rtol=0, atol=0)
        torch.testing.assert_close(deep.write[~selected], normal.write[~selected], rtol=0, atol=0)
        torch.testing.assert_close(deep.log_decay, normal.log_decay, rtol=0, atol=0)
        self.assertGreater((deep.erase[selected] - normal.erase[selected]).abs().sum().item(), 0)
        self.assertGreater((deep.write[selected] - normal.write[selected]).abs().sum().item(), 0)
        altered = WriteFeedback(self.feedback.mask, self.feedback.saved_value + 10)
        changed = self.model.build_update_inputs(self.z, self.u, self.state, altered)
        self.assertGreater((changed.erase[selected] - deep.erase[selected]).abs().sum().item(), 0)

    def test_update_matches_core_and_split_chunk(self):
        with torch.no_grad():
            self.model.msm.proj.weight.normal_(0, 0.03)
            self.model.cooperation[-1].weight.normal_(0, 0.03)
        p = self.model.build_update_inputs(self.z, self.u, self.state, self.feedback)
        expected = gdn2_update(self.state.matrix, p.key, p.value, p.log_decay, p.erase, p.write)
        full = self.model.update(self.z, self.u, self.state, self.feedback)
        torch.testing.assert_close(full.matrix, expected)
        partial = self.model.update(self.z[:, :1], self.u[:, :1], self.state,
                                    WriteFeedback(self.feedback.mask[:, :1], self.feedback.saved_value[:, :1]))
        split = self.model.update(self.z[:, 1:], self.u[:, 1:], partial,
                                  WriteFeedback(self.feedback.mask[:, 1:], self.feedback.saved_value[:, 1:]))
        torch.testing.assert_close(full.matrix, split.matrix, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(full.last_feature, split.last_feature)

    def test_padding_nan_and_all_padding_preserve_state(self):
        state = self.model.update(self.z, self.u, self.state)
        mask = torch.tensor([[True, True, False], [False, False, False]])
        z = torch.where(mask[:, :, None, None], self.z, float("nan")).requires_grad_()
        u = torch.where(mask[:, :, None, None], self.u, float("nan")).requires_grad_()
        selected = torch.tensor([[[False], [True], [False]], [[False], [False], [False]]])
        saved = torch.where(selected.unsqueeze(-1), self.feedback.saved_value, float("nan")).requires_grad_()
        feedback = WriteFeedback(selected, saved)
        new = self.model.update(z, u, state, feedback, mask)
        read = self.model.read(z, state, mask)
        self.assertTrue(torch.isfinite(new.matrix).all())
        self.assertTrue(torch.isfinite(read).all())
        torch.testing.assert_close(new.matrix[1], state.matrix[1], rtol=0, atol=0)
        torch.testing.assert_close(new.last_feature[1], state.last_feature[1], rtol=0, atol=0)
        torch.testing.assert_close(new.last_feature[0], self.model.ssi.feature_fusion(self.z)[0, 1])
        self.assertEqual(torch.count_nonzero(read[~mask]).item(), 0)
        (new.matrix.square().mean() + read.square().mean()).backward()
        for x in (z, u, saved):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(torch.count_nonzero(x.grad[~mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(saved.grad[~selected.squeeze(-1)]).item(), 0)

    def test_cross_chunk_gradients_and_batch_isolation(self):
        z = self.z.clone().requires_grad_()
        u = self.u.clone().requires_grad_()
        new = self.model.update(z, u, self.state, self.feedback)
        out = self.model.read(self.z, new)
        out[0].square().mean().backward()
        for x in (z, u):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad[0].abs().sum().item(), 0)
            self.assertEqual(torch.count_nonzero(x.grad[1]).item(), 0)
        self.assertIsNotNone(self.model.q_proj.weight.grad)
        self.assertIsNotNone(self.model.msm.proj.weight.grad)
        self.assertIsNotNone(self.model.cooperation[-1].weight.grad)

    def test_ablation_switches_and_mask_validation(self):
        cfg = replace(self.cfg, sensory_use_ssi=False, sensory_use_msm=False,
                      sensory_use_deep_update=False)
        model = SensoryMemory(cfg)
        self.assertIsNone(model.ssi)
        self.assertIsNone(model.msm)
        self.assertIsNone(model.cooperation)
        a = model.update(self.z, self.u, self.state)
        b = model.update(self.z, self.u, self.state, self.feedback)
        torch.testing.assert_close(a.matrix, b.matrix, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.model.update(self.z, self.u, self.state, self.feedback,
                              torch.zeros_like(self.mask))
        with self.assertRaises(ValueError):
            self.model.read(self.z, self.state, torch.tensor([[True, False, True], [True]*3]))

    def test_bfloat16_autocast_and_default_shapes(self):
        cfg = MemoryConfig()
        model = SensoryMemory(cfg)
        z = torch.randn(1, 27, 17, 512, dtype=torch.bfloat16)
        state = init_state(cfg, 1, "cpu", z.dtype).sensory
        with torch.autocast("cpu", dtype=torch.bfloat16):
            new = model.update(z, z, state)
            out = model.read(z, new)
        self.assertEqual(new.matrix.dtype, torch.float32)
        self.assertEqual(new.last_feature.dtype, torch.bfloat16)
        self.assertEqual(new.matrix.shape, (1, 17, 4, 32, 32))
        self.assertEqual(out.shape, (1, 27, 17, 512))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out).all())

    def test_empty_chunk(self):
        z = self.z[:, :0]
        self.assertEqual(self.model.read(z, self.state).shape, z.shape)
        self.assertIs(self.model.update(z, z, self.state), self.state)


if __name__ == "__main__":
    unittest.main()
