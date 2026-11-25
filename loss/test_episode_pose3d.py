"""Run: python -m unittest loss.test_episode_pose3d -v."""

import unittest

import torch

from .episode_pose3d import EpisodePoseLoss
from .pose3d import loss_mpjpe, n_mpjpe, loss_velocity


class EpisodePoseLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.loss = EpisodePoseLoss()
        self.p = torch.randn(2, 2, 4, 17, 3)
        self.y = torch.randn_like(self.p)
        self.mask = torch.ones(2, 2, 4, dtype=torch.bool)
        self.ids = torch.arange(8).reshape(1, 2, 4).expand(2, -1, -1)

    def test_original_objective_and_gradients(self):
        p = self.p.clone().requires_grad_()
        out = self.loss(p, self.y, self.mask, self.ids)
        ref_p = self.p.flatten(1, 2).clone().requires_grad_()
        ref_y = self.y.flatten(1, 2)
        ref_y = ref_y - ref_y[..., :1, :]
        pos = loss_mpjpe(ref_p, ref_y)
        scale = n_mpjpe(ref_p, ref_y)
        vel = loss_velocity(ref_p, ref_y)
        expected = pos + 0.5 * scale + 20.0 * vel
        for actual, ref in zip(out[:4], (expected, pos, scale, vel)):
            torch.testing.assert_close(actual, ref)
        out.total.backward()
        expected.backward()
        torch.testing.assert_close(p.grad.flatten(1, 2), ref_p.grad, rtol=2e-5, atol=2e-6)
        self.assertEqual((out.valid_frames.item(), out.valid_pairs.item()), (16, 14))

    def test_flattened_and_windowed_inputs_match(self):
        a = self.loss(self.p, self.y, self.mask, self.ids)
        b = self.loss(*(x.flatten(1, 2) for x in (self.p, self.y, self.mask, self.ids)))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_window_boundary_velocity_and_gradient(self):
        p = torch.zeros(1, 2, 2, 17, 3)
        p[:, 1, :, :, 0] = 3
        p.requires_grad_()
        ids = torch.arange(4).reshape(1, 2, 2)
        out = self.loss(p, torch.zeros_like(p), torch.ones_like(ids, dtype=torch.bool), ids)
        self.assertEqual(out.valid_pairs.item(), 3)
        torch.testing.assert_close(out.velocity, torch.tensor(1.0))
        out.velocity.backward()
        self.assertLess(p.grad[0, 0, -1, :, 0].max().item(), 0)
        self.assertGreater(p.grad[0, 1, 0, :, 0].min().item(), 0)
        self.assertFalse(p.grad[0, 0, 0].any() or p.grad[0, 1, 1].any())

    def test_no_velocity_between_episodes_or_nonconsecutive_ids(self):
        loss = EpisodePoseLoss(root_rel=False)
        p = torch.zeros(2, 4, 17, 3)
        p[0, :, :, 0] = torch.tensor([0., 1., 100., 101.])[:, None]
        p[1, :, :, 0] = 1000
        ids = torch.tensor([[4, 5, 9, 10], [11, 12, 12, 10]])
        out = loss(p, torch.zeros_like(p), torch.ones_like(ids, dtype=torch.bool), ids)
        self.assertEqual(out.valid_pairs.item(), 3)
        torch.testing.assert_close(out.velocity, torch.tensor(2/3))

    def test_unequal_valid_lengths_and_padding(self):
        mask = self.mask.clone()
        mask[0, 1, 2:] = False
        mask[1] = False
        mask[1, 0, 0] = True
        p = torch.where(mask[..., None, None], self.p, float('nan')).requires_grad_()
        y = torch.where(mask[..., None, None], self.y, float('nan')).requires_grad_()
        ids = torch.where(mask, self.ids, -1)
        out = self.loss(p, y, mask, ids)
        real_p = torch.cat((self.p[0].flatten(0, 1)[:6], self.p[1, 0, :1]))
        real_y = torch.cat((self.y[0].flatten(0, 1)[:6], self.y[1, 0, :1]))
        real_y = real_y - real_y[..., :1, :]
        torch.testing.assert_close(out.position, loss_mpjpe(real_p[None], real_y[None]))
        torch.testing.assert_close(out.scale, n_mpjpe(real_p[None], real_y[None]))
        torch.testing.assert_close(out.velocity, loss_velocity(real_p[None, :6], real_y[None, :6]))
        self.assertEqual((out.valid_frames.item(), out.valid_pairs.item()), (7, 5))
        out.total.backward()
        for x in (p, y):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertFalse(x.grad[~mask].any())

    def test_all_padding_is_finite_graph_connected_zero(self):
        p = torch.full_like(self.p, float('nan')).requires_grad_()
        out = self.loss(p, torch.full_like(p, float('nan')), torch.zeros_like(self.mask),
                        torch.full_like(self.ids, -1))
        for x in out:
            self.assertEqual(x.item(), 0)
        out.total.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertFalse(p.grad.any())

    def test_zero_pose_and_single_valid_frame(self):
        p = torch.zeros_like(self.p).requires_grad_()
        mask = torch.zeros_like(self.mask)
        mask[0, 0, 0] = True
        out = self.loss(p, self.y, mask, torch.where(mask, self.ids, -1))
        self.assertEqual(out.valid_pairs.item(), 0)
        self.assertEqual(out.velocity.item(), 0)
        self.assertGreater(out.position.item(), 0)
        torch.testing.assert_close(out.position, out.scale)
        out.total.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_mask_gap_does_not_bridge_invalid_frame(self):
        mask = self.mask.clone()
        mask[0, 0, 2] = False
        ids = torch.where(mask, self.ids, -1)
        out = self.loss(self.p, self.y, mask, ids)
        self.assertEqual(out.valid_frames.item(), 15)
        self.assertEqual(out.valid_pairs.item(), 12)

    def test_root_target_only_and_scale_alignment(self):
        y = self.y - self.y[..., :1, :]
        p = 2*y
        out = self.loss(p, self.y, self.mask, self.ids)
        self.assertGreater(out.position.item(), 0)
        self.assertLess(out.scale.item(), 2e-6)
        p = y + 1
        out = self.loss(p, self.y, self.mask, self.ids)
        torch.testing.assert_close(out.position, torch.tensor(3.).sqrt())
        torch.testing.assert_close(out.velocity, torch.tensor(0.), atol=1e-6, rtol=0)
        raw = EpisodePoseLoss(root_rel=False)(self.y, self.y, self.mask, self.ids)
        torch.testing.assert_close(raw.total, torch.tensor(0.))

    def test_mixed_precision_and_default_episode_dimensions(self):
        p = torch.randn(1, 2, 243, 17, 3).to(torch.bfloat16).requires_grad_()
        y = torch.randn_like(p)
        ids = torch.arange(486).reshape(1, 2, 243)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = self.loss(p, y, mask, ids)
        expected = self.loss(p.float(), y.float(), mask, ids)
        for x, ref in zip(out[:4], expected[:4]):
            self.assertEqual(x.dtype, torch.float32)
            torch.testing.assert_close(x, ref, rtol=0, atol=0)
        self.assertEqual(out.valid_pairs.item(), 485)
        out.total.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_weight_configuration_and_validation(self):
        out = EpisodePoseLoss(0.0, 0.0)(self.p, self.y, self.mask, self.ids)
        torch.testing.assert_close(out.total, out.position, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            EpisodePoseLoss(lambda_scale=float('nan'))
        with self.assertRaises(ValueError):
            EpisodePoseLoss(scale_eps=0)
        with self.assertRaises(ValueError):
            self.loss(self.p, self.y, self.mask.float(), self.ids)
        with self.assertRaises(ValueError):
            self.loss(self.p, self.y, self.mask, self.ids - 1)
        with self.assertRaises(ValueError):
            self.loss(self.p, self.y[..., :16, :], self.mask, self.ids)


if __name__ == '__main__':
    unittest.main()
