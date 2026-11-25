"""Run: python -m unittest model.memory.test_memory_system -v.

Tests the complete post-backbone pipeline with the original-style Linear head.
The full MotionAGFormer encoder is deliberately outside this test boundary.
"""

import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import torch
from torch import nn

from .config import MemoryConfig
from .memory_system import MemorySystem


def assert_state_equal(a, b, row=None):
    if isinstance(a, torch.Tensor):
        if row is not None:
            a, b = a[row], b[row]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        for f in fields(a):
            assert_state_equal(getattr(a, f.name), getattr(b, f.name), row)


class MemorySystemTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.cfg = MemoryConfig(enabled=True, window_size=12, chunk_size=4,
                                dim_rep=12, working_capacity=3, write_stride=2,
                                proxy_capacity=4, explicit_key_dim=4, explicit_value_dim=6,
                                sensory_heads=2, sensory_key_dim=3, sensory_value_dim=5,
                                proxy_dim=8, pose_query_dim=8, pose_query_heads=2,
                                fusion_hidden_dim=8, decoder_hidden_dim=8,
                                proxy_min_exposure=1)
        self.system = MemorySystem(self.cfg)
        self.head = nn.Linear(12, 3)
        self.z = torch.randn(2, 12, 17, 12)
        self.ids = torch.arange(12).expand(2, -1)
        self.mask = torch.ones(2, 12, dtype=torch.bool)
        self.streams = torch.tensor([10, 11])

    def run_window(self, z=None, state=None, ids=None, mask=None, streams=None):
        return self.system(self.z if z is None else z, state,
                           self.ids if ids is None else ids,
                           self.mask if mask is None else mask,
                           self.streams if streams is None else streams, self.head)

    def test_disabled_unchanged_head_and_parameter_keys(self):
        system = MemorySystem(replace(self.cfg, enabled=False))
        self.assertEqual(len(system.state_dict()), 0)
        pose, state = system(self.z, None, self.ids, self.mask, self.streams, self.head)
        torch.testing.assert_close(pose, self.head(self.z), rtol=0, atol=0)
        self.assertFalse(state.sensory.valid.any() or state.working.valid.any() or state.long_term.valid.any())

    def test_full_loop_and_empty_first_chunk(self):
        pose, state = self.run_window()
        self.assertEqual(pose.shape, (2, 12, 17, 3))
        self.assertTrue(torch.isfinite(pose).all())
        torch.testing.assert_close(pose[:, :4], self.head(self.z[:, :4]))
        self.assertGreater((pose[:, 4:] - self.head(self.z[:, 4:])).abs().sum().item(), 0)
        self.assertTrue(state.sensory.valid.all())
        self.assertEqual(state.working.valid.sum(1).tolist(), [3, 3])
        self.assertTrue(state.long_term.valid.any(1).all())
        self.assertTrue((state.long_term.exposure.sum(1) > 0).all())
        self.assertEqual(state.last_frame_id.tolist(), [11, 11])
        self.assertEqual(state.stream_ids.tolist(), [10, 11])

    def test_read_predict_write_order_and_feedback(self):
        events = []
        modules = (self.system.sensory, self.system.working, self.system.long_term)
        s, w, l = modules
        def record(name, fn):
            def wrapper(*args, **kwargs):
                events.append(name)
                if name == 's_write':
                    feedback = args[3]
                    self.assertTrue((feedback.mask.sum(1) == 2).all())
                return fn(*args, **kwargs)
            return wrapper
        from contextlib import ExitStack
        with ExitStack() as stack:
            for module, method, name in ((s, 'read', 's_read'), (w, 'encode_query', 'q'),
                    (l, 'get_kv', 'l_read'), (self.head, 'forward', 'predict'),
                    (w, 'update', 'w_write'), (s, 'update', 's_write'), (l, 'update', 'l_write')):
                stack.enter_context(patch.object(module, method, side_effect=record(name, getattr(module, method))))
            self.run_window()
        self.assertEqual(events, ['s_read','q','l_read','predict','w_write','s_write','l_write'] * 3)

    def test_cross_window_equivalence_and_no_mutation(self):
        z2 = torch.randn_like(self.z)
        # Snapshot via independent detached clones, not aliases.
        import copy
        with torch.no_grad():
            p1, state_ng = self.run_window()
            snapshot = copy.deepcopy(state_ng)
            p2, state2 = self.run_window(z2, state_ng, self.ids + 12)
            assert_state_equal(state_ng, snapshot)
            large = MemorySystem(replace(self.cfg, window_size=24))
            large.load_state_dict(self.system.state_dict())
            p_all, combined = large(torch.cat((self.z, z2), 1), None,
                torch.arange(24).expand(2,-1), torch.ones(2,24,dtype=torch.bool), self.streams, self.head)
        torch.testing.assert_close(p_all, torch.cat((p1, p2),1))
        assert_state_equal(state2, combined)
        self.assertEqual(state2.last_frame_id.tolist(), [23,23])
        self.assertTrue((state2.long_term.valid.sum(1) <= self.cfg.proxy_capacity).all())

    def test_partial_reset_and_suffix_padding(self):
        with torch.no_grad():
            _, old = self.run_window()
            z = torch.randn_like(self.z)
            ids = self.ids + 12
            ids = ids.clone(); ids[1] = self.ids[1]
            mask = self.mask.clone(); mask[1, 5:] = False
            streams = torch.tensor([10, 20])
            p, new = self.run_window(z, old, ids, mask, streams)
            p_fresh, fresh = self.system(z[1:], None, ids[1:], mask[1:], streams[1:], self.head)
        torch.testing.assert_close(p[1], p_fresh[0])
        for f in fields(new):
            a,b = getattr(new,f.name),getattr(fresh,f.name)
            if isinstance(a,torch.Tensor):
                torch.testing.assert_close(a[1],b[0])
            else:
                for g in fields(a): torch.testing.assert_close(getattr(a,g.name)[1],getattr(b,g.name)[0])
        self.assertEqual(new.last_frame_id.tolist(), [23,4])
        self.assertFalse(p[1,5:].any())

    def test_all_padding_state_identity_and_invalid_windows(self):
        with torch.no_grad():
            _, state = self.run_window()
            mask = torch.zeros_like(self.mask)
            pose, unchanged = self.run_window(torch.full_like(self.z,float('nan')),state,self.ids,mask,torch.tensor([20,21]))
        self.assertFalse(pose.any())
        assert_state_equal(state,unchanged)
        with self.assertRaises(ValueError): self.run_window(state=state,ids=self.ids+10)
        with self.assertRaises(ValueError): self.run_window(state=state,ids=self.ids+13)

    def test_gradients_from_future_chunk_and_padding(self):
        mask = self.mask.clone(); mask[1,9:] = False
        z = torch.where(mask[:,:,None,None],self.z,float('nan')).requires_grad_()
        p, state = self.run_window(z,mask=mask)
        self.assertTrue(torch.isfinite(p).all())
        p[0,-4:].square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad[0,:4].abs().sum().item(),0)
        self.assertFalse(z.grad[1].any())
        for param in (self.system.sensory.q_proj.weight, self.system.sensory.k_proj.weight,
                      self.system.sensory.v_proj.weight, self.system.sensory.cooperation[-1].weight,
                      self.system.working.key_encoder.weight, self.system.working.feature_proj.weight,
                      self.system.long_term.input_proj.weight, self.system.long_term.key_proj.weight,
                      self.system.explicit_reader.out_proj.weight, self.system.pose_reader.queries,
                      self.system.fusion.gate[-1].weight, self.system.decoder.mlp[0].weight,self.head.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(),0)
        self.assertFalse(state.working.relevance.requires_grad)

    def test_cross_window_gradient_and_external_head_ownership(self):
        first = self.z.clone().requires_grad_()
        _, state = self.run_window(first)
        second = torch.randn_like(first).requires_grad_()
        p, _ = self.run_window(second, state, self.ids + 12)
        p[:, :4].square().mean().backward()
        for z in (first, second):
            self.assertIsNotNone(z.grad)
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertNotIn(id(self.head.weight), {id(p) for p in self.system.parameters()})

    def test_default_shapes_bf16_and_bounded_state(self):
        cfg = MemoryConfig(enabled=True)
        model, head = MemorySystem(cfg), nn.Linear(512,3)
        z = torch.randn(1,243,17,512,dtype=torch.bfloat16)
        ids = torch.arange(243)[None]
        mask = torch.ones(1,243,dtype=torch.bool)
        streams = torch.tensor([7])
        with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
            p,state = model(z,None,ids,mask,streams,head)
            p2,state2 = model(z,state,ids+243,mask,streams,head)
        self.assertEqual(p.shape,(1,243,17,3))
        self.assertEqual(p2.shape,p.shape)
        self.assertTrue(torch.isfinite(p).all() and torch.isfinite(p2).all())
        self.assertEqual(state2.sensory.matrix.dtype,torch.float32)
        self.assertEqual(state2.working.keys.shape,(1,9,17,64))
        self.assertEqual(state2.long_term.proxy.shape,(1,16,17,128))
        self.assertEqual(state2.last_frame_id.item(),485)
        self.assertTrue(state2.long_term.valid.any())


if __name__ == '__main__':
    unittest.main()
