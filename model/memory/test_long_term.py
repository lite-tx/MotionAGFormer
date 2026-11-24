"""Run: python -m unittest model.memory.test_long_term -v."""

import unittest
from dataclasses import replace, fields

import torch
from torch.nn import functional as F

from .config import MemoryConfig
from .state import EvictedMemory, ReadStats, init_state
from .working import WorkingMemory
from .long_term import LongTermMemory


class LongTermMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(25)
        self.cfg = MemoryConfig(window_size=12, chunk_size=4, dim_rep=12,
                                working_capacity=3, write_stride=2,
                                proxy_capacity=2, explicit_key_dim=4,
                                explicit_value_dim=6, proxy_dim=8,
                                proxy_min_exposure=1)
        self.model = LongTermMemory(self.cfg)
        self.state = init_state(self.cfg, 2, "cpu").long_term
        self.mask = torch.ones(2, 4, dtype=torch.bool)

    def records(self, axes, start=0, valid=None):
        e = len(axes)
        keys = F.one_hot(torch.tensor(axes, dtype=torch.long), 4).float()
        keys = keys[None,:,None,:].expand(2,e,17,4).clone()
        return EvictedMemory(keys, torch.randn(2,e,17,6),
                             torch.arange(start,start+e).expand(2,e),
                             torch.zeros(2,e), torch.ones(2,e,dtype=torch.long),
                             torch.ones(2,e,dtype=torch.bool) if valid is None else valid)

    def stats(self, usage=None, mask=None):
        count = (self.mask if mask is None else mask).sum(1)*17
        return ReadStats(torch.zeros(2,3), torch.zeros(2,2) if usage is None else usage, count)

    def update(self, state=None, evicted=None, now=4, usage=None, mask=None):
        mask = self.mask if mask is None else mask
        return self.model.update(self.state if state is None else state, evicted,
                                 self.stats(usage, mask), torch.arange(now,now+4).expand(2,4), mask)

    def test_empty_slots_and_projection(self):
        k,v = self.model.get_kv(self.state)
        self.assertEqual(k.shape,(2,2,17,4))
        self.assertEqual(v.shape,(2,2,17,6))
        self.assertFalse(k.any() or v.any())
        e = self.records([0])
        state = self.update(evicted=e)
        k,v = self.model.get_kv(state)
        torch.testing.assert_close(k[:,0],e.keys[:,0])
        self.assertTrue(state.valid[:,0].all())
        self.assertFalse(state.valid[:,1].any())
        self.assertEqual(state.source_range[:,0].tolist(),[[0,0]]*2)
        self.assertEqual(state.created_at[:,0].tolist(),[7,7])
        self.assertFalse(self.state.valid.any())

    def test_similar_candidates_group_and_match(self):
        e = self.records([0,0,1])
        state = self.update(evicted=e)
        self.assertEqual(state.support_count.tolist(),[[2,1]]*2)
        self.assertEqual(state.source_range.tolist(),[[[0,1],[2,2]]]*2)
        old = state.proxy.clone()
        more = self.records([0,0],start=4)
        new = self.update(state,more,now=8)
        self.assertEqual(new.support_count.tolist(),[[4,1]]*2)
        self.assertEqual(new.source_range[:,0].tolist(),[[0,5]]*2)
        self.assertGreater((new.proxy[:,0]-old[:,0]).abs().sum().item(),0)
        torch.testing.assert_close(new.proxy[:,1],old[:,1],rtol=0,atol=0)
        torch.testing.assert_close(state.proxy,old,rtol=0,atol=0)

    def test_lfu_uses_normalized_usage_and_resets_metadata(self):
        state = self.update(evicted=self.records([0,1]))
        state = replace(state,usage=torch.tensor([[10.,5.]]*2),
                        exposure=torch.tensor([[1000,70]]*2))
        new = self.update(state,self.records([2],start=4),now=8)
        # Slot 0 has higher total usage but lower usage per exposure.
        self.assertEqual(new.source_range.tolist(),[[[4,4],[1,1]]]*2)
        self.assertEqual(new.support_count.tolist(),[[1,1]]*2)
        self.assertEqual(new.usage[:,0].tolist(),[0.,0.])
        self.assertEqual(new.exposure[:,0].tolist(),[0,0])
        self.assertEqual(new.created_at[:,0].tolist(),[11,11])

    def test_protection_and_bounded_rejection(self):
        state = self.update(evicted=self.records([0,1]))
        self.model = LongTermMemory(replace(self.cfg,proxy_min_exposure=1000))
        new = self.update(state,self.records([2,3],start=4),now=8)
        torch.testing.assert_close(new.proxy,state.proxy,rtol=0,atol=0)
        self.assertEqual(new.exposure.tolist(),[[68,68]]*2)
        self.model = LongTermMemory(self.cfg)
        new = self.update(evicted=self.records([0,1,2,3]))
        self.assertEqual(new.valid.sum(1).tolist(),[2,2])
        self.assertEqual(new.support_count.tolist(),[[1,1]]*2)

    def test_matched_slots_reserved_before_novel_allocation(self):
        state = self.update(evicted=self.records([0,1]))
        more = self.records([2,0,1],start=4)
        new = self.update(state,more,now=8)
        self.assertEqual(new.support_count.tolist(),[[2,2]]*2)
        self.assertEqual(new.source_range.tolist(),[[[0,5],[1,6]]]*2)

    def test_stats_only_and_all_padding(self):
        state = self.update(evicted=self.records([0,1]))
        mask = torch.tensor([[True]*4,[False]*4])
        new = self.update(state,None,now=8,usage=torch.tensor([[2.,3.],[float('nan'),float('nan')]]),mask=mask)
        torch.testing.assert_close(new.proxy,state.proxy,rtol=0,atol=0)
        self.assertEqual(new.usage.tolist(),[[2.,3.],[0.,0.]])
        self.assertEqual(new.exposure.tolist(),[[68,68],[0,0]])
        self.assertIs(self.update(new,None,now=12,mask=torch.zeros_like(mask)),new)

    def test_pum_gradients_through_multiple_evictions(self):
        first = self.records([0])
        first.keys.requires_grad_(); first.values.requires_grad_()
        state = self.update(evicted=first)
        more = self.records([0,0],start=4)
        # Nonidentical keys within the same matching region.
        more.keys[:,1,:,1] = 0.3
        more.keys.requires_grad_(); more.values.requires_grad_()
        state = self.update(state,more,now=8)
        k,v = self.model.get_kv(state)
        (k.square().mean()+v.square().mean()).backward()
        for x in (first.keys,first.values,more.keys,more.values,
                  self.model.write_query.weight,self.model.update_gate.weight,
                  self.model.input_proj.weight,self.model.key_proj.weight,self.model.value_proj.weight):
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertFalse(state.usage.requires_grad)

    def test_batch_isolation_and_invalid_nan_records(self):
        e = self.records([0,1],valid=torch.tensor([[True,False],[False,True]]))
        e.keys[~e.valid] = float('nan'); e.values[~e.valid] = float('nan')
        e.keys.requires_grad_(); e.values.requires_grad_()
        state = self.update(evicted=e)
        k,v = self.model.get_kv(state)
        self.assertTrue(torch.isfinite(k).all() and torch.isfinite(v).all())
        self.assertEqual(state.source_range[:,0].tolist(),[[0,0],[1,1]])
        v[0].square().sum().backward()
        self.assertTrue(torch.isfinite(e.values.grad).all())
        self.assertFalse(e.values.grad[1].any())
        self.assertFalse(e.values.grad[~e.valid].any())

    def test_actual_working_memory_transfer_and_bf16(self):
        cfg = MemoryConfig()
        working = WorkingMemory(cfg)
        long_term = LongTermMemory(cfg)
        state = init_state(cfg,1,'cpu',torch.bfloat16)
        w,l = state.working,state.long_term
        mask = torch.ones(1,27,dtype=torch.bool)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            for i in range(7):
                z = torch.randn(1,27,17,512,dtype=torch.bfloat16)
                ids = torch.arange(27)[None]+i*27
                stats = ReadStats(torch.zeros(1,9),torch.zeros(1,16),torch.tensor([459]))
                w,_,e = working.update(w,working.encode_query(z),working.encode_value(z,z[...,:3]),ids,stats,mask)
                l = long_term.update(l,e,stats,ids,mask)
                k,v = long_term.get_kv(l)
                self.assertEqual(k.shape,(1,16,17,64))
                self.assertEqual(v.shape,(1,16,17,128))
                self.assertEqual(k.dtype,torch.bfloat16)
                self.assertTrue(torch.isfinite(v).all())
                self.assertLessEqual(l.valid.sum().item(),16)
        self.assertGreater(l.support_count.sum().item(),0)
        self.assertEqual(l.proxy.dtype,torch.bfloat16)

    def test_autocast_routing_does_not_disable_key_gradient(self):
        e = self.records([0,0,1])
        e.keys.requires_grad_(); e.values.requires_grad_()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            state = self.update(evicted=e)
            k,v = self.model.get_kv(state)
            loss = k.square().sum()+v.square().sum()
        loss.backward()
        grad = self.model.key_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_validation_and_no_mutation(self):
        e = self.records([0,1])
        state = self.update(evicted=e)
        old = {f.name:getattr(state,f.name).clone() for f in fields(state)}
        self.update(state,self.records([2],start=4),now=8)
        for name,value in old.items():
            torch.testing.assert_close(getattr(state,name),value,rtol=0,atol=0)
        with self.assertRaises(ValueError):
            self.update(state,e,now=4)
        with self.assertRaises(ValueError):
            self.update(evicted=self.records([0],start=8),now=4)
        with self.assertRaises(ValueError):
            replace(self.cfg,proxy_match_threshold=float('nan'))


if __name__ == '__main__':
    unittest.main()
