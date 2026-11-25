"""Run: python -m unittest model.memory.test_retrieval -v."""

import unittest
from dataclasses import replace, fields

import torch
from torch.nn import functional as F

from .config import MemoryConfig
from .retrieval import JointExplicitReader, PoseQueryReader
from .state import init_state


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.cfg = MemoryConfig(window_size=12, chunk_size=4, dim_rep=12,
                                working_capacity=3, write_stride=2, proxy_capacity=2,
                                explicit_key_dim=4, explicit_value_dim=6, proxy_dim=8,
                                pose_query_dim=8, pose_query_heads=2)
        self.reader = JointExplicitReader(self.cfg)
        self.pose_reader = PoseQueryReader(self.cfg)
        self.state = init_state(self.cfg, 2, 'cpu')
        self.q = F.normalize(torch.randn(2,4,17,4), dim=-1)
        self.z = torch.randn(2,4,17,12)
        self.ids = torch.arange(4,8).expand(2,-1)
        self.mask = torch.ones(2,4,dtype=torch.bool)
        self.k_l = F.normalize(torch.randn(2,2,17,4),dim=-1)
        self.v_l = torch.randn(2,2,17,6)
        self.l_valid = torch.tensor([[True,False],[True,True]])
        self.w = replace(self.state.working,
                         keys=F.normalize(torch.randn(2,3,17,4),dim=-1),
                         values=torch.randn(2,3,17,6),
                         valid=torch.tensor([[True,False,True],[False,True,False]]),
                         timestamps=torch.tensor([[0,-1,3],[-1,2,-1]]))

    def read(self, q=None, w=None, kl=None, vl=None, lv=None, mask=None):
        return self.reader(self.q if q is None else q, self.w if w is None else w,
                           self.k_l if kl is None else kl, self.v_l if vl is None else vl,
                           self.l_valid if lv is None else lv, self.ids,
                           self.mask if mask is None else mask)

    def test_joint_softmax_and_conservation_of_attention_mass(self):
        rw,rl,stats = self.read()
        keys = torch.cat((self.w.keys,self.k_l),1)
        values = torch.cat((self.w.values,self.v_l),1)
        valid = torch.cat((self.w.valid,self.l_valid),1)
        scores = torch.einsum('btjd,bnjd->btjn',self.q,keys)*2
        a = scores.masked_fill(~valid[:,None,None,:],-torch.inf).softmax(-1)
        expected = self.reader.out_proj(torch.einsum('btjn,bnjd->btjd',a,values))
        torch.testing.assert_close(rw+rl,expected)
        torch.testing.assert_close(stats.working_usage,a[...,:3].sum((1,2)))
        torch.testing.assert_close(stats.long_term_usage,a[...,3:].sum((1,2)))
        torch.testing.assert_close(stats.working_usage.sum(1)+stats.long_term_usage.sum(1),
                                   stats.query_count.float())
        self.assertFalse(stats.working_usage.requires_grad)

    def test_sources_compete_not_two_softmaxes(self):
        w = replace(self.w,keys=torch.ones_like(self.w.keys),values=torch.ones_like(self.w.values),
                    valid=torch.tensor([[True,False,False]]*2),
                    timestamps=torch.tensor([[1,-1,-1]]*2))
        lv = torch.tensor([[True,True]]*2)
        rw,rl,stats = self.read(q=torch.ones_like(self.q),w=w,kl=torch.ones_like(self.k_l),
                                vl=torch.ones_like(self.v_l),lv=lv)
        base = self.reader.out_proj(torch.ones(2,4,17,6))
        torch.testing.assert_close(rw,base/3)
        torch.testing.assert_close(rl,2*base/3)
        torch.testing.assert_close(stats.working_usage[:,0],torch.full((2,),68/3))

    def test_empty_and_padding_nan_gradients(self):
        w = replace(self.w,keys=torch.full_like(self.w.keys,float('nan')).requires_grad_(),
                    values=torch.full_like(self.w.values,float('nan')).requires_grad_(),
                    valid=torch.zeros_like(self.w.valid))
        mask = torch.tensor([[True,False,False,False],[False]*4])
        q = torch.where(mask[:,:,None,None],self.q,float('nan')).requires_grad_()
        kl = torch.full_like(self.k_l,float('nan')).requires_grad_()
        vl = torch.full_like(self.v_l,float('nan')).requires_grad_()
        rw,rl,stats = self.read(q=q,w=w,kl=kl,vl=vl,lv=torch.zeros_like(self.l_valid),mask=mask)
        self.assertFalse(rw.any() or rl.any() or stats.working_usage.any() or stats.long_term_usage.any())
        self.assertEqual(stats.query_count.tolist(),[17,0])
        (rw.sum()+rl.sum()).backward()
        for x in (q,w.keys,w.values,kl,vl):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertFalse(x.grad.any())

    def test_relative_time_bias_only_for_working(self):
        w = replace(self.w,keys=torch.ones_like(self.w.keys),valid=torch.tensor([[True,True,False]]*2),
                    timestamps=torch.tensor([[0,3,-1]]*2))
        q,kl = torch.ones_like(self.q),torch.ones_like(self.k_l)
        lv = torch.tensor([[True,False]]*2)
        _,_,neutral = self.read(q=q,w=w,kl=kl,lv=lv)
        with torch.no_grad(): self.reader.time_slope.fill_(-1)
        _,_,biased = self.read(q=q,w=w,kl=kl,lv=lv)
        self.assertTrue((biased.working_usage[:,1]>biased.working_usage[:,0]).all())
        self.assertTrue((biased.long_term_usage[:,0]>neutral.long_term_usage[:,0]).all())
        self.assertFalse(biased.working_usage[:,2].any())

    def test_read_gradients_joint_and_batch_isolation(self):
        q = self.q.clone().requires_grad_()
        w = replace(self.w,keys=self.w.keys.clone().requires_grad_(),
                    values=self.w.values.clone().requires_grad_())
        kl,vl = self.k_l.clone().requires_grad_(),self.v_l.clone().requires_grad_()
        rw,rl,_ = self.read(q=q,w=w,kl=kl,vl=vl)
        (rw[0,:,3].square().mean()+rl[0,:,3].square().mean()).backward()
        for x in (q,w.keys,w.values,kl,vl):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad[0].abs().sum().item(),0)
            self.assertFalse(x.grad[1].any())
            self.assertFalse(x.grad[0,:,:3].any() or x.grad[0,:,4:].any())
        self.assertIsNotNone(self.reader.time_slope.grad)

    def test_pose_roles_and_frame_locality(self):
        reader = self.pose_reader
        self.assertEqual(reader.body_bias.shape,(6,17))
        self.assertTrue((reader.body_bias[0]==0).all())
        self.assertEqual(reader.body_bias[2,11].item(),0)
        self.assertLess(reader.body_bias[2,14].item(),0)
        self.assertEqual(reader.body_bias[4,4].item(),0)
        r = [torch.randn_like(self.z) for _ in range(3)]
        valid = torch.ones(2,3,dtype=torch.bool)
        out = reader(self.z,*r,valid,self.mask)
        modified = self.z.clone(); modified[0,1]+=3
        changed = reader(modified,*r,valid,self.mask)
        self.assertGreater((out[0,1]-changed[0,1]).abs().sum().item(),0)
        torch.testing.assert_close(out[1],changed[1],rtol=0,atol=0)
        torch.testing.assert_close(out[0,[0,2,3]],changed[0,[0,2,3]],rtol=0,atol=0)

    def test_pose_correction_empty_and_missing_sources(self):
        valid = torch.tensor([[True,False,False],[False]*3])
        mask = torch.tensor([[True,True,False,False],[True]*4])
        z = torch.where(mask[:,:,None,None],self.z,float('nan')).requires_grad_()
        rs = torch.randn_like(z).requires_grad_()
        missing = torch.full_like(z,float('nan')).requires_grad_()
        out = self.pose_reader(z,rs,missing,missing,valid,mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(out[1].any() or out[0,2:].any())
        self.assertGreater(out[0,:2].abs().sum().item(),0)
        out.square().sum().backward()
        for x in (z,rs,missing): self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(missing.grad.any())
        self.assertFalse(z.grad[1].any() or z.grad[0,2:].any())
        self.assertGreater(self.pose_reader.queries.grad.abs().sum().item(),0)

    def test_state_not_mutated_and_validation(self):
        before = {f.name:getattr(self.w,f.name).clone() for f in fields(self.w)}
        self.read()
        for name,x in before.items(): torch.testing.assert_close(getattr(self.w,name),x,rtol=0,atol=0)
        with self.assertRaises(ValueError): self.read(w=replace(self.w,timestamps=self.w.timestamps+10))
        with self.assertRaises(ValueError): self.read(mask=self.mask.float())
        with self.assertRaises(ValueError): replace(self.cfg,pose_query_heads=3)
        with self.assertRaises(ValueError): replace(self.cfg,pose_body_bias=float('nan'))
        with self.assertRaises(ValueError): self.pose_reader(self.z,self.z,self.z,self.z,self.mask)

    def test_empty_chunk(self):
        rw,rl,stats = self.reader(self.q[:,:0],self.w,self.k_l,self.v_l,self.l_valid,self.ids[:,:0])
        self.assertEqual(rw.shape,(2,0,17,12)); self.assertEqual(rl.shape,rw.shape)
        self.assertFalse(stats.query_count.any())
        z = self.z[:,:0]
        self.assertEqual(self.pose_reader(z,z,z,z,torch.ones(2,3,dtype=torch.bool)).shape,z.shape)

    def test_default_shapes_bf16(self):
        cfg = MemoryConfig()
        reader,pose_reader = JointExplicitReader(cfg),PoseQueryReader(cfg)
        state = init_state(cfg,1,'cpu',torch.bfloat16)
        w = replace(state.working,keys=torch.randn_like(state.working.keys),
                    values=torch.randn_like(state.working.values),
                    timestamps=torch.arange(9)[None],valid=torch.ones(1,9,dtype=torch.bool))
        q = torch.randn(1,27,17,64,dtype=torch.bfloat16)
        z = torch.randn(1,27,17,512,dtype=torch.bfloat16)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            rw,rl,stats = reader(q,w,torch.randn(1,16,17,64,dtype=q.dtype),
                                 torch.randn(1,16,17,128,dtype=q.dtype),
                                 torch.ones(1,16,dtype=torch.bool),torch.arange(27)[None]+10)
            out = pose_reader(z,z,rw,rl,torch.ones(1,3,dtype=torch.bool))
        self.assertEqual(out.shape,z.shape)
        self.assertEqual(out.dtype,z.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(stats.query_count.tolist(),[459])
        torch.testing.assert_close(stats.working_usage.sum(1)+stats.long_term_usage.sum(1),
                                   stats.query_count.float())

    def test_live_working_proxy_reader_loop(self):
        from .working import WorkingMemory
        from .long_term import LongTermMemory

        working = WorkingMemory(self.cfg)
        long_term = LongTermMemory(self.cfg)
        w,l = self.state.working,self.state.long_term
        head = torch.nn.Linear(12,3)
        features = [torch.randn_like(self.z).requires_grad_() for _ in range(4)]
        for i,z in enumerate(features):
            ids = torch.arange(4).expand(2,-1)+i*4
            q = working.encode_query(z,self.mask)
            kl,vl = long_term.get_kv(l)
            rw,rl,stats = self.reader(q,w,kl,vl,l.valid,ids,self.mask)
            if i == 0:
                self.assertFalse(rw.any() or rl.any())
            if i == 2:
                self.assertTrue((stats.working_usage.sum(1)>0).all())
                self.assertTrue((stats.long_term_usage.sum(1)>0).all())
            if i > 0:
                torch.testing.assert_close(stats.working_usage.sum(1)+stats.long_term_usage.sum(1),
                                           stats.query_count.float())
            # Standalone test harness: a simple residual and head, not V1 Fusion.
            zero = torch.zeros_like(z)
            availability = torch.stack((torch.zeros(2,dtype=torch.bool),w.valid.any(1),l.valid.any(1)),1)
            rp = self.pose_reader(z,zero,rw,rl,availability,self.mask)
            u = z+rw+rl+rp
            pose = head(u)
            old_l = l
            w,feedback,evicted = working.update(w,q,working.encode_value(z,pose),ids,stats,self.mask)
            l = long_term.update(l,evicted,stats,ids,self.mask)
            self.assertLessEqual(w.valid.sum(1).max().item(),3)
            self.assertLessEqual(l.valid.sum(1).max().item(),2)
            self.assertTrue((feedback.mask.sum(1)==2).all())
            if i == 1:
                self.assertTrue(l.valid.any())
                self.assertFalse(old_l.valid.any())
        pose.square().mean().backward()
        for param in (working.key_encoder.weight,working.feature_proj.weight,
                      long_term.input_proj.weight,long_term.key_proj.weight,
                      long_term.value_proj.weight,self.reader.out_proj.weight,
                      self.pose_reader.queries):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(),0)
        self.assertGreater(sum(z.grad.abs().sum().item() for z in features[:-1] if z.grad is not None),0)


if __name__ == '__main__':
    unittest.main()
