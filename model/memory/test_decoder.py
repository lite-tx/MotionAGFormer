"""Run: python -m unittest model.memory.test_decoder -v."""

import unittest
from dataclasses import replace

import torch
from torch import nn

from .config import MemoryConfig
from .decoder import AdaptiveMemoryFusion, ResidualMLPDecoder
from .long_term import LongTermMemory
from .retrieval import JointExplicitReader, PoseQueryReader
from .state import init_state
from .working import WorkingMemory


class DecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.cfg = MemoryConfig(window_size=12, chunk_size=4, dim_rep=12,
                                working_capacity=3, write_stride=2,
                                proxy_capacity=2, explicit_key_dim=4,
                                explicit_value_dim=6, proxy_dim=8,
                                pose_query_dim=8, pose_query_heads=2,
                                fusion_hidden_dim=8, decoder_hidden_dim=8)
        self.fusion = AdaptiveMemoryFusion(self.cfg)
        self.decoder = ResidualMLPDecoder(self.cfg)
        self.z = torch.randn(2,4,17,12)
        self.reads = [torch.randn_like(self.z) for _ in range(4)]
        self.valid = torch.ones(2,3,dtype=torch.bool)
        self.mask = torch.ones(2,4,dtype=torch.bool)

    def test_formula_and_joint_explicit_weight(self):
        h,g = self.fusion(self.z,*self.reads,self.valid,return_weights=True)
        rs,rw,rl,rp = self.reads
        expected = self.z+g[...,1:2]*rs+g[...,2:3]*(rw+rl)+g[...,3:4]*rp
        torch.testing.assert_close(h,expected)
        torch.testing.assert_close(g.sum(-1),torch.ones_like(g[...,0]))
        torch.testing.assert_close(g[...,0],torch.full_like(g[...,0],0.9))
        h2,g2 = self.fusion(self.z,rs,rw+rl,torch.zeros_like(rl),rp,self.valid,return_weights=True)
        torch.testing.assert_close(h,h2)
        torch.testing.assert_close(g,g2)

    def test_all_availability_combinations(self):
        for bits in range(8):
            v = torch.tensor([[bool(bits & (1<<i)) for i in range(3)]]*2)
            h,g = self.fusion(self.z,*self.reads,v,return_weights=True)
            allowed = [True, bool(bits&1), bool(bits&6), bool(bits)]
            self.assertFalse(g[..., [not x for x in allowed]].any())
            torch.testing.assert_close(g.sum(-1),torch.ones_like(g[...,0]))
            if not bits:
                torch.testing.assert_close(h,self.z,rtol=0,atol=0)
                torch.testing.assert_close(self.decoder(h,v),self.z,rtol=0,atol=0)

    def test_padding_and_empty_identity_nan_safe(self):
        mask = torch.tensor([[True,True,False,False],[True]*4])
        valid = torch.tensor([[True,False,True],[False]*3])
        z = torch.where(mask[:,:,None,None],self.z,float('nan')).requires_grad_()
        rs,rw,rl,rp = [torch.where((mask & active[:,None])[:,:,None,None],r,float('nan')).requires_grad_()
                       for r,active in zip(self.reads,[valid[:,0],valid[:,1],valid[:,2],valid.any(1)])]
        h,g = self.fusion(z,rs,rw,rl,rp,valid,mask,return_weights=True)
        u = self.decoder(h,valid,mask,memory_strength=g[...,1:].sum(-1,keepdim=True))
        self.assertTrue(torch.isfinite(u[mask]).all())
        torch.testing.assert_close(u[1],z[1],rtol=0,atol=0)
        torch.testing.assert_close(u[~mask],z[~mask],equal_nan=True,rtol=0,atol=0)
        self.assertFalse(g[~mask][...,1:].any())
        u[mask].square().mean().backward()
        for x in (z,rs,rw,rl,rp): self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(rw.grad.any())
        self.assertFalse(rs.grad[1].any() or rl.grad[1].any() or rp.grad[1].any())
        self.assertFalse(z.grad[~mask].any())

    def test_decoder_formula_and_skip_strength(self):
        h = self.z.clone().requires_grad_()
        delta = self.decoder.mlp(self.decoder.norm(h))
        strength = torch.rand(*h.shape[:3],1)
        out = self.decoder(h,self.valid,memory_strength=strength)
        torch.testing.assert_close(out,h+0.1*strength*delta)
        torch.testing.assert_close(self.decoder(h,self.valid),h+0.1*delta)
        torch.testing.assert_close(self.decoder(h,self.valid,memory_strength=torch.zeros_like(strength)),h,rtol=0,atol=0)
        torch.testing.assert_close(self.decoder(h,self.valid,memory_strength=torch.ones_like(strength)),self.decoder(h,self.valid))
        with self.assertRaises(ValueError): self.decoder(h,self.valid,memory_strength=strength+2)

    def test_backward_into_reads_gate_decoder_and_head(self):
        reads = [r.clone().requires_grad_() for r in self.reads]
        z = self.z.clone().requires_grad_()
        head = nn.Linear(12,3)
        h,g = self.fusion(z,*reads,self.valid,return_weights=True)
        u = self.decoder(h,self.valid,memory_strength=g[...,1:].sum(-1,keepdim=True))
        head(u).square().mean().backward()
        # Final gate weight is zero-initialized; earlier gate layers can learn
        # after the first optimizer step. Readers and decoder learn immediately.
        for x in (z,*reads,self.fusion.gate[-1].weight,self.decoder.gain_logit,
                  self.decoder.mlp[0].weight,head.weight):
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(),0)

    def test_per_joint_per_frame_and_batch_isolation(self):
        with torch.no_grad(): self.fusion.gate[-1].weight.normal_(0,.1)
        def run(z, reads):
            h,g = self.fusion(z,*reads,self.valid,return_weights=True)
            return self.decoder(h,self.valid,memory_strength=g[...,1:].sum(-1,keepdim=True))
        old = run(self.z,self.reads)
        modified = [r.clone() for r in self.reads]
        modified[0][0,1,3] += 5
        new = run(self.z,modified)
        diff = old-new
        self.assertGreater(diff[0,1,3].abs().sum().item(),0)
        diff[0,1,3] = 0
        torch.testing.assert_close(diff,torch.zeros_like(diff),rtol=0,atol=0)

    def test_default_shape_bf16(self):
        cfg = MemoryConfig()
        fusion,decoder = AdaptiveMemoryFusion(cfg),ResidualMLPDecoder(cfg)
        z = torch.randn(2,27,17,512,dtype=torch.bfloat16)
        valid = torch.tensor([[True,True,True],[False]*3])
        with torch.autocast('cpu',dtype=torch.bfloat16):
            h,g = fusion(z,z,z,z,z,valid,return_weights=True)
            u = decoder(h,valid,memory_strength=g[...,1:].sum(-1,keepdim=True))
        self.assertEqual(u.shape,z.shape)
        self.assertEqual(u.dtype,z.dtype)
        self.assertEqual(g.dtype,torch.float32)
        self.assertTrue(torch.isfinite(u).all())
        torch.testing.assert_close(u[1],z[1],rtol=0,atol=0)

    def test_empty_chunk_and_validation(self):
        z = self.z[:,:0]
        h,g = self.fusion(z,z,z,z,z,self.valid,return_weights=True)
        self.assertEqual(h.shape,z.shape)
        self.assertEqual(g.shape,(2,0,17,4))
        self.assertEqual(self.decoder(h,self.valid).shape,z.shape)
        with self.assertRaises(ValueError): self.fusion(self.z,*self.reads,self.mask)
        with self.assertRaises(ValueError): self.decoder(self.z,self.valid,self.mask.float())
        with self.assertRaises(ValueError): replace(self.cfg,fusion_initial_memory_mass=0)
        with self.assertRaises(ValueError): replace(self.cfg,decoder_initial_gain=float('nan'))

    def test_live_explicit_read_fusion_predict_update(self):
        working, long_term = WorkingMemory(self.cfg), LongTermMemory(self.cfg)
        reader, pose_reader = JointExplicitReader(self.cfg), PoseQueryReader(self.cfg)
        w,l = init_state(self.cfg,2,'cpu').working,init_state(self.cfg,2,'cpu').long_term
        head = nn.Linear(12,3)
        features = [torch.randn_like(self.z).requires_grad_() for _ in range(4)]
        for i,z in enumerate(features):
            ids = torch.arange(4).expand(2,-1)+i*4
            q = working.encode_query(z,self.mask)
            kl,vl = long_term.get_kv(l)
            rw,rl,stats = reader(q,w,kl,vl,l.valid,ids,self.mask)
            # Isolate this stage: real W/L + readers, sensory unavailable.
            rs = torch.zeros_like(z)
            available = torch.stack((torch.zeros(2,dtype=torch.bool),w.valid.any(1),l.valid.any(1)),1)
            rp = pose_reader(z,rs,rw,rl,available,self.mask)
            h,g = self.fusion(z,rs,rw,rl,rp,available,self.mask,return_weights=True)
            u = self.decoder(h,available,self.mask,memory_strength=g[...,1:].sum(-1,keepdim=True))
            pose = head(u)
            if i == 0: torch.testing.assert_close(pose,head(z),rtol=0,atol=0)
            if i >= 2: self.assertTrue((stats.long_term_usage.sum(1)>0).all())
            w,_,e = working.update(w,q,working.encode_value(z,pose),ids,stats,self.mask)
            l = long_term.update(l,e,stats,ids,self.mask)
        pose.square().mean().backward()
        for p in (working.key_encoder.weight,long_term.input_proj.weight,
                  reader.out_proj.weight,pose_reader.queries,self.fusion.gate[-1].weight,
                  self.decoder.mlp[0].weight,head.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertTrue(any(z.grad is not None and z.grad.abs().sum()>0 for z in features[:-1]))


if __name__ == '__main__':
    unittest.main()
