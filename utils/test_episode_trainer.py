"""Run: python -m unittest utils.test_episode_trainer -v."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from loss.episode_pose3d import EpisodePoseLoss
from model.memory.config import MemoryConfig
from model.memory.memory_system import MemorySystem
from utils.episode_trainer import build_episode_model, build_optimizer, forward_episode, train_episode_epoch


def batch(lengths, offset=0, windows=3, t=4):
    b = len(lengths)
    x = torch.randn(b, windows, t, 17, 3)
    y = torch.randn_like(x)
    ids = torch.arange(windows*t).expand(b, -1)
    mask = ids < torch.tensor(lengths)[:, None]
    ids = torch.where(mask, ids+offset, -1)
    return dict(x=x, y=y, valid_mask=mask.reshape(b,windows,t),
                frame_ids=ids.reshape(b,windows,t), stream_id=torch.arange(b)+offset)


class SmallMemoryModel(nn.Module):
    """Real memory system and linear encoder/head; not the full AGFormer."""
    def __init__(self, enabled=True):
        super().__init__()
        cfg = MemoryConfig(enabled=enabled, window_size=4, chunk_size=2, dim_rep=12,
                           working_capacity=2, write_stride=1, proxy_capacity=3,
                           explicit_key_dim=4, explicit_value_dim=6, proxy_dim=8,
                           sensory_heads=2, sensory_key_dim=3, sensory_value_dim=4,
                           pose_query_dim=8, pose_query_heads=2, fusion_hidden_dim=8,
                           decoder_hidden_dim=8, proxy_min_exposure=1)
        self.embed = nn.Linear(3,12)
        self.head = nn.Linear(12,3)
        self.memory_system = MemorySystem(cfg)
        self.calls = []

    def forward_window(self, x, state, ids, mask, streams):
        self.calls.append((x.shape[0], state is None))
        return self.memory_system(self.embed(x), state, ids, mask, streams, self.head)


class EpisodeTrainerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.criterion = EpisodePoseLoss()

    def test_factory_configuration_contract(self):
        # Stub only the factory boundary; the original timm dependency is not
        # required for this configuration test.
        args = SimpleNamespace(num_joints=17, dim_out=3, dim_in=3, root_rel=True,
                               add_velocity=False, use_proj_as_2d=False,
                               n_frames=243, dim_rep=512, memory={'enabled': False})
        factory = Mock(return_value=object())
        with patch.dict('sys.modules', {'utils.learning': SimpleNamespace(load_model=factory)}):
            self.assertIs(build_episode_model(args), factory.return_value)
            cfg = factory.call_args.kwargs['memory_config']
            self.assertFalse(cfg.enabled)
            self.assertEqual((cfg.window_size,cfg.chunk_size,cfg.dim_rep),(243,27,512))
            args.memory = {'enabled': True, 'chunk_size': 81, 'working_capacity': 9}
            build_episode_model(args)
            self.assertTrue(factory.call_args.kwargs['memory_config'].enabled)
            args.lambda_lv = 1
            with self.assertRaises(ValueError): build_episode_model(args)

    def test_active_rows_and_episode_reset(self):
        model = SmallMemoryModel()
        b = batch([12, 3, 7])
        p = forward_episode(model, b)
        self.assertEqual(p.shape,b['y'].shape)
        self.assertEqual(model.calls,[(3,True),(2,False),(1,False)])
        self.assertFalse(p[~b['valid_mask']].any())
        # Each sample gives the same predictions on its own for this encoder.
        for i in range(3):
            q = forward_episode(model,{k:v[i:i+1] for k,v in b.items()})
            torch.testing.assert_close(p[i],q[0],rtol=2e-5,atol=2e-6)
        self.assertTrue(model.calls[-2][1] or model.calls[-3][1])

    def test_future_window_loss_reaches_earlier_input(self):
        model = SmallMemoryModel()
        b = batch([12, 3])
        b['x'].requires_grad_()
        p = forward_episode(model,b)
        p[0,-1].square().mean().backward()
        self.assertGreater(b['x'].grad[0,0].abs().sum().item(),0)
        self.assertFalse(b['x'].grad[1].any())
        for param in (model.memory_system.sensory.k_proj.weight,
                      model.memory_system.working.key_encoder.weight,
                      model.memory_system.long_term.input_proj.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(),0)

    def test_weighted_accumulation_matches_combined_batch(self):
        a,b,c = batch([12,4]),batch([6],offset=20),batch([3],offset=40)
        model = SmallMemoryModel(enabled=False)
        other = copy.deepcopy(model)
        opt,_ = build_optimizer(model)
        opt2,_ = build_optimizer(other)
        stats = train_episode_epoch(model,[a,b,c],opt,self.criterion,'cpu',accumulation_steps=2)
        combined = {k:torch.cat((a[k],b[k]),dim=0) for k in a}
        ref = train_episode_epoch(other,[combined,c],opt2,self.criterion,'cpu')
        self.assertEqual(stats.optimizer_steps,2)
        self.assertEqual(stats.valid_frames,25)
        self.assertEqual(stats.valid_pairs,21)
        for x,y in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
        for x,y in zip(stats[:4],ref[:4]): self.assertAlmostEqual(x,y,places=5)

    def test_real_memory_optimizer_step_and_scheduler(self):
        model = SmallMemoryModel()
        opt,sched = build_optimizer(model,memory_lr=1e-3)
        before = [p.detach().clone() for p in (model.embed.weight,model.memory_system.sensory.k_proj.weight,
                    model.memory_system.working.key_encoder.weight,model.memory_system.long_term.input_proj.weight)]
        stats = train_episode_epoch(model,[batch([12,7]),batch([8],offset=30)],opt,self.criterion,'cpu',
                                    accumulation_steps=2,grad_clip=1.0)
        self.assertEqual(stats.optimizer_steps,1)
        for old,new in zip(before,(model.embed.weight,model.memory_system.sensory.k_proj.weight,
                    model.memory_system.working.key_encoder.weight,model.memory_system.long_term.input_proj.weight)):
            self.assertGreater((old-new).abs().sum().item(),0)
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'],4.95e-4)
        self.assertAlmostEqual(opt.param_groups[1]['lr'],9.9e-4)
        self.assertEqual(sum(first for _,first in model.calls),2)

    def test_all_padding_skips_optimizer_and_encoder(self):
        model = SmallMemoryModel()
        opt,_ = build_optimizer(model)
        b = batch([0,0])
        stats = train_episode_epoch(model,[b],opt,self.criterion,'cpu')
        self.assertEqual(stats,(0.,0.,0.,0.,0,0,0))
        self.assertEqual(model.calls,[])
        self.assertFalse(opt.state)

    def test_partial_padding_sanitized_before_encoder(self):
        model = SmallMemoryModel()
        b = batch([5,0])
        b['x'][~b['valid_mask']] = float('nan')
        p = forward_episode(model,b)
        self.assertTrue(torch.isfinite(p).all())
        self.assertEqual(model.calls,[(1,True),(1,False)])
        p.square().mean().backward()
        self.assertTrue(torch.isfinite(model.embed.weight.grad).all())

    def test_bf16_training_and_invalid_episode(self):
        model = SmallMemoryModel()
        opt,_ = build_optimizer(model)
        b = batch([12,5])
        stats = train_episode_epoch(model,[b],opt,self.criterion,'cpu',precision='bf16',grad_clip=1.0)
        self.assertEqual(stats.optimizer_steps,1)
        self.assertTrue(torch.isfinite(torch.tensor(stats.total)))
        b['frame_ids'][0,1,0] += 1
        with self.assertRaises(ValueError): forward_episode(model,b)
        with self.assertRaises(ValueError): train_episode_epoch(model,[],opt,self.criterion,'cpu',precision='fp16')

    def test_nonfinite_loss_rejected_before_step(self):
        model = SmallMemoryModel(enabled=False)
        opt,_ = build_optimizer(model)
        b = batch([4])
        b['y'][0,0,0,1,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            train_episode_epoch(model,[b],opt,self.criterion,'cpu')
        self.assertFalse(opt.state)


if __name__ == '__main__':
    unittest.main()
