"""Run: python -m unittest utils.test_episode_checkpoint -v."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from loss.episode_pose3d import EpisodePoseLoss
from utils.episode_checkpoint import load_pretrained, load_weights, resume_checkpoint, save_checkpoint
from utils.episode_trainer import build_optimizer, train_episode_epoch
from utils.test_episode_trainer import SmallMemoryModel, batch


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'checkpoint.pt'
        torch.manual_seed(61)
        self.base = SmallMemoryModel(enabled=False)
        self.model = SmallMemoryModel()
        self.opt, self.scheduler = build_optimizer(self.model)
        self.generator = torch.Generator().manual_seed(63)
        self.config = {'epochs':60,'seed':61,'memory':{'enabled':True}}

    def test_original_prefixed_checkpoint_and_memory_initialization(self):
        before = {k:v.clone() for k,v in self.model.state_dict().items() if k.startswith('memory_system.')}
        torch.save({'model':{'module.'+k:v for k,v in self.base.state_dict().items()}},self.path)
        loaded, missing = load_pretrained(self.path,self.model)
        self.assertEqual(loaded,len(self.base.state_dict()))
        self.assertEqual(set(missing),set(before))
        for k,v in self.base.state_dict().items(): torch.testing.assert_close(self.model.state_dict()[k],v)
        for k,v in before.items(): torch.testing.assert_close(self.model.state_dict()[k],v)
        load_weights(self.path,self.base)
        with self.assertRaises(ValueError): load_weights(self.path,self.model)

    def test_shape_missing_unexpected_and_prefix_collision(self):
        original = self.base.state_dict()
        key = next(iter(original))
        for modified in ({k:v for k,v in original.items() if k!=key},
                         dict(original,unexpected=torch.ones(1)),
                         {**original,key:torch.ones(1)},
                         {**original,'module.'+key:original[key]}):
            torch.save(modified,self.path)
            with self.assertRaises(ValueError): load_pretrained(self.path,self.model)

    def test_epoch_resume_reproduces_rng_and_optimizer_update(self):
        criterion = EpisodePoseLoss()
        a = batch([12,7]); b = batch([9,6],offset=20)
        train_episode_epoch(self.model,[a],self.opt,criterion,'cpu')
        self.scheduler.step()
        save_checkpoint(self.path,self.model,self.opt,self.scheduler,next_epoch=1,best_mpjpe=41.5,
                        config=self.config,generator=self.generator,device='cpu')
        expected_rng = (random.random(),np.random.rand(),torch.rand(4),torch.rand(4,generator=self.generator))
        train_episode_epoch(self.model,[b],self.opt,criterion,'cpu')
        other = SmallMemoryModel(); opt,sched = build_optimizer(other)
        gen = torch.Generator()
        epoch,best = resume_checkpoint(self.path,other,opt,sched,config={**self.config,'epochs':70},generator=gen,device='cpu')
        self.assertEqual((epoch,best),(1,41.5)); self.assertAlmostEqual(opt.param_groups[0]['lr'],4.95e-4)
        self.assertEqual(random.random(),expected_rng[0]); self.assertEqual(np.random.rand(),expected_rng[1])
        torch.testing.assert_close(torch.rand(4),expected_rng[2]); torch.testing.assert_close(torch.rand(4,generator=gen),expected_rng[3])
        train_episode_epoch(other,[b],opt,criterion,'cpu')
        for p,q in zip(self.model.parameters(),other.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(list(self.path.parent.glob('*.tmp')),[])
        load_weights(self.path,other)
        with self.assertRaises(ValueError): resume_checkpoint(self.path,other,opt,sched,
                    config={**self.config,'seed':0},generator=gen,device='cpu')


if __name__ == '__main__':
    unittest.main()
