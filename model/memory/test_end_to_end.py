"""Actual AGFormer regression (requires the original repository dependencies).

Small dimensions keep the test suitable for CPU; the full Base can be
profiled separately with verify_memory.py. No mock encoder or memory module.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.episode_checkpoint import load_pretrained
from utils.episode_trainer import build_episode_model, forward_episode
from verify_memory import make_batch, read_config, verify


CONFIG = Path(__file__).resolve().parents[2]/'configs/h36m/MotionAGFormer-memory-base.yaml'


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(111)
        self.cfg = read_config(CONFIG,tiny=True)

    def test_actual_backbone_future_loss_and_memory_gradients(self):
        model = build_episode_model(SimpleNamespace(**self.cfg))
        batch = make_batch(self.cfg,1)
        batch['x'].requires_grad_()
        prediction = forward_episode(model,batch)
        self.assertEqual(prediction.shape,(1,2,18,17,3))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction[:,1].square().mean().backward()
        self.assertGreater(batch['x'].grad[:,0].abs().sum().item(),0.)
        for prefix in ('layers.','head.','memory_system.sensory.', 'memory_system.working.',
                       'memory_system.long_term.','memory_system.explicit_reader.',
                       'memory_system.pose_reader.','memory_system.fusion.','memory_system.decoder.'):
            grads = [p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads,prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads),prefix)
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0.,prefix)

    def test_original_format_backbone_checkpoint_and_skip(self):
        base_cfg = dict(self.cfg,memory=dict(self.cfg['memory'],enabled=False))
        base = build_episode_model(SimpleNamespace(**base_cfg)).eval()
        model = build_episode_model(SimpleNamespace(**self.cfg)).eval()
        initial_memory = {k:v.clone() for k,v in model.state_dict().items() if k.startswith('memory_system.')}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original_format.pt'
            torch.save({'model':{'module.'+k:v for k,v in base.state_dict().items()}},path)
            loaded,missing = load_pretrained(path,model)
        self.assertEqual(loaded,len(base.state_dict()))
        self.assertEqual(set(missing),set(initial_memory))
        for k,v in base.state_dict().items():
            torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        for k,v in initial_memory.items():
            torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        batch = make_batch(self.cfg,1)
        with torch.no_grad():
            a,b = forward_episode(base,batch),forward_episode(model,batch)
        torch.testing.assert_close(a[:,0,:3],b[:,0,:3])
        self.assertGreater((a[:,0,3:]-b[:,0,3:]).abs().sum().item(),0.)

    def test_actual_backbone_bf16_optimizer_smoke(self):
        report = verify(self.cfg,'cpu',precision='bf16',steps=1)
        self.assertEqual(report['valid_frames'],36)
        self.assertEqual(report['losses'][0]['optimizer_steps'],1)
        self.assertTrue(all(v>0 for v in report['parameter_updates_l1'].values()))


if __name__=='__main__':
    unittest.main()
