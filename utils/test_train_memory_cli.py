"""CLI smoke test with real memory/trainer/evaluation and a small encoder."""

import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train_memory
from utils import test_episode_evaluation
from utils.test_episode_trainer import SmallMemoryModel


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class TrainingCLITest(unittest.TestCase):
    def test_train_resume_evaluate_and_output_guard(self):
        # Reuse the synthetic raw-format H36M fixture, not a stub Dataset.
        fixture = test_episode_evaluation.EvaluationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        out = fixture.path.parent/'run'
        config = Config(data_root=str(fixture.path.parent),dt_file=fixture.path.name,
                        n_frames=4,windows_per_episode=3,episode_stride=4,
                        flip=True,num_workers=0,precision='fp32',batch_size=2,
                        lambda_scale=0.5,lambda_3d_velocity=20.,root_rel=True,
                        learning_rate=5e-4,weight_decay=.01,memory_lr=None,lr_decay=.99,
                        epochs=1,accumulation_steps=1,grad_clip=1.)
        # timm/easydict are absent from the isolated CPU test environment.
        # Substitute the original factory/config boundary only.
        tools = SimpleNamespace(get_config=lambda _: Config(config),set_random_seed=torch.manual_seed)
        original_tools = sys.modules.get('utils.tools')
        sys.modules['utils.tools'] = tools
        def restore_tools():
            if original_tools is None:
                sys.modules.pop('utils.tools', None)
            else:
                sys.modules['utils.tools'] = original_tools
        self.addCleanup(restore_tools)
        def run(*extra):
            args = ['train_memory.py','--device','cpu','--output-dir',str(out),*extra]
            with patch('utils.episode_trainer.build_episode_model',side_effect=lambda _: SmallMemoryModel()), \
                    patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()) as output:
                train_memory.main()
            return output.getvalue()
        run()
        self.assertTrue((out/'best_epoch.pth.tr').is_file())
        latest = out/'latest_epoch.pth.tr'
        self.assertEqual(torch.load(latest,weights_only=True)['next_epoch'],1)
        with self.assertRaises(FileExistsError): run()
        config.epochs=2
        run('--resume',str(latest))
        self.assertEqual(torch.load(latest,weights_only=True)['next_epoch'],2)
        self.assertEqual(len((out/'metrics.jsonl').read_text().splitlines()),2)
        report = run('--eval-only','--checkpoint',str(latest),'--reset-memory')
        metrics = json.loads(report[report.index('{'):])
        self.assertFalse(metrics['carry_memory'])
        self.assertEqual(metrics['valid_frames'],16)


if __name__ == '__main__':
    unittest.main()
