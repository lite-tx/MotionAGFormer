"""Run: python -m unittest utils.test_episode_evaluation -v."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.reader.h36m_episode import H36MEpisodeDataset, build_episode_loader
from utils.episode_evaluation import StreamMetrics, aligned_error, evaluate_h36m, flip_h36m, restore_pose
from utils.test_episode_trainer import SmallMemoryModel


class TrackingModel(nn.Module):
    """Observe the evaluator's state ownership; predict input xyz unchanged."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_window(self, x, state, ids, mask, streams):
        prior = state
        token = object()
        self.calls.append((int(streams.item()), prior, token, x.clone()))
        return x, token


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'data.pkl'
        source = np.array(['A']*13+['B']*3+['s_09_act_05_subact_02_cam01']*2)
        n = len(source)
        rng = np.random.default_rng(54)
        xy = rng.uniform(100, 900, (n,17,2)).astype(np.float32)
        conf = np.ones((n,17),dtype=np.float32)
        factor = np.linspace(0.8,1.2,n)
        x = np.concatenate((xy/500-1,conf[...,None]),axis=-1)
        gt = restore_pose(x,factor)+10
        raw = dict(source=source,camera_name=np.array(['55011271']*n),
                   joint_2d=xy,confidence=conf,joint3d_image=np.zeros((n,17,3),np.float32),
                   action=np.array(['walk']*13+['sit']*5),
                   **{'2.5d_factor':factor,'joints_2.5d_image':gt})
        with self.path.open('wb') as f: pickle.dump({'test':raw,'train':raw},f)
        self.ds = H36MEpisodeDataset(self.path,'test',window_size=4,windows_per_episode=2,
                                      include_eval_metadata=True)

    def test_stream_continuity_tta_and_metrics(self):
        model = TrackingModel()
        result = evaluate_h36m(model,build_episode_loader(self.ds,1),'cpu')
        self.assertLess(result['mpjpe'],1e-4)
        self.assertLess(result['p_mpjpe'],1e-4)
        self.assertLess(result['acceleration'],1e-4)
        self.assertEqual(result['valid_frames'],16)
        self.assertEqual(result['excluded_frames'],2)
        self.assertEqual(result['acceleration_triples'],12)
        self.assertEqual(result['actions']['walk']['frames'],13)
        self.assertEqual(len(model.calls),12)
        for i in range(0,len(model.calls),2):
            a,b = model.calls[i:i+2]
            self.assertEqual(a[0],b[0]); self.assertIsNot(a[2],b[2])
            torch.testing.assert_close(b[3],flip_h36m(a[3]))
            if i==0 or model.calls[i-2][0]!=a[0]:
                self.assertIsNone(a[1]); self.assertIsNone(b[1])
            else:
                self.assertIs(a[1],model.calls[i-2][2]); self.assertIs(b[1],model.calls[i-1][2])

    def test_reset_diagnostic_and_single_row_enforcement(self):
        model = TrackingModel()
        result = evaluate_h36m(model,build_episode_loader(self.ds,1),'cpu',flip=False,carry_memory=False)
        self.assertTrue(all(c[1] is None for c in model.calls))
        self.assertFalse(result['carry_memory'] or result['flip'])
        with self.assertRaises(ValueError): evaluate_h36m(model,build_episode_loader(self.ds,2),'cpu')
        no_meta = H36MEpisodeDataset(self.path,'test',window_size=4)
        with self.assertRaises(ValueError): evaluate_h36m(model,build_episode_loader(no_meta,1),'cpu')

    def test_alignment_and_coordinate_restore(self):
        rng = np.random.default_rng(5)
        gt = rng.normal(size=(5,17,3))
        rot = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        pred = 3*gt @ rot + 2
        np.testing.assert_allclose(aligned_error(pred,gt),0,atol=1e-12)
        self.assertTrue(np.all(aligned_error(-gt,gt)>0))
        collapsed = aligned_error(np.zeros_like(gt),gt)
        np.testing.assert_allclose(collapsed,np.linalg.norm(gt-gt.mean(1,keepdims=True),axis=-1).mean(1))
        p = restore_pose(gt,np.ones(5)*2)
        np.testing.assert_allclose(p[:,1:],gt[:,1:]*1000)
        self.assertFalse(p[:,0].any())

    def test_action_macro_and_acceleration_counts(self):
        m = StreamMetrics()
        p = np.zeros((5,17,3)); g = p.copy()
        p[:,:,0] = np.array([0,1,4,9,16])[:,None]
        m.update(p[:2],g[:2],np.arange(2),'A',0)
        m.update(p[2:],g[2:],np.arange(2,5),'A',0)
        m.update(np.ones((1,17,3)),np.zeros((1,17,3)),np.array([0]),'B',1)
        r = m.result()
        self.assertAlmostEqual(r['mpjpe'],(6+np.sqrt(3))/2)
        self.assertAlmostEqual(r['acceleration'],2)
        self.assertEqual(r['acceleration_triples'],3)
        self.assertIsNone(r['actions']['B']['acceleration'])

    def test_real_memory_stream_eval(self):
        model = SmallMemoryModel()
        r = evaluate_h36m(model,build_episode_loader(self.ds,1),'cpu',precision='bf16')
        self.assertTrue(np.isfinite(r['mpjpe']) and np.isfinite(r['p_mpjpe']))
        self.assertEqual(r['valid_frames'],16)


if __name__ == '__main__':
    unittest.main()
