"""Run: python -m unittest data.reader.test_h36m_episode -v."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from .h36m_episode import H36MEpisodeDataset, build_episode_loader


def fixture():
    # Same source under two cameras, a short source, then a repeated name.
    source = np.array(['A']*13 + ['A']*7 + ['B']*3 + ['A']*2)
    camera = np.array(['54138969']*13 + ['55011271']*7 + ['55011271']*3 + ['54138969']*2)
    n = len(source)
    coords = np.arange(n*17*3, dtype=np.float32).reshape(n,17,3)
    return dict(source=source, camera_name=camera, joint_2d=coords[..., :2],
                joint3d_image=coords+20, confidence=np.full((n,17),0.7,dtype=np.float32))


class EpisodeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'h36m.pkl'
        self.raw = fixture()
        self.save()

    def save(self):
        with self.path.open('wb') as f:
            pickle.dump({'train':self.raw, 'test':self.raw}, f)

    def dataset(self, split='train', **kwargs):
        options = dict(window_size=6, windows_per_episode=2, flip=False)
        options.update(kwargs)
        return H36MEpisodeDataset(self.path, split, **options)

    def test_normalization_matches_original_and_confidence(self):
        ds = self.dataset()
        for i in (0, 12, 13, 24):
            height = 1002 if self.raw['camera_name'][i]=='54138969' else 1000
            xy = self.raw['joint_2d'][i].astype(np.float32)/1000*2-[1,height/1000]
            gt = self.raw['joint3d_image'][i].astype(np.float32)/1000*2
            gt[..., :2] = gt[..., :2]-[1,height/1000]
            np.testing.assert_allclose(ds.x[i,:,:2],xy,atol=2e-7)
            np.testing.assert_allclose(ds.y[i],gt,atol=2e-7)
            np.testing.assert_allclose(ds.x[i,:,2],0.7)
        del self.raw['confidence']; self.save()
        self.assertTrue((self.dataset().x[...,2]==1).all())

    def test_training_overlap_only_between_episodes(self):
        ds = self.dataset(episode_stride=2)
        self.assertEqual([(e.start,e.stop) for e in ds.episodes],[(0,12),(2,13),(13,20),(20,23),(23,25)])
        a,b = ds[0],ds[1]
        self.assertEqual(a['frame_ids'].tolist(),[list(range(6)),list(range(6,12))])
        self.assertEqual(b['frame_ids'].flatten().tolist(),list(range(2,13))+[-1])
        self.assertEqual(a['sequence_id'],b['sequence_id'])
        self.assertNotEqual(a['stream_id'],b['stream_id'])
        self.assertTrue((a['global_frame_ids'][1,0]==a['global_frame_ids'][0,-1]+1).item())

    def test_test_frames_covered_once_in_order_and_streams(self):
        ds = self.dataset('test', flip=True)
        self.assertEqual([(e.start,e.stop) for e in ds.episodes],[(0,12),(12,13),(13,20),(20,23),(23,25)])
        seen = []
        for sample in ds:
            seen.extend(sample['global_frame_ids'][sample['valid_mask']].tolist())
            self.assertFalse(sample['flipped'])
        self.assertEqual(seen,list(range(25)))
        self.assertEqual(ds[0]['stream_id'],ds[1]['stream_id'])
        self.assertNotEqual(ds[1]['stream_id'],ds[2]['stream_id'])
        self.assertNotEqual(ds[0]['stream_id'],ds[4]['stream_id'])
        self.assertEqual(ds[1]['frame_ids'][0,0].item(),12)

    def test_suffix_padding_is_not_a_real_frame(self):
        ds = self.dataset('test')
        sample = ds[2]
        self.assertEqual(sample['x'].shape,(2,6,17,3))
        self.assertEqual(sample['valid_mask'].sum().item(),7)
        self.assertEqual(sample['frame_ids'][1].tolist(),[6,-1,-1,-1,-1,-1])
        self.assertTrue((sample['global_frame_ids'][~sample['valid_mask']]==-1).all())
        self.assertFalse(sample['y'][~sample['valid_mask']].any())
        torch.testing.assert_close(sample['x'][1,1:],sample['x'][1,0].expand(5,-1,-1))
        self.assertFalse(ds[4]['valid_mask'][1].any())

    def test_episode_flip_shared_by_all_windows_and_labels(self):
        ds = self.dataset(flip=True)
        with patch('torch.rand',return_value=torch.tensor(0.1)) as draw:
            flipped = ds[0]
            draw.assert_called_once()
        with patch('torch.rand',return_value=torch.tensor(0.9)):
            plain = ds[0]
        self.assertTrue(flipped['flipped']); self.assertFalse(plain['flipped'])
        order = [0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]
        for key in ('x','y'):
            expected = plain[key][:,:,order,:].clone()
            expected[...,0] *= -1
            torch.testing.assert_close(flipped[key],expected)
        for key in ('frame_ids','global_frame_ids','valid_mask'):
            torch.testing.assert_close(flipped[key],plain[key])
        torch.testing.assert_close(ds[0]['x'][0,0,:,2],plain['x'][0,0,:,2])

    def test_loader_shuffles_episodes_and_collates(self):
        train = build_episode_loader(self.dataset(),2,generator=torch.Generator().manual_seed(19))
        test = build_episode_loader(self.dataset('test'),2)
        self.assertIsInstance(train.sampler,RandomSampler)
        self.assertIsInstance(test.sampler,SequentialSampler)
        batch = next(iter(train))
        self.assertEqual(batch['x'].shape,(2,2,6,17,3))
        self.assertEqual(batch['frame_ids'].shape,(2,2,6))
        self.assertEqual(batch['stream_id'].shape,(2,))
        self.assertEqual(batch['valid_mask'].dtype,torch.bool)
        seen = [i for b in train for i in b['episode_id'].tolist()]
        self.assertEqual(sorted(seen),list(range(len(train.dataset))))

    def test_invalid_configuration_and_metadata(self):
        for kwargs in (dict(episode_stride=13),dict(windows_per_episode=0)):
            with self.assertRaises(ValueError): self.dataset(**kwargs)
        with self.assertRaises(ValueError): self.dataset('test',episode_stride=2)
        self.raw['camera_name'][0] = 'unknown'; self.save()
        with self.assertRaises(ValueError): self.dataset()

    def test_real_memory_scheduler_consumes_episode_metadata(self):
        from model.memory.config import MemoryConfig
        from model.memory.memory_system import MemorySystem
        ds = self.dataset('test')
        # Long episode followed by its short tail, with an entirely padded
        # second window. Use the actual post-backbone memory system.
        cfg = MemoryConfig(enabled=True,window_size=6,chunk_size=3,dim_rep=12,
                           working_capacity=3,write_stride=1,proxy_capacity=4,
                           sensory_heads=2,sensory_key_dim=3,sensory_value_dim=4,
                           explicit_key_dim=4,explicit_value_dim=6,proxy_dim=8,
                           pose_query_dim=8,pose_query_heads=2,fusion_hidden_dim=8,
                           decoder_hidden_dim=8,proxy_min_exposure=1)
        system = MemorySystem(cfg)
        embed,head = nn.Linear(3,12),nn.Linear(12,3)
        state = None
        for batch in build_episode_loader(ds,1):
            for w in range(ds.windows_per_episode):
                z = embed(batch['x'][:,w])
                pose,state = system(z,state,batch['frame_ids'][:,w],batch['valid_mask'][:,w],batch['stream_id'],head)
                self.assertTrue(torch.isfinite(pose).all())
                self.assertFalse(pose[~batch['valid_mask'][:,w]].any())
            mask = batch['valid_mask'].flatten()
            self.assertEqual(state.last_frame_id.item(),batch['frame_ids'].flatten()[mask][-1].item())
        self.assertEqual(state.stream_ids.item(),3)
        self.assertEqual(state.last_frame_id.item(),1)


if __name__ == '__main__':
    unittest.main()
