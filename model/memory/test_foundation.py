"""CPU smoke tests for the V1 foundation.

Run from the repository root: python -m unittest model.memory.test_foundation
The backbone tests additionally require the original repository dependencies.
"""

import unittest
from dataclasses import replace

import torch
from torch import nn

from .config import MemoryConfig
from .memory_system import MemorySystem
from .state import init_state


class MemoryFoundationTest(unittest.TestCase):
    def setUp(self):
        self.config = MemoryConfig(window_size=6, chunk_size=3, dim_rep=16)
        self.system = MemorySystem(self.config)
        self.z = torch.randn(2, 6, 17, 16)
        self.head = nn.Linear(16, 3)
        self.ids = torch.arange(6).expand(2, -1)
        self.mask = torch.ones(2, 6, dtype=torch.bool)
        self.streams = torch.tensor([10, 11])

    def run_window(self, state=None, ids=None, mask=None, streams=None):
        return self.system(
            self.z, state,
            self.ids if ids is None else ids,
            self.mask if mask is None else mask,
            self.streams if streams is None else streams,
            self.head,
        )

    def test_baseline_and_chunks(self):
        pose, state = self.run_window()
        torch.testing.assert_close(pose, self.head(self.z), rtol=0, atol=0)
        self.assertEqual(state.sensory.matrix.shape, (2, 17, 4, 32, 32))
        self.assertEqual(state.sensory.matrix.dtype, torch.float32)
        self.assertFalse(state.working.valid.any())
        self.assertFalse(state.long_term.valid.any())
        chunks = list(self.system.iter_chunks(self.z, self.ids, self.mask))
        self.assertEqual(len(chunks), 2)
        torch.testing.assert_close(torch.cat([c[0] for c in chunks], dim=1), self.z)

    def test_stream_reset_and_padding(self):
        _, state = self.run_window()
        state.sensory.matrix.fill_(7)
        state.sensory.valid.fill_(True)

        mask = self.mask.clone()
        mask[1, 3:] = False
        next_ids = self.ids + 6
        next_ids = next_ids.clone()
        next_ids[1] = self.ids[1]
        _, next_state = self.run_window(
            state, next_ids, mask, torch.tensor([10, 20])
        )
        self.assertTrue((next_state.sensory.matrix[0] == 7).all())
        self.assertTrue((next_state.sensory.matrix[1] == 0).all())
        self.assertTrue((state.sensory.matrix == 7).all())  # no in-place reset
        self.assertEqual(next_state.last_frame_id.tolist(), [11, 2])
        self.assertEqual(next_state.stream_ids.tolist(), [10, 20])

        _, empty_state = self.run_window(
            next_state, self.ids, torch.zeros_like(mask), torch.tensor([30, 31])
        )
        torch.testing.assert_close(empty_state.last_frame_id, next_state.last_frame_id)
        torch.testing.assert_close(empty_state.stream_ids, next_state.stream_ids)
        torch.testing.assert_close(empty_state.sensory.matrix, next_state.sensory.matrix)

    def test_overlap_and_invalid_mask(self):
        _, state = self.run_window()
        with self.assertRaises(ValueError):
            self.run_window(state, self.ids + 3)
        with self.assertRaises(ValueError):
            self.run_window(state, self.ids + 7)
        bad_mask = self.mask.clone()
        bad_mask[0, 1] = False
        with self.assertRaises(ValueError):
            self.run_window(mask=bad_mask)

    def test_enabled_memory_smoke(self):
        system = MemorySystem(replace(self.config, enabled=True, write_stride=3))
        pose, state = system(self.z, None, self.ids, self.mask, self.streams, self.head)
        self.assertEqual(pose.shape, (2, 6, 17, 3))
        self.assertTrue(torch.isfinite(pose).all())
        self.assertTrue(state.sensory.valid.all())
        self.assertEqual(state.working.valid.sum(1).tolist(), [2, 2])

    def test_fp32_matrix_with_half_features(self):
        state = init_state(self.config, 2, "cpu", torch.float16)
        self.assertEqual(state.sensory.matrix.dtype, torch.float32)
        self.assertEqual(state.working.keys.dtype, torch.float16)
        self.assertEqual(state.long_term.proxy.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
