"""Chunk-level orchestration of the three historical memories.

Only the encoded window is partitioned. A chunk reads its incoming state,
refines and predicts, then commits Working -> Sensory / Long-term updates.
The caller owns the state and may carry it into the next contiguous window.
"""

from dataclasses import replace

import torch
from torch import nn

from .config import MemoryConfig
from .state import advance_state, prepare_state


class MemorySystem(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        # Disabled mode must not introduce new parameter or buffer keys.
        if config.enabled:
            from .sensory import SensoryMemory
            from .working import WorkingMemory
            from .long_term import LongTermMemory
            from .retrieval import JointExplicitReader, PoseQueryReader
            from .decoder import AdaptiveMemoryFusion, ResidualMLPDecoder

            self.sensory = SensoryMemory(config)
            self.working = WorkingMemory(config)
            self.long_term = LongTermMemory(config)
            self.explicit_reader = JointExplicitReader(config)
            self.pose_reader = PoseQueryReader(config)
            self.fusion = AdaptiveMemoryFusion(config)
            self.decoder = ResidualMLPDecoder(config)

    def iter_chunks(self, z, frame_ids, valid_mask):
        """Partition encoded features (never the raw backbone input)."""
        size = self.config.chunk_size
        for start in range(0, z.shape[1], size):
            end = start + size
            yield z[:, start:end], frame_ids[:, start:end], valid_mask[:, start:end]

    def _process_chunk(self, z, state, ids, mask, streams, head):
        """Internal step; inputs and stream continuity validated by forward."""
        valid = mask[:, :, None, None]
        # Mask before the linear head too: a NaN padding feature could otherwise
        # poison parameter gradients even when the loss ignores padded frames.
        z = torch.where(valid, z, 0.0)
        if not mask.any():
            return torch.where(valid, head(z), 0.0), state

        s, w, l = state.sensory, state.working, state.long_term
        available = torch.stack((s.valid, w.valid.any(1), l.valid.any(1)), dim=1)

        # READ: every branch sees the same pre-prediction snapshot.
        rs = self.sensory.read(z, s, mask)
        q = self.working.encode_query(z, mask)
        kl, vl = self.long_term.get_kv(l)
        rw, rl, stats = self.explicit_reader(q, w, kl, vl, l.valid, ids, mask)
        rp = self.pose_reader(z, rs, rw, rl, available, mask)

        # PREDICT: keep the original regression head external (one owner).
        h, weights = self.fusion(z, rs, rw, rl, rp, available, mask,
                                 return_weights=True)
        u = self.decoder(h, available, mask,
                         memory_strength=weights[..., 1:].sum(-1, keepdim=True))
        pose = torch.where(valid, head(u), 0.0)

        # WRITE: the actual retained values condition Deep Update; evictions
        # are transferred after reading, with statistics in old slot order.
        v = self.working.encode_value(z, pose, mask)
        w_new, feedback, evicted = self.working.update(w, q, v, ids, stats, mask)
        s_new = self.sensory.update(z, u, s, feedback, mask)
        l_new = self.long_term.update(l, evicted, stats, ids, mask)
        next_state = replace(state, sensory=s_new, working=w_new, long_term=l_new)
        return pose, advance_state(next_state, ids, mask, streams)

    def forward(self, z, state, frame_ids, valid_mask, stream_ids, head):
        """Return pose [B,T,J,pose_dim] and a new, per-stream MemoryState.

        V1 accepts contiguous, non-overlapping windows with suffix padding.
        In enabled mode padded predictions are zero; the backbone itself
        remains bidirectional and can see padded/window context. There is
        no implicit detach or hidden persistent video state in this module.
        """
        state = prepare_state(
            self.config, z, state, frame_ids, valid_mask, stream_ids
        )
        if not self.config.enabled:
            # Exact original regression computation in baseline mode.
            return head(z), advance_state(state, frame_ids, valid_mask, stream_ids)

        outputs = []
        for chunk, ids, mask in self.iter_chunks(z, frame_ids, valid_mask):
            pose, state = self._process_chunk(chunk, state, ids, mask, stream_ids, head)
            outputs.append(pose)
        return torch.cat(outputs, dim=1), state
