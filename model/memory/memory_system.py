"""Post-backbone memory scheduling boundary.

This commit provides only the foundation and original-head bypass.
Subsequent commits will add the actual sensory/working/long-term modules.
Enabling memory before they exist fails explicitly rather than silently
returning an unenhanced prediction.
"""

from torch import nn

from .config import MemoryConfig
from .state import advance_state, prepare_state


class MemorySystem(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

    def iter_chunks(self, z, frame_ids, valid_mask):
        """Partition encoded features (never the raw backbone input)."""
        size = self.config.chunk_size
        for start in range(0, z.shape[1], size):
            end = start + size
            yield z[:, start:end], frame_ids[:, start:end], valid_mask[:, start:end]

    def forward(self, z, state, frame_ids, valid_mask, stream_ids, head):
        if self.config.enabled:
            raise NotImplementedError(
                "Memory V1 foundation only: the memory read/update modules "
                "have not been implemented yet. Use MemoryConfig(enabled=False)."
            )
        state = prepare_state(
            self.config, z, state, frame_ids, valid_mask, stream_ids
        )
        # Keep the exact original regression computation in baseline mode.
        pose = head(z)
        state = advance_state(state, frame_ids, valid_mask, stream_ids)
        return pose, state
