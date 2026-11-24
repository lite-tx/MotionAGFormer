"""SAMA-inspired structure interaction and motion-dependent GDN-2 decay.

These are task-specific adaptations, not the original Mamba/SSI/MSM blocks.
"""

import math

import torch
from torch import nn

from model.modules.graph import CONNECTIONS


class StructureAwareInteraction(nn.Module):
    """Shared, trainable 17-joint graph; state context is read-only."""

    def __init__(self, num_joints=17, initial_gain=0.1, non_edge_weight=0.01):
        super().__init__()
        if num_joints != 17:
            raise ValueError("SSI requires the MotionAGFormer 17-joint topology.")
        if not 0 < initial_gain < 1 or not 0 < non_edge_weight < 1:
            raise ValueError("SSI initialization weights must be in (0, 1).")
        prior = torch.full((num_joints, num_joints), non_edge_weight)
        prior.fill_diagonal_(1.0)
        for joint, neighbors in CONNECTIONS.items():
            for neighbor in neighbors:
                prior[joint, neighbor] = 1.0
                prior[neighbor, joint] = 1.0
        # Row softmax of log prior initializes strong skeleton connections
        # while keeping non-edges learnable (not a hard attention mask).
        self.adj_logits = nn.Parameter(prior.log())
        gain_logit = math.log(initial_gain / (1.0 - initial_gain))
        self.feature_gain = nn.Parameter(torch.tensor(gain_logit))
        self.state_gain = nn.Parameter(torch.tensor(gain_logit))

    def adjacency(self):
        return self.adj_logits.float().softmax(dim=-1)

    def feature_fusion(self, z):
        """[B, C, J, D] -> [B, C, J, D], with no temporal mixing."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            context = torch.einsum("ij,btjd->btid", self.adjacency(), z.float())
            result = z.float() + self.feature_gain.float().sigmoid() * context
        return result.to(z.dtype)

    def state_context(self, matrix):
        """[B, J, H, K, V] -> temporary FP32 read context."""
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            context = torch.einsum("ij,bjhkv->bihkv", self.adjacency(), matrix.float())
            return matrix.float() + self.state_gain.float().sigmoid() * context


class MotionAdaptiveModulator(nn.Module):
    """Modulate the key-channel log-decay with a per-frame, per-joint scalar."""

    def __init__(self, dim_rep):
        super().__init__()
        self.proj = nn.Linear(2 * dim_rep, 1)
        # gamma = 2 * sigmoid(0) = 1: neutral at initialization.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z, last_feature, last_valid, valid_mask):
        """Return gamma [B, C, J, 1, 1] for suffix-padded chunks.

        z and last_feature are SSI features, not the original encoder output.
        No previous valid frame (e.g. stream start) implies gamma = 1.
        """
        if z.shape[1] == 0:
            return z.new_ones((*z.shape[:3], 1, 1), dtype=torch.float32)
        z = torch.where(valid_mask[:, :, None, None], z, 0.0)
        last = torch.where(last_valid[:, None, None], last_feature, 0.0)
        previous = torch.cat((last[:, None], z[:, :-1]), dim=1)
        prev_valid = torch.cat((last_valid[:, None], valid_mask[:, :-1]), dim=1)
        paired = valid_mask & prev_valid
        # Sanitize both inputs before projection for safe padding gradients.
        pair = torch.cat((z, previous), dim=-1)
        pair = torch.where(paired[:, :, None, None], pair, 0.0)
        gamma = 2.0 * self.proj(pair).float().sigmoid()
        gamma = torch.where(paired[:, :, None, None], gamma, 1.0)
        return gamma.unsqueeze(-1)
