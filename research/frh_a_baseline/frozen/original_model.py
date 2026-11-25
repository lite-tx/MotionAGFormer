"""MotionBERT-Lite with a same-joint XMem-style bank and package-level GRU."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class MotionBertXMemory(nn.Module):
    """Do not carry state across unrelated input sequences.

    The state is per batch element. In particular, a normal and a mirrored
    test stream may share the batch dimension but never share their contents.
    """

    def __init__(self, backbone: nn.Module, key_dim: int = 64,
                 value_dim: int = 512, hidden_dim: int = 64,
                 fusion_dim: int = 256, fusion_heads: int = 8):
        super().__init__()
        self.backbone = backbone
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.key_proj = nn.Linear(512, key_dim)
        self.selection_proj = nn.Linear(512, key_dim)
        self.shrinkage_proj = nn.Linear(512, 1)
        self.value_proj = nn.Sequential(nn.Linear(1024, value_dim), nn.GELU())
        self.pool_score = nn.Linear(512, 1)
        self.gru = nn.GRUCell(512, hidden_dim)
        self.fusion_in = nn.Linear(512 + value_dim + hidden_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.joint_attention = nn.MultiheadAttention(
            fusion_dim, fusion_heads, dropout=0.0, batch_first=True)
        self.fusion_out = nn.Linear(fusion_dim, 512)
        # Preserve nonzero training gradients through all memory paths on the
        # first step; the first packet still bypasses the branch exactly.
        nn.init.normal_(self.fusion_out.weight, std=1.0e-3)
        nn.init.zeros_(self.fusion_out.bias)

    def read_bank(self, feature: torch.Tensor, state: dict) -> torch.Tensor:
        """XMem's selected/shrunken squared-distance affinity, per joint."""
        # q,e: B,J,T,Ck; keys: B,J,N,Ck; values: B,J,N,Cv
        q = self.key_proj(feature).permute(0, 2, 1, 3)
        e = torch.sigmoid(self.selection_proj(feature)).permute(0, 2, 1, 3)
        k, v, shrink = state['key'], state['value'], state['shrinkage']
        a_sq = torch.matmul(k.square(), e.transpose(-1, -2))
        two_ab = 2.0 * torch.matmul(k, (q * e).transpose(-1, -2))
        b_sq = (e * q.square()).sum(dim=-1).unsqueeze(-2)
        similarity = (-a_sq + two_ab - b_sq) * shrink / math.sqrt(self.key_dim)
        weights = torch.softmax(similarity, dim=-2)
        result = torch.matmul(weights.transpose(-1, -2), v)
        return result.permute(0, 2, 1, 3).contiguous()

    def forward_clip(self, inputs: torch.Tensor, state: Optional[dict] = None):
        feature = self.backbone.get_representation(inputs)
        batch, frames, joints, channels = feature.shape
        if joints != 17 or channels != 512:
            raise ValueError(f'unexpected feature shape: {tuple(feature.shape)}')
        if state is None:
            old_hidden = feature.new_zeros(batch, joints, self.hidden_dim)
            fused = feature
        else:
            old_hidden = state['hidden']
            if old_hidden.shape != (batch, joints, self.hidden_dim):
                raise ValueError('state batch/joint shape does not match current packet')
            readout = self.read_bank(feature, state)
            prior = old_hidden[:, None].expand(-1, frames, -1, -1)
            x = self.fusion_in(torch.cat((feature, readout, prior), dim=-1))
            x = x.reshape(batch * frames, joints, -1)
            z = self.fusion_norm(x)
            attention, _ = self.joint_attention(z, z, z, need_weights=False)
            fused = feature + self.fusion_out(x + attention).reshape(batch, frames, joints, 512)
        prediction = self.backbone.head(fused)

        # One learned temporal summary and one GRU update per joint/package.
        pool_weights = torch.softmax(self.pool_score(fused), dim=1)
        summary = (pool_weights * fused).sum(dim=1)
        hidden = self.gru(summary.reshape(batch * joints, 512),
                          old_hidden.reshape(batch * joints, self.hidden_dim))
        hidden = hidden.reshape(batch, joints, self.hidden_dim)
        key = self.key_proj(feature).permute(0, 2, 1, 3).contiguous()
        shrinkage = (self.shrinkage_proj(feature).square() + 1.0).permute(0, 2, 1, 3).contiguous()
        value = self.value_proj(torch.cat((feature, fused), dim=-1)).permute(0, 2, 1, 3).contiguous()
        if state is not None:
            key = torch.cat((state['key'], key), dim=2)
            value = torch.cat((state['value'], value), dim=2)
            shrinkage = torch.cat((state['shrinkage'], shrinkage), dim=2)
        return prediction, {'hidden': hidden, 'key': key, 'value': value,
                            'shrinkage': shrinkage}

    def forward_group(self, clips: torch.Tensor) -> torch.Tensor:
        """Train one contiguous group: L,B,T,J,C; gradient remains within group."""
        state = None
        predictions = []
        for clip in clips.unbind(dim=0):
            prediction, state = self.forward_clip(clip, state)
            predictions.append(prediction)
        return torch.stack(predictions, dim=0)

    def forward(self, clips):
        # Two independent groups per GPU preserve the original eight-group
        # global batch on four GPUs. Each group starts with its own empty state.
        if isinstance(clips, (list, tuple)):
            return tuple(self.forward_group(group) for group in clips)
        return self.forward_group(clips)
