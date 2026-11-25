"""Four controlled variants of the prior XMem-inspired Lite model."""
from __future__ import annotations

import torch

from original_model import MotionBertXMemory


MODES = ('lite', 'gru', 'bank', 'full')


class AblationModel(MotionBertXMemory):
    def __init__(self, backbone, mode):
        if mode not in MODES:
            raise ValueError(f'unknown experiment mode: {mode}')
        super().__init__(backbone)
        self.mode = mode
        self.use_gru = mode in ('gru', 'full')
        self.use_bank = mode in ('bank', 'full')

    def forward_clip(self, inputs: torch.Tensor, state=None):
        # Baseline is exactly the original Lite network; the unused auxiliary
        # parameters remain allocated for identical initialization only.
        if self.mode == 'lite':
            return self.backbone(inputs), None

        feature = self.backbone.get_representation(inputs)
        batch, frames, joints, channels = feature.shape
        if joints != 17 or channels != 512:
            raise ValueError(f'unexpected feature shape: {tuple(feature.shape)}')
        if self.use_gru and state is not None:
            old_hidden = state['hidden']
            if old_hidden.shape != (batch, joints, self.hidden_dim):
                raise ValueError('hidden state has an unexpected shape')
        else:
            old_hidden = feature.new_zeros(batch, joints, self.hidden_dim)

        if state is None:
            fused = feature
        else:
            readout = (self.read_bank(feature, state) if self.use_bank
                       else torch.zeros_like(feature))
            prior = old_hidden[:, None].expand(-1, frames, -1, -1)
            x = self.fusion_in(torch.cat((feature, readout, prior), dim=-1))
            x = x.reshape(batch * frames, joints, -1)
            z = self.fusion_norm(x)
            attention, _ = self.joint_attention(z, z, z, need_weights=False)
            fused = feature + self.fusion_out(x + attention).reshape(batch, frames, joints, 512)
        prediction = self.backbone.head(fused)

        new_state = {}
        if self.use_gru:
            pool_weights = torch.softmax(self.pool_score(fused), dim=1)
            summary = (pool_weights * fused).sum(dim=1)
            hidden = self.gru(summary.reshape(batch * joints, 512),
                              old_hidden.reshape(batch * joints, self.hidden_dim))
            new_state['hidden'] = hidden.reshape(batch, joints, self.hidden_dim)
        if self.use_bank:
            key = self.key_proj(feature).permute(0, 2, 1, 3).contiguous()
            shrink = (self.shrinkage_proj(feature).square() + 1.0).permute(0, 2, 1, 3).contiguous()
            value = self.value_proj(torch.cat((feature, fused), dim=-1)).permute(0, 2, 1, 3).contiguous()
            if state is not None:
                key = torch.cat((state['key'], key), dim=2)
                value = torch.cat((state['value'], value), dim=2)
                shrink = torch.cat((state['shrinkage'], shrink), dim=2)
            new_state.update(key=key, value=value, shrinkage=shrink)
        return prediction, new_state

    def active_named_parameters(self):
        bank_prefixes = ('key_proj.', 'selection_proj.', 'shrinkage_proj.', 'value_proj.')
        gru_prefixes = ('pool_score.', 'gru.')
        for name, param in self.named_parameters():
            if self.mode == 'lite' and not name.startswith('backbone.'):
                continue
            if not self.use_bank and name.startswith(bank_prefixes):
                continue
            if not self.use_gru and name.startswith(gru_prefixes):
                continue
            yield name, param
