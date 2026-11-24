"""Chunk-level sensory memory: read old state, predict externally, then write.

Q/K use SSI-enhanced encoder features; V and base gates use decoded features.
MSM modulates log-decay; Working Memory feedback adjusts erase/write logits.
All learned projections are shared across joints and independent of stream ID.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MemoryConfig
from .gdnv2 import gdn2_read, gdn2_update
from .sama import MotionAdaptiveModulator, StructureAwareInteraction
from .state import SensoryState, WriteFeedback


@dataclass
class SensoryUpdateInputs:
    key: Tensor
    value: Tensor
    log_decay: Tensor
    erase: Tensor
    write: Tensor
    feature: Tensor  # SSI feature used for the next MSM boundary


class SensoryMemory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        d = config.dim_rep
        h, k, v = config.sensory_heads, config.sensory_key_dim, config.sensory_value_dim
        self.h, self.dk, self.dv = h, k, v
        self.ssi = StructureAwareInteraction(
            config.num_joints, config.ssi_initial_gain, config.ssi_non_edge_weight
        ) if config.sensory_use_ssi else None
        self.msm = MotionAdaptiveModulator(d) if config.sensory_use_msm else None
        self.q_proj = nn.Linear(d, h * k, bias=False)
        self.k_proj = nn.Linear(d, h * k, bias=False)
        self.v_proj = nn.Linear(d, h * v)
        self.decay_proj = nn.Linear(d, h * k)
        self.erase_proj = nn.Linear(d, h * k)
        self.write_proj = nn.Linear(d, h * v)
        self.out_proj = nn.Linear(h * v, d, bias=False)

        # -softplus(bias) = log(initial_decay). Per-channel learned decay.
        decay_bias = math.log(math.expm1(-math.log(config.sensory_initial_decay)))
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, decay_bias)
        self.cooperation = None
        if config.sensory_use_deep_update:
            hidden = max(h * k, h * v)
            self.cooperation = nn.Sequential(
                nn.Linear(d + config.explicit_value_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, h * (k + v)),
            )
            # Initial Deep Update == Normal Update; learn the correction.
            nn.init.zeros_(self.cooperation[-1].weight)
            nn.init.zeros_(self.cooperation[-1].bias)

    def _validate(self, z, state, valid_mask):
        cfg = self.config
        if z.ndim != 4 or z.shape[2:] != (cfg.num_joints, cfg.dim_rep):
            raise ValueError("Z must have shape [B, C, J, D] matching MemoryConfig.")
        if not z.is_floating_point():
            raise TypeError("Z must be floating point.")
        b, c, j, d = z.shape
        if (state.matrix.shape != (b, j, self.h, self.dk, self.dv)
                or state.matrix.dtype != torch.float32 or state.matrix.device != z.device
                or state.last_feature.shape != (b, j, d)
                or state.last_feature.dtype != z.dtype or state.last_feature.device != z.device
                or state.valid.shape != (b,) or state.valid.dtype != torch.bool
                or state.valid.device != z.device):
            raise ValueError("SensoryState has incompatible shape, dtype, or device.")
        if valid_mask is None:
            valid_mask = torch.ones((b, c), dtype=torch.bool, device=z.device)
        if (valid_mask.shape != (b, c) or valid_mask.dtype != torch.bool
                or valid_mask.device != z.device):
            raise ValueError("valid_mask must be bool [B, C] on Z.device.")
        if ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
            raise ValueError("Sensory Memory V1 requires suffix padding.")
        return valid_mask

    def _structure(self, z, mask):
        z = torch.where(mask[:, :, None, None], z, 0.0)
        return self.ssi.feature_fusion(z) if self.ssi is not None else z

    def read(self, z: Tensor, state: SensoryState, valid_mask=None):
        """Parallel historical read [B, C, J, D], without state mutation."""
        mask = self._validate(z, state, valid_mask) & state.valid[:, None]
        z_ctx = self._structure(z, mask)
        q = self.q_proj(z_ctx).unflatten(-1, (self.h, self.dk))
        matrix = torch.where(state.valid[:, None, None, None, None], state.matrix, 0.0)
        if self.ssi is not None:
            matrix = self.ssi.state_context(matrix)
        r = gdn2_read(matrix, q, mask).flatten(-2)
        out = self.out_proj(r.to(z.dtype)).to(z.dtype)
        return torch.where(mask[:, :, None, None], out, 0.0)

    def build_update_inputs(self, z, u, state, write_feedback=None, valid_mask=None):
        """Construct activated GDN-2 inputs. Exposed for diagnostics/tests."""
        mask = self._validate(z, state, valid_mask)
        if u.shape != z.shape or u.device != z.device or not u.is_floating_point():
            raise ValueError("U must be a floating point tensor matching Z's shape and device.")
        z_ctx = self._structure(z, mask)
        u = torch.where(mask[:, :, None, None], u, 0.0)
        key = self.k_proj(z_ctx).unflatten(-1, (self.h, self.dk))
        value = self.v_proj(u).unflatten(-1, (self.h, self.dv))
        g = -F.softplus(self.decay_proj(u).float()).unflatten(-1, (self.h, self.dk))
        b = self.erase_proj(u).float()
        w = self.write_proj(u).float()
        if self.msm is not None:
            g = g * self.msm(z_ctx, state.last_feature, state.valid, mask)

        if write_feedback is not None:
            saved, written = write_feedback.saved_value, write_feedback.mask
            expected = (*z.shape[:3], self.config.explicit_value_dim)
            if (written.shape != (*z.shape[:2], 1) or written.dtype != torch.bool
                    or written.device != z.device or saved.shape != expected
                    or saved.device != z.device or not saved.is_floating_point()):
                raise ValueError("WriteFeedback has incompatible shape, dtype, or device.")
            if (written.squeeze(-1) & ~mask).any():
                raise ValueError("WriteFeedback cannot mark padded frames as saved.")
            if self.cooperation is not None:
                # Sanitize unsaved values before the MLP, not after it.
                written = written.unsqueeze(-1)  # [B, C, 1, 1]
                saved = torch.where(written, saved.to(u.dtype), 0.0)
                selected_u = torch.where(written, u, 0.0)
                delta = self.cooperation(torch.cat((selected_u, saved), dim=-1)).float()
                db, dw = delta.split((self.h * self.dk, self.h * self.dv), dim=-1)
                b = b + torch.where(written, db, 0.0)
                w = w + torch.where(written, dw, 0.0)

        return SensoryUpdateInputs(
            key, value, g,
            b.sigmoid().unflatten(-1, (self.h, self.dk)),
            w.sigmoid().unflatten(-1, (self.h, self.dv)),
            z_ctx,
        )

    def update(self, z: Tensor, u: Tensor, state: SensoryState,
               write_feedback: Optional[WriteFeedback] = None, valid_mask=None):
        """Post-prediction write. Return a new state; preserve masked rows."""
        mask = self._validate(z, state, valid_mask)
        p = self.build_update_inputs(z, u, state, write_feedback, mask)
        if z.shape[1] == 0:
            return state
        active = mask.any(dim=1)
        # Never read stale invalid state; do not advance all-padding rows.
        old = torch.where(state.valid[:, None, None, None, None], state.matrix, 0.0)
        matrix = gdn2_update(old, p.key, p.value, p.log_decay, p.erase, p.write, mask)
        matrix = torch.where(active[:, None, None, None, None], matrix, state.matrix)
        index = (mask.long().sum(dim=1) - 1).clamp_min(0)
        last = p.feature[torch.arange(z.shape[0], device=z.device), index]
        last = torch.where(active[:, None, None], last, state.last_feature)
        return SensoryState(matrix, last, state.valid | active)
