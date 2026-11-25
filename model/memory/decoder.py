"""Availability-aware residual fusion and lightweight pose feature decoder.

The explicit reader has already normalized W and L jointly. Fusion therefore
uses a single explicit-memory option; neither source is renormalized here.
All learned operations are per frame and joint, not new temporal mixers.
"""

import math

import torch
from torch import nn

from .config import MemoryConfig


def _availability(z, memory_valid, valid_mask, config):
    if (z.ndim != 4 or z.shape[2:] != (config.num_joints, config.dim_rep)
            or not z.is_floating_point()):
        raise ValueError("Feature must be floating point [B,C,J,D].")
    b, c = z.shape[:2]
    if (memory_valid.shape != (b, 3) or memory_valid.dtype != torch.bool
            or memory_valid.device != z.device):
        raise ValueError("memory_valid must be bool [B,3] in S,W,L order.")
    if valid_mask is None:
        valid_mask = torch.ones((b, c), dtype=torch.bool, device=z.device)
    if (valid_mask.shape != (b, c) or valid_mask.dtype != torch.bool
            or valid_mask.device != z.device):
        raise ValueError("valid_mask must be bool [B,C] on the feature device.")
    sources = valid_mask[:, :, None] & memory_valid[:, None, :]
    return valid_mask, sources


class AdaptiveMemoryFusion(nn.Module):
    """Z + gated S, (W + L), and pose correction; skip is always available."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        d, hidden = config.dim_rep, config.fusion_hidden_dim
        self.gate = nn.Sequential(nn.Linear(4*d, hidden), nn.GELU(), nn.Linear(hidden, 4))
        # Prefer the original feature initially, but leave nonzero memory
        # coefficients so gradients can reach the memory readers immediately.
        mass = config.fusion_initial_memory_mass
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[0] = math.log(3*(1-mass)/mass)

    def forward(self, z, r_s, r_w, r_l, r_pose, memory_valid,
                valid_mask=None, *, return_weights=False):
        """Return H [B,C,J,D], optionally weights [B,C,J,4] (skip,S,E,pose).

        memory_valid [B,3] describes the *incoming* S/W/L states. Invalid
        sources and padding receive zero mass. Z is an unweighted residual;
        skip is a choice to add no memory, not a multiplier of Z.
        Padded features and rows with no memories pass through unchanged.
        """
        _, sources = _availability(z, memory_valid, valid_mask, self.config)
        for r in (r_s, r_w, r_l, r_pose):
            if r.shape != z.shape or r.device != z.device or not r.is_floating_point():
                raise ValueError("Readouts must be floating point and match Z's shape/device.")
        active = sources.any(-1)
        s, w, l = (torch.where(sources[..., i, None, None], r.to(z.dtype), 0.0)
                   for i, r in enumerate((r_s, r_w, r_l)))
        p = torch.where(active[:, :, None, None], r_pose.to(z.dtype), 0.0)
        z_safe = torch.where(active[:, :, None, None], z, 0.0)
        explicit = w + l
        logits = self.gate(torch.cat((z_safe, s, explicit, p), dim=-1)).float()
        allowed = torch.stack((torch.ones_like(active), sources[..., 0],
                               sources[..., 1:].any(-1), active), dim=-1).unsqueeze(2)
        weights = logits.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        # FP32 weighted accumulation, then restore backbone feature dtype.
        delta = (weights[..., 1:2] * s.float() +
                 weights[..., 2:3] * explicit.float() +
                 weights[..., 3:4] * p.float())
        h = (z_safe.float() + delta).to(z.dtype)
        h = torch.where(active[:, :, None, None], h, z)
        return (h, weights) if return_weights else h


class ResidualMLPDecoder(nn.Module):
    """Lightweight post-fusion MLP; the original 3D regression head is external."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        d, hidden = config.dim_rep, config.decoder_hidden_dim
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        gain = config.decoder_initial_gain
        self.gain_logit = nn.Parameter(torch.tensor(math.log(gain/(1-gain))))

    def forward(self, h, memory_valid, valid_mask=None, *, memory_strength=None):
        """Return U [B,C,J,D] with exact identity on unavailable/padded rows.

        Optionally pass fusion weights[...,1:].sum(-1,keepdim=True) as
        memory_strength [B,C,J,1]. This also suppresses decoder refinement
        when the fusion gate chooses skip, rather than bypassing that choice.
        If omitted, available memories enable the original residual formula.
        """
        _, sources = _availability(h, memory_valid, valid_mask, self.config)
        active = sources.any(-1)[:, :, None, None]
        strength = active.float()
        if memory_strength is not None:
            if (memory_strength.shape != (*h.shape[:3], 1)
                    or memory_strength.device != h.device or not memory_strength.is_floating_point()):
                raise ValueError("memory_strength must be floating point [B,C,J,1].")
            strength = torch.where(active, memory_strength.float(), 0.0)
            if ((strength < 0) | (strength > 1.000001) | ~torch.isfinite(strength)).any():
                raise ValueError("Active memory_strength must be finite and in [0,1].")
        # Sanitize the inputs before any arithmetic on inactive positions.
        # Do not mask zero-strength active joints before the MLP: the gate
        # still needs the derivative of the correction with respect to strength.
        safe = torch.where(active, h, 0.0)
        delta = self.mlp(self.norm(safe)).float()
        gain = self.gain_logit.float().sigmoid()
        result = (safe.float() + gain * strength * delta).to(h.dtype)
        return torch.where(active, result, h)
