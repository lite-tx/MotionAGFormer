"""Masked, episode-level version of the original H36M training objective.

Keep MotionAGFormer's MPJPE + N-MPJPE + velocity objective. No new memory
regularizers are introduced. The loss never receives or writes MemoryState.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class EpisodeLossResult(NamedTuple):
    total: Tensor
    position: Tensor
    scale: Tensor
    velocity: Tensor
    valid_frames: Tensor  # scalar int64; for weighted epoch logging
    valid_pairs: Tensor   # scalar int64; velocity uses a different denominator


class EpisodePoseLoss(nn.Module):
    """Original three-term objective, with valid-frame and continuity masks.

    Accepts [B,W,T,J,3] (batched windows) or [B,T,J,3] (already flattened
    episodes); metadata has the same leading dimensions before J,3.
    Each batch row is one independent episode; never concatenate batch rows.
    By default only targets are made root-relative (joint 0), matching the
    H36M trainer. Raw predictions retain root supervision. Inputs/targets
    must already use the original normalized coordinate system.
    """

    def __init__(self, lambda_scale=0.5, lambda_3d_velocity=20.0,
                 *, root_rel=True, scale_eps=1e-8):
        super().__init__()
        for name, value in (("lambda_scale", lambda_scale),
                            ("lambda_3d_velocity", lambda_3d_velocity)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(scale_eps) or scale_eps <= 0:
            raise ValueError("scale_eps must be finite and positive.")
        self.lambda_scale = float(lambda_scale)
        self.lambda_3d_velocity = float(lambda_3d_velocity)
        self.root_rel = bool(root_rel)
        self.scale_eps = float(scale_eps)

    def forward(self, predicted: Tensor, target: Tensor,
                valid_mask: Tensor, frame_ids: Tensor) -> EpisodeLossResult:
        if (predicted.ndim not in (4, 5) or predicted.shape != target.shape
                or predicted.shape[-2:] != (17, 3)
                or not predicted.is_floating_point() or not target.is_floating_point()
                or target.device != predicted.device):
            raise ValueError("predicted and target must be floating point [B,(W,)T,17,3] on the same device.")
        leading = predicted.shape[:-2]
        if (valid_mask.shape != leading or valid_mask.dtype != torch.bool
                or valid_mask.device != predicted.device
                or frame_ids.shape != leading or frame_ids.dtype != torch.long
                or frame_ids.device != predicted.device):
            raise ValueError("valid_mask (bool) and frame_ids (int64) must match the leading frame dimensions.")
        if any(size <= 0 for size in leading):
            raise ValueError("Episode batch and time dimensions must be nonempty.")
        if (valid_mask & (frame_ids < 0)).any():
            raise ValueError("Valid frames require nonnegative frame IDs.")
        if predicted.ndim == 5:
            predicted, target = predicted.flatten(1, 2), target.flatten(1, 2)
            valid_mask, frame_ids = valid_mask.flatten(1, 2), frame_ids.flatten(1, 2)

        # Sanitize padding before subtraction, scale fitting or finite
        # differences. Losses are computed in FP32 under mixed precision.
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            mask = valid_mask[..., None, None]
            p = torch.where(mask, predicted.float(), 0.0)
            y = torch.where(mask, target.float(), 0.0)
            if self.root_rel:
                y = y - y[..., :1, :]
            pf, yf = p[valid_mask], y[valid_mask]  # [F,J,3], possibly empty
            frames = valid_mask.sum()
            frame_denom = frames.clamp_min(1) * p.shape[-2]
            position = torch.linalg.vector_norm(pf - yf, dim=-1).sum() / frame_denom

            # Original per-frame least-squares scale. The epsilon only
            # changes the degenerate/tiny-denominator case.
            denominator = pf.square().sum(dim=-1).mean(dim=-1, keepdim=True)
            numerator = (pf * yf).sum(dim=-1).mean(dim=-1, keepdim=True)
            scale_factor = numerator / denominator.clamp_min(self.scale_eps)
            scale = torch.linalg.vector_norm(
                pf * scale_factor.unsqueeze(-1) - yf, dim=-1
            ).sum() / frame_denom

            # Flatten W,T, not B. In particular, the final frame of a
            # window and the first of the next form an ordinary valid pair.
            pair_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]
                         & (frame_ids[:, 1:] - frame_ids[:, :-1] == 1))
            pairs = pair_mask.sum()
            pv = p[:, 1:][pair_mask] - p[:, :-1][pair_mask]
            yv = y[:, 1:][pair_mask] - y[:, :-1][pair_mask]
            velocity = torch.linalg.vector_norm(pv - yv, dim=-1).sum() / (pairs.clamp_min(1) * p.shape[-2])
            # Empty masks produce graph-connected zero losses (no mean of
            # empty tensors). The trainer should skip an all-padding step.
            total = position + self.lambda_scale * scale + self.lambda_3d_velocity * velocity
        return EpisodeLossResult(total, position, scale, velocity, frames, pairs)
