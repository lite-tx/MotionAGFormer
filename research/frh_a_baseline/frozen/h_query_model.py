"""Query-only H conditioning for the frozen incremental XMem reference."""
from __future__ import annotations

import math

import torch
from torch import nn

from train_controlled import ControlledXMem


class HQueryXMem(ControlledXMem):
    """Add a zero-initialized H residual without changing any bank writer.

    The original forward_clip, source-frame deduplication, fusion, two GRUs,
    and first-window bypass are inherited without modification.  Only query
    construction inside the original selected/shrunken squared-distance read
    is different.  Key, selection and shrinkage still follow the reference.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__(backbone, arm="joint")
        if self.hidden_dim != 64 or self.key_dim != 64:
            raise ValueError("the registered query residual requires 64D H and keys")
        self.h_query_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        # Linear's default CPU initialization consumes RNG even when followed
        # by zeros_.  Preserve the reference construction's RNG state.
        with torch.random.fork_rng(devices=[]):
            self.h_query_proj = nn.Linear(self.hidden_dim, self.key_dim, bias=False)
            nn.init.zeros_(self.h_query_proj.weight)

    def read_bank(self, feature: torch.Tensor, state: dict) -> torch.Tensor:
        """Original global-bank L2 read, with q=q0+W LN(H_old) only."""
        batch, frames, joints, channels = feature.shape
        if joints != 17 or channels != 512:
            raise ValueError("feature must have shape (B,T,17,512)")
        key = state["key"]
        value = state["value"]
        shrink = state["shrinkage"]
        valid = state["valid"]
        if key.ndim != 4 or key.shape[:2] != (batch, joints):
            raise ValueError("key must have shape (B,17,N,key_dim)")
        n_hist = key.shape[2]
        if key.shape[-1] != self.key_dim or value.shape != (batch, joints, n_hist, 512):
            raise ValueError("bank key/value dimensions do not match")
        if shrink.shape != (batch, joints, n_hist, 1) or valid.shape != (batch, n_hist):
            raise ValueError("bank shrinkage/valid dimensions do not match")
        if valid.dtype != torch.bool:
            raise ValueError("bank valid must be boolean")
        if n_hist == 0:
            return feature.new_zeros(batch, frames, joints, 512)

        keys = key.reshape(batch, joints * n_hist, self.key_dim)
        values = value.reshape(batch, joints * n_hist, 512)
        scale = shrink.reshape(batch, 1, joints * n_hist) / math.sqrt(self.key_dim)
        bank_mask = valid[:, None, :].expand(batch, joints, n_hist).reshape(batch, joints * n_hist)
        has_history = bank_mask.any(dim=1)
        safe_mask = bank_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | ~has_history

        hidden = state["hidden"]
        if hidden.shape != (batch, joints, self.hidden_dim):
            raise ValueError("hidden must have shape (B,17,64)")
        hidden = torch.where(has_history[:, None, None], hidden, torch.zeros_like(hidden))
        residual = self.h_query_proj(self.h_query_norm(hidden))
        q = (self.key_proj(feature) + residual[:, None]).reshape(
            batch, frames * joints, self.key_dim)
        e = torch.sigmoid(self.selection_proj(feature)).reshape(batch, frames * joints, self.key_dim)
        keys_t = keys.transpose(1, 2)
        keys_sq_t = keys.square().transpose(1, 2)
        outputs = []
        for start in range(0, frames * joints, self.query_chunk):
            qi = q[:, start:start + self.query_chunk]
            ei = e[:, start:start + self.query_chunk]
            q_square = (ei * qi.square()).sum(dim=-1, keepdim=True)
            distances = torch.bmm(ei, keys_sq_t) - 2.0 * torch.bmm(ei * qi, keys_t) + q_square
            logits = (-distances * scale).masked_fill(~safe_mask[:, None, :], float("-inf"))
            weights = torch.softmax(logits, dim=-1)
            chunk = torch.bmm(weights, values)
            chunk = torch.where(has_history[:, None, None], chunk, torch.zeros_like(chunk))
            outputs.append(chunk)
        return torch.cat(outputs, dim=1).reshape(batch, frames, joints, 512)
