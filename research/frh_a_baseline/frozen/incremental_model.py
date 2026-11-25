"""MotionAGFormer-XS with cross-joint memory and two per-new-frame GRUs.

The backbone always processes the original complete input window.  Only the
state update and bank write use source-frame indices to collapse resampling
duplicates and skip source frames already processed in the same state chain.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from original_model import MotionBertXMemory


class IncrementalXMem(MotionBertXMemory):
    """One hidden state per sample and joint; bank entries per unique frame.

    ``frame_indices`` is preferably a CPU integer tensor: the metadata is
    small and keeping it on CPU avoids a GPU synchronization for each clip.
    Each row must correspond to one video/camera and be nondecreasing; the
    caller resets ``state`` at group/video boundaries.  No tensor is detached
    within a state chain, so group losses propagate through the previous clips.
    """

    def __init__(self, backbone: nn.Module, query_chunk: int = 51):
        super().__init__(backbone)
        if not isinstance(query_chunk, int) or query_chunk <= 0:
            raise ValueError("query_chunk must be a positive integer")
        self.query_chunk = query_chunk
        self.value_gru = nn.GRUCell(512, self.hidden_dim)

    def active_named_parameters(self):
        # The package-wide temporal pooling belongs to the previous design.
        for name, parameter in self.named_parameters():
            if not name.startswith("pool_score."):
                yield name, parameter

    def read_bank(self, feature: torch.Tensor, state: dict) -> torch.Tensor:
        """All history-frame/joint keys compete in one softmax per query.

        Only the *query* dimension is chunked; every chunk sees all valid keys,
        so it is the same global softmax as a non-chunked evaluation.
        """
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

        # Original storage remains (B,J,N,C).  Flatten both historical axes;
        # each frame/joint query sees every valid historical frame/joint pair.
        keys = key.reshape(batch, joints * n_hist, self.key_dim)
        values = value.reshape(batch, joints * n_hist, 512)
        scale = shrink.reshape(batch, 1, joints * n_hist) / math.sqrt(self.key_dim)
        bank_mask = valid[:, None, :].expand(batch, joints, n_hist).reshape(batch, joints * n_hist)
        has_history = bank_mask.any(dim=1)

        # In a mixed batch an empty row must not softmax only -inf.  A dummy
        # entry is exposed only for that row; its result is then zeroed.
        safe_mask = bank_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | ~has_history
        q = self.key_proj(feature).reshape(batch, frames * joints, self.key_dim)
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

    @staticmethod
    def _new_groups(frame_indices: torch.Tensor, frontier: torch.Tensor, batch: int, frames: int):
        """CPU metadata: one ordered list of positions per newly seen frame."""
        if frame_indices is None or tuple(frame_indices.shape) != (batch, frames):
            raise ValueError("frame_indices must have shape (B,T)")
        if frame_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("frame_indices must be integer source-frame indices")
        rows = frame_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        previous = frontier.detach().to(device="cpu", dtype=torch.long).tolist()
        selected, next_frontier = [], []
        for row, last_seen in zip(rows, previous):
            if any(idx < 0 for idx in row) or any(a > b for a, b in zip(row, row[1:])):
                raise ValueError("source-frame indices must be nonnegative and nondecreasing")
            groups = []
            for pos, source_idx in enumerate(row):
                if source_idx <= last_seen:
                    continue
                if groups and groups[-1][0] == source_idx:
                    groups[-1][1].append(pos)
                else:
                    groups.append((source_idx, [pos]))
            selected.append(groups)
            next_frontier.append(groups[-1][0] if groups else last_seen)
        return selected, torch.tensor(next_frontier, dtype=torch.long)

    @staticmethod
    def _pool_new(tensor: torch.Tensor, groups, width: int, device: torch.device):
        """Mean the occurrences of each source frame, independently per joint."""
        batch, frames, joints, channels = tensor.shape
        if width == 0:
            return tensor.new_zeros(batch, 0, joints, channels)
        duplicate = any(len(positions) != 1 for row in groups for _, positions in row)
        if not duplicate:
            indices = [[row[i][1][0] if i < len(row) else 0 for i in range(width)] for row in groups]
            index = torch.tensor(indices, device=device, dtype=torch.long)
            return torch.gather(tensor, 1, index[:, :, None, None].expand(batch, width, joints, channels))
        # Pool only positions belonging to the same source ID.  The weights do
        # not require gradients; bmm preserves gradient to every occurrence.
        weights = torch.zeros((batch, width, frames), dtype=tensor.dtype, device=device)
        for b, row in enumerate(groups):
            for i, (_, positions) in enumerate(row):
                weights[b, i, positions] = 1.0 / len(positions)
        pooled = torch.bmm(weights, tensor.reshape(batch, frames, joints * channels))
        return pooled.reshape(batch, width, joints, channels)

    def forward_clip(self, inputs: torch.Tensor, state: Optional[dict] = None,
                     frame_indices: Optional[torch.Tensor] = None):
        feature = self.backbone.get_representation(inputs)
        batch, frames, joints, channels = feature.shape
        if joints != 17 or channels != 512:
            raise ValueError("backbone feature must have shape (B,T,17,512)")
        device = feature.device

        if state is None:
            old_hidden = feature.new_zeros(batch, joints, self.hidden_dim)
            frontier = torch.full((batch,), -1, dtype=torch.long)
            old_key = feature.new_zeros(batch, joints, 0, self.key_dim)
            old_value = feature.new_zeros(batch, joints, 0, 512)
            old_shrink = feature.new_zeros(batch, joints, 0, 1)
            old_valid = torch.zeros(batch, 0, dtype=torch.bool, device=device)
            fused = feature
        else:
            old_key, old_value = state["key"], state["value"]
            old_shrink, old_valid = state["shrinkage"], state["valid"]
            if old_valid.device != device:
                raise ValueError("bank validity mask must be on the feature device")
            if state["hidden"].shape != (batch, joints, self.hidden_dim):
                raise ValueError("hidden must have shape (B,17,hidden_dim)")
            frontier = state["frontier"].detach().to(device="cpu", dtype=torch.long)
            if frontier.shape != (batch,):
                raise ValueError("frontier must have shape (B,)")
            has_history = old_valid.any(dim=1)
            old_hidden = torch.where(has_history[:, None, None], state["hidden"],
                                     torch.zeros_like(state["hidden"]))
            readout = self.read_bank(feature, state)
            prior = old_hidden[:, None].expand(-1, frames, -1, -1)
            x = self.fusion_in(torch.cat((feature, readout, prior), dim=-1))
            x = x.reshape(batch * frames, joints, -1)
            z = self.fusion_norm(x)
            attention, _ = self.joint_attention(z, z, z, need_weights=False)
            candidate = feature + self.fusion_out(x + attention).reshape(batch, frames, joints, 512)
            # Match the existing first-window path for any empty-history row.
            fused = torch.where(has_history[:, None, None, None], candidate, feature)

        # All 27 outputs are computed before any new state or memory is used.
        prediction = self.backbone.head(fused)
        groups, new_frontier = self._new_groups(frame_indices, frontier, batch, frames)
        width = max(map(len, groups), default=0)
        if width == 0:
            # Prediction is still emitted even if this window has no new frame.
            return prediction, {"hidden": old_hidden, "key": old_key, "value": old_value,
                                "shrinkage": old_shrink, "valid": old_valid,
                                "frontier": new_frontier}

        key = self.key_proj(feature)
        shrink = self.shrinkage_proj(feature).square() + 1.0
        value = self.value_proj(torch.cat((feature, fused), dim=-1))
        frame_fused = self._pool_new(fused, groups, width, device)
        frame_key = self._pool_new(key, groups, width, device)
        frame_value = self._pool_new(value, groups, width, device)
        frame_shrink = self._pool_new(shrink, groups, width, device)
        validity = torch.tensor([[i < len(row) for i in range(width)] for row in groups],
                                dtype=torch.bool, device=device)

        hidden = old_hidden
        for i in range(width):
            ordinary = self.gru(frame_fused[:, i].reshape(batch * joints, 512),
                                hidden.reshape(batch * joints, self.hidden_dim))
            deep = self.value_gru(frame_value[:, i].reshape(batch * joints, 512), ordinary)
            candidate_hidden = deep.reshape(batch, joints, self.hidden_dim)
            hidden = torch.where(validity[:, i, None, None], candidate_hidden, hidden)

        # The same pooled Value that updates value_gru is stored in the bank.
        new_valid = validity[:, None, :, None]
        add_key = frame_key.permute(0, 2, 1, 3).contiguous().masked_fill(~new_valid, 0.0)
        add_value = frame_value.permute(0, 2, 1, 3).contiguous().masked_fill(~new_valid, 0.0)
        add_shrink = frame_shrink.permute(0, 2, 1, 3).contiguous().masked_fill(~new_valid, 0.0)
        return prediction, {
            "hidden": hidden,
            "key": torch.cat((old_key, add_key), dim=2),
            "value": torch.cat((old_value, add_value), dim=2),
            "shrinkage": torch.cat((old_shrink, add_shrink), dim=2),
            "valid": torch.cat((old_valid, validity), dim=1),
            "frontier": new_frontier,
        }

    def forward_group(self, clips: torch.Tensor, frame_indices: torch.Tensor):
        if clips.ndim != 5 or frame_indices.shape != clips.shape[:3]:
            raise ValueError("expected clips (L,B,T,J,C) and frame_indices (L,B,T)")
        state = None
        outputs = []
        for clip, indices in zip(clips.unbind(dim=0), frame_indices.unbind(dim=0)):
            prediction, state = self.forward_clip(clip, state=state, frame_indices=indices)
            outputs.append(prediction)
        return torch.stack(outputs, dim=0)

    def forward(self, clips: torch.Tensor, frame_indices: torch.Tensor):
        return self.forward_group(clips, frame_indices)
