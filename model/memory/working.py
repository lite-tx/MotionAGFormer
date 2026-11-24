"""Frame-managed, joint-level explicit KV memory (RMem-inspired).

The joint reader is external: it supplies attention mass in old slot order.
Only management metadata is detached. Selected KV features and transferred
records retain their autograd paths. No state is modified in-place.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import MemoryConfig
from .state import EvictedMemory, ReadStats, WorkingState, WriteFeedback


def _pad(x, length, fill=0):
    return torch.cat((x, x.new_full((length - x.shape[0], *x.shape[1:]), fill)))


def _mask(mask, b, c, device):
    if mask is None:
        return torch.ones(b, c, dtype=torch.bool, device=device)
    if mask.shape != (b, c) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("valid_mask must be bool [B, C] on the feature device.")
    return mask


class WorkingMemory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        d, k, v = config.dim_rep, config.explicit_key_dim, config.explicit_value_dim
        self.key_encoder = nn.Linear(d, k, bias=False)
        self.feature_proj = nn.Linear(d, v)
        self.pose_encoder = nn.Sequential(nn.Linear(config.pose_dim, v), nn.GELU(), nn.Linear(v, v))
        self.value_fusion = nn.Sequential(nn.Linear(2 * v, v), nn.GELU(), nn.Linear(v, v))

    def _check_feature(self, z):
        if (z.ndim != 4 or z.shape[2:] != (self.config.num_joints, self.config.dim_rep)
                or not z.is_floating_point()):
            raise ValueError("Z must be floating point [B, C, J, D].")

    def encode_query(self, z, valid_mask=None):
        """The same normalized projection is used for current Q and stored K."""
        self._check_feature(z)
        mask = _mask(valid_mask, *z.shape[:2], z.device)[:, :, None, None]
        q = self.key_encoder(torch.where(mask, z, 0.0))
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(z.dtype)
        return torch.where(mask, q, 0.0)

    def encode_value(self, z, pose, valid_mask=None):
        """Observation feature plus predicted 3D pose, never ground-truth pose."""
        self._check_feature(z)
        if (pose.shape != (*z.shape[:3], self.config.pose_dim)
                or pose.device != z.device or not pose.is_floating_point()):
            raise ValueError("pose must be floating point [B, C, J, pose_dim].")
        mask = _mask(valid_mask, *z.shape[:2], z.device)[:, :, None, None]
        e_z = self.feature_proj(torch.where(mask, z, 0.0))
        e_p = self.pose_encoder(torch.where(mask, pose.to(z.dtype), 0.0))
        v = e_z + self.value_fusion(torch.cat((e_z, e_p), dim=-1))
        return torch.where(mask, v.to(z.dtype), 0.0)

    def select_frames(self, timestamps, valid_mask):
        """Zero-based stream frame IDs: stride=9 admits IDs 8, 17, 26, ... ."""
        return valid_mask & (timestamps.remainder(self.config.write_stride) == self.config.write_stride - 1)

    def retention_score(self, relevance, age):
        """Adapted RMem score on valid old records only; age in valid frames."""
        age = age.float().clamp_min(1)
        freshness = (2.0 * age.sum().clamp_min(1).log() / age).sqrt()
        return relevance.float() + self.config.working_freshness_weight * freshness

    def _validate(self, state, key, value, timestamps, stats, valid_mask):
        cfg = self.config
        if key.ndim != 4:
            raise ValueError("key must have shape [B, C, J, dk].")
        b, c, j, dk = key.shape
        n, dv = cfg.working_capacity, cfg.explicit_value_dim
        if b <= 0 or c > cfg.chunk_size or (j, dk) != (cfg.num_joints, cfg.explicit_key_dim):
            raise ValueError("key shape does not match MemoryConfig.")
        mask = _mask(valid_mask, b, c, key.device)
        if not key.is_floating_point():
            raise TypeError("key must be floating point.")
        checks = (
            (value, (b, c, j, dv), key.dtype),
            (timestamps, (b, c), torch.long),
            (state.keys, (b, n, j, dk), key.dtype),
            (state.values, (b, n, j, dv), key.dtype),
            (state.timestamps, (b, n), torch.long),
            (state.relevance, (b, n), torch.float32),
            (state.age, (b, n), torch.long),
            (state.valid, (b, n), torch.bool),
            (stats.working_usage, (b, n), None),
            (stats.query_count, (b,), torch.long),
        )
        for x, shape, dtype in checks:
            if (x.shape != shape or x.device != key.device
                    or (dtype is not None and x.dtype != dtype)
                    or (dtype is None and not x.is_floating_point())):
                raise ValueError("WorkingState, candidate, or ReadStats layout mismatch.")
        if ((~mask[:, :-1]) & mask[:, 1:]).any() or (mask & (timestamps < 0)).any():
            raise ValueError("Expected nonnegative IDs with suffix padding.")
        if (mask[:, 1:] & (timestamps[:, 1:] - timestamps[:, :-1] != 1)).any():
            raise ValueError("Valid frame IDs must be consecutive within a chunk.")
        if (state.valid & ((state.age < 1) | (state.timestamps < 0))).any():
            raise ValueError("Occupied slots must have positive age and nonnegative IDs.")
        if ((stats.query_count < 0) | (stats.query_count > mask.sum(1) * j)).any():
            raise ValueError("query_count must count valid joint queries (no head multiplier).")
        if c and (state.valid & mask.any(1, keepdim=True)
                  & (state.timestamps >= timestamps[:, :1])).any():
            raise ValueError("Working Memory can only contain earlier frames.")
        return mask

    def update(self, state: WorkingState, key, value, timestamps,
               read_stats: ReadStats, valid_mask=None):
        """Return (new bank, aligned write feedback, padded evicted records).

        Admission is based on absolute IDs, not chunk-local offsets. A new
        record is protected this update, and old records compete only when
        space is needed. Every evicted record is returned intact to long-term
        memory. The fixed-size bank is packed in ascending timestamp order.
        All-padding rows are preserved exactly. Empty E is represented by
        tensors of shape [B, 0, ...], not None.
        """
        mask = self._validate(state, key, value, timestamps, read_stats, valid_mask)
        b, c, j, _ = key.shape
        n = self.config.working_capacity
        selected = self.select_frames(timestamps, mask)
        if (selected.sum(1) > n).any():
            raise ValueError("Too many admitted frames for Working Memory capacity.")
        feedback = WriteFeedback(selected.unsqueeze(-1),
                                 torch.where(selected[:, :, None, None], value, 0.0))
        # Relevance is management metadata, not a differentiable attention loss.
        with torch.no_grad():
            usage = torch.where(state.valid, read_stats.working_usage.float(), 0.0)
            count = read_stats.query_count
            measured = usage / count.clamp_min(1).float().unsqueeze(-1)
            rho = self.config.working_relevance_momentum
            relevance = torch.where(count[:, None] > 0,
                                    rho * state.relevance + (1 - rho) * measured,
                                    state.relevance)
            age = state.age + mask.sum(1, keepdim=True)

        rows, removed = [], []
        for i in range(b):
            old = state.valid[i].nonzero(as_tuple=True)[0]
            new = selected[i].nonzero(as_tuple=True)[0]
            # Sort by timestamp before a stable score sort: oldest loses ties.
            old = old[state.timestamps[i, old].argsort(stable=True)]
            overflow = max(0, old.numel() + new.numel() - n)
            score = self.retention_score(relevance[i, old], age[i, old])
            ranked = score.argsort(stable=True)
            drop = old[ranked[:overflow]]
            keep = old[ranked[overflow:]]
            keep = keep[state.timestamps[i, keep].argsort(stable=True)]
            drop = drop[state.timestamps[i, drop].argsort(stable=True)]
            removed.append((state.keys[i, drop], state.values[i, drop],
                            state.timestamps[i, drop], relevance[i, drop], age[i, drop]))
            if not mask[i].any():
                rows.append((state.keys[i], state.values[i], state.timestamps[i],
                             state.relevance[i], state.age[i], state.valid[i]))
                continue
            k = torch.cat((state.keys[i, keep], key[i, new]))
            v = torch.cat((state.values[i, keep], value[i, new]))
            ts = torch.cat((state.timestamps[i, keep], timestamps[i, new]))
            rel = torch.cat((relevance[i, keep], relevance.new_zeros(new.numel())))
            ages = torch.cat((age[i, keep], age.new_ones(new.numel())))
            valid = torch.ones_like(ts, dtype=torch.bool)
            rows.append((_pad(k, n), _pad(v, n), _pad(ts, n, -1),
                         _pad(rel, n), _pad(ages, n), _pad(valid, n, False)))

        bank = WorkingState(*(torch.stack(col) for col in zip(*rows)))
        e = max(row[2].numel() for row in removed)
        padded = [tuple(_pad(x, e, -1 if p == 2 else 0) for p, x in enumerate(row))
                  + (_pad(torch.ones_like(row[2], dtype=torch.bool), e, False),)
                  for row in removed]
        evicted = EvictedMemory(*(torch.stack(col) for col in zip(*padded)))
        return bank, feedback, evicted
