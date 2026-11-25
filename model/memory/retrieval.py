"""Joint explicit-memory read and Cutie-inspired pose refinement.

Readers never mutate historical state. Working/proxy slots compete in one
softmax; their contributions and statistics retain the old slot ordering.
Pose queries are temporary per-frame tokens, not persistent video memory.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import MemoryConfig
from .state import ReadStats, WorkingState


def _valid_mask(mask, b, c, device):
    if mask is None:
        return torch.ones((b, c), device=device, dtype=torch.bool)
    if mask.shape != (b, c) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("valid_mask must be bool [B,C] on the feature device.")
    return mask


class JointExplicitReader(nn.Module):
    """Same-joint retrieval from one combined working/long-term bank."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.out_proj = nn.Linear(config.explicit_value_dim, config.dim_rep, bias=False)
        # Learned relative-time bias for real frames only; neutral initially.
        self.time_slope = nn.Parameter(torch.zeros(()))

    def forward(self, query, working: WorkingState, key_l, value_l, long_valid,
                timestamps, valid_mask=None):
        """Return R_w, R_l [B,C,J,D] and ReadStats in OLD bank slot order.

        query [B,C,J,dk]; long KV [B,L,J,dk/dv]; timestamps [B,C].
        Stats count valid joint queries (not heads), and use unrenormalized
        source contributions. Empty rows yield exactly zero, not NaN.
        """
        cfg = self.config
        if query.ndim != 4 or not query.is_floating_point():
            raise ValueError("query must be floating point [B,C,J,dk].")
        b, c, j, dk = query.shape
        n, l, dv = cfg.working_capacity, cfg.proxy_capacity, cfg.explicit_value_dim
        if b <= 0 or c > cfg.chunk_size or (j, dk) != (cfg.num_joints, cfg.explicit_key_dim):
            raise ValueError("query shape does not match MemoryConfig.")
        mask = _valid_mask(valid_mask, b, c, query.device)
        checks = ((working.keys, (b,n,j,dk), None), (working.values, (b,n,j,dv), None),
                  (key_l, (b,l,j,dk), None), (value_l, (b,l,j,dv), None),
                  (working.valid, (b,n), torch.bool), (long_valid, (b,l), torch.bool),
                  (working.timestamps, (b,n), torch.long), (timestamps, (b,c), torch.long))
        for x, shape, dtype in checks:
            if (x.shape != shape or x.device != query.device or
                    (x.dtype != dtype if dtype is not None else not x.is_floating_point())):
                raise ValueError("Explicit bank or timestamp layout mismatch.")
        if (mask & (timestamps < 0)).any() or (working.valid & (working.timestamps < 0)).any():
            raise ValueError("Valid frame IDs must be nonnegative.")
        if c and (working.valid & mask.any(1, keepdim=True) &
                  (working.timestamps >= timestamps[:, :1])).any():
            raise ValueError("Working Memory must precede the current chunk.")

        slot_valid = torch.cat((working.valid, long_valid), dim=1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = torch.where(mask[:, :, None, None], query.float(), 0.0)
            keys = torch.cat((working.keys.float(), key_l.float()), dim=1)
            values = torch.cat((working.values.float(), value_l.float()), dim=1)
            keys = torch.where(slot_valid[:, :, None, None], keys, 0.0)
            values = torch.where(slot_valid[:, :, None, None], values, 0.0)
            q = F.normalize(q, dim=-1, eps=1e-6)
            keys = F.normalize(keys, dim=-1, eps=1e-6)
            scores = torch.einsum("btjd,bnjd->btjn", q, keys) * math.sqrt(dk)
            current = torch.where(mask, timestamps, 0)
            history = torch.where(working.valid, working.timestamps, 0)
            delta = (current[:, :, None] - history[:, None, :]).clamp_min(0).float()
            time_bias = self.time_slope.float() * torch.log1p(delta / cfg.write_stride)
            bias = torch.cat((time_bias, time_bias.new_zeros((b,c,l))), dim=-1)
            scores = scores + bias[:, :, None, :]
            allowed = mask[:, :, None, None] & slot_valid[:, None, None, :]
            # A finite sentinel avoids all-masked softmax NaNs. Explicitly
            # zero invalid weights afterwards, including entirely empty rows.
            weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(dim=-1)
            weights = torch.where(allowed, weights, 0.0)
            aw, al = weights.split((n,l), dim=-1)
            rw = torch.einsum("btjn,bnjd->btjd", aw, values[:, :n])
            rl = torch.einsum("btjn,bnjd->btjd", al, values[:, n:])
            stats = ReadStats(aw.detach().sum((1,2)), al.detach().sum((1,2)),
                              mask.long().sum(1) * j)
        return (self.out_proj(rw.to(query.dtype)).to(query.dtype),
                self.out_proj(rl.to(query.dtype)).to(query.dtype), stats)


class _Attention(nn.Module):
    """Small multi-head attention with FP32 scores and optional soft body bias."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, q, kv, bias=None):
        shape = (self.heads, self.head_dim)
        a = self.q(q).unflatten(-1, shape).transpose(1,2)
        b = self.k(kv).unflatten(-1, shape).transpose(1,2)
        v = self.v(kv).unflatten(-1, shape).transpose(1,2)
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (a.float() @ b.float().transpose(-2,-1)) / math.sqrt(self.head_dim)
            if bias is not None:
                scores = scores + bias.float()
            result = scores.softmax(dim=-1) @ v.float()
        return self.out(result.transpose(1,2).flatten(-2).to(q.dtype))


class PoseQueryReader(nn.Module):
    """Per-frame joint -> pose tokens -> joint correction, inspired by Cutie."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        if config.num_joints != 17:
            raise ValueError("PoseQueryReader requires the MotionAGFormer 17-joint order.")
        self.config = config
        d, p, h = config.dim_rep, config.pose_query_dim, config.pose_query_heads
        if p % h:
            raise ValueError("pose_query_dim must be divisible by pose_query_heads.")
        self.local_proj = nn.Linear(4*d, p)
        self.queries = nn.Parameter(torch.empty(config.pose_queries, p))
        nn.init.normal_(self.queries, std=0.02)
        self.local_norm = nn.LayerNorm(p)
        self.query_norm = nn.LayerNorm(p)
        self.self_norm = nn.LayerNorm(p)
        self.ffn_norm = nn.LayerNorm(p)
        self.feedback_norm = nn.LayerNorm(p)
        self.collect = _Attention(p,h)
        self.interact = _Attention(p,h)
        self.feedback = _Attention(p,h)
        self.ffn = nn.Sequential(nn.Linear(p,2*p), nn.GELU(), nn.Linear(2*p,p))
        self.out_proj = nn.Linear(p,d,bias=False)
        # H36M order: global, torso/head, left/right arm, left/right leg.
        groups = [range(17), (0,7,8,9,10), (11,12,13), (14,15,16), (4,5,6), (1,2,3)]
        prior = torch.zeros(config.pose_queries,17)
        for i in range(min(config.pose_queries,len(groups))):
            prior[i].fill_(-config.pose_body_bias)
            prior[i,list(groups[i])] = 0.0
        # Additional queries beyond the six canonical roles are global.
        self.register_buffer("body_bias",prior,persistent=False)

    def forward(self, z, r_s, r_w, r_l, memory_valid, valid_mask=None):
        """Return only the pose-level correction [B,C,J,D].

        memory_valid [B,3] is per-stream availability in S,W,L order.
        A query is local to one frame; no temporal or cross-stream mixing.
        All-empty memory and padded frames produce exact zero correction.
        """
        cfg = self.config
        if z.ndim != 4 or z.shape[2:] != (cfg.num_joints,cfg.dim_rep) or not z.is_floating_point():
            raise ValueError("Z must be floating point [B,C,J,D].")
        b,c,j,d = z.shape
        mask = _valid_mask(valid_mask,b,c,z.device)
        if memory_valid.shape != (b,3) or memory_valid.dtype != torch.bool or memory_valid.device != z.device:
            raise ValueError("memory_valid must be bool [B,3] in sensory, working, long-term order.")
        for r in (r_s,r_w,r_l):
            if r.shape != z.shape or r.device != z.device or not r.is_floating_point():
                raise ValueError("All readouts must match Z's shape and device.")
        active = mask & memory_valid.any(-1,keepdim=True)
        if c == 0:
            return torch.zeros_like(z)
        parts = [torch.where(active[:,:,None,None], z, 0.0)]
        for i,r in enumerate((r_s,r_w,r_l)):
            use = mask & memory_valid[:,i,None]
            parts.append(torch.where(use[:,:,None,None],r.to(z.dtype),0.0))
        local = self.local_proj(torch.cat(parts,dim=-1)).reshape(b*c,j,cfg.pose_query_dim)
        local_ctx = self.local_norm(local)
        q = self.queries.to(local.dtype).unsqueeze(0).expand(b*c,-1,-1)
        q = q + self.collect(self.query_norm(q),local_ctx,self.body_bias)
        q_ctx = self.self_norm(q)
        q = q + self.interact(q_ctx,q_ctx)
        q = q + self.ffn(self.ffn_norm(q))
        delta = self.feedback(local_ctx,self.feedback_norm(q),self.body_bias.transpose(0,1))
        correction = self.out_proj(delta).reshape(b,c,j,d).to(z.dtype)
        return torch.where(active[:,:,None,None],correction,0.0)
