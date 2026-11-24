"""Persistent pose proxies: TCPFormer-inspired PUM and bounded XMem-like management.

Only the latent proxy is stored. Its projected keys share the Working Memory
retrieval space. Hard routing and usage statistics are not differentiable;
proxy initialization, aggregation, updates, and projected KV retain gradients.
Read the old bank before updating it with the current chunk's evictions.
"""

import math
from dataclasses import fields

import torch
from torch import nn
from torch.nn import functional as F

from .config import MemoryConfig
from .state import EvictedMemory, LongTermState, ReadStats


class LongTermMemory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        k, v, p = config.explicit_key_dim, config.explicit_value_dim, config.proxy_dim
        self.input_proj = nn.Linear(k + v, p, bias=False)
        self.key_proj = nn.Linear(p, k, bias=False)
        self.value_proj = nn.Linear(p, v, bias=False)
        self.write_query = nn.Linear(p, k, bias=False)
        self.update_gate = nn.Linear(2 * p, 1)
        # Initialize a shared key subspace rather than comparing unrelated
        # random embeddings. If p >= k, new slots initially preserve their key.
        n = min(k, p)
        with torch.no_grad():
            self.input_proj.weight[:n].zero_()
            self.input_proj.weight[:n, :n].copy_(torch.eye(n))
            self.key_proj.weight.zero_()
            self.key_proj.weight[:n, :n].copy_(torch.eye(n))
            self.write_query.weight.copy_(self.key_proj.weight)
            g = config.proxy_initial_update
            self.update_gate.weight.zero_()
            self.update_gate.bias.fill_(math.log(g / (1.0 - g)))

    def _key(self, proxy):
        # Routing is no-grad; do not cache a detached autocast weight that
        # could suppress gradients in a later get_kv in the same AMP context.
        with torch.autocast(device_type=proxy.device.type, enabled=False):
            return F.normalize(F.linear(proxy.float(), self.key_proj.weight.float()),
                               dim=-1, eps=1e-6)

    def get_kv(self, state: LongTermState):
        """Project [B,L,J,dp] to explicit [B,L,J,dk/dv]; mask empty slots."""
        mask = state.valid[:, :, None, None]
        proxy = torch.where(mask, state.proxy, 0.0)
        k = self._key(proxy).to(proxy.dtype)
        v = self.value_proj(proxy).to(proxy.dtype)
        return torch.where(mask, k, 0.0), torch.where(mask, v, 0.0)

    def encode_candidates(self, keys, values):
        return self.input_proj(torch.cat((keys, values), dim=-1)).to(keys.dtype)

    @staticmethod
    def pose_similarity(candidate_keys, proxy_keys):
        """Full-pose cosine in the shared key space, not a learned hard router."""
        with torch.autocast(device_type=candidate_keys.device.type, enabled=False):
            a = F.normalize(candidate_keys.flatten(1).float(), dim=-1, eps=1e-6)
            b = F.normalize(proxy_keys.flatten(1).float(), dim=-1, eps=1e-6)
            return a @ b.transpose(0, 1)

    def pum(self, proxy, keys, candidates):
        """[J,dp] and [E,J,dk/dp] -> updated [J,dp].

        Each joint attends over the same assigned frames with its own weights.
        A bounded interpolation gate prevents unbounded residual accumulation.
        Even for one candidate (attention=1), the gate controls overwriting.
        """
        q = F.normalize(self.write_query(proxy).float(), dim=-1, eps=1e-6)
        with torch.autocast(device_type=proxy.device.type, enabled=False):
            k = F.normalize(keys.float(), dim=-1, eps=1e-6)
            scores = torch.einsum("jd,ejd->je", q, k) * math.sqrt(k.shape[-1])
            weights = scores.softmax(dim=-1)
            summary = torch.einsum("je,ejd->jd", weights, candidates.float())
        gate_input = torch.cat((proxy, summary.to(proxy.dtype)), dim=-1).mean(dim=0)
        gate = self.update_gate(gate_input).float().sigmoid()
        return (proxy.float() + gate * (summary - proxy.float())).to(proxy.dtype)

    def _validate(self, state, evicted, stats, ids, mask):
        cfg = self.config
        if state.proxy.ndim != 4 or not state.proxy.is_floating_point():
            raise ValueError("proxy must be floating point [B,L,J,dp].")
        b, l, j, p = state.proxy.shape
        device, dtype = state.proxy.device, state.proxy.dtype
        if b <= 0 or (l, j, p) != (cfg.proxy_capacity, cfg.num_joints, cfg.proxy_dim):
            raise ValueError("LongTermState shape does not match MemoryConfig.")
        if ids.ndim != 2 or ids.shape[0] != b or ids.shape[1] > cfg.chunk_size:
            raise ValueError("timestamps must be [B,C].")
        c = ids.shape[1]
        if mask is None:
            mask = torch.ones((b, c), device=device, dtype=torch.bool)
        checks = [(ids, (b,c), torch.long), (mask, (b,c), torch.bool),
                  (state.usage, (b,l), torch.float32), (state.exposure, (b,l), torch.long),
                  (state.created_at, (b,l), torch.long), (state.last_update, (b,l), torch.long),
                  (state.source_range, (b,l,2), torch.long), (state.support_count, (b,l), torch.long),
                  (state.valid, (b,l), torch.bool), (stats.long_term_usage, (b,l), None),
                  (stats.query_count, (b,), torch.long)]
        if evicted is not None:
            if evicted.keys.ndim != 4:
                raise ValueError("Evicted keys must be [B,E,J,dk].")
            e = evicted.keys.shape[1]
            checks += [(evicted.keys, (b,e,j,cfg.explicit_key_dim), dtype),
                       (evicted.values, (b,e,j,cfg.explicit_value_dim), dtype),
                       (evicted.timestamps, (b,e), torch.long), (evicted.relevance, (b,e), torch.float32),
                       (evicted.age, (b,e), torch.long), (evicted.valid, (b,e), torch.bool)]
        for x, shape, kind in checks:
            if (x.shape != shape or x.device != device or
                    (x.dtype != kind if kind is not None else not x.is_floating_point())):
                raise ValueError("LongTermState, ReadStats or input layout mismatch.")
        if ((~mask[:, :-1]) & mask[:, 1:]).any() or (mask & (ids < 0)).any():
            raise ValueError("Expected nonnegative frame IDs with suffix padding.")
        if (mask[:,1:] & (ids[:,1:] - ids[:,:-1] != 1)).any():
            raise ValueError("Valid IDs must be consecutive within a chunk.")
        active = mask.any(1)
        if ((stats.query_count < 0) | (stats.query_count > mask.sum(1)*j)).any():
            raise ValueError("query_count counts valid joint queries, without a head multiplier.")
        if (state.valid & ((state.exposure < 0) | (state.support_count < 1)
                           | (state.created_at < 0) | (state.last_update < state.created_at)
                           | (state.source_range[...,0] < 0)
                           | (state.source_range[...,1] < state.source_range[...,0])
                           | (state.source_range[...,1] > state.last_update))).any():
            raise ValueError("Invalid occupied proxy metadata.")
        if c and (state.valid & active[:,None] & (state.last_update >= ids[:,:1])).any():
            raise ValueError("Proxy state must precede the current chunk.")
        if evicted is not None:
            if (evicted.valid & ~active[:,None]).any():
                raise ValueError("An inactive row cannot receive evicted records.")
            if (evicted.valid & ((evicted.timestamps < 0) | (evicted.age < 1))).any():
                raise ValueError("Invalid evicted record metadata.")
            if c and (evicted.valid & (evicted.timestamps >= ids[:,:1])).any():
                raise ValueError("Evicted records must precede the current chunk.")
        return mask

    def _route(self, old_proxy, valid, usage, exposure, created, keys, candidates):
        """Return slot -> candidate list and seed slots, on a fixed snapshot.

        Reserve existing matches before allocating novel poses. New slots and
        matched slots are protected for this update. Novel candidates that
        cannot be admitted are dropped, never buffered without bound.
        """
        cfg = self.config
        groups, seeds = {}, {}
        with torch.no_grad():
            occupied = valid.nonzero(as_tuple=True)[0].tolist()
            bank = {s: self._key(old_proxy[s]) for s in occupied}
            novel = []
            for i in range(keys.shape[0]):
                if occupied:
                    sims = self.pose_similarity(keys[i:i+1], torch.stack([bank[s] for s in occupied]))[0]
                    idx = int(sims.argmax().item())
                    if sims[idx].item() >= cfg.proxy_match_threshold:
                        groups.setdefault(occupied[idx], []).append(i)
                        continue
                novel.append(i)
            # Novel poses can group with seeds created in this same update.
            for i in novel:
                slots = list(seeds)
                if slots:
                    sims = self.pose_similarity(keys[i:i+1], torch.stack([bank[s] for s in slots]))[0]
                    idx = int(sims.argmax().item())
                    if sims[idx].item() >= cfg.proxy_match_threshold:
                        groups[slots[idx]].append(i)
                        continue
                free = [s for s in range(cfg.proxy_capacity) if s not in bank]
                if free:
                    slot = free[0]
                else:
                    eligible = [s for s in occupied if s not in groups and
                                exposure[s].item() >= cfg.proxy_min_exposure]
                    if not eligible:
                        continue
                    # Lowest normalized usage, then oldest creation, then slot.
                    slot = min(eligible, key=lambda s: (
                        usage[s].item() / max(1, exposure[s].item()), created[s].item(), s))
                seeds[slot] = i
                groups[slot] = [i]
                bank[slot] = self._key(candidates[i])
        return groups, seeds

    def update(self, state: LongTermState, evicted: EvictedMemory, read_stats: ReadStats,
               timestamps, valid_mask=None):
        """Commit post-prediction state; statistics refer to OLD slot order.

        Exposure counts the actual valid joint queries per occupied slot.
        A zero-eviction chunk still updates retrieval statistics. No in-place
        writes to incoming tensors and no feature detaches on accepted paths.
        """
        mask = self._validate(state, evicted, read_stats, timestamps, valid_mask)
        if not mask.any():
            return state
        rows = []
        names = [f.name for f in fields(LongTermState)]
        for b in range(state.proxy.shape[0]):
            row = {name: getattr(state, name)[b] for name in names}
            if not mask[b].any():
                rows.append(tuple(row.values()))
                continue
            now = timestamps[b, mask[b]][-1]
            with torch.no_grad():
                count = read_stats.query_count[b]
                usage = torch.where(row["valid"] & (count > 0), read_stats.long_term_usage[b].float(), 0.0)
                if (usage < 0).any() or not torch.isfinite(usage).all():
                    raise ValueError("Usage mass must be finite and nonnegative.")
                row["usage"] = row["usage"] + usage
                row["exposure"] = row["exposure"] + row["valid"].long() * count
            indices = None if evicted is None else evicted.valid[b].nonzero(as_tuple=True)[0]
            if indices is None or indices.numel() == 0:
                rows.append(tuple(row.values()))
                continue
            indices = indices[evicted.timestamps[b, indices].argsort(stable=True)]
            keys, values, times = evicted.keys[b, indices], evicted.values[b, indices], evicted.timestamps[b, indices]
            candidates = self.encode_candidates(keys, values)
            groups, seeds = self._route(row["proxy"], row["valid"], row["usage"],
                                         row["exposure"], row["created_at"], keys, candidates)
            # Functional slot assembly preserves gradients through old and new features.
            slots = []
            meta = {name: row[name].clone() for name in names if name != "proxy"}
            for s in range(self.config.proxy_capacity):
                if s not in groups:
                    slots.append(row["proxy"][s])
                    continue
                group = groups[s]
                if s in seeds:
                    seed = seeds[s]
                    proxy = candidates[seed]
                    rest = group[1:]
                    if rest:
                        proxy = self.pum(proxy, keys[rest], candidates[rest])
                    meta["usage"][s] = 0
                    meta["exposure"][s] = 0
                    meta["created_at"][s] = now
                    meta["support_count"][s] = len(group)
                    meta["source_range"][s] = torch.stack((times[group].min(), times[group].max()))
                    meta["valid"][s] = True
                else:
                    proxy = self.pum(row["proxy"][s], keys[group], candidates[group])
                    meta["support_count"][s] += len(group)
                    meta["source_range"][s] = torch.stack((
                        torch.minimum(row["source_range"][s,0], times[group].min()),
                        torch.maximum(row["source_range"][s,1], times[group].max())))
                meta["last_update"][s] = now
                slots.append(proxy)
            row["proxy"] = torch.stack(slots)
            row.update(meta)
            rows.append(tuple(row[name] for name in names))
        return LongTermState(*(torch.stack(col) for col in zip(*rows)))
