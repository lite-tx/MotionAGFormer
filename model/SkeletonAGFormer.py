"""AnyTop-inspired, topology-conditioned MotionAGFormer-XS.

No fixed J, joint-index embedding, detector identifier, or per-layout weights.
A batch shares a skeleton definition; bucket different layouts in separate
batches. The temporal branch is inherited from MotionAGFormer (fixed T).
"""
from collections import OrderedDict

import torch
from torch import nn

from model.MotionAGFormer import AGFormerBlock
from model.skeleton import get_skeleton, sample_layout, eval_layout


class TopologyAttention(nn.Module):
    """AnyTop-style query/key-relative distance and directed-relation attention."""
    def __init__(self, dim, heads, qkv_bias=False, attn_drop=0., drop=0., max_dist=16):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.d = heads, dim // heads
        self.max_dist = max_dist
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.distance_q = nn.Embedding(max_dist + 1, dim)
        self.distance_k = nn.Embedding(max_dist + 1, dim)
        self.relation_q = nn.Embedding(6, dim)
        self.relation_k = nn.Embedding(6, dim)
        for e in (self.distance_q, self.distance_k, self.relation_q, self.relation_k):
            nn.init.normal_(e.weight, std=.02)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))

    def forward(self, x, relation, distance):
        b, t, j, c = x.shape
        q, k, v = self.qkv(x).reshape(b,t,j,3,self.heads,self.d).permute(3,0,4,1,2,5).unbind(0)
        dist = distance.clamp_max(self.max_dist)
        eq = (self.distance_q(dist) + self.relation_q(relation)).reshape(j,j,self.heads,self.d)
        ek = (self.distance_k(dist) + self.relation_k(relation)).reshape(j,j,self.heads,self.d)
        logits = q @ k.transpose(-1,-2)
        logits = logits + torch.einsum("bhtid,ijhd->bhtij", q, eq)
        logits = logits + torch.einsum("bhtjd,ijhd->bhtij", k, ek)
        weight = self.attn_drop((logits * self.d ** -.5).softmax(dim=-1))
        return self.proj((weight @ v).permute(0,2,3,1,4).reshape(b,t,j,c))


class DynamicGCN(nn.Module):
    """Feature-wise normalization, shared graph weights, actual input adjacency."""
    def __init__(self, dim):
        super().__init__()
        self.U = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        for layer in (self.U, self.V):
            nn.init.normal_(layer.weight, std=(2. / dim) ** .5)

    def forward(self, x, adjacency):
        aggregate = torch.einsum("ij,btjc->btic", adjacency, self.V(x))
        return torch.relu(x + self.norm(aggregate + self.U(x)))


class SpatialBlock(nn.Module):
    def __init__(self, dim, mixer, mlp_ratio, drop, layer_scale, init_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mixer = mixer
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim,hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden,dim), nn.Dropout(drop))
        self.scale1 = nn.Parameter(torch.full((dim,), init_scale)) if layer_scale else 1.
        self.scale2 = nn.Parameter(torch.full((dim,), init_scale)) if layer_scale else 1.

    def forward(self, x, *condition):
        x = x + self.scale1 * self.mixer(self.norm1(x), *condition)
        return x + self.scale2 * self.mlp(self.norm2(x))


class SkeletonAGFormerBlock(nn.Module):
    def __init__(self, dim, heads, n_frames, mlp_ratio=4., qkv_bias=False, attn_drop=0.,
                 drop=0., layer_scale=True, init_scale=1e-5, neighbour_num=2,
                 use_temporal_similarity=True, temporal_connection_len=1, use_adaptive_fusion=True):
        super().__init__()
        self.att_spatial = SpatialBlock(dim, TopologyAttention(dim, heads, qkv_bias, attn_drop, drop),
                                       mlp_ratio, drop, layer_scale, init_scale)
        self.graph_spatial = SpatialBlock(dim, DynamicGCN(dim), mlp_ratio, drop, layer_scale, init_scale)
        temporal = dict(dim=dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop, num_heads=heads,
                        qkv_bias=qkv_bias, use_layer_scale=layer_scale, layer_scale_init_value=init_scale,
                        mode="temporal", neighbour_num=neighbour_num, n_frames=n_frames,
                        use_temporal_similarity=use_temporal_similarity,
                        temporal_connection_len=temporal_connection_len)
        self.att_temporal = AGFormerBlock(mixer_type="attention", **temporal)
        self.graph_temporal = AGFormerBlock(mixer_type="graph", **temporal)
        self.fusion = nn.Linear(dim * 2, 2) if use_adaptive_fusion else None
        if self.fusion is not None:
            nn.init.zeros_(self.fusion.weight)
            nn.init.constant_(self.fusion.bias, .5)

    def forward(self, x, relation, distance, adjacency):
        a = self.att_temporal(self.att_spatial(x, relation, distance))
        g = self.graph_temporal(self.graph_spatial(x, adjacency))
        if self.fusion is None:
            return (a + g) * .5
        w = self.fusion(torch.cat((a,g),-1)).softmax(-1)
        return a * w[..., :1] + g * w[..., 1:]


class SkeletonReadout(nn.Module):
    """Predict requested joints, including ones absent from the observation.

    No fixed learned output IDs: queries use reference geometry. The exact-name
    skip preserves observed-joint features but cross-attention also handles
    missing/new joints. The same readout is used for every target topology.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, source, target, source_rest, target_rest, embed):
        b,t,j,c = x.shape
        k = len(target.names)
        q = embed(target_rest).view(1,1,k,c).expand(b,t,k,c)
        indices = {name:i for i,name in enumerate(source.names)}
        matched = [indices.get(name, -1) for name in target.names]
        idx = torch.tensor([max(i,0) for i in matched], device=x.device)
        mask = x.new_tensor([i >= 0 for i in matched]).view(1,1,k,1)
        q = q + x.index_select(2, idx) * mask
        # Fixed geometry is metadata, never inferred from target ground truth.
        # Geometry distance is an additive attention bias, not hard masking.
        bias = -torch.cdist(target_rest.float(), source_rest.float()).square().to(x.dtype)
        x_flat = x.reshape(b*t,j,c)
        y = self.attention(q.reshape(b*t,k,c), x_flat, x_flat,
                           attn_mask=bias, need_weights=False)[0].reshape(b,t,k,c)
        return self.norm(q+y)


class SkeletonAGFormer(nn.Module):
    def __init__(self, n_layers=12, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3,
                 num_heads=8, n_frames=27, mlp_ratio=4., qkv_bias=False, attn_drop=0.,
                 drop=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 neighbour_num=2, use_temporal_similarity=True, temporal_connection_len=1,
                 use_adaptive_fusion=True, canonical="h36m", train_topology_aug=True,
                 topology_eval_layout="canonical", detector_noise=0.):
        super().__init__()
        if dim_in != 3:
            raise ValueError("v1 expects xy + confidence; pad missing confidence with ones")
        self.n_frames = n_frames
        self.canonical = get_skeleton(canonical)
        self.train_topology_aug = train_topology_aug
        self.topology_eval_layout = topology_eval_layout
        self.detector_noise = float(detector_noise)
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        # Continuous, index-independent anatomy encoding. This is a lightweight
        # adaptation of AnyTop's semantic/rest-pose enrichment, NOT its T5 encoder.
        self.anatomy_embed = nn.Sequential(nn.Linear(3, dim_feat), nn.GELU(), nn.Linear(dim_feat,dim_feat))
        self.layers = nn.ModuleList([
            SkeletonAGFormerBlock(dim_feat, num_heads, n_frames, mlp_ratio, qkv_bias, attn_drop,
                                  drop, use_layer_scale, layer_scale_init_value, neighbour_num,
                                  use_temporal_similarity, temporal_connection_len, use_adaptive_fusion)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim_feat)
        self.readout = SkeletonReadout(dim_feat, num_heads)
        self.rep_logit = nn.Sequential(OrderedDict([("fc",nn.Linear(dim_feat,dim_rep)),("act",nn.Tanh())]))
        self.head = nn.Linear(dim_rep,dim_out)

    def forward(self, x, return_rep=False, *, skeleton=None, target_skeleton=None):
        if x.ndim != 4 or x.shape[1] != self.n_frames or x.shape[-1] not in (2,3):
            raise ValueError("expected [B, configured T, J, 2 or 3]")
        if x.shape[-1] == 2:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        # Legacy training/evaluation entry: resample source input only; labels
        # and output remain canonical, making MPJPE comparable across layouts.
        if skeleton is None:
            if x.shape[2] != len(self.canonical.names):
                raise ValueError("explicit SkeletonSpec required for noncanonical input")
            layout = (sample_layout(self.canonical) if self.training and self.train_topology_aug
                      else eval_layout(self.canonical, "canonical" if self.training else self.topology_eval_layout))
            x = layout.apply(x)
            skeleton = layout.spec
        if x.shape[2] != len(skeleton.names):
            raise ValueError("joint count does not match SkeletonSpec")
        if target_skeleton is None:
            target_skeleton = self.canonical
        if self.training and self.detector_noise > 0:
            # Synthetic noise robustness only. Real detector generalization
            # requires separate SH/CPN/unseen-detector data and evaluation.
            sigma = torch.rand((x.shape[0],1,1,1),device=x.device,dtype=x.dtype) * self.detector_noise
            x = torch.cat((x[..., :2] + sigma * torch.randn_like(x[..., :2]), x[..., 2:]), dim=-1)
        rest, relation, distance, adjacency = skeleton.tensors(x.device,x.dtype)
        target_rest = x.new_tensor(target_skeleton.rest)
        x = self.joints_embed(x) + self.anatomy_embed(rest)[None,None]
        for layer in self.layers:
            x = layer(x, relation, distance, adjacency)
        x = self.readout(self.norm(x), skeleton, target_skeleton, rest, target_rest, self.anatomy_embed)
        rep = self.rep_logit(x)
        return rep if return_rep else self.head(rep)
