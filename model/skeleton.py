"""Explicit, order-independent human skeleton definitions and layout augmentation.

Reference coordinates are schematic, shared anatomy metadata (not per-example
ground truth). x: anatomical left positive; y: up; z: anterior. The root is
defined by parent -1. All nodes must belong to one rooted tree.
"""
from dataclasses import dataclass
import random
from typing import Tuple

import torch


@dataclass(frozen=True)
class SkeletonSpec:
    names: Tuple[str, ...]
    parents: Tuple[int, ...]
    rest: Tuple[Tuple[float, float, float], ...]

    def __post_init__(self):
        j = len(self.names)
        if not j or len(set(self.names)) != j or len(self.parents) != j or len(self.rest) != j:
            raise ValueError("names, parents and rest must describe the same unique joints")
        if self.parents.count(-1) != 1 or any(p < -1 or p >= j for p in self.parents):
            raise ValueError("expected one rooted tree")
        if not torch.isfinite(torch.tensor(self.rest)).all() or any(len(r) != 3 for r in self.rest):
            raise ValueError("rest must contain finite xyz coordinates")
        for i in range(j):
            seen = set()
            while i != -1:
                if i in seen:
                    raise ValueError("cycle in skeleton")
                seen.add(i)
                i = self.parents[i]

    @property
    def root(self):
        return self.parents.index(-1)

    def reindex(self, order):
        order = tuple(order)
        if sorted(order) != list(range(len(self.names))):
            raise ValueError("order must be a permutation")
        inverse = {old: new for new, old in enumerate(order)}
        return SkeletonSpec(tuple(self.names[i] for i in order),
                            tuple(-1 if self.parents[i] == -1 else inverse[self.parents[i]] for i in order),
                            tuple(self.rest[i] for i in order))

    def tensors(self, device, dtype):
        """Rest positions, six directed relation types, shortest-path distances, adjacency."""
        j = len(self.names)
        adj = torch.eye(j, dtype=torch.float32)
        ancestors = []
        for i, p in enumerate(self.parents):
            if p >= 0:
                adj[i, p] = adj[p, i] = 1
            chain = []
            while p != -1:
                chain.append(p)
                p = self.parents[p]
            ancestors.append(chain)
        # self, parent, child, ancestor, descendant, other
        rel = torch.full((j, j), 5, dtype=torch.long)
        dist = torch.empty((j, j), dtype=torch.long)
        for i in range(j):
            path_i = {node: d for d, node in enumerate([i] + ancestors[i])}
            for k in range(j):
                if i == k:
                    rel[i, k] = 0
                elif k == self.parents[i]:
                    rel[i, k] = 1
                elif i == self.parents[k]:
                    rel[i, k] = 2
                elif k in ancestors[i]:
                    rel[i, k] = 3
                elif i in ancestors[k]:
                    rel[i, k] = 4
                for d, node in enumerate([k] + ancestors[k]):
                    if node in path_i:
                        dist[i, k] = d + path_i[node]
                        break
        degree = adj.sum(-1).rsqrt()
        adj = degree[:, None] * adj * degree[None, :]
        return (torch.tensor(self.rest, device=device, dtype=dtype),
                rel.to(device), dist.to(device), adj.to(device=device, dtype=dtype))


# MotionBERT / standard H36M index convention, also used by posetrack2h36m.
H36M = SkeletonSpec(
    ("pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
     "spine", "thorax", "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
     "right_shoulder", "right_elbow", "right_wrist"),
    (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15),
    ((0., 0., 0.), (-.13, -.05, 0.), (-.14, -.47, 0.), (-.14, -.88, 0.),
     (.13, -.05, 0.), (.14, -.47, 0.), (.14, -.88, 0.), (0., .19, 0.),
     (0., .38, 0.), (0., .53, 0.), (0., .69, 0.), (.22, .38, 0.),
     (.46, .38, 0.), (.69, .38, 0.), (-.22, .38, 0.), (-.46, .38, 0.), (-.69, .38, 0.))
)
# Repository data.const.H36M_TO_MPI index mapping, not an assertion of
# anatomically identical labels. Verify the source data convention before
# claiming cross-dataset zero-shot; H36M is the primary experiment.
H36M_TO_MPI = (14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4)
MPI = H36M.reindex(sorted(range(17), key=lambda i: H36M_TO_MPI[i]))


def get_skeleton(name):
    if name == "h36m":
        return H36M
    if name == "mpi":
        return MPI
    raise ValueError("unknown canonical skeleton: " + name)


@dataclass(frozen=True)
class Layout:
    """A transformation from a source skeleton. Each row is (a, b, alpha).
    Real joints use a=b; synthetic points linearly interpolate a->b.
    """
    spec: SkeletonSpec
    rows: Tuple[Tuple[int, int, float], ...]

    def apply(self, x):
        if x.ndim != 4 or x.shape[-1] not in (2, 3):
            raise ValueError("expected [B,T,J,2 or 3] (xy[,confidence])")
        a = torch.tensor([r[0] for r in self.rows], device=x.device)
        b = torch.tensor([r[1] for r in self.rows], device=x.device)
        alpha = x.new_tensor([r[2] for r in self.rows]).view(1, 1, -1, 1)
        xa, xb = x.index_select(2, a), x.index_select(2, b)
        xy = torch.lerp(xa[..., :2], xb[..., :2], alpha)
        if x.shape[-1] == 2:
            return xy
        return torch.cat((xy, torch.minimum(xa[..., 2:], xb[..., 2:])), -1)


def make_layout(source, drop=(), insert=(), order=None):
    """Drop joints with nearest-ancestor reconnection; insert midpoints on retained edges."""
    by_name = {n: i for i, n in enumerate(source.names)}
    dropped = {by_name[n] for n in drop}
    if source.root in dropped:
        raise ValueError("training/eval layout must retain pelvis")
    keep = [i for i in range(len(source.names)) if i not in dropped]
    index = {old: i for i, old in enumerate(keep)}
    names = [source.names[i] for i in keep]
    rest = [source.rest[i] for i in keep]
    rows = [(i, i, 0.) for i in keep]
    parents = []
    for i in keep:
        p = source.parents[i]
        while p in dropped:
            p = source.parents[p]
        parents.append(-1 if p == -1 else index[p])
    for a_name, b_name in insert:
        a, b = by_name[a_name], by_name[b_name]
        if a not in index or b not in index or parents[index[b]] != index[a]:
            raise ValueError("insert requires a retained directed edge")
        midpoint = tuple((u + v) / 2 for u, v in zip(source.rest[a], source.rest[b]))
        new_idx = len(names)
        names.append(a_name + "__" + b_name + "_mid")
        rest.append(midpoint)
        rows.append((a, b, .5))
        parents.append(index[a])
        parents[index[b]] = new_idx
    spec = SkeletonSpec(tuple(names), tuple(parents), tuple(rest))
    if order is not None:
        spec = spec.reindex(order)
        rows = [rows[i] for i in order]
    return Layout(spec, tuple(rows))


# Explicitly disjoint held-out operations: no elbow removals or distal midpoint
# insertion during training. These are synthetic stress tests, not independent
# detector observations or proof of generalization to an external dataset.
TRAIN_DROP = ("spine", "neck", "right_knee", "left_knee")
TRAIN_EDGES = (("right_hip", "right_knee"), ("left_hip", "left_knee"),
               ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"))
TEST_DROP = ("left_elbow", "right_elbow", "spine", "neck")
TEST_EDGES = (("right_knee", "right_ankle"), ("left_knee", "left_ankle"),
              ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"))


def sample_layout(source, rng=None):
    rng = rng or random
    # Keep exact 17-joint layout in 25% of batches.
    if rng.random() < .25:
        return make_layout(source)
    drop = rng.sample(TRAIN_DROP, rng.randint(0, 3))
    edges = [e for e in TRAIN_EDGES if not any(n in drop for n in e)]
    insert = rng.sample(edges, rng.randint(0, len(edges)))
    layout = make_layout(source, drop, insert)
    order = list(range(len(layout.rows)))
    rng.shuffle(order)
    return Layout(layout.spec.reindex(order), tuple(layout.rows[i] for i in order))


def eval_layout(source, name):
    if name == "canonical":
        return make_layout(source)
    if name == "permuted":
        return make_layout(source, order=list(reversed(range(len(source.names)))))
    if name == "holdout_sparse":
        return make_layout(source, drop=TEST_DROP)
    if name == "holdout_dense":
        return make_layout(source, insert=TEST_EDGES)
    if name == "holdout_mixed":
        return make_layout(source, drop=("left_elbow", "right_elbow"), insert=TEST_EDGES[:2])
    raise ValueError("unknown layout: " + name)
