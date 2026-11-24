"""Central configuration for the post-backbone memory system.

Defaults are design starting points, not experimentally validated settings.
The foundation commit deliberately leaves memory computation disabled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = False
    window_size: int = 243
    chunk_size: int = 27
    num_joints: int = 17
    dim_rep: int = 512
    pose_dim: int = 3

    working_capacity: int = 9
    write_stride: int = 9
    working_relevance_momentum: float = 0.8
    working_freshness_weight: float = 0.3
    proxy_capacity: int = 16
    explicit_key_dim: int = 64
    explicit_value_dim: int = 128

    sensory_heads: int = 4
    sensory_key_dim: int = 32
    sensory_value_dim: int = 32
    sensory_use_ssi: bool = True
    sensory_use_msm: bool = True
    sensory_use_deep_update: bool = True
    sensory_initial_decay: float = 0.95
    ssi_initial_gain: float = 0.1
    ssi_non_edge_weight: float = 0.01
    proxy_dim: int = 128
    pose_queries: int = 6
    pose_query_dim: int = 128

    def __post_init__(self):
        dimensions = (
            self.window_size, self.chunk_size, self.num_joints,
            self.dim_rep, self.pose_dim, self.working_capacity,
            self.write_stride, self.proxy_capacity, self.explicit_key_dim,
            self.explicit_value_dim, self.sensory_heads,
            self.sensory_key_dim, self.sensory_value_dim, self.proxy_dim,
            self.pose_queries, self.pose_query_dim,
        )
        if any(type(n) is not int or n <= 0 for n in dimensions):
            raise ValueError("All memory dimensions and capacities must be positive integers.")
        if self.chunk_size > self.window_size:
            raise ValueError("chunk_size cannot exceed window_size.")
        max_admitted = (self.chunk_size + self.write_stride - 1) // self.write_stride
        if self.working_capacity < max_admitted:
            raise ValueError("Working capacity must accommodate one chunk's admitted frames.")
        if self.enabled and (self.num_joints != 17 or self.pose_dim != 3):
            raise ValueError("Memory V1 is designed for the 17-joint, 3D pose protocol.")
        for name in ("sensory_initial_decay", "ssi_initial_gain", "ssi_non_edge_weight"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be strictly between zero and one.")
        if not 0.0 <= self.working_relevance_momentum < 1.0:
            raise ValueError("working_relevance_momentum must be in [0, 1).")
        if not 0.0 <= self.working_freshness_weight < float("inf"):
            raise ValueError("working_freshness_weight must be finite and nonnegative.")
