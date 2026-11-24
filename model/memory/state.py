"""Per-stream dynamic state and communication contracts.

These tensors are not parameters or model buffers. States are passed
explicitly between windows. Metadata is indexed by batch slot, not shared
across videos. The caller is responsible for stable stream identities
(including person and camera view).
"""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from .config import MemoryConfig


@dataclass
class SensoryState:
    matrix: Tensor            # [B, J, H, dk_s, dv_s], float32
    last_feature: Tensor      # [B, J, D], previous valid SSI feature
    valid: Tensor             # [B], bool


@dataclass
class WorkingState:
    keys: Tensor              # [B, N, J, dk_e]
    values: Tensor            # [B, N, J, dv_e]
    timestamps: Tensor        # [B, N], int64
    relevance: Tensor         # [B, N], float32
    age: Tensor               # [B, N], int64
    valid: Tensor             # [B, N], bool


@dataclass
class LongTermState:
    proxy: Tensor             # [B, L, J, dp]
    usage: Tensor             # [B, L], float32
    exposure: Tensor          # [B, L], int64
    created_at: Tensor        # [B, L], int64
    last_update: Tensor       # [B, L], int64
    source_range: Tensor      # [B, L, 2], int64
    support_count: Tensor     # [B, L], int64
    valid: Tensor             # [B, L], bool


@dataclass
class ReadStats:
    working_usage: Tensor     # [B, N], accumulated attention mass
    long_term_usage: Tensor   # [B, L], accumulated attention mass
    query_count: Tensor       # [B], number of valid query positions


@dataclass
class WriteFeedback:
    mask: Tensor              # [B, C, 1], bool
    saved_value: Tensor       # [B, C, J, dv_e]


@dataclass
class EvictedMemory:
    keys: Tensor              # [B, E, J, dk_e]
    values: Tensor            # [B, E, J, dv_e]
    timestamps: Tensor        # [B, E]
    relevance: Tensor         # [B, E]
    age: Tensor               # [B, E]
    valid: Tensor             # [B, E], bool; E may be zero


@dataclass
class MemoryState:
    sensory: SensoryState
    working: WorkingState
    long_term: LongTermState
    last_frame_id: Tensor     # [B], int64; -1 denotes no consumed frame
    stream_ids: Tensor        # [B], int64; -1 denotes unassigned slot


def init_state(config: MemoryConfig, batch_size: int, device, dtype=torch.float32):
    """Create empty, independent states, with no valid historical records."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Memory features must have a floating-point dtype.")

    b, j = batch_size, config.num_joints
    n, l = config.working_capacity, config.proxy_capacity

    def zeros(shape, kind=dtype):
        return torch.zeros(shape, device=device, dtype=kind)

    def ids(shape):
        return torch.full(shape, -1, device=device, dtype=torch.long)

    return MemoryState(
        sensory=SensoryState(
            matrix=zeros((b, j, config.sensory_heads,
                          config.sensory_key_dim, config.sensory_value_dim),
                         torch.float32),
            last_feature=zeros((b, j, config.dim_rep)),
            valid=zeros((b,), torch.bool),
        ),
        working=WorkingState(
            keys=zeros((b, n, j, config.explicit_key_dim)),
            values=zeros((b, n, j, config.explicit_value_dim)),
            timestamps=ids((b, n)),
            relevance=zeros((b, n), torch.float32),
            age=zeros((b, n), torch.long),
            valid=zeros((b, n), torch.bool),
        ),
        long_term=LongTermState(
            proxy=zeros((b, l, j, config.proxy_dim)),
            usage=zeros((b, l), torch.float32),
            exposure=zeros((b, l), torch.long),
            created_at=ids((b, l)),
            last_update=ids((b, l)),
            source_range=ids((b, l, 2)),
            support_count=zeros((b, l), torch.long),
            valid=zeros((b, l), torch.bool),
        ),
        last_frame_id=ids((b,)),
        stream_ids=ids((b,)),
    )


def _check_layout(old, fresh, path="state"):
    if isinstance(fresh, Tensor):
        if not isinstance(old, Tensor):
            raise TypeError(f"{path} must be a Tensor.")
        if (old.shape != fresh.shape or old.device != fresh.device
                or old.dtype != fresh.dtype):
            raise ValueError(f"{path} has incompatible shape, device, or dtype.")
        return
    if type(old) is not type(fresh):
        raise TypeError(f"{path} has an incompatible state type.")
    for field in fields(fresh):
        _check_layout(getattr(old, field.name), getattr(fresh, field.name),
                      f"{path}.{field.name}")


def _reset_rows(old, fresh, reset):
    if isinstance(fresh, Tensor):
        mask = reset.reshape((reset.shape[0],) + (1,) * (fresh.ndim - 1))
        return torch.where(mask, fresh, old)
    return replace(old, **{
        field.name: _reset_rows(getattr(old, field.name),
                               getattr(fresh, field.name), reset)
        for field in fields(fresh)
    })


def prepare_state(config, z, state, frame_ids, valid_mask, stream_ids):
    """Validate the V1 non-overlapping, contiguous-window protocol.

    Padding is a suffix. A changed active stream resets only its own row.
    An all-padding row never resets or advances its existing state.
    Batch size/order must remain aligned; stream changes reset, not migrate.
    """
    if z.ndim != 4 or z.shape[1] == 0:
        raise ValueError("z must have shape [B, T, J, D] with T > 0.")
    b, t, j, d = z.shape
    if (t, j, d) != (config.window_size, config.num_joints, config.dim_rep):
        raise ValueError("Encoded window shape does not match MemoryConfig.")
    for name, value, shape, kind in (
        ("frame_ids", frame_ids, (b, t), torch.long),
        ("valid_mask", valid_mask, (b, t), torch.bool),
        ("stream_ids", stream_ids, (b,), torch.long),
    ):
        if (not isinstance(value, Tensor) or value.shape != shape
                or value.dtype != kind or value.device != z.device):
            raise ValueError(f"{name} must have shape {shape}, dtype {kind}, and z.device.")

    if ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
        raise ValueError("V1 requires valid frames followed by suffix padding.")
    if (valid_mask & (frame_ids < 0)).any():
        raise ValueError("Valid frame IDs must be nonnegative.")
    active = valid_mask.any(dim=1)
    if (active & (stream_ids < 0)).any():
        raise ValueError("Active stream IDs must be nonnegative.")
    consecutive = frame_ids[:, 1:] - frame_ids[:, :-1] == 1
    if (valid_mask[:, 1:] & ~consecutive).any():
        raise ValueError("Valid frame IDs within a window must be consecutive.")

    fresh = init_state(config, b, z.device, z.dtype)
    if state is None:
        return fresh
    _check_layout(state, fresh)

    same = active & (state.stream_ids == stream_ids) & (state.last_frame_id >= 0)
    if (same & (frame_ids[:, 0] != state.last_frame_id + 1)).any():
        raise ValueError("Overlapping, reversed, or gapped windows cannot reuse V1 state.")

    reset = active & (state.stream_ids != stream_ids)
    if reset.any():
        state = _reset_rows(state, fresh, reset)
    return state


def advance_state(state, frame_ids, valid_mask, stream_ids):
    """Record the last valid consumed frame, without modifying the input state."""
    counts = valid_mask.long().sum(dim=1)
    active = counts > 0
    last = frame_ids.gather(1, (counts - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    return replace(
        state,
        last_frame_id=torch.where(active, last, state.last_frame_id),
        stream_ids=torch.where(active, stream_ids, state.stream_ids),
    )
