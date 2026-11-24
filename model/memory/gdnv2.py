"""Gated DeltaNet-2 matrix-state reference for MotionAGFormer memory.

This is an independent PyTorch implementation of the published recurrence,
not the full GDN-2 language-model layer. The adaptation reads the incoming
state for an entire chunk, predicts, and only then updates the state.

Layout: state [B, J, H, K, V]; sequences [B, C, J, H, K or V].
All joints and heads are independent here; only time is recurrent.
Projection, SSI, MSM, and cooperation gates belong to the caller.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def _check_state(state: Tensor):
    if state.ndim != 5 or state.dtype != torch.float32:
        raise ValueError("state must be float32 with shape [B, J, H, K, V].")
    if any(size <= 0 for size in state.shape):
        raise ValueError("All state dimensions must be positive.")
    return state.shape


def _check_tensor(name, x, shape, state):
    if x.shape != shape or x.device != state.device:
        raise ValueError(f"{name} must have shape {tuple(shape)} on state.device.")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be floating point.")


def _check_mask(mask, shape, state):
    if mask is not None and (
        mask.shape != shape or mask.dtype != torch.bool or mask.device != state.device
    ):
        raise ValueError(f"valid_mask must be bool with shape {tuple(shape)} on state.device.")


def _as_fp32(x, mask):
    x = x.float()
    # Sanitize padding before any arithmetic, including normalization. Merely
    # masking the output can still propagate NaNs through backward.
    return x if mask is None else torch.where(mask, x, 0.0)


def _normalize(x, enabled):
    return F.normalize(x, p=2, dim=-1, eps=1e-6) if enabled else x


def gdn2_read(state: Tensor, query: Tensor, valid_mask=None, *, normalize_q=True):
    """Read the SAME incoming state for every query in a chunk, without writing.

    query: [B, C, J, H, K]; valid_mask: optional [B, C].
    Returns float32 [B, C, J, H, V]. The caller owns output projection/casting.
    """
    b, j, h, dk, _ = _check_state(state)
    if query.ndim != 5:
        raise ValueError("query must have shape [B, C, J, H, K].")
    c = query.shape[1]
    _check_tensor("query", query, (b, c, j, h, dk), state)
    _check_mask(valid_mask, (b, c), state)
    mask = None if valid_mask is None else valid_mask[:, :, None, None, None]
    with torch.autocast(device_type=state.device.type, enabled=False):
        q = _normalize(_as_fp32(query, mask), normalize_q)
        out = torch.einsum("btjhk,bjhkv->btjhv", q, state)
        return out if mask is None else torch.where(mask, out, 0.0)


def _step(state, k, v, log_decay, erase, write):
    # D(alpha) S; e = b * k; z = w * v.
    decayed = log_decay.exp().unsqueeze(-1) * state
    erased_value = ((erase * k).unsqueeze(-1) * decayed).sum(dim=-2)
    correction = write * v - erased_value
    return decayed + k.unsqueeze(-1) * correction.unsqueeze(-2)


def gdn2_step(state: Tensor, key: Tensor, value: Tensor, log_decay: Tensor,
              erase: Tensor, write: Tensor, valid_mask=None, *, normalize_k=True):
    """One GDN-2 transition; state [B, J, H, K, V], mask [B].

    key/log_decay/erase: [B, J, H, K]; value/write: [B, J, H, V].
    Caller supplies activated gates: log_decay <= 0, erase/write in [0, 1]
    for V1. No hidden clamping or gate generation is performed here.
    """
    b, j, h, dk, dv = _check_state(state)
    for name, x, dim in (("key", key, dk), ("value", value, dv),
                         ("log_decay", log_decay, dk), ("erase", erase, dk),
                         ("write", write, dv)):
        _check_tensor(name, x, (b, j, h, dim), state)
    _check_mask(valid_mask, (b,), state)
    mask = None if valid_mask is None else valid_mask[:, None, None, None]
    with torch.autocast(device_type=state.device.type, enabled=False):
        k = _normalize(_as_fp32(key, mask), normalize_k)
        v, g, e, w = (_as_fp32(x, mask) for x in (value, log_decay, erase, write))
        updated = _step(state, k, v, g, e, w)
        return updated if mask is None else torch.where(mask.unsqueeze(-1), updated, state)


def gdn2_update(state: Tensor, key: Tensor, value: Tensor, log_decay: Tensor,
                erase: Tensor, write: Tensor, valid_mask=None, *, normalize_k=True):
    """Apply a chunk of ordered transitions and return only its final FP32 state.

    key/log_decay/erase: [B, C, J, H, K]; value/write: [B, C, J, H, V].
    valid_mask: optional [B, C]. Invalid frames do not decay or write state.
    An empty chunk is the identity. No in-place state writes or detach calls.
    This deliberately serial reference will later be compared with optimized kernels.
    """
    b, j, h, dk, dv = _check_state(state)
    if key.ndim != 5:
        raise ValueError("key must have shape [B, C, J, H, K].")
    c = key.shape[1]
    for name, x, dim in (("key", key, dk), ("value", value, dv),
                         ("log_decay", log_decay, dk), ("erase", erase, dk),
                         ("write", write, dv)):
        _check_tensor(name, x, (b, c, j, h, dim), state)
    _check_mask(valid_mask, (b, c), state)
    mask = None if valid_mask is None else valid_mask[:, :, None, None, None]
    with torch.autocast(device_type=state.device.type, enabled=False):
        k = _normalize(_as_fp32(key, mask), normalize_k)
        v, g, e, w = (_as_fp32(x, mask) for x in (value, log_decay, erase, write))
        for t in range(c):
            updated = _step(state, k[:, t], v[:, t], g[:, t], e[:, t], w[:, t])
            state = updated if mask is None else torch.where(
                mask[:, t].unsqueeze(-1), updated, state
            )
        return state
