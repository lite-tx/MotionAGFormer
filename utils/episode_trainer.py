"""Single-device, episode-level training for memory-v1 (H36M).

One optimizer step follows complete episode unrolls. Never carry a training
state across batches or optimizer steps. No truncation within an episode.
"""

import math
from dataclasses import fields, replace
from itertools import islice
from typing import NamedTuple

import torch
from torch import nn

from loss.episode_pose3d import EpisodePoseLoss
from model.memory.config import MemoryConfig


class EpochStats(NamedTuple):
    total: float
    position: float
    scale: float
    velocity: float
    valid_frames: int
    valid_pairs: int
    optimizer_steps: int


def build_episode_model(args):
    """Reuse the original factory; memory.enabled=False is the matched baseline."""
    from utils.learning import load_model

    if args.num_joints != 17 or args.dim_out != 3 or args.dim_in != 3:
        raise ValueError('V1 trainer requires H36M 17-joint xyz / xy-confidence input.')
    if not args.root_rel or args.add_velocity or args.use_proj_as_2d:
        raise ValueError('V1 requires root_rel=True, add_velocity=False, use_proj_as_2d=False.')
    if any(getattr(args, name, 0) != 0 for name in ('lambda_lv', 'lambda_lg', 'lambda_a', 'lambda_av')):
        raise ValueError('V1 trainer supports only position, scale and velocity losses.')
    options = dict(getattr(args, 'memory', {}))
    options.setdefault('enabled', True)
    options.setdefault('window_size', args.n_frames)
    options.setdefault('chunk_size', min(27, args.n_frames))
    options.setdefault('num_joints', args.num_joints)
    options.setdefault('dim_rep', args.dim_rep)
    options.setdefault('pose_dim', args.dim_out)
    return load_model(args, memory_config=MemoryConfig(**options))


def build_optimizer(model, learning_rate=5e-4, weight_decay=0.01, *, memory_lr=None, lr_decay=0.99):
    """AdamW and per-epoch exponential decay; default equal LR for all parameters."""
    for name, value in (('learning_rate', learning_rate), ('lr_decay', lr_decay)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if lr_decay > 1 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Expected lr_decay <= 1 and nonnegative finite weight_decay.')
    if memory_lr is not None and (not math.isfinite(memory_lr) or memory_lr <= 0):
        raise ValueError('memory_lr must be finite and positive.')
    base, memory = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (memory if name.startswith('memory_system.') else base).append(p)
    groups = []
    if base:
        groups.append({'params': base, 'lr': learning_rate, 'name': 'backbone_head'})
    if memory:
        groups.append({'params': memory, 'lr': learning_rate if memory_lr is None else memory_lr,
                       'name': 'memory'})
    if not groups:
        raise ValueError('No trainable parameters.')
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)


def _validate(batch):
    x, y = batch['x'], batch['y']
    mask, ids, streams = batch['valid_mask'], batch['frame_ids'], batch['stream_id']
    if (x.ndim != 5 or x.shape != y.shape or x.shape[-2:] != (17, 3)
            or any(n <= 0 for n in x.shape[:3])
            or not x.is_floating_point() or not y.is_floating_point()):
        raise ValueError('Expected x and y [B,W,T,17,3].')
    if (mask.shape != x.shape[:3] or mask.dtype != torch.bool
            or ids.shape != mask.shape or ids.dtype != torch.long
            or streams.shape != (x.shape[0],) or streams.dtype != torch.long
            or any(t.device != x.device for t in (y, mask, ids, streams))):
        raise ValueError('Episode metadata shape, dtype, or device mismatch.')
    m, t = mask.flatten(1), ids.flatten(1)
    if ((~m[:, :-1]) & m[:, 1:]).any():
        raise ValueError('Episode padding must be a suffix, including across windows.')
    if (m & (t < 0)).any() or (m.any(1) & (streams < 0)).any():
        raise ValueError('Valid frames require nonnegative frame and stream IDs.')
    if (m[:, 1:] & (t[:, 1:] - t[:, :-1] != 1)).any():
        raise ValueError('Episode frame IDs must be contiguous and non-overlapping.')
    return int(m.sum().item()), int((m[:, 1:] & m[:, :-1]).sum().item())


def _select_state(state, rows):
    """Functional batch selection; gradients stay connected to prior windows."""
    if isinstance(state, torch.Tensor):
        return state.index_select(0, rows)
    return replace(state, **{f.name: _select_state(getattr(state, f.name), rows)
                             for f in fields(state)})


def forward_episode(model, batch):
    """Return [B,W,T,17,3]; reset state at entry, keep all episode gradients.

    Fully padded rows never enter the encoder, so they cannot affect its
    BatchNorm. Partial windows still use edge-repeat padding; no backbone
    mask is implied. Single device only: do not bypass DP/DDP forward hooks.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        raise ValueError('Use an unwrapped single-device model for episode training V1.')
    _validate(batch)
    x, mask, ids, streams = (batch[k] for k in ('x', 'valid_mask', 'frame_ids', 'stream_id'))
    b, windows, t, j, _ = x.shape
    state, previous, outputs = None, None, []
    for w in range(windows):
        active = mask[:, w].any(1)
        rows = active.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            ref = outputs[-1] if outputs else x
            outputs.append(ref.new_zeros((b, t, j, 3)))
            continue
        if state is not None and rows.numel() != previous.numel():
            state = _select_state(state, active[previous].nonzero(as_tuple=True)[0])
        m = mask[rows, w]
        window = x[rows, w]
        last = window[torch.arange(rows.numel(), device=x.device), m.sum(1)-1]
        window = torch.where(m[..., None, None], window, last[:, None])
        p, state = model.forward_window(window, state, ids[rows, w], m, streams[rows])
        p = torch.where(m[..., None, None], p, 0.0)
        outputs.append(p.new_zeros((b, t, j, 3)).index_copy(0, rows, p))
        previous = rows
    return torch.stack(outputs, dim=1)


def train_episode_epoch(model, loader, optimizer, criterion: EpisodePoseLoss, device,
                        *, accumulation_steps=1, precision='fp32', grad_clip=None):
    """Train on complete episodes, with count-weighted microbatch accumulation.

    FP32 or BF16 autocast (no FP16 scaler in V1). For unequal lengths, frame
    losses and velocity losses use their own effective-batch denominators.
    Scheduler.step() belongs to the caller, once per completed epoch.
    """
    device = torch.device(device)
    if type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError('accumulation_steps must be a positive integer.')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('V1 supports fp32 or bf16 precision.')
    if precision == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support BF16; use fp32.')
    if precision == 'bf16' and device.type not in ('cpu', 'cuda'):
        raise ValueError('BF16 V1 is supported on CPU/CUDA only.')
    if grad_clip is not None and (not math.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError('grad_clip must be finite and positive, or None.')
    model.train()
    sums = [0., 0., 0.]
    frames = pairs = steps = 0
    iterator = iter(loader)
    params = [p for g in optimizer.param_groups for p in g['params']]
    while True:
        group = list(islice(iterator, accumulation_steps))
        if not group:
            break
        counts = [_validate(batch) for batch in group]
        nf, npairs = (sum(c[i] for c in counts) for i in (0, 1))
        optimizer.zero_grad(set_to_none=True)
        if not nf:
            continue
        for batch, (f, p) in zip(group, counts):
            if not f:
                continue
            batch = {k: batch[k].to(device, non_blocking=True)
                     for k in ('x', 'y', 'valid_mask', 'frame_ids', 'stream_id')}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                pred = forward_episode(model, batch)
                result = criterion(pred, batch['y'], batch['valid_mask'], batch['frame_ids'])
            loss = (result.position + criterion.lambda_scale * result.scale) * (f / nf)
            loss = loss + criterion.lambda_3d_velocity * result.velocity * (p / max(1, npairs))
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('Nonfinite episode loss; optimizer step aborted.')
            loss.backward()
            sums[0] += result.position.detach().item() * f
            sums[1] += result.scale.detach().item() * f
            sums[2] += result.velocity.detach().item() * p
            frames += f
            pairs += p
        # Even when clipping is disabled, reject nonfinite gradients.
        torch.nn.utils.clip_grad_norm_(params, float('inf') if grad_clip is None else grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        steps += 1
    pos, scale, vel = sums[0]/max(1, frames), sums[1]/max(1, frames), sums[2]/max(1, pairs)
    return EpochStats(pos + criterion.lambda_scale*scale + criterion.lambda_3d_velocity*vel,
                      pos, scale, vel, frames, pairs, steps)
