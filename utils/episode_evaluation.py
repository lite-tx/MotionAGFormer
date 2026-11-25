"""H36M continuous-stream evaluation (not the original clip coverage).

Single-stream ordered evaluation avoids mixing state across Batch rows.
The original coordinate restoration, exclusions and action-macro metrics
are retained. All real frames are evaluated once, including sequence tails.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from data.reader.h36m_episode import _FLIP_ORDER, _text


_BLOCKED = {'s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01'}


def flip_h36m(x):
    """Reflect x coordinates and exchange left/right joints (out of place)."""
    result = x[..., list(_FLIP_ORDER), :].clone()
    result[..., 0] = -result[..., 0]
    return result


def restore_pose(pred, factors):
    """Original root-zero + denormalize + 2.5D factor + root-relative protocol.

    All four H36M cameras have width 1000. The image-center translation
    cancels after root subtraction, so only the 500 scale remains.
    """
    pred = np.asarray(pred, dtype=np.float64).copy()
    pred[:, 0] = 0.0
    return pred * (500.0 * np.asarray(factors).reshape(-1, 1, 1))


def aligned_error(pred, gt):
    """Per-frame similarity-aligned MPJPE (proper rotation, scale, translation).

    Degenerate collapsed poses align to the target centroid; no zero-norm
    division. Nondegenerate cases are equivalent to pose3d.p_mpjpe.
    """
    p = pred - pred.mean(axis=1, keepdims=True)
    g = gt - gt.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(p.transpose(0, 2, 1) @ g)
    sign = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    u[:, :, -1] *= sign[:, None]
    s[:, -1] *= sign
    rotation = u @ vt
    norm2 = (p*p).sum(axis=(1, 2))
    scale = np.divide(s.sum(axis=1), norm2, out=np.zeros_like(norm2), where=norm2 > 0)
    return np.linalg.norm(scale[:, None, None] * (p @ rotation) - g, axis=-1).mean(1)


class StreamMetrics:
    """Accumulate errors per action; acceleration has its own triple count."""

    def __init__(self):
        self.joint_sum = defaultdict(lambda: np.zeros(17, dtype=np.float64))
        self.pa_sum = defaultdict(float)
        self.frames = defaultdict(int)
        self.acc_sum = defaultdict(float)
        self.triples = defaultdict(int)
        self.tail = None

    def update(self, pred, gt, ids, action, stream_id):
        if not (len(pred) and pred.shape == gt.shape == (len(ids), 17, 3)):
            raise ValueError('Expected nonempty aligned prediction/target frames.')
        if not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise FloatingPointError('Nonfinite H36M evaluation pose.')
        if np.any(np.diff(ids) != 1):
            raise ValueError('Evaluation frames must be contiguous within a stream.')
        error = np.asarray(pred, dtype=np.float64) - gt
        self.joint_sum[action] += np.linalg.norm(error, axis=-1).sum(0)
        self.pa_sum[action] += aligned_error(pred, gt).sum()
        self.frames[action] += len(ids)
        if self.tail is not None:
            previous_stream, previous_action, previous_id, previous_error = self.tail
            if (stream_id == previous_stream and action == previous_action
                    and ids[0] == previous_id + 1):
                error = np.concatenate((previous_error, error))
        if len(error) >= 3:
            acc = np.linalg.norm(error[:-2] - 2*error[1:-1] + error[2:], axis=-1).mean(1)
            self.acc_sum[action] += acc.sum()
            self.triples[action] += len(acc)
        self.tail = (stream_id, action, int(ids[-1]), error[-2:].copy())

    def result(self):
        if not self.frames:
            raise ValueError('No eligible evaluation frames.')
        actions = {}
        for a in sorted(self.frames):
            n, t = self.frames[a], self.triples[a]
            actions[a] = {'mpjpe': float(self.joint_sum[a].mean()/n),
                          'p_mpjpe': float(self.pa_sum[a]/n),
                          'acceleration': float(self.acc_sum[a]/t) if t else None,
                          'frames': n, 'acceleration_triples': t}
        acc = [v['acceleration'] for v in actions.values() if v['acceleration'] is not None]
        return {'protocol': 'h36m_continuous_full_frames_v1',
                'mpjpe': float(np.mean([v['mpjpe'] for v in actions.values()])),
                'p_mpjpe': float(np.mean([v['p_mpjpe'] for v in actions.values()])),
                'acceleration': float(np.mean(acc)) if acc else None,
                'joint_errors': np.mean([self.joint_sum[a]/n for a,n in self.frames.items()], axis=0).tolist(),
                'valid_frames': sum(self.frames.values()),
                'acceleration_triples': sum(self.triples.values()), 'actions': actions}


@torch.no_grad()
def evaluate_h36m(model, loader, device, *, flip=True, precision='fp32', carry_memory=True):
    """Evaluate ordered test episodes with two independent TTA memory states."""
    ds = loader.dataset
    if (loader.batch_size != 1 or not isinstance(loader.sampler, SequentialSampler)
            or ds.split != 'test' or ds.eval_metadata is None or loader.drop_last):
        raise ValueError('Use the ordered test EpisodeDataset with metadata, batch_size=1, drop_last=False.')
    device = torch.device(device)
    if precision not in ('fp32', 'bf16') or (precision == 'bf16' and
            (device.type not in ('cpu', 'cuda') or
             (device.type == 'cuda' and not torch.cuda.is_bf16_supported()))):
        raise ValueError('Unsupported evaluation precision/device.')
    model.eval()
    metrics = StreamMetrics()
    meta = ds.eval_metadata
    state, flipped_state, last_stream = None, None, None
    next_global, next_local, excluded = 0, 0, 0
    for episode_index, batch in enumerate(loader):
        if int(batch['episode_id'].item()) != episode_index or bool(batch['flipped'].item()):
            raise ValueError('Test episodes must be ordered and unaugmented.')
        stream = int(batch['stream_id'].item())
        if stream != last_stream:
            state, flipped_state, next_local = None, None, 0
            last_stream = stream
        x, mask, ids, streams = (batch[k].to(device) for k in ('x','valid_mask','frame_ids','stream_id'))
        for w in range(x.shape[1]):
            m = mask[:, w]
            if not m.any():
                continue
            local = ids[:, w][m].cpu().numpy()
            global_ids = batch['global_frame_ids'][0,w][m[0].cpu()].numpy()
            if (not np.array_equal(local, np.arange(next_local, next_local+len(local))) or
                    not np.array_equal(global_ids, np.arange(next_global, next_global+len(local)))):
                raise ValueError('Evaluation contains a gap, overlap or reordered frame.')
            next_local += len(local)
            next_global += len(local)
            if not carry_memory:
                state, flipped_state = None, None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                p, state = model.forward_window(x[:,w], state, ids[:,w], m, streams)
                if flip:
                    fp, flipped_state = model.forward_window(flip_h36m(x[:,w]), flipped_state,
                                                              ids[:,w], m, streams)
                    p = (p.float() + flip_h36m(fp).float()) * 0.5
            source = _text(meta['source'][global_ids[0]])
            if any(_text(v) != source for v in meta['source'][global_ids]):
                raise ValueError('Source changes inside an evaluation window.')
            if source in _BLOCKED or source[:-6] in _BLOCKED:
                excluded += len(local)
                metrics.tail = None
                continue
            action = _text(meta['action'][global_ids[0]])
            if any(_text(v) != action for v in meta['action'][global_ids]):
                raise ValueError('Action changes inside an evaluation window.')
            pred = restore_pose(p[m].float().cpu().numpy(), meta['factor'][global_ids])
            gt = np.asarray(meta['gt'][global_ids], dtype=np.float64)
            gt = gt - gt[:, :1]
            metrics.update(pred, gt, local, action, stream)
    if next_global != len(ds.x):
        raise ValueError('Evaluation did not cover every original test frame.')
    result = metrics.result()
    result.update(excluded_frames=excluded, flip=bool(flip), carry_memory=bool(carry_memory))
    return result
