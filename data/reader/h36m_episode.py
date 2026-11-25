"""Continuous H36M episodes for the stateful MotionAGFormer V1.

Reads the original MotionBERT-format PKL, not the identity-free clip PKLs.
One sample is an independent episode; its windows are chronological and
non-overlapping. Only independent episodes may be shuffled or augmented.
"""

import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_CAMERA_HEIGHT = {"54138969": 1002, "60457274": 1002,
                  "55011271": 1000, "58860488": 1000}
# Same joint permutation as utils.data.flip_data for H36M (root = 0).
_FLIP_ORDER = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)


@dataclass(frozen=True)
class Episode:
    sequence_id: int
    sequence_start: int  # index in the original split
    start: int           # absolute index in the original split
    stop: int            # exclusive, real frames only
    source: str
    camera: str


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _normalize(raw):
    """Same camera-coordinate normalization and confidence as DataReaderH36M.

    3D targets remain in normalized image coordinates; the trainer applies
    the original per-frame root subtraction. No 3D labels enter the input.
    """
    xy = np.asarray(raw['joint_2d'])
    gt = np.asarray(raw['joint3d_image'])
    source = np.asarray(raw['source'])
    camera = np.asarray(raw['camera_name'])
    n = len(source)
    if (source.shape != (n,) or camera.shape != (n,) or xy.ndim != 3
            or gt.ndim != 3 or xy.shape[:2] != (n, 17) or gt.shape[:2] != (n, 17)
            or xy.shape[-1] < 2 or gt.shape[-1] < 3 or n == 0):
        raise ValueError('Expected aligned, nonempty H36M frame arrays with 17 joints.')
    if 'confidence' in raw:
        confidence = np.asarray(raw['confidence'], dtype=np.float32)
        if confidence.ndim == 2:
            confidence = confidence[..., None]
        if confidence.shape != (n, 17, 1):
            raise ValueError('confidence must have shape [N,17] or [N,17,1].')
    else:
        confidence = np.ones((n, 17, 1), dtype=np.float32)
    x = np.empty((n, 17, 3), dtype=np.float32)
    x[..., :2] = xy[..., :2]
    x[..., 2:] = confidence
    y = gt[..., :3].astype(np.float32, copy=True)
    x[..., :2] /= 1000
    x[..., :2] *= 2
    y /= 1000
    y *= 2
    x[..., 0] -= 1
    y[..., 0] -= 1
    for name in np.unique(camera):
        height = _CAMERA_HEIGHT.get(_text(name))
        if height is None:
            raise ValueError(f'Unknown H36M camera: {name!r}')
        rows = camera == name
        x[rows, :, 1] -= height / 1000
        y[rows, :, 1] -= height / 1000
    return x, y, source, camera


def _episodes(source, camera, length, stride):
    """Split contiguous source/camera runs, including the final run.

    Use regular full episodes and at most one additional partial episode
    per run to cover its tail. Never join disjoint runs with the same name.
    """
    boundaries = np.flatnonzero((source[1:] != source[:-1]) |
                                (camera[1:] != camera[:-1])) + 1
    starts = np.concatenate(([0], boundaries, [len(source)]))
    result = []
    for sequence_id, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        lo, hi = int(lo), int(hi)
        offsets = list(range(0, max(0, hi - lo - length) + 1, stride))
        if offsets[-1] + length < hi - lo:
            offsets.append(offsets[-1] + stride)
        for offset in offsets:
            start = lo + offset
            result.append(Episode(sequence_id, lo, start, min(start + length, hi),
                                  _text(source[lo]), _text(camera[lo])))
    return result


class H36MEpisodeDataset(Dataset):
    """Return fixed-shape [W,T,17,3] tensors with real-frame metadata.

    Frame IDs are zero-based within a source/camera run, not within each
    episode. Train stream IDs are unique per episode; test IDs are stable per
    contiguous run. Always reset training state at the start of each episode,
    including when the same sample is drawn again in a later epoch.
    """

    def __init__(self, path, split='train', window_size=243,
                 windows_per_episode=2, episode_stride=None, flip=True,
                 include_eval_metadata=False):
        if split not in ('train', 'test'):
            raise ValueError('split must be train or test.')
        if any(type(v) is not int or v <= 0 for v in (window_size, windows_per_episode)):
            raise ValueError('Window dimensions must be positive integers.')
        self.split = split
        self.window_size = window_size
        self.windows_per_episode = windows_per_episode
        self.length = window_size * windows_per_episode
        if episode_stride is None:
            episode_stride = max(1, window_size // 3) if split == 'train' else self.length
        if type(episode_stride) is not int or not 0 < episode_stride <= self.length:
            raise ValueError('episode_stride must be in [1, episode length].')
        if split == 'test' and episode_stride != self.length:
            raise ValueError('Test episodes must cover frames once, without overlap.')
        self.episode_stride = episode_stride
        self.flip = bool(flip) and split == 'train'
        # The original H36M pickle must be from a trusted source.
        with open(path, 'rb') as f:
            raw = pickle.load(f)[split]
        self.x, self.y, source, camera = _normalize(raw)
        self.episodes = _episodes(source, camera, self.length, episode_stride)
        self.eval_metadata = None
        if include_eval_metadata:
            if split != 'test':
                raise ValueError('Evaluation metadata is only available for the test split.')
            n = len(source)
            action = np.asarray(raw['action'])
            factor = np.asarray(raw['2.5d_factor'])
            gt = np.asarray(raw['joints_2.5d_image'])
            if (action.shape != (n,) or factor.size != n or gt.shape != (n, 17, 3)):
                raise ValueError('Invalid H36M evaluation metadata layout.')
            self.eval_metadata = {'action': action, 'factor': factor.reshape(n),
                                  'gt': gt, 'source': source}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = self.episodes[index]
        count = e.stop - e.start
        x = np.empty((self.length, 17, 3), dtype=np.float32)
        y = np.zeros_like(x)
        x[:count] = self.x[e.start:e.stop]
        y[:count] = self.y[e.start:e.stop]
        # Edge-repeat inputs for the fixed-size encoder, but never mark a
        # repeated frame as real or allow it to write history or receive loss.
        x[count:] = x[count - 1]
        flipped = self.flip and bool(torch.rand(()) < 0.5)
        if flipped:
            x = x[:, _FLIP_ORDER, :]
            y = y[:, _FLIP_ORDER, :]
            x[..., 0] *= -1
            y[..., 0] *= -1
        mask = np.arange(self.length) < count
        ids = np.full(self.length, -1, dtype=np.int64)
        global_ids = ids.copy()
        ids[:count] = np.arange(e.start - e.sequence_start, e.stop - e.sequence_start)
        global_ids[:count] = np.arange(e.start, e.stop)
        shape = (self.windows_per_episode, self.window_size)
        return {
            'x': torch.from_numpy(x.reshape(*shape, 17, 3)),
            'y': torch.from_numpy(y.reshape(*shape, 17, 3)),
            'frame_ids': torch.from_numpy(ids.reshape(shape)),
            'global_frame_ids': torch.from_numpy(global_ids.reshape(shape)),
            'valid_mask': torch.from_numpy(mask.reshape(shape)),
            'stream_id': index if self.split == 'train' else e.sequence_id,
            'sequence_id': e.sequence_id,
            'episode_id': index,
            'source': e.source,
            'camera_name': e.camera,
            'flipped': flipped,
        }


def build_episode_loader(dataset, batch_size, *, num_workers=0,
                         pin_memory=False, generator=None):
    """Shuffle whole episodes only; preserve ordered evaluation samples.

    num_workers=0 is a conservative default for large in-memory PKLs on
    spawn-based platforms. batch_size counts episodes, not windows.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == 'train',
                      num_workers=num_workers, pin_memory=pin_memory,
                      generator=generator, drop_last=False)
