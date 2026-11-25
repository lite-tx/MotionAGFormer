"""Source-frame metadata for the unchanged MotionAGFormer 3DHP windows.

The stored train data and all 27 position slots are kept intact.  A separate
array maps each slot back to its physical source-frame index; state is reset
between groups, so indices only need to be unique within a single source.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def train_clip_source_indices(source_length: int, clip_index: int,
                              clip_length: int = 27, stride: int = 9) -> np.ndarray:
    """Mirror MPI3DHP.partition / resample without changing input contents."""
    source_length, clip_index = int(source_length), int(clip_index)
    if source_length <= 0 or clip_index < 0 or clip_length <= 0 or stride <= 0:
        raise ValueError('source length, clip length, stride and clip index are invalid')
    start = clip_index * stride
    if start >= source_length:
        raise ValueError('clip starts after the source sequence')
    present = min(clip_length, source_length - start)
    if present == clip_length:
        relative = np.arange(clip_length, dtype=np.int64)
    else:
        # Same floating-point operation as the original dataset's resample.
        relative = np.floor(np.linspace(0, present, num=clip_length,
                                        endpoint=False)).astype(np.int64)
        relative = np.clip(relative, 0, present - 1)
    return relative + start


def contiguous_groups_and_indices(sources: Sequence[dict], num_clips: int,
                                  clip_length: int = 27, stride: int = 9,
                                  max_group: int = 8):
    """Group adjacent original clips by source and cover each original once."""
    if num_clips < 0 or max_group <= 0:
        raise ValueError('invalid dataset or maximum group length')
    table = np.empty((num_clips, clip_length), dtype=np.int64)
    groups: list[list[int]] = []
    next_offset = 0
    for source in sources:
        offset, count, length = (int(source['offset']), int(source['clips']),
                                 int(source['frames']))
        expected_count = (length + stride - 1) // stride
        if length <= 0 or offset != next_offset or count != expected_count:
            raise ValueError(f'invalid source descriptor: {source}')
        if offset + count > num_clips:
            raise ValueError('source extends beyond dataset')
        for k in range(count):
            table[offset + k] = train_clip_source_indices(length, k, clip_length, stride)
        group_ids = range(offset, offset + count)
        groups.extend([list(range(i, min(i + max_group, offset + count)))
                       for i in range(group_ids.start, group_ids.stop, max_group)])
        next_offset = offset + count
    if next_offset != num_clips:
        raise ValueError('source descriptors do not cover original clips')
    flat = [clip for group in groups for clip in group]
    if flat != list(range(num_clips)):
        raise ValueError('contiguous grouping must cover each original clip once')
    return groups, table


def evaluation_source_indices(start_3d: int, end_3d: int, source_length: int,
                              pad: int, causal_shift: int = 0,
                              clip_length: int = 27) -> np.ndarray:
    """Mirror ChunkedGenerator.get_batch's temporal range and edge padding."""
    start, end = int(start_3d) - int(pad) - int(causal_shift), int(end_3d) + int(pad) - int(causal_shift)
    if source_length <= 0 or end <= start or end - start != clip_length:
        raise ValueError('unexpected generator window or source length')
    if int(start_3d) < 0 or int(end_3d) > source_length:
        raise ValueError('test target outside its source video')
    return np.clip(np.arange(start, end, dtype=np.int64), 0, int(source_length) - 1)


def group_digest(groups: Sequence[Sequence[int]]) -> str:
    h = hashlib.sha256()
    for group in groups:
        h.update((','.join(map(str, group)) + '\n').encode('ascii'))
    return h.hexdigest()
