"""Strict model initialization and epoch-boundary checkpointing for V1.

No dynamic video memory is serialized: training episodes reset at entry.
Only trusted checkpoints should be used; restricted tensor loading is required.
"""

import math
import os
import random
import tempfile
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch


def _read(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _weights(payload):
    if not isinstance(payload, Mapping):
        raise ValueError('Checkpoint must be a mapping.')
    raw = payload.get('model', payload.get('state_dict', payload))
    if not isinstance(raw, Mapping):
        raise ValueError('Checkpoint model must be a state dict.')
    result = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Model state dict must contain string keys and tensors.')
        key = key[7:] if key.startswith('module.') else key
        if key in result:
            raise ValueError('Duplicate checkpoint key after module-prefix removal.')
        result[key] = value
    return result


def _load_checked(model, weights, pretrained=False):
    expected = model.state_dict()
    missing = set(expected) - set(weights)
    extra = set(weights) - set(expected)
    mismatch = [k for k in set(expected) & set(weights) if expected[k].shape != weights[k].shape]
    if pretrained:
        allowed_missing = {k for k in expected if k.startswith('memory_system.')}
        # Warm-start expects an original backbone/head, never a partial memory checkpoint.
        if missing != allowed_missing or extra or mismatch:
            raise ValueError(f'Original pretrained checkpoint mismatch: missing={sorted(missing-allowed_missing)}, '
                             f'extra={sorted(extra)}, shape={sorted(mismatch)}, memory_present={sorted(allowed_missing-missing)}')
    elif missing or extra or mismatch:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shape={sorted(mismatch)}')
    model.load_state_dict(weights, strict=not pretrained)
    return len(weights), sorted(missing)


def load_pretrained(path, model):
    """Require every original backbone/head tensor; leave memory newly initialized."""
    return _load_checked(model, _weights(_read(path)), pretrained=True)


def load_weights(path, model):
    """Exact-model evaluation load; accepts original module-prefixed baseline weights."""
    return _load_checked(model, _weights(_read(path)))


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f'Nonserializable config value: {type(value)}')


def _rng(generator, device):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None}


def _restore_rng(state, generator, device):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda'] is not None:
        if torch.device(device).type != 'cuda' or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG restore requires the same device count; use a weights-only load otherwise.')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, *, next_epoch, best_mpjpe,
                    config, generator, device):
    """Call after scheduler.step(); saved LR is the next epoch's LR."""
    if type(next_epoch) is not int or next_epoch < 0 or not math.isfinite(best_mpjpe):
        raise ValueError('Invalid completed epoch or best metric.')
    payload = {'format': 'memory-v1-epoch-1', 'next_epoch': next_epoch,
               'best_mpjpe': float(best_mpjpe), 'config': _plain(config),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'rng': _rng(generator, device),
               'device_type': torch.device(device).type}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resume_checkpoint(path, model, optimizer, scheduler, *, config, generator, device):
    """Restore full optimizer/scheduler/RNG state at an epoch boundary."""
    payload = _read(path)
    if payload.get('format') != 'memory-v1-epoch-1':
        raise ValueError('Not an episode-training checkpoint; use pretrained/weights loading.')
    old, new = dict(payload['config']), _plain(config)
    for key in ('epochs', 'name'):
        old.pop(key, None)
        new.pop(key, None)
    if old != new:
        raise ValueError('Resume config differs from saved config (only epochs/name may change).')
    if payload['device_type'] != torch.device(device).type:
        raise ValueError('Resume requires the saved device type; use weights-only loading to migrate.')
    cuda = payload['rng']['cuda']
    if cuda is not None and len(cuda) != torch.cuda.device_count():
        raise ValueError('Resume requires the saved CUDA device count.')
    _load_checked(model, _weights(payload))
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    _restore_rng(payload['rng'], generator, device)
    return payload['next_epoch'], payload['best_mpjpe']
