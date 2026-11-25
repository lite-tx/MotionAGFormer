"""One random-start MotionAGFormer-XS + 3DHP memory comparison arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ablation_model import AblationModel, MODES
from common_3dhp import (AGFAdapter, CONFIG, DATA_ROOT, Fusion, GROUPS_PER_STEP,
                         MAX_EPOCHS, PATIENCE, ROOT_RUN, SEED, UPSTREAM, clip_loss,
                         denormalize, flip_data, get_cfg, load_bucket, load_model,
                         make_train_data, mpjpe_cal, seed_all, sha256)
from criteria import BestTracker


def create_model(mode):
    seed_all(SEED)
    cfg = get_cfg()
    backbone = AGFAdapter(load_model(cfg))  # No checkpoint or pretrained weights.
    return AblationModel(backbone, mode), cfg


def digest_backbone(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.backbone.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(repr(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def train_batch(model, optimizer, cfg, dataset, groups, selected, epoch, device):
    count = sum(len(groups[i]) for i in selected)
    buckets = {size: [i for i in selected if len(groups[i]) == size] for size in (1, 2, 3)}
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for size, ids in buckets.items():
        if not ids:
            continue
        x, y = load_bucket(dataset, groups, ids, epoch, device)
        output = model.forward_group(x)
        loss = sum(clip_loss(output[t], y[t], cfg) * len(ids) for t in range(size))
        if not torch.isfinite(loss):
            raise ValueError(f'nonfinite training loss in a length-{size} bucket')
        total += loss.detach().to(torch.float64)
        (loss / count).backward()
    grads = [(n, p.grad) for n, p in model.active_named_parameters() if p.grad is not None]
    if not grads or not torch.stack([torch.isfinite(g).all() for _, g in grads]).all():
        bad = [n for n, g in grads if not torch.isfinite(g).all()]
        raise ValueError(f'missing or nonfinite gradients: {bad[:10]}')
    optimizer.step()
    return float(total.item()), count


def evaluate(model, test_dataset, cfg, device):
    """Official target windows/center score, with state reset per video.

    Memory states are additional to the original baseline: all valid windows
    are visited in official order, so previous windows may overlap the current
    one. Normal and flip inputs have separate states along the batch axis.
    """
    model.eval()
    count = len(test_dataset)
    if count != 2875:
        raise ValueError(f'unexpected number of 3DHP test frames: {count}')
    predictions = torch.empty((count, 1, 17, 3), device=device, dtype=torch.float32)
    targets = torch.empty_like(predictions)
    names = []
    state, previous_seq, previous_center = None, None, None
    finished_sequences = set()
    left, right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(count):
            _, gt, inputs, seq, scale, _ = test_dataset[i]
            seq = str(seq)
            center = int(test_dataset.generator.pairs[i][1])
            if seq != previous_seq:
                if seq in finished_sequences:
                    raise ValueError(f'test source {seq} is not contiguous')
                if previous_seq is not None:
                    finished_sequences.add(previous_seq)
                state, previous_center = None, None
            if previous_center is not None and center <= previous_center:
                raise ValueError(f'test centers out of temporal order in {seq}')
            x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            prediction, state = model.forward_clip(x, state)
            augmented_back = flip_data(prediction[1], left, right)
            result = (prediction[0] + augmented_back) * (0.5 * float(scale))
            predictions[i, 0] = result[(cfg.n_frames - 1) // 2]
            targets[i, 0] = torch.as_tensor(gt[0], dtype=torch.float32, device=device)
            names.append(seq)
            previous_seq, previous_center = seq, center
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError('nonfinite 3DHP evaluation predictions or targets')
    predictions[:, :, 14] = 0
    targets[:, :, 14] = 0
    pred_mm = denormalize(predictions, names)
    pred_mm = pred_mm - pred_mm[..., 14:15, :]
    targets = targets - targets[..., 14:15, :]
    p1 = float(mpjpe_cal(pred_mm, targets).item())
    per_frame = torch.norm(pred_mm - targets, dim=-1).mean(dim=(-1, -2)).cpu().numpy()
    per_seq = {seq: float(np.mean(per_frame[np.asarray(names) == seq])) for seq in sorted(set(names))}
    if not math.isfinite(p1):
        raise ValueError('nonfinite 3DHP P1')
    return p1, per_seq, time.perf_counter() - start


def emit(run_dir, item):
    item['wall_time'] = time.time()
    text = json.dumps(item, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(text + '\n')


def atomic_checkpoint(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def provenance():
    here = Path(__file__).resolve().parent
    files = [here / n for n in ('train_3dhp_scratch.py', 'common_3dhp.py', 'original_model.py',
                                'ablation_model.py', 'criteria.py', 'preflight.py', 'launch_four.py')]
    files.extend([CONFIG, DATA_ROOT / 'data_train_3dhp.npz', DATA_ROOT / 'data_test_3dhp.npz',
                  UPSTREAM / 'train_3dhp.py', UPSTREAM / 'data/reader/motion_dataset.py',
                  UPSTREAM / 'data/reader/generator_3dhp.py', UPSTREAM / 'utils/data.py',
                  UPSTREAM / 'utils/utils_3dhp.py', UPSTREAM / 'loss/pose3d.py'])
    files.extend(sorted((UPSTREAM / 'model').rglob('*.py')))
    return {str(p): sha256(p) for p in files}


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('this arm requires exactly one selected GPU')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    model, cfg = create_model(args.mode)
    initial = digest_backbone(model)
    if initial != args.expected_backbone_sha256:
        raise ValueError('initial random backbone differs from preflight')
    model = model.to(device)
    active = list(model.active_named_parameters())
    # Exact optimizer family used by upstream train_3dhp.py. That code does
    # not pass the yaml weight_decay to Adam, hence effective decay is zero.
    optimizer = torch.optim.Adam((p for _, p in active), lr=float(cfg.learning_rate), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.lr_decay))
    tracker = BestTracker(patience=PATIENCE)
    dataset, groups, sources = make_train_data(cfg)
    test_dataset = Fusion(cfg, train=False)
    run_dir = ROOT_RUN / args.mode
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'mode': args.mode, 'initialization': 'random_no_checkpoint', 'seed': SEED,
                'backbone': 'MotionAGFormer-XS', 'dataset': 'MPI-INF-3DHP',
                'initial_backbone_sha256': initial, 'window_frames': 27, 'train_stride': 9,
                'groups_per_optimizer_step': GROUPS_PER_STEP, 'train_sources': len(sources),
                'train_clips': len(dataset), 'train_groups': len(groups),
                'test_valid_frames': len(test_dataset), 'test_window_center_only': True,
                'test_memory': 'previous official windows of the same video, including overlaps',
                'tta': 'normal and flipped stream states independent',
                'loss': 'position + 0.5 scale + 20 velocity',
                'optimizer': 'Adam, amsgrad=True, effective weight_decay=0',
                'learning_rate': float(cfg.learning_rate), 'lr_decay': float(cfg.lr_decay),
                'epochs_maximum': MAX_EPOCHS, 'patience': PATIENCE,
                'selection': 'official 3DHP mean MPJPE on valid test frames',
                'active_parameters': sum(p.numel() for _, p in active),
                'provenance': provenance()}
    (run_dir / 'run_config.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    emit(run_dir, {'event': 'training_started', **manifest})
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        order = list(range(len(groups)))
        random.Random(SEED + epoch).shuffle(order)
        steps = (len(order) + GROUPS_PER_STEP - 1) // GROUPS_PER_STEP
        total_loss, total_clips = 0.0, 0
        for step in range(steps):
            selected = order[step * GROUPS_PER_STEP:(step + 1) * GROUPS_PER_STEP]
            batch_loss, count = train_batch(model, optimizer, cfg, dataset, groups, selected, epoch, device)
            total_loss += batch_loss
            total_clips += count
            if (step + 1) % 100 == 0 or step + 1 == steps:
                emit(run_dir, {'event': 'train_progress', 'epoch': epoch, 'step': step + 1,
                     'steps_per_epoch': steps, 'clips_seen': total_clips,
                     'mean_loss': total_loss / total_clips,
                     'seconds': time.perf_counter() - start,
                     'peak_gib': torch.cuda.max_memory_allocated(device) / 2**30})
        if total_clips != len(dataset):
            raise ValueError(f'expected {len(dataset)} clips, saw {total_clips}')
        p1, per_seq, eval_seconds = evaluate(model, test_dataset, cfg, device)
        improved, stop = tracker.update(epoch, p1)
        scheduler.step()
        item = {'mode': args.mode, 'epoch': epoch, 'p1_mm': p1,
                'tracker': tracker.state_dict(), 'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'initial_backbone_sha256': initial, 'seed': SEED,
                'selection': manifest['selection'], 'provenance': manifest['provenance']}
        if improved:
            atomic_checkpoint(run_dir / 'best.pt', item)
        atomic_checkpoint(run_dir / 'last.pt', item)
        emit(run_dir, {'event': 'epoch_complete', 'epoch': epoch, 'P1_mm': p1,
             'per_sequence': per_seq, 'best_epoch': tracker.best_epoch, 'best_P1_mm': tracker.best_p1,
             'bad_epochs': tracker.bad_epochs, 'patience': tracker.patience,
             'improved': improved, 'stop': stop, 'train_clips': total_clips,
             'mean_train_loss': total_loss / total_clips, 'evaluation_seconds': eval_seconds,
             'seconds': time.perf_counter() - start})
        if stop:
            break
    emit(run_dir, {'event': 'training_complete', 'best_epoch': tracker.best_epoch,
                   'best_P1_mm': tracker.best_p1, 'last_epoch': epoch,
                   'early_stopped': tracker.bad_epochs >= tracker.patience})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--expected-backbone-sha256', required=True)
    run(parser.parse_args())
