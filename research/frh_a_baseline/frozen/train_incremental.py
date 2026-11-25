"""One complete, random-start MotionAGFormer-XS overlap-memory run.

Normal execution covers every original clip per epoch and evaluates all 2,875
official valid 3DHP targets.  ``--preflight-max-steps`` is explicitly marked
as a partial preflight; it cannot produce a best model or a benchmark result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ORIGINAL = Path(os.environ.get('XMEM_ORIGINAL_DIR',
                               '/home/xzy/3DHPE/motionagformer_xs_3dhp_xmem_20261109'))
sys.path.insert(0, str(ORIGINAL))
import common_3dhp as base  # noqa: E402
from criteria import BestTracker  # noqa: E402
from train_3dhp_scratch import atomic_checkpoint, digest_backbone, emit  # noqa: E402
from data_index import contiguous_groups_and_indices, evaluation_source_indices, group_digest  # noqa: E402
from incremental_model import IncrementalXMem  # noqa: E402

SEED = 20261109
GROUPS_PER_STEP = 12
MAX_GROUP = 8
MAX_EPOCHS = 60
PATIENCE = 15
ORIGINAL_BACKBONE_SHA256 = 'c0313d75c1eaa944f42c2b71d0974d76e932b117266728fe7794bf05e3113754'
DEFAULT_RUN_ROOT = Path('/data/xzy/3dhpe_experiments/xmem_overlap_20261116')


def make_train_data(cfg):
    """Retain all original samples, replacing only the source-local grouping."""
    dataset, _, sources = base.make_train_data(cfg)
    groups, indices = contiguous_groups_and_indices(
        sources, len(dataset), clip_length=int(cfg.n_frames),
        stride=int(cfg.stride), max_group=MAX_GROUP)
    return dataset, groups, sources, indices


def train_batch(model, optimizer, cfg, dataset, groups, source_indices,
                selected, epoch, device):
    count = sum(len(groups[i]) for i in selected)
    if count <= 0:
        raise ValueError('empty training step')
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for size in range(1, MAX_GROUP + 1):
        ids = [i for i in selected if len(groups[i]) == size]
        if not ids:
            continue
        # The original loader preserves the data and applies one flip per group.
        x, y = base.load_bucket(dataset, groups, ids, epoch, device)
        group_indices = np.asarray([groups[i] for i in ids], dtype=np.int64)
        # Keep integer metadata on CPU: no GPU-to-CPU synchronisation is needed.
        frame_ids = torch.from_numpy(source_indices[group_indices].transpose(1, 0, 2).copy())
        output = model.forward_group(x, frame_ids)
        if output.shape != y.shape:
            raise ValueError(f'prediction/target shape mismatch: {output.shape} vs {y.shape}')
        loss = sum(base.clip_loss(output[t], y[t], cfg) * len(ids) for t in range(size))
        if not torch.isfinite(loss):
            raise ValueError(f'nonfinite loss in length-{size} bucket')
        total += loss.detach().to(torch.float64)
        (loss / count).backward()
    gradients = [(n, p.grad) for n, p in model.active_named_parameters() if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for _, g in gradients]).all():
        raise ValueError(f'missing or nonfinite active gradients: {[n for n, g in gradients if not torch.isfinite(g).all()][:10]}')
    # A single-window group legitimately has no loss depending on its newly
    # created state.  Once any multi-window group is present, the following
    # parameters must have an autograd path from later-window pose losses.
    # Zero-valued (but present and finite) gradients are allowed.
    if any(len(groups[i]) >= 2 for i in selected):
        required = ('gru.', 'value_gru.', 'key_proj.', 'selection_proj.',
                    'shrinkage_proj.', 'value_proj.')
        active = list(model.active_named_parameters())
        missing = []
        for prefix in required:
            members = [(n, p) for n, p in active if n.startswith(prefix)]
            if not members:
                missing.append(prefix + '(missing parameter group)')
            else:
                missing.extend(n for n, p in members if p.grad is None)
        if missing:
            raise ValueError(f'missing gradients in multi-window memory paths: {missing}')
    optimizer.step()
    return float(total.item()), count


def evaluate(model, test_dataset, cfg, device):
    """Use the same official windows, flip TTA, root alignment and P1 as before."""
    model.eval()
    count = len(test_dataset)
    if count != 2875:
        raise ValueError(f'unexpected official valid 3DHP target count: {count}')
    predictions = torch.empty((count, 1, 17, 3), device=device, dtype=torch.float32)
    targets = torch.empty_like(predictions)
    names = []
    state, previous_seq, previous_center = None, None, None
    finished_sequences = set()
    left, right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
    generator = test_dataset.generator
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(count):
            _, gt, inputs, seq, scale, _ = test_dataset[i]
            seq = str(seq)
            pair = generator.pairs[i]
            center, end = int(pair[1]), int(pair[2])
            if seq != previous_seq:
                if seq in finished_sequences:
                    raise ValueError(f'test video is not contiguous: {seq}')
                if previous_seq is not None:
                    finished_sequences.add(previous_seq)
                state, previous_center = None, None
            if previous_center is not None and center <= previous_center:
                raise ValueError(f'test centers not increasing within {seq}')
            source_length = len(generator.poses_2d[seq])
            source_idx = evaluation_source_indices(center, end, source_length,
                                                    generator.pad,
                                                    generator.causal_shift,
                                                    int(cfg.n_frames))
            x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            if x.shape != (2, int(cfg.n_frames), 17, 3):
                raise ValueError(f'unexpected flip-TTA input shape: {x.shape}')
            # Both streams describe the same source times; their state rows remain separate.
            frame_ids = torch.from_numpy(np.stack((source_idx, source_idx)).copy())
            prediction, state = model.forward_clip(x, state=state, frame_indices=frame_ids)
            mirrored = base.flip_data(prediction[1], left, right)
            result = (prediction[0] + mirrored) * (0.5 * float(scale))
            predictions[i, 0] = result[(cfg.n_frames - 1) // 2]
            targets[i, 0] = torch.as_tensor(gt[0], dtype=torch.float32, device=device)
            names.append(seq)
            previous_seq, previous_center = seq, center
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError('nonfinite prediction or target during evaluation')
    predictions[:, :, 14] = 0
    targets[:, :, 14] = 0
    pred_mm = base.denormalize(predictions, names)
    pred_mm = pred_mm - pred_mm[..., 14:15, :]
    targets = targets - targets[..., 14:15, :]
    p1 = float(base.mpjpe_cal(pred_mm, targets).item())
    per_frame = torch.norm(pred_mm - targets, dim=-1).mean(dim=(-1, -2)).cpu().numpy()
    per_seq = {seq: float(np.mean(per_frame[np.asarray(names) == seq]))
               for seq in sorted(set(names))}
    if not math.isfinite(p1):
        raise ValueError('nonfinite P1')
    return p1, per_seq, time.perf_counter() - started


def _provenance(data_root):
    # Hash complete source/config/data files rather than recording just names.
    files = [Path(__file__).resolve(), Path(__file__).with_name('data_index.py'),
             Path(__file__).with_name('incremental_model.py'), base.CONFIG,
             ORIGINAL / 'common_3dhp.py', ORIGINAL / 'original_model.py',
             ORIGINAL / 'train_3dhp_scratch.py', ORIGINAL / 'criteria.py',
             base.UPSTREAM / 'data/reader/motion_dataset.py',
             base.UPSTREAM / 'data/reader/generator_3dhp.py',
             Path(data_root) / 'data_train_3dhp.npz', Path(data_root) / 'data_test_3dhp.npz']
    files.extend(sorted((base.UPSTREAM / 'model').rglob('*.py')))
    return {str(p): base.sha256(p) for p in files}


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('select exactly one available GPU with CUDA_VISIBLE_DEVICES')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('formal training and GPU preflight require a CUDA device')
    if not 1 <= args.epoch_limit <= MAX_EPOCHS or args.query_chunk <= 0:
        raise ValueError('invalid epoch limit or query chunk')
    if args.preflight_max_steps < 0:
        raise ValueError('invalid preflight step limit')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.DATA_ROOT = Path(args.data_dir).resolve()
    base.seed_all(SEED)
    cfg = base.get_cfg()
    backbone = base.AGFAdapter(base.load_model(cfg))
    model = IncrementalXMem(backbone, query_chunk=args.query_chunk)
    initial_hash = digest_backbone(model)
    if initial_hash != ORIGINAL_BACKBONE_SHA256:
        raise ValueError(f'random backbone mismatch vs prior experiment: {initial_hash}')
    model = model.to(device)
    active = list(model.active_named_parameters())
    optimizer = torch.optim.Adam((p for _, p in active), lr=float(cfg.learning_rate), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.lr_decay))
    tracker = BestTracker(patience=PATIENCE)
    run_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RUN_ROOT / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        hashes = _provenance(base.DATA_ROOT)
        dataset, groups, sources, source_indices = make_train_data(cfg)
        test_data = base.Fusion(cfg, train=False)
        is_preflight = args.preflight_max_steps > 0
        manifest = {'mode': 'full_incremental_xmem', 'preflight_only': is_preflight,
                    'initialization': 'random_no_checkpoint', 'seed': SEED,
                    'backbone': 'MotionAGFormer-XS', 'dataset': 'MPI-INF-3DHP',
                    'initial_backbone_sha256': initial_hash, 'window_frames': 27,
                    'train_stride': 9, 'group_window_stride': 1, 'max_windows_per_group': MAX_GROUP,
                    'groups_per_optimizer_step': GROUPS_PER_STEP, 'train_sources': len(sources),
                    'train_clips': len(dataset), 'train_groups': len(groups),
                    'groups_sha256': group_digest(groups), 'test_valid_frames': len(test_data),
                    'test_window_center_only': True, 'query_chunk': args.query_chunk,
                    'memory': 'per source-frame/joint; newly observed only; cross-joint full-bank read',
                    'normal_and_value_gru': 'per-new-frame sequential shared hidden',
                    'state_gap_policy': 'carry state; no unobserved-frame inputs',
                    'flip': 'one consistent train-group flip; independent normal/flip test state',
                    'loss': 'all-window position + 0.5 scale + 20 velocity',
                    'optimizer': 'Adam, amsgrad=True, effective weight_decay=0',
                    'learning_rate': float(cfg.learning_rate), 'lr_decay': float(cfg.lr_decay),
                    'epochs_maximum': args.epoch_limit, 'patience': PATIENCE,
                    'selection': '3DHP mean P1 over official valid test frames',
                    'active_parameters': sum(p.numel() for _, p in active),
                    'data_loading_and_provenance_seconds': time.perf_counter() - started,
                    'provenance': hashes}
        (run_dir / 'run_config.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        emit(run_dir, {'event': 'training_started', **manifest})
        for epoch in range(1, args.epoch_limit + 1):
            model.train()
            epoch_start = time.perf_counter()
            order = list(range(len(groups)))
            random.Random(SEED + epoch).shuffle(order)
            steps = math.ceil(len(order) / GROUPS_PER_STEP)
            total_loss, total_clips = 0.0, 0
            for step in range(steps):
                selected = order[step * GROUPS_PER_STEP:(step + 1) * GROUPS_PER_STEP]
                value, count = train_batch(model, optimizer, cfg, dataset, groups,
                                           source_indices, selected, epoch, device)
                total_loss += value
                total_clips += count
                if is_preflight or step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                    emit(run_dir, {'event': 'train_progress', 'epoch': epoch,
                                   'step': step + 1, 'steps_per_epoch': steps,
                                   'clips_seen': total_clips,
                                   'mean_loss': total_loss / total_clips,
                                   'seconds': time.perf_counter() - epoch_start,
                                   'peak_gib': torch.cuda.max_memory_allocated(device) / 2**30})
                if is_preflight and step + 1 >= args.preflight_max_steps:
                    emit(run_dir, {'event': 'preflight_complete', 'epoch': epoch,
                                   'steps': step + 1, 'clips_seen': total_clips,
                                   'mean_step_seconds': (time.perf_counter() - epoch_start) / (step + 1),
                                   'no_formal_result': True})
                    return
            if total_clips != len(dataset):
                raise ValueError(f'expected {len(dataset)} train clips, saw {total_clips}')
            p1, per_seq, eval_seconds = evaluate(model, test_data, cfg, device)
            improved, stop = tracker.update(epoch, p1)
            scheduler.step()
            item = {'mode': manifest['mode'], 'epoch': epoch, 'p1_mm': p1,
                    'tracker': tracker.state_dict(),
                    'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'initial_backbone_sha256': initial_hash, 'seed': SEED,
                    'groups_sha256': manifest['groups_sha256'],
                    'selection': manifest['selection'], 'provenance': hashes}
            if improved:
                atomic_checkpoint(run_dir / 'best.pt', item)
            atomic_checkpoint(run_dir / 'last.pt', item)
            emit(run_dir, {'event': 'epoch_complete', 'epoch': epoch, 'P1_mm': p1,
                           'per_sequence': per_seq, 'best_epoch': tracker.best_epoch,
                           'best_P1_mm': tracker.best_p1, 'bad_epochs': tracker.bad_epochs,
                           'patience': tracker.patience, 'improved': improved, 'stop': stop,
                           'train_clips': total_clips, 'mean_train_loss': total_loss / total_clips,
                           'evaluation_seconds': eval_seconds,
                           'seconds': time.perf_counter() - epoch_start,
                           'peak_gib': torch.cuda.max_memory_allocated(device) / 2**30})
            if stop:
                break
        emit(run_dir, {'event': 'training_complete', 'best_epoch': tracker.best_epoch,
                       'best_P1_mm': tracker.best_p1, 'last_epoch': epoch,
                       'early_stopped': tracker.bad_epochs >= tracker.patience})
    except Exception as exc:
        emit(run_dir, {'event': 'run_failed', 'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(base.DATA_ROOT))
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch-limit', type=int, default=MAX_EPOCHS)
    parser.add_argument('--query-chunk', type=int, default=51)
    parser.add_argument('--preflight-max-steps', type=int, default=0,
                        help='explicit partial speed check, never writes benchmark checkpoints')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
