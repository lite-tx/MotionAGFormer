"""Shared, fixed-recipe 150-epoch training and final evaluation wrappers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

REPO = Path(__file__).resolve().parent
os.environ['XMEM_ORIGINAL_DIR'] = str(REPO)

import numpy as np
import torch
import train_controlled as reference

original = reference.original
base = reference.base
SEED = 20261111
EPOCHS = 150
UPDATES_PER_EPOCH = 1224
CLIPS_PER_EPOCH = 117024
GROUPS = 14680
GROUP_SHA = 'e24d914b203df27d95a9a463bd58e0a5d03536f503733ddb54c39a907089a0b8'
INITIAL_REFERENCE_SHA = '005075a8875e806d9da045a26c3ae41420d6eb3a7447ddd5a47ad58f872b2b65'
PARENT_SHA = '1dacd59fe0ce009b6293ce5b120fbc90d413a8c2184127291257e841ff44dec7'
TEST_SHA = 'f01b0605944ead1eca5f731c90daa7deaa23ae86d3f084ff9f45a2d434e24018'
SAVE_EPOCHS = (60, 90, 120, 150)
VARIANTS = ('baseline', 'h_query', 'matrix_h')


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(output, event, **payload):
    row = {'event': event, 'timestamp_utc': now(), **payload}
    with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def rng_state():
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}


def restore_rng(saved):
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])


def tensor_dict_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(repr(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure(data_dir):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select one assigned CUDA device')
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise RuntimeError('This registration uses an assigned NVIDIA L40')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.DATA_ROOT = data_dir.resolve()
    if base.sha256(base.DATA_ROOT / 'data_train_3dhp.npz') != reference.TRAIN_SHA:
        raise ValueError('Training data differs from the accepted run')
    cfg = base.get_cfg()
    if base.sha256(base.CONFIG) != '812b6bfc089b8b7b6c5c32dc3ff8f47249383469b787b54c2adc56ef62186585':
        raise ValueError('Original XS configuration changed')
    if float(cfg.learning_rate) != .0005 or float(cfg.lr_decay) != .99:
        raise ValueError('Unexpected registered learning-rate recipe')
    expected = json.loads((REPO / 'frozen_source_hashes.json').read_text())
    for name, wanted in expected.items():
        if base.sha256(REPO / name) != wanted:
            raise ValueError(f'Frozen reference changed: {name}')
    return cfg, torch.device('cuda:0'), expected


def build_model(cfg, variant, device):
    # Build the old initial model as a byte-level oracle, then reset to the
    # identical seed before constructing each candidate.
    oracle, _, _, oracle_digest = reference.build(cfg, 'joint', SEED, torch.device('cpu'))
    if oracle_digest != INITIAL_REFERENCE_SHA:
        raise ValueError('Accepted seed11 initialization cannot be reproduced')
    reference_state = {name: value.detach().clone() for name, value in oracle.state_dict().items()}
    oracle_rng = rng_state()
    del oracle
    base.SEED = SEED
    base.seed_all(SEED)
    backbone = base.AGFAdapter(base.load_model(cfg))
    if variant == 'baseline':
        model = reference.ControlledXMem(backbone, 'joint')
    elif variant == 'h_query':
        from h_query_model import HQueryXMem
        model = HQueryXMem(backbone)
    elif variant == 'matrix_h':
        from matrix_h_model import MatrixHXMem
        model = MatrixHXMem(backbone)
    else:
        raise ValueError(variant)
    excluded = ('gru.', 'value_gru.') if variant == 'matrix_h' else ()
    shared = {}
    current = model.state_dict()
    for name, wanted in reference_state.items():
        if name.startswith(excluded):
            continue
        if name not in current or not torch.equal(wanted, current[name].cpu()):
            raise ValueError(f'Shared initialization differs: {name}')
        shared[name] = current[name]
    actual_rng = rng_state()
    if not torch.equal(actual_rng['cpu_rng'], oracle_rng['cpu_rng']) or not torch.equal(
            actual_rng['cuda_rng'], oracle_rng['cuda_rng']):
        raise ValueError('New module construction changed the shared torch RNG stream')
    initial_digest = reference.model_digest(model)
    active = dict(model.active_named_parameters())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in active)
    expected_parameters = {'baseline': 3728508, 'h_query': 3732604, 'matrix_h': 3524479}
    count = sum(parameter.numel() for parameter in active.values())
    if count != expected_parameters[variant]:
        raise ValueError(f'Unexpected active parameter count: {variant} {count}')
    info = {'reference_initial_sha256': oracle_digest, 'initial_model_sha256': initial_digest,
            'shared_initial_sha256': tensor_dict_digest(shared), 'shared_state_tensors': len(shared),
            'active_parameters': count, 'initial_cpu_rng_sha256': hashlib.sha256(
                actual_rng['cpu_rng'].numpy().tobytes()).hexdigest()}
    model.to(device=device, dtype=torch.float32)
    active = dict(model.active_named_parameters())
    optimizer = torch.optim.Adam(active.values(), lr=.0005, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
    return model, optimizer, scheduler, info


def source_hashes():
    return {path.name: base.sha256(path) for path in sorted(REPO.glob('*.py'))}


def upstream_provenance():
    values = original._provenance(base.DATA_ROOT)
    for directory in ('utils', 'loss', 'data'):
        for path in sorted((base.UPSTREAM / directory).rglob('*.py')):
            values[str(path)] = base.sha256(path)
    return values


def make_manifest(args, variant, init, frozen, groups):
    parent = None
    if args.resume_from is not None:
        if variant != 'baseline' or base.sha256(args.resume_from) != PARENT_SHA:
            raise ValueError('Only the accepted baseline epoch60 may be extended')
        parent = {'path': str(args.resume_from.resolve()), 'sha256': PARENT_SHA, 'epoch': 60}
    return {'schema': 1, 'variant': variant, 'seed': SEED, 'epochs': EPOCHS, 'scope': 'all',
            'clips': CLIPS_PER_EPOCH, 'groups': GROUPS, 'groups_per_step': 12, 'max_group': 8,
            'group_sha256': original.group_digest(groups), 'optimizer_updates_per_epoch': UPDATES_PER_EPOCH,
            'train_data_sha256': reference.TRAIN_SHA, 'fp32': True, 'tf32': False,
            'optimizer': 'Adam AMSGrad', 'lr': .0005, 'weight_decay': 0., 'decay': .99,
            'current_pose_loss_weight': .25, 'primary_checkpoint_epoch': EPOCHS,
            'save_epochs': list(SAVE_EPOCHS), 'formal_test_selection': False,
            'initialization': init, 'parent': parent, 'frozen_sources_sha256': frozen,
            'sources_sha256': source_hashes(),
            'upstream_provenance': upstream_provenance()}


def restore_own(path, model, optimizer, scheduler, manifest):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['run_config'] != manifest:
        raise ValueError('Current checkpoint registration differs from this run')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved)
    epoch = int(saved['epoch'])
    steps = {int(value['step']) for value in optimizer.state.values() if 'step' in value}
    if steps != {epoch * UPDATES_PER_EPOCH}:
        raise ValueError(f'Checkpoint optimizer step mismatch: {steps}')
    if not math.isclose(optimizer.param_groups[0]['lr'], .0005 * .99**epoch, rel_tol=1e-12):
        raise ValueError('Restored learning rate does not match completed epochs')
    return epoch


def batch_signature(selected, groups, epoch):
    rows = [{'group': int(group_id), 'windows': [int(i) for i in groups[group_id]],
             'flip': random.Random(SEED + 1000003 * epoch + group_id).random() > .5}
            for group_id in selected]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def diagnose(model, cfg, dataset, groups, indices, epoch, device, output):
    """One predeclared training group; no test labels, optimization or RNG drift."""
    saved_rng = rng_state()
    was_training = model.training
    started = time.perf_counter()
    state = None
    rows = []
    group_id = next(i for i, group in enumerate(groups) if len(group) == 8)
    model.eval()
    try:
        x, _ = base.load_bucket(dataset, groups, [group_id], epoch, device)
        frames = torch.from_numpy(indices[np.asarray([groups[group_id]])].transpose(1, 0, 2).copy())
        with torch.no_grad():
            for t in range(x.shape[0]):
                prediction, state = model.forward_clip(x[t], state=state, frame_indices=frames[t])
                hidden = state['hidden']
                if not torch.isfinite(prediction).all() or not torch.isfinite(hidden).all():
                    raise ValueError('Nonfinite state in registered training-side diagnostic')
                row = {'window': t, 'hidden_rms': float(hidden.square().mean().sqrt()),
                       'hidden_absmax': float(hidden.abs().max()), 'bank_entries': int(state['valid'].sum())}
                if hasattr(model, 'h_query_proj'):
                    delta = model.h_query_proj(model.h_query_norm(hidden))
                    row['next_query_delta_rms'] = float(delta.square().mean().sqrt())
                rows.append(row)
            result = {'epoch': epoch, 'source': 'training_only', 'group_id': group_id,
                      'window_ids': [int(i) for i in groups[group_id]], 'rows': rows,
                      'seconds': time.perf_counter() - started}
            if hasattr(model, 'matrix_size') or type(model).__name__ == 'MatrixHXMem':
                sample = hidden[0, [0, 5, 10]].reshape(3, 8, 8)
                result['sampled_state_singular_values'] = torch.linalg.svdvals(sample).cpu().tolist()
                result['spectral_scope'] = 'state_matrix_only_not_transition_or_stability_certificate'
        save_json(output / f'diagnostics_epoch{epoch:03d}.json', result)
    finally:
        model.train(was_training)
        restore_rng(saved_rng)


def parser(description):
    value = argparse.ArgumentParser(description=description)
    value.add_argument('--data-dir', type=Path, required=True)
    value.add_argument('--checkpoint-dir', type=Path, required=True)
    value.add_argument('--output-dir', type=Path, required=True)
    value.add_argument('--seed', type=int, choices=(SEED,), default=SEED)
    return value


def main_train(variant):
    arguments = parser(f'Train registered {variant} to a fixed total150 epochs')
    arguments.add_argument('--resume-from', type=Path)
    arguments.add_argument('--stop-after-epoch', type=int)
    arguments.add_argument('--check-diagnostics', action='store_true')
    args = arguments.parse_args()
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= EPOCHS:
        arguments.error('Invalid operational smoke endpoint')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None and args.resume_from.resolve().parent == args.checkpoint_dir.resolve():
        raise ValueError('Parent and child checkpoint directories must differ')
    output = args.output_dir
    try:
        cfg, device, frozen = configure(args.data_dir)
        loading_started = time.perf_counter()
        dataset, groups, sources, indices = original.make_train_data(cfg)
        included = reference.training_group_ids(groups, sources, 'all')
        if len(dataset) != CLIPS_PER_EPOCH or len(included) != GROUPS or len(sources) != 128:
            raise ValueError('Unexpected original full-data window/group/source counts')
        if original.group_digest(groups) != GROUP_SHA or included != list(range(GROUPS)):
            raise ValueError('Registered source groups differ')
        loading_seconds = time.perf_counter() - loading_started
        model, optimizer, scheduler, init = build_model(cfg, variant, device)
        manifest = make_manifest(args, variant, init, frozen, groups)
        last = args.checkpoint_dir / 'last.pt'
        complete_path = output / 'training_complete.json'
        start_epoch = 0
        restore_info = None
        if last.exists():
            start_epoch = restore_own(last, model, optimizer, scheduler, manifest)
            restore_info = {'kind': 'own_epoch_checkpoint', 'epoch': start_epoch,
                            'checkpoint_sha256': base.sha256(last)}
        elif args.resume_from is not None:
            from check_resume import load_verified_parent
            start_epoch, restore_info = load_verified_parent(
                args.resume_from, model, optimizer, scheduler, source_dir=REPO)
            if start_epoch != 60:
                raise ValueError('Expected accepted epoch60 parent')
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if start_epoch == EPOCHS and complete['checkpoint_sha256'] == base.sha256(last):
                emit(output, 'training_already_complete', epoch=EPOCHS)
                return
            raise ValueError('Completion marker does not match checkpoint')
        if start_epoch == EPOCHS:
            complete = {'epoch': EPOCHS, 'variant': variant, 'seed': SEED,
                        'checkpoint_sha256': base.sha256(last),
                        'model_sha256': reference.model_digest(model),
                        'retained_optimizer_updates': EPOCHS * UPDATES_PER_EPOCH,
                        'timestamp_utc': now()}
            save_json(complete_path, complete)
            emit(output, 'training_completion_recovered', **complete)
            return
        if start_epoch > EPOCHS:
            raise ValueError('Checkpoint exceeds registered training endpoint')
        expected_lr = .0005 * .99**start_epoch
        if not math.isclose(optimizer.param_groups[0]['lr'], expected_lr, rel_tol=1e-12):
            raise ValueError('Incorrect initial/restored learning rate')
        save_json(output / 'run_config.json', manifest)
        emit(output, 'initialized', variant=variant, seed=SEED, total_epochs=EPOCHS,
             resume_epoch=start_epoch, restore=restore_info, initialization=init,
             data_loading_seconds=loading_seconds, train_clips=CLIPS_PER_EPOCH,
             train_groups=GROUPS, updates_per_epoch=UPDATES_PER_EPOCH,
             cuda_device=torch.cuda.get_device_name(0),
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
             cpu_affinity=sorted(os.sched_getaffinity(0)), torch_version=torch.__version__)
        if args.check_diagnostics:
            before = reference.model_digest(model)
            diagnose(model, cfg, dataset, groups, indices, start_epoch, device, output)
            if reference.model_digest(model) != before:
                raise ValueError('Training-side diagnostics changed parameters or buffers')
            emit(output, 'diagnostic_preflight_passed', epoch=start_epoch)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start_epoch + 1, EPOCHS + 1):
            epoch_started = time.perf_counter()
            model.train()
            order = included.copy()
            random.Random(SEED + epoch).shuffle(order)
            total_steps = math.ceil(len(order) / original.GROUPS_PER_STEP)
            if total_steps != UPDATES_PER_EPOCH:
                raise ValueError('Incorrect update count')
            loss_sum, seen_clips = 0., 0
            expected_lr = .0005 * .99**(epoch - 1)
            if not math.isclose(optimizer.param_groups[0]['lr'], expected_lr, rel_tol=1e-12):
                raise ValueError(f'Learning-rate clock differs at epoch{epoch}')
            for number, offset in enumerate(range(0, len(order), original.GROUPS_PER_STEP), 1):
                selected = order[offset:offset + original.GROUPS_PER_STEP]
                value, count = reference.step(model, optimizer, cfg, dataset, groups, indices,
                                               selected, epoch, device, current_pose_weight=.25)
                loss_sum += value
                seen_clips += count
                if number in (1, total_steps) or number % 100 == 0:
                    emit(output, 'train_progress', variant=variant, epoch=epoch,
                         steps=[number, total_steps], clips=seen_clips,
                         retained_optimizer_updates=(epoch - 1) * UPDATES_PER_EPOCH + number,
                         learning_rate=optimizer.param_groups[0]['lr'],
                         mean_loss=loss_sum / seen_clips, seconds=time.perf_counter() - epoch_started,
                         batch_sha256=batch_signature(selected, groups, epoch),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if seen_clips != CLIPS_PER_EPOCH:
                raise ValueError('Epoch did not cover every registered training window exactly once')
            scheduler.step()
            payload = reference.checkpoint_payload(model, optimizer, scheduler, epoch, manifest)
            original.atomic_checkpoint(last, payload)
            if epoch in SAVE_EPOCHS:
                original.atomic_checkpoint(args.checkpoint_dir / f'epoch_{epoch:03d}.pt', payload)
            elapsed = time.perf_counter() - epoch_started
            emit(output, 'epoch_end', variant=variant, epoch=epoch, clips=seen_clips,
                 retained_optimizer_updates=epoch * UPDATES_PER_EPOCH,
                 mean_loss=loss_sum / seen_clips, seconds=elapsed,
                 next_learning_rate=optimizer.param_groups[0]['lr'],
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if epoch in SAVE_EPOCHS:
                diagnose(model, cfg, dataset, groups, indices, epoch, device, output)
            if epoch == EPOCHS:
                complete = {'epoch': EPOCHS, 'variant': variant, 'seed': SEED,
                            'checkpoint_sha256': base.sha256(last),
                            'model_sha256': reference.model_digest(model),
                            'retained_optimizer_updates': EPOCHS * UPDATES_PER_EPOCH,
                            'timestamp_utc': now()}
                save_json(complete_path, complete)
                emit(output, 'training_complete', **complete)
                return
            if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
                smoke = {'variant': variant, 'epoch': epoch, 'primary_total_epochs': EPOCHS,
                         'checkpoint_sha256': base.sha256(last), 'epoch_seconds': elapsed,
                         'data_loading_seconds': loading_seconds,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                         'timestamp_utc': now(), 'status': 'one_epoch_smoke_complete_not_full_training'}
                save_json(output / 'smoke_complete.json', smoke)
                emit(output, 'smoke_complete', **smoke)
                return
    except BaseException as error:
        save_json(output / 'training_failure.json', {'timestamp_utc': now(),
                  'type': type(error).__name__, 'message': str(error), 'variant': variant})
        emit(output, 'training_failed', variant=variant, error_type=type(error).__name__, error=str(error))
        raise


def main_eval(variant):
    args = parser(f'Evaluate registered {variant} epoch150 only').parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir
    try:
        cfg, device, _ = configure(args.data_dir)
        if base.sha256(base.DATA_ROOT / 'data_test_3dhp.npz') != TEST_SHA:
            raise ValueError('Official test archive differs')
        checkpoint = args.checkpoint_dir / 'last.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['epoch'] != EPOCHS or saved['run_config']['variant'] != variant:
            raise ValueError('Only this registered epoch150 endpoint may be evaluated')
        if saved['run_config']['sources_sha256'] != source_hashes():
            raise ValueError('Evaluation sources differ from training')
        if saved['run_config']['upstream_provenance'] != upstream_provenance():
            raise ValueError('Upstream model/data/score sources changed after training started')
        model, _, _, _ = build_model(cfg, variant, device)
        model.load_state_dict(saved['model'], strict=True)
        before = reference.model_digest(model)
        emit(output, 'formal_evaluation_started', variant=variant, epoch=EPOCHS,
             checkpoint_sha256=base.sha256(checkpoint))
        test_data = base.Fusion(cfg, train=False)
        score, per_sequence, seconds = original.evaluate(model, test_data, cfg, device)
        if reference.model_digest(model) != before:
            raise ValueError('Evaluation modified model state')
        result = {'variant': variant, 'seed': SEED, 'epoch': EPOCHS, 'mpjpe_mm': score,
                  'per_sequence_mpjpe_mm': per_sequence, 'test_targets': 2875, 'root_joint': 14,
                  'normal_flip_histories': 'independent', 'checkpoint_sha256': base.sha256(checkpoint),
                  'model_sha256': before, 'test_data_sha256': TEST_SHA,
                  'seconds': seconds, 'timestamp_utc': now(), 'selection': 'fixed_epoch150'}
        save_json(output / 'formal_result.json', result)
        emit(output, 'formal_evaluation_complete', **result)
    except BaseException as error:
        save_json(output / 'evaluation_failure.json', {'timestamp_utc': now(), 'variant': variant,
                  'type': type(error).__name__, 'message': str(error)})
        emit(output, 'formal_evaluation_failed', variant=variant, error=str(error))
        raise
