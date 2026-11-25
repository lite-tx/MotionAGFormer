"""Preflight for the real H36M MotionAGFormer; never saves training checkpoints.

Default: full Base, two windows, synthetic data, inference only. Explicitly
request training steps or real data on a machine with sufficient resources.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import default_collate

from data.reader.h36m_episode import H36MEpisodeDataset
from loss.episode_pose3d import EpisodePoseLoss
from utils.episode_checkpoint import load_pretrained
from utils.episode_trainer import build_episode_model, build_optimizer, forward_episode, train_episode_epoch


def read_config(path, tiny=False):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['memory'] = dict(cfg.get('memory', {}), enabled=True)
    if tiny:
        # Same actual AGFormer operators, reduced size for a CPU smoke test.
        cfg.update(n_layers=2, dim_feat=32, dim_rep=64, num_heads=4,
                   n_frames=18, windows_per_episode=2, episode_stride=6, drop_path=0.)
        cfg['memory'] = dict(enabled=True, chunk_size=3, working_capacity=3,
                             write_stride=1, proxy_capacity=4, sensory_heads=2,
                             sensory_key_dim=8, sensory_value_dim=8, explicit_key_dim=16,
                             explicit_value_dim=24, proxy_dim=32, pose_query_dim=32,
                             pose_query_heads=4, fusion_hidden_dim=32, decoder_hidden_dim=32,
                             proxy_min_exposure=1, proxy_match_threshold=0.)
    return cfg


def make_batch(cfg, batch_size, data=None):
    w, t = cfg['windows_per_episode'], cfg['n_frames']
    if data is not None:
        ds = H36MEpisodeDataset(data, split='train', window_size=t,
                                windows_per_episode=w, episode_stride=cfg['episode_stride'], flip=False)
        if len(ds) < batch_size:
            raise ValueError('Not enough real episodes for the requested batch.')
        indices = sorted(range(len(ds)), key=lambda i: ds.episodes[i].stop-ds.episodes[i].start,
                         reverse=True)[:batch_size]
        return default_collate([ds[i] for i in indices])
    x = torch.randn(batch_size, w, t, 17, 3) * .2
    y = torch.cat((x[..., :2], x[..., :1] * .5), dim=-1)
    ids = torch.arange(w*t).reshape(1,w,t).expand(batch_size,-1,-1)
    return dict(x=x, y=y, frame_ids=ids, valid_mask=torch.ones_like(ids, dtype=torch.bool),
                stream_id=torch.arange(batch_size))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(fn, device):
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    info = {'seconds': time.perf_counter()-start}
    if device.type == 'cuda':
        info['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
        info['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
    return result, info


def verify(cfg, device, *, batch_size=1, precision='fp32', steps=0, data=None, pretrained=None, seed=111):
    device = torch.device(device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Select an available CPU or CUDA device.')
    if precision not in ('fp32','bf16') or (precision=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported()):
        raise ValueError('Unsupported precision on this device.')
    if batch_size <= 0 or steps < 0:
        raise ValueError('batch_size must be positive and steps nonnegative.')
    torch.manual_seed(seed)
    base_cfg = dict(cfg, memory=dict(cfg['memory'], enabled=False))
    baseline = build_episode_model(SimpleNamespace(**base_cfg))
    torch.manual_seed(seed)
    model = build_episode_model(SimpleNamespace(**cfg))
    if pretrained is not None:
        load_pretrained(pretrained, baseline)
        load_pretrained(pretrained, model)
    baseline_state = baseline.state_dict()
    augmented_state = model.state_dict()
    if set(baseline_state) != {k for k in augmented_state if not k.startswith('memory_system.')}:
        raise AssertionError('Original state_dict keys changed.')
    for k,v in baseline_state.items():
        torch.testing.assert_close(v, augmented_state[k], rtol=0, atol=0)
    del baseline_state, augmented_state
    params = lambda m: sum(p.numel() for p in m.parameters())
    result = {'device': str(device), 'precision': precision, 'data': 'synthetic' if data is None else 'real_train_pkl',
              'pretrained': pretrained is not None, 'layers': cfg['n_layers'], 'window_size': cfg['n_frames'],
              'windows': cfg['windows_per_episode'], 'batch_size': batch_size,
              'baseline_params': params(baseline), 'memory_model_params': params(model),
              'memory_params': {n:params(m) for n,m in model.memory_system.named_children()}}
    batch = make_batch(cfg,batch_size,data)
    batch = {k:batch[k] for k in ('x','y','valid_mask','frame_ids','stream_id')}
    result['valid_frames'] = int(batch['valid_mask'].sum())
    inference_batch = {k:v.to(device) for k,v in batch.items()}

    def infer(net):
        net.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
            return forward_episode(net,inference_batch).float().cpu()

    baseline.to(device)
    expected,result['baseline_inference'] = measure(lambda:infer(baseline),device)
    del baseline
    if device.type=='cuda':
        torch.cuda.empty_cache()
    model.to(device)
    predicted,result['memory_inference'] = measure(lambda:infer(model),device)
    if predicted.shape != batch['y'].shape or not torch.isfinite(predicted).all():
        raise AssertionError('Nonfinite or incorrectly shaped prediction.')
    c = model.memory_system.config.chunk_size
    torch.testing.assert_close(predicted[:,0,:c], expected[:,0,:c],
                               rtol=2e-2 if precision=='bf16' else 1e-4,
                               atol=3e-3 if precision=='bf16' else 1e-5)
    result['first_chunk_max_abs_diff'] = float((predicted[:,0,:c]-expected[:,0,:c]).abs().max())
    result['prediction_shape'] = list(predicted.shape)
    result['losses'] = []
    if steps:
        optimizer,_ = build_optimizer(model,cfg['learning_rate'],cfg['weight_decay'],
                                      memory_lr=cfg.get('memory_lr'),lr_decay=cfg['lr_decay'])
        criterion = EpisodePoseLoss(cfg['lambda_scale'],cfg['lambda_3d_velocity'],root_rel=cfg['root_rel'])
        watched = ('joints_embed.weight','head.weight','memory_system.sensory.k_proj.weight',
                   'memory_system.working.key_encoder.weight','memory_system.long_term.input_proj.weight')
        before = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in watched}
        del inference_batch
        for _ in range(steps):
            stats,timing = measure(lambda:train_episode_epoch(model,[batch],optimizer,criterion,device,
                                    precision=precision,grad_clip=cfg.get('grad_clip')),device)
            if stats.optimizer_steps != 1:
                raise AssertionError('Optimizer step did not complete.')
            result['losses'].append(dict(stats._asdict(),**timing))
        result['parameter_updates_l1'] = {n:float((p.detach().cpu()-before[n]).abs().sum())
                                          for n,p in model.named_parameters() if n in before}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/h36m/MotionAGFormer-memory-base.yaml')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision',choices=('fp32','bf16'),default='fp32')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--steps',type=int,default=0,help='Repeat one episode batch; smoke test, not an accuracy experiment.')
    parser.add_argument('--tiny',action='store_true',help='Use smaller real AGFormer for CPU training checks.')
    parser.add_argument('--data',type=Path,help='Optional trusted original H36M PKL; otherwise synthetic input.')
    parser.add_argument('--pretrained',type=Path,help='Optional original checkpoint; strict original tensor checks.')
    parser.add_argument('--seed',type=int,default=111)
    args = parser.parse_args()
    if args.tiny and args.pretrained:
        parser.error('--tiny cannot use a full-size pretrained checkpoint.')
    cfg = read_config(args.config,args.tiny)
    report = verify(cfg,args.device,batch_size=args.batch_size,precision=args.precision,steps=args.steps,
                    data=args.data,pretrained=args.pretrained,seed=args.seed)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
