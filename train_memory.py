"""H36M episode training/evaluation entry point, separate from train.py."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/h36m/MotionAGFormer-memory-base.yaml')
    p.add_argument('--output-dir', default='checkpoint/memory-v1')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--pretrained', help='Initialize from original MotionAGFormer weights')
    mode.add_argument('--resume', help='Resume an episode checkpoint at the next epoch')
    mode.add_argument('--checkpoint', help='Exact model checkpoint for --eval-only')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--reset-memory', action='store_true', help='Eval-only diagnostic: reset each window')
    args = p.parse_args()
    if args.eval_only != bool(args.checkpoint) or (args.reset_memory and not args.eval_only):
        p.error('--eval-only requires --checkpoint; --checkpoint and --reset-memory require --eval-only')
    return args


def main():
    opts = parse_args()
    from data.reader.h36m_episode import H36MEpisodeDataset, build_episode_loader
    from loss.episode_pose3d import EpisodePoseLoss
    from utils.episode_checkpoint import load_pretrained, load_weights, resume_checkpoint, save_checkpoint
    from utils.episode_evaluation import evaluate_h36m
    from utils.episode_trainer import build_episode_model, build_optimizer, train_episode_epoch
    from utils.tools import get_config, set_random_seed

    args = get_config(opts.config)
    args.seed = opts.seed
    device = torch.device(opts.device)
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(opts.seed)
    path = Path(args.data_root) / args.dt_file
    model = build_episode_model(args).to(device)
    print('Parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    test = H36MEpisodeDataset(path, split='test', window_size=args.n_frames,
                             windows_per_episode=args.windows_per_episode,
                             flip=False, include_eval_metadata=True)
    test_loader = build_episode_loader(test, 1, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    def evaluate():
        return evaluate_h36m(model, test_loader, device, flip=args.flip,
                             precision=args.precision, carry_memory=not opts.reset_memory)
    if opts.eval_only:
        load_weights(opts.checkpoint, model)
        print(json.dumps(evaluate(), indent=2, allow_nan=False), flush=True)
        return

    train = H36MEpisodeDataset(path, split='train', window_size=args.n_frames,
                              windows_per_episode=args.windows_per_episode,
                              episode_stride=args.episode_stride, flip=args.flip)
    train_loader = build_episode_loader(train, args.batch_size, num_workers=args.num_workers,
                                        pin_memory=device.type=='cuda', generator=generator)
    criterion = EpisodePoseLoss(args.lambda_scale, args.lambda_3d_velocity, root_rel=args.root_rel)
    optimizer, scheduler = build_optimizer(model, args.learning_rate, args.weight_decay,
                                           memory_lr=args.memory_lr, lr_decay=args.lr_decay)
    start, best = 0, float('inf')
    if opts.pretrained:
        loaded, new = load_pretrained(opts.pretrained, model)
        print(f'Loaded {loaded} original tensors; {len(new)} memory tensors newly initialized.', flush=True)
    elif opts.resume:
        start, best = resume_checkpoint(opts.resume, model, optimizer, scheduler,
                                         config=args, generator=generator, device=device)
    if args.epochs <= start:
        raise ValueError('Configured epochs must exceed resume epoch.')
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs} | LR {[g["lr"] for g in optimizer.param_groups]}', flush=True)
        stats = train_episode_epoch(model, train_loader, optimizer, criterion, device,
                                    accumulation_steps=args.accumulation_steps, precision=args.precision,
                                    grad_clip=args.grad_clip)
        if not stats.optimizer_steps:
            raise RuntimeError('No optimizer steps completed.')
        metrics = evaluate()
        improved = metrics['mpjpe'] < best
        best = min(best, metrics['mpjpe'])
        scheduler.step()  # save the next epoch's LR, not the just-completed one
        kwargs = dict(next_epoch=epoch+1, best_mpjpe=best, config=args,
                      generator=generator, device=device)
        if improved:
            save_checkpoint(out/'best_epoch.pth.tr', model, optimizer, scheduler, **kwargs)
        save_checkpoint(out/'latest_epoch.pth.tr', model, optimizer, scheduler, **kwargs)
        report = {'epoch':epoch+1, 'train':stats._asdict(), 'evaluation':metrics,
                  'best_mpjpe':best, 'next_lr':[g['lr'] for g in optimizer.param_groups]}
        line = json.dumps(report, allow_nan=False)
        print(line, flush=True)
        with (out/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')


if __name__ == '__main__':
    main()
