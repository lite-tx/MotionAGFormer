"""GPU preflight for H-query; partial smoke is not a benchmark run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

REPO = Path(__file__).resolve().parent
os.environ["XMEM_ORIGINAL_DIR"] = str(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import torch

import train_controlled as ref
from h_query_model import HQueryXMem

SEED = 20261111
EXPECTED_INITIAL = "005075a8875e806d9da045a26c3ae41420d6eb3a7447ddd5a47ad58f872b2b65"


def emit(event, **fields):
    print(json.dumps({"event": event, "wall_time": time.time(), **fields}), flush=True)


def shared_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("h_query_"):
            continue
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode("ascii"))
        h.update(repr(tuple(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def build_a(cfg, device):
    ref.base.SEED = SEED
    ref.base.seed_all(SEED)
    backbone = ref.base.AGFAdapter(ref.base.load_model(cfg))
    model = HQueryXMem(backbone)
    initial = shared_digest(model)
    model.to(device=device, dtype=torch.float32)
    active = dict(model.active_named_parameters())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in active)
    optimizer = torch.optim.Adam(active.values(), lr=float(cfg.learning_rate), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.lr_decay))
    return model, optimizer, scheduler, initial


def max_shared_difference(c, a, gradients=False):
    names = dict(c.named_parameters()) if gradients else c.state_dict()
    other = dict(a.named_parameters()) if gradients else a.state_dict()
    differences = []
    exact = True
    for name, value in names.items():
        if gradients:
            value, right = value.grad, other[name].grad
            if value is None or right is None:
                assert value is None and right is None, name
                continue
        else:
            right = other[name]
        exact &= torch.equal(value, right)
        if value.is_floating_point():
            differences.append((value - right).abs().reshape(-1))
        else:
            assert torch.equal(value, right), name
    return {"exact": exact,
            "max_abs": float(torch.cat(differences).max()) if differences else 0.}


def synthetic_bank_checks(c, a, device):
    generator = torch.Generator(device=device).manual_seed(3389)
    rand = lambda *shape: torch.randn(*shape, device=device, generator=generator)
    feature = rand(2, 3, 17, 512)
    state = {
        "key": rand(2, 17, 5, 64).requires_grad_(),
        "value": rand(2, 17, 5, 512).requires_grad_(),
        "shrinkage": (rand(2, 17, 5, 1).square() + 1).requires_grad_(),
        "valid": torch.tensor([[True] * 5, [False] * 5], device=device),
        "hidden": rand(2, 17, 64).requires_grad_(),
    }
    before = c.read_bank(feature, state)
    zero = a.read_bank(feature, state)
    assert torch.equal(before, zero)
    assert torch.count_nonzero(zero[1]) == 0
    with torch.no_grad():
        a.h_query_proj.weight.copy_(torch.eye(64, device=device) * .1)
    changed = a.read_bank(feature, state)
    difference = float((changed[0] - zero[0]).abs().max())
    assert difference > 0
    changed.square().mean().backward()
    hidden_grad = float(state["hidden"].grad[0].norm())
    assert hidden_grad > 0 and torch.count_nonzero(state["hidden"].grad[1]) == 0
    assert torch.isfinite(state["key"].grad).all() and torch.isfinite(state["value"].grad).all()
    empty = {name: value[:, :, :0] if name in ("key", "value", "shrinkage") else value
             for name, value in state.items()}
    empty["valid"] = state["valid"][:, :0]
    assert torch.equal(c.read_bank(feature, empty), a.read_bank(feature, empty))
    with torch.no_grad():
        a.h_query_proj.weight.zero_()
    c.zero_grad(set_to_none=True)
    a.zero_grad(set_to_none=True)
    return {"zero_readout_exact": True, "mixed_empty_row_zero": True,
            "empty_bank_exact": True, "nonzero_residual_readout_max_change": difference,
            "history_hidden_gradient_norm": hidden_grad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output-json", type=Path, default=REPO.parent / "preflight/h_query_preflight.json")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("at least two real updates are required")
    started = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one CUDA GPU must be selected")
    if torch.cuda.get_device_name() != "NVIDIA L40":
        raise RuntimeError("registered smoke requires NVIDIA L40")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    cfg = ref.base.get_cfg()

    c, c_opt, _, c_initial = ref.build(cfg, "joint", SEED, device)
    c_rng_cpu = torch.get_rng_state().clone()
    c_rng_cuda = torch.cuda.get_rng_state().clone()
    a, a_opt, _, a_initial = build_a(cfg, device)
    assert c_initial == a_initial == EXPECTED_INITIAL
    assert torch.equal(c_rng_cpu, torch.get_rng_state())
    assert torch.equal(c_rng_cuda, torch.cuda.get_rng_state())
    initial_difference = max_shared_difference(c, a)
    assert initial_difference["exact"]
    added = sum(p.numel() for name, p in a.named_parameters() if name.startswith("h_query_"))
    assert added == 4096
    emit("initialization_ok", shared_initial=c_initial, added_parameters=added)
    synthetic = synthetic_bank_checks(c, a, device)
    emit("synthetic_bank_ok", **synthetic)

    loading_started = time.perf_counter()
    dataset, groups, sources, indices = ref.original.make_train_data(cfg)
    included = ref.training_group_ids(groups, sources, "all")
    assert len(dataset) == 117024 and len(included) == 14680
    assert ref.original.group_digest([groups[i] for i in included]) == "e24d914b203df27d95a9a463bd58e0a5d03536f503733ddb54c39a907089a0b8"
    loading_seconds = time.perf_counter() - loading_started
    order = included.copy()
    random.Random(SEED + 1).shuffle(order)
    first = order[:ref.original.GROUPS_PER_STEP]
    longest = max(len(groups[i]) for i in first)
    same_length = [i for i in first if len(groups[i]) == longest]
    x, _ = ref.base.load_bucket(dataset, groups, same_length, 1, device)
    frames = torch.from_numpy(indices[np.asarray([groups[i] for i in same_length])].transpose(1, 0, 2).copy())
    c.eval()
    a.eval()
    with torch.no_grad():
        c_pose = c.forward_group(x, frames)
        a_pose = a.forward_group(x, frames)
    forward_difference = float((c_pose - a_pose).abs().max())
    assert torch.equal(c_pose, a_pose)
    del x, frames, c_pose, a_pose
    emit("real_forward_ok", max_abs=forward_difference, data_loading_seconds=loading_seconds)

    c.train()
    a.train()
    rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    c_loss, c_count = ref.step(c, c_opt, cfg, dataset, groups, indices, first, 1, device, .25)
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda)
    a_loss, a_count = ref.step(a, a_opt, cfg, dataset, groups, indices, first, 1, device, .25)
    assert c_count == a_count and c_loss == a_loss
    gradients = max_shared_difference(c, a, gradients=True)
    update = max_shared_difference(c, a)
    assert gradients["exact"] and update["exact"], (gradients, update)
    first_new_gradient = float(a.h_query_proj.weight.grad.norm())
    first_new_weight = float(a.h_query_proj.weight.norm())
    assert first_new_gradient > 0 and first_new_weight > 0
    emit("zero_init_optimizer_ok", shared_gradients=gradients, shared_update=update,
         new_gradient=first_new_gradient, new_weight=first_new_weight)

    del c, c_opt
    torch.cuda.empty_cache()
    times, losses = [], [a_loss]
    for number in range(1, args.steps):
        chosen = order[number * ref.original.GROUPS_PER_STEP:(number + 1) * ref.original.GROUPS_PER_STEP]
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        loss, count = ref.step(a, a_opt, cfg, dataset, groups, indices, chosen, 1, device, .25)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - step_started)
        losses.append(loss)
        if number in (1, args.steps - 1):
            emit("real_smoke_progress", updates=number + 1, loss=loss, windows=count,
                 step_seconds=times[-1])

    result = {
        "status": "passed", "scope": "partial_real_training_smoke_not_formal_result",
        "seed": SEED, "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "shared_initial_sha256": c_initial,
        "additional_parameters": added, "shared_initial_exact": initial_difference,
        "cpu_and_cuda_rng_preserved": True, "synthetic_bank": synthetic,
        "real_forward_max_abs": forward_difference, "shared_gradient_first_step": gradients,
        "shared_parameter_and_buffer_first_step": update, "first_query_gradient_norm": first_new_gradient,
        "first_query_weight_norm": first_new_weight, "real_updates": args.steps,
        "all_losses": losses, "data_loading_seconds": loading_seconds,
        "median_step_seconds": statistics.median(times),
        "linear_epoch_seconds": statistics.median(times) * 1224,
        "linear_150_epochs_hours": statistics.median(times) * 1224 * 150 / 3600,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_saved": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    emit("preflight_complete", output=str(args.output_json), **result)


if __name__ == "__main__":
    main()
