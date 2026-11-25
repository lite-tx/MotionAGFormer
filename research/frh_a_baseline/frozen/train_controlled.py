"""Train matched original, memory-bank-only, and GRU-only study arms."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import train_incremental as original

base = original.base
SEEDS = (20261109, 20261110, 20261111)
TRAIN_SHA = "1082d49aa56d458bc33b7bd4ad4966d23e4c530de7110fb8da36783062b97f6d"
MANIFEST_SHA = "965f72ac9f2334c86c2ea87b5318d18b6238c91e6bf2f0fbe1cef916c14ca4fd"
VALIDATION_SUBJECTS = ("S7", "S8")
ARMS = ("joint", "r", "h")
BANK_PREFIXES = ("key_proj.", "selection_proj.", "shrinkage_proj.")
HIDDEN_PREFIXES = ("gru.", "value_gru.")


class ControlledXMem(original.IncrementalXMem):
    def __init__(self, backbone, arm):
        super().__init__(backbone, query_chunk=51)
        self.arm = arm
        if arm not in ARMS:
            raise ValueError(f"unknown model arm {arm}")

    def active_named_parameters(self):
        omitted = {"joint": (), "r": HIDDEN_PREFIXES, "h": BANK_PREFIXES}[self.arm]
        for name, tensor in super().active_named_parameters():
            if not name.startswith(omitted):
                yield name, tensor

    def read_bank(self, feature, state):
        if self.arm == "h":
            return torch.zeros_like(feature)
        return super().read_bank(feature, state)

    def forward_clip(self, inputs, state=None, frame_indices=None):
        if state is not None and self.arm == "r":
            state = {**state, "hidden": torch.zeros_like(state["hidden"])}
        prediction, next_state = super().forward_clip(inputs, state=state, frame_indices=frame_indices)
        if self.arm == "r":
            next_state["hidden"] = torch.zeros_like(next_state["hidden"])
        if self.arm == "h":
            # Preserve whether the fusion path has past observations. The GRUs
            # still evolve normally; no stored K/V is available to prediction.
            next_state["valid"] = next_state["valid"].any(dim=1, keepdim=True)
            for field in ("key", "shrinkage", "value"):
                next_state[field] = next_state[field][:, :, :1].detach()
        return prediction, next_state


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build(cfg, arm, seed, device, reference=False):
    base.SEED = seed
    base.seed_all(seed)
    backbone = base.AGFAdapter(base.load_model(cfg))
    model = (original.IncrementalXMem(backbone, query_chunk=51) if reference else ControlledXMem(backbone, arm))
    if seed == original.SEED and original.digest_backbone(model) != original.ORIGINAL_BACKBONE_SHA256:
        raise ValueError("first seed does not reproduce archived initialization")
    digest = model_digest(model)
    model.to(device=device, dtype=torch.float32)
    active = dict(model.active_named_parameters())
    if not reference:
        for name, param in model.named_parameters():
            param.requires_grad_(name in active)
    optimizer = torch.optim.Adam(active.values(), lr=float(cfg.learning_rate), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.lr_decay))
    return model, optimizer, scheduler, digest


def training_group_ids(groups, sources, scope):
    edges = np.array([r["offset"] + r["clips"] for r in sources])
    subject = [r["seq"].split()[0] for r in sources]
    if set(subject) != {f"S{n}" for n in range(1, 9)}:
        raise ValueError("unexpected 3DHP training subjects")
    result = []
    for group_id, group in enumerate(groups):
        source = int(np.searchsorted(edges, group[0], side="right"))
        if group[-1] >= edges[source]:
            raise ValueError("group crosses a camera trajectory")
        if scope == "all" or subject[source] not in VALIDATION_SUBJECTS:
            result.append(group_id)
    return result


def group_predictions(model, clips, frames, supervise_current_pose):
    if not supervise_current_pose:
        return model.forward_group(clips, frames), None
    features = []
    hook = model.backbone.model.register_forward_hook(lambda _model, _args, output: features.append(output))
    try:
        native = model.forward_group(clips, frames)
    finally:
        hook.remove()
    if len(features) != clips.shape[0]:
        raise ValueError("expected a single backbone representation per window")
    # The group starts without history, so its first native pose already uses the current window.
    current = torch.stack([native[0], *(model.backbone.head(feature) for feature in features[1:])])
    return native, current


def step(model, optimizer, cfg, dataset, groups, source_indices, selected, epoch, device, current_pose_weight=0.):
    count = sum(len(groups[i]) for i in selected)
    if count <= 0:
        raise ValueError("empty training step")
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for size in range(1, original.MAX_GROUP + 1):
        ids = [i for i in selected if len(groups[i]) == size]
        if not ids:
            continue
        x, y = base.load_bucket(dataset, groups, ids, epoch, device)
        ids_by_clip = np.asarray([groups[i] for i in ids], dtype=np.int64)
        frames = torch.from_numpy(source_indices[ids_by_clip].transpose(1, 0, 2).copy())
        pose, current = group_predictions(model, x, frames, current_pose_weight > 0. and size > 1)
        if pose.shape != y.shape:
            raise ValueError("training pose and labels differ in shape")
        loss = sum(base.clip_loss(pose[t], y[t], cfg) * len(ids) for t in range(size))
        if current is not None:
            if current.shape != y.shape:
                raise ValueError("current-window pose and labels differ in shape")
            auxiliary = sum(base.clip_loss(current[t], y[t], cfg) * len(ids) for t in range(size))
            loss = (1. - current_pose_weight) * loss + current_pose_weight * auxiliary
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        total += loss.detach().to(torch.float64)
        (loss / count).backward()
    active = list(model.active_named_parameters())
    gradients = [p.grad for _, p in active if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise ValueError("missing or nonfinite gradients")
    if any(len(groups[i]) >= 2 for i in selected):
        required = HIDDEN_PREFIXES + BANK_PREFIXES + ("value_proj.",)
        missing = [name for name, p in active if name.startswith(required) and p.grad is None]
        if missing:
            raise ValueError(f"active memory parameters did not receive gradients: {missing}")
    optimizer.step()
    return float(total.item()), count


def load_validation(manifest, indices):
    if base.sha256(manifest) != MANIFEST_SHA:
        raise ValueError("unexpected validation manifest hash")
    rows = [r for r in json.loads(manifest.read_bytes())["pairs"] if r["split"] == "holdout"]
    if len(rows) != 288 or {r["subject"] for r in rows} != set(VALIDATION_SUBJECTS):
        raise ValueError("validation targets do not match the original human split")
    for row in rows:
        clip = row["source_offset"] + row["window_index"]
        if clip != row["clip_id"] or indices[clip, 13] != row["source_centre"]:
            raise ValueError(f"validation target outside source window: {row['pair_id']}")
    return rows


@torch.inference_mode()
def validate(model, dataset, indices, rows, device):
    model.eval()
    by_view = defaultdict(dict)
    for row in rows:
        key = row["subject"], row["sequence"], row["camera"], row["source_offset"]
        by_view[key][row["clip_id"]] = row
    score = {}
    left, right = dataset.left_joints, dataset.right_joints
    for (_, _, _, offset), targets in sorted(by_view.items()):
        state = None
        for clip in range(offset, max(targets) + 1):
            x = np.asarray(dataset.poses_2d[clip], dtype=np.float32)
            flipped = base.flip_data(x, left, right)
            x = torch.from_numpy(np.stack((x, flipped))).to(device)
            ids = torch.from_numpy(np.stack((indices[clip], indices[clip])))
            pose, state = model.forward_clip(x, state=state, frame_indices=ids)
            row = targets.get(clip)
            if row is None:
                continue
            pred = 0.5 * (pose[0, 13] + base.flip_data(pose[1:2, 13], left, right)[0])
            y = torch.as_tensor(dataset.normalized_poses3d[clip][13], device=device, dtype=torch.float32)
            residual = pred - pred[14:15] - (y - y[14:15])
            value = float(residual.norm(dim=-1).mean()) * 1024
            if not math.isfinite(value):
                raise ValueError(f"nonfinite validation target {row['pair_id']}")
            score[row["pair_id"]] = value
    if set(score) != {r["pair_id"] for r in rows}:
        raise ValueError("validation replay did not cover the target set")
    times = defaultdict(list)
    for row in rows:
        times[row["subject"], row["sequence"], row["band"], row["source_centre"]].append(score[row["pair_id"]])
    if any(len(values) != 2 for values in times.values()):
        raise ValueError("validation targets require synchronized two-camera observations")
    time_score = {coords: float(np.mean(values)) for coords, values in times.items()}
    means = defaultdict(list)
    for (person, video, band, _), value in time_score.items():
        for key in ("all", person, str(band), f"{person} {video}", f"{person} {video} {band}"):
            means[key].append(value)
    return {key: float(np.mean(values)) for key, values in means.items()}, score


@torch.inference_mode()
def verify_mask(model, dataset, groups, indices, group_id, device):
    model.eval()
    x, _ = base.load_bucket(dataset, groups, [group_id], 1, device)
    clip_a, clip_b = groups[group_id][:2]
    _, state = model.forward_clip(x[0], frame_indices=torch.from_numpy(indices[clip_a:clip_a + 1]))
    same_input = {"state": state, "frame_indices": torch.from_numpy(indices[clip_b:clip_b + 1])}
    expected, _ = model.forward_clip(x[1], **same_input)
    if model.arm == "r":
        altered = {**state, "hidden": torch.full_like(state["hidden"], 3.)}
    else:
        altered = {**state, "key": torch.full_like(state["key"], 3.), "value": torch.full_like(state["value"], -3.)}
    actual, _ = model.forward_clip(x[1], **{**same_input, "state": altered})
    if not torch.equal(actual, expected):
        raise ValueError(f"removed branch still influences output for {model.arm}")


def preflight(cfg, dataset, groups, source_indices, order, validation, device, seed, current_pose_weight=0.):
    def test(arm, reference=False, weight=0.):
        model, opt, _, initial = build(cfg, arm, seed, device, reference)
        selected = order[:original.GROUPS_PER_STEP]
        train_batch = original.train_batch if reference else step
        extra = {} if reference else {"current_pose_weight": weight}
        total, count = train_batch(model, opt, cfg, dataset, groups, source_indices, selected, 1, device, **extra)
        metrics = {
            "initial": initial, "updated": model_digest(model), "clip_count": count,
            "sum_loss": total, "num_parameters_updated": sum(p.grad is not None for p in model.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if not reference and arm != "joint":
            verify_mask(model, dataset, groups, source_indices, selected[0], device)
            metrics["removed_branch_invariant"] = True
        if not reference and arm == "joint":
            summary, raw = validate(model, dataset, source_indices, validation, device)
            metrics["validation_examples"] = len(raw)
            metrics["validation_score_mm"] = summary["all"]
        return metrics

    baseline = test("joint", reference=True)
    metrics = {arm: test(arm) for arm in ARMS}
    for key in ("initial", "updated", "clip_count", "sum_loss", "num_parameters_updated"):
        if baseline[key] != metrics["joint"][key]:
            raise ValueError(f"native and controlled joint training step differ on {key}")
    if {x["initial"] for x in metrics.values()} != {baseline["initial"]} or (
        len({m["updated"] for m in metrics.values()}) != len(ARMS)
    ):
        raise ValueError("branch ablations do not start from the same state or do not alter training")
    result = {"native_step_exact": True, "reference": baseline, "arms": metrics}
    if current_pose_weight:
        modified = {arm: test(arm, weight=current_pose_weight) for arm in ARMS}
        for arm in ARMS:
            for key in ("initial", "clip_count", "num_parameters_updated"):
                if modified[arm][key] != metrics[arm][key]:
                    raise ValueError(f"auxiliary pose changes {key} for {arm}")
            if modified[arm]["updated"] == metrics[arm]["updated"]:
                raise ValueError(f"the auxiliary loss has no optimizer effect for {arm}")
        result["current_pose_loss_weight"] = current_pose_weight
        result["current_pose_auxiliary_arms"] = modified
    return result


def checkpoint_payload(model, opt, scheduler, epoch, manifest):
    return {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "run_config": manifest, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
    }


def resume_if_present(path, model, opt, scheduler, manifest):
    if not path.exists():
        return 0
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["run_config"] != manifest:
        raise ValueError("saved training settings differ from requested settings")
    model.load_state_dict(saved["model"], strict=True)
    opt.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    return int(saved["epoch"])


def save_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="joint")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--scope", choices=("validation", "all"), default="validation")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--validation-epochs", type=int, nargs="*", default=())
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--current-pose-loss-weight", type=float, default=0.)
    args = parser.parse_args()
    if args.epochs < 1 or not set(args.validation_epochs).issubset(range(1, args.epochs + 1)) or (
        args.scope != "validation" and args.validation_epochs
    ) or not (
        0. <= args.current_pose_loss_weight < 1.
    ):
        parser.error("invalid validation schedule or training scope")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA L40":
        raise RuntimeError("select exactly one idle L40")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_path = base.DATA_ROOT / "data_train_3dhp.npz"
    if base.sha256(train_path) != TRAIN_SHA:
        raise ValueError("unexpected training data archive")
    device = torch.device("cuda:0")
    cfg = base.get_cfg()
    dataset, groups, sources, indices = original.make_train_data(cfg)
    included = training_group_ids(groups, sources, args.scope)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        torch.cuda.reset_peak_memory_stats()
        rows = load_validation(args.validation_manifest, indices)
        selected = min((r["source_centre"], r["subject"], r["sequence"], r["band"]) for r in rows)
        examples = [r for r in rows if (r["source_centre"], r["subject"], r["sequence"], r["band"]) == selected]
        result = preflight(cfg, dataset, groups, indices, included, examples, device, args.seed, args.current_pose_loss_weight)
        result["train_groups"] = len(included)
        result["train_clips"] = sum(len(groups[group]) for group in included)
        save_json(output / "preflight.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    validation = load_validation(args.validation_manifest, indices) if args.validation_epochs else []
    model, opt, scheduler, initial = build(cfg, args.arm, args.seed, device)
    files = [Path(__file__), Path(original.__file__), Path(original.__file__).with_name("incremental_model.py"),
             Path(original.__file__).with_name("data_index.py"), Path(base.__file__), base.CONFIG]
    manifests = {
        "arm": args.arm, "seed": args.seed, "epochs": args.epochs, "scope": args.scope,
        "validation_epochs": list(args.validation_epochs), "validation_manifest_sha256": MANIFEST_SHA if validation else None,
        "groups_per_step": original.GROUPS_PER_STEP, "groups": len(included),
        "clips": sum(len(groups[id_]) for id_ in included), "group_sha256": original.group_digest([groups[i] for i in included]),
        "initial_model_sha256": initial, "active_parameters": sum(p.numel() for _, p in model.active_named_parameters()),
        "train_data_sha256": TRAIN_SHA, "fp32": True, "optimizer": "Adam AMSGrad", "lr": float(cfg.learning_rate),
        "decay": float(cfg.lr_decay), "sources_sha256": {x.name: base.sha256(x) for x in files},
    }
    if args.current_pose_loss_weight:
        manifests["current_pose_loss_weight"] = args.current_pose_loss_weight
    last = output / "last.pt"
    complete = output / "complete.json"
    start = resume_if_present(last, model, opt, scheduler, manifests)
    if complete.exists():
        if start == args.epochs and json.loads(complete.read_bytes())["checkpoint_sha256"] == base.sha256(last):
            return
        raise ValueError("training completion marker differs from the checkpoint")
    save_json(output / "run_config.json", manifests)
    original.emit(output, {"event": "initialized", "resume_epoch": start, **manifests})
    for epoch in range(start + 1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = included.copy()
        random.Random(args.seed + epoch).shuffle(order)
        total_steps = math.ceil(len(order) / original.GROUPS_PER_STEP)
        total_loss, clips = 0., 0
        for number, idx in enumerate(range(0, len(order), original.GROUPS_PER_STEP), 1):
            selected = order[idx:idx + original.GROUPS_PER_STEP]
            value, count = step(model, opt, cfg, dataset, groups, indices, selected, epoch, device, args.current_pose_loss_weight)
            total_loss += value
            clips += count
            if number in (1, total_steps) or number % 100 == 0:
                original.emit(output, {"event": "train_progress", "epoch": epoch, "steps": [number, total_steps],
                                       "clips": clips, "mean_loss": total_loss / clips,
                                       "seconds": time.perf_counter() - epoch_start})
        if clips != manifests["clips"]:
            raise ValueError("an epoch did not cover all applicable training groups")
        if epoch in args.validation_epochs:
            summary, raw = validate(model, dataset, indices, validation, device)
            save_json(output / f"validation_{epoch:03d}.json", {"epoch": epoch, "summary_mm": summary, "window_mm": raw})
            original.emit(output, {"event": "validation", "epoch": epoch, "summary_mm": summary})
        scheduler.step()
        original.atomic_checkpoint(last, checkpoint_payload(model, opt, scheduler, epoch, manifests))
        original.emit(output, {"event": "epoch_end", "epoch": epoch, "mean_loss": total_loss / clips,
                               "seconds": time.perf_counter() - epoch_start})
    completed = {"epoch": args.epochs, "arm": args.arm, "seed": args.seed, "scope": args.scope,
                 "checkpoint_sha256": base.sha256(last), "model_sha256": model_digest(model)}
    save_json(complete, completed)
    original.emit(output, {"event": "complete", **completed})


if __name__ == "__main__":
    main()
