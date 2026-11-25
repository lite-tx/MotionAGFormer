"""Evaluate a trained SkeletonAGFormer on actual native skeleton arrays.

NPZ x: [N,T,J,2 or 3], y: [N,T,K,3] in training-compatible normalized
coordinates; optional mm_per_unit: scalar or [N,T] valid physical conversion.
JSON skeleton files: {"names":[...],"parents":[...],"rest":[[x,y,z],...]}.
This is a generic diagnostic, NOT the official H36M/3DHP evaluation protocol.
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.skeleton import SkeletonSpec
from utils.learning import load_model
from utils.tools import get_config


def read_spec(path):
    with open(path,encoding="utf-8") as f:
        d = json.load(f)
    return SkeletonSpec(tuple(d["names"]),tuple(d["parents"]),tuple(tuple(r) for r in d["rest"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--source-skeleton",required=True)
    p.add_argument("--target-skeleton")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    config = get_config(args.config)
    if config.model_name != "SkeletonAGFormer":
        raise ValueError("expected SkeletonAGFormer configuration")
    model = load_model(config)
    ckpt = torch.load(args.checkpoint,map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict({k.removeprefix("module.") if hasattr(k,"removeprefix") else (k[7:] if k.startswith("module.") else k):v for k,v in state.items()},strict=True)
    model.to(args.device).eval()
    source = read_spec(args.source_skeleton)
    target = read_spec(args.target_skeleton) if args.target_skeleton else model.canonical
    with np.load(args.data,allow_pickle=False) as d:
        x = torch.as_tensor(d["x"],dtype=torch.float32)
        y = torch.as_tensor(d["y"],dtype=torch.float32)
        scale = torch.as_tensor(d["mm_per_unit"],dtype=torch.float32) if "mm_per_unit" in d else None
    if x.ndim != 4 or y.ndim != 4 or x.shape[:2] != y.shape[:2] or x.shape[2] != len(source.names) or y.shape[2:] != (len(target.names),3):
        raise ValueError("array shapes disagree with source/target metadata")
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("nonfinite coordinates; provide valid measurements or explicit confidences")
    if scale is not None:
        if scale.ndim == 0:
            scale = scale.expand(x.shape[:2])
        if scale.shape != x.shape[:2] or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError("mm_per_unit must be positive scalar or [N,T]")
    dataset = TensorDataset(x,y,scale if scale is not None else torch.ones(x.shape[:2]))
    total,n = 0.,0
    with torch.no_grad():
        for xb,yb,s in DataLoader(dataset,batch_size=args.batch_size):
            xb,yb,s = xb.to(args.device),yb.to(args.device),s.to(args.device)
            pred = model(xb,skeleton=source,target_skeleton=target)
            pred = pred - pred[:,:,target.root:target.root+1]
            yb = yb - yb[:,:,target.root:target.root+1]
            err = (pred-yb).norm(dim=-1)*s[...,None]
            total += err.sum().item()
            n += err.numel()
    if not n:
        raise ValueError("empty evaluation set")
    print(json.dumps({"source_joints":len(source.names),"target_joints":len(target.names),
                      "mpjpe":total/n,"unit":"mm" if scale is not None else "normalized",
                      "protocol":"native NPZ root-relative diagnostic"},indent=2))


if __name__ == "__main__":
    main()
