"""Evaluation-only fixed-skeleton comparator using the same observed inputs."""
import torch


def project_to_fixed(x, source, target):
    """Keep observed named joints; reconstruct missing coordinates from nearest
    available reference positions with inverse-distance interpolation. No 3D GT
    or hidden 2D coordinates are used. Added synthetic points can contribute.
    """
    index = {name:i for i,name in enumerate(source.names)}
    dist = torch.cdist(x.new_tensor(target.rest).float(), x.new_tensor(source.rest).float())
    rows = []
    for i,name in enumerate(target.names):
        if name in index:
            rows.append(x[:,:,index[name]])
        else:
            d, ids = dist[i].topk(min(3, len(source.names)), largest=False)
            weights = (1. / d.clamp_min(1e-4))
            weights = (weights / weights.sum()).to(device=x.device,dtype=x.dtype)
            neighbors = x.index_select(2, ids.to(x.device))
            xy = (neighbors[..., :2] * weights.view(1,1,-1,1)).sum(2)
            if x.shape[-1] == 3:
                conf = neighbors[..., 2:].amin(2)
                rows.append(torch.cat((xy,conf), -1))
            else:
                rows.append(xy)
    return torch.stack(rows, dim=2)
