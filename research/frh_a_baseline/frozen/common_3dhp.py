"""Original MotionAGFormer 3DHP data and a source-local group index."""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

UPSTREAM = Path('/home/xzy/3DHPE/memory_study_preserved_20261105/stage1/upstream_ag')
DATA_ROOT = Path('/home/xzy/3DHPE/data/motion3d')
CONFIG = UPSTREAM / 'configs/mpi/MotionAGFormer-xsmall.yaml'
ROOT_RUN = Path('/data/xzy/3dhpe_experiments/motionagformer_xs_3dhp_xmem_20261109')
SEED = 20261109
GROUPS_PER_STEP = 32
MAX_EPOCHS = 60
PATIENCE = 15
sys.path.insert(0, str(UPSTREAM))
from data.reader.motion_dataset import Fusion, MPI3DHP  # noqa: E402
from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity  # noqa: E402
from utils.data import denormalize, flip_data  # noqa: E402
from utils.learning import load_model  # noqa: E402
from utils.tools import get_config  # noqa: E402
from utils.utils_3dhp import mpjpe_cal  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cfg():
    cfg = get_config(str(CONFIG))
    cfg.data_root = str(DATA_ROOT) + '/'
    if (cfg.n_frames, cfg.num_joints, cfg.dim_feat, cfg.dim_rep, cfg.stride) != (27, 17, 64, 512, 9):
        raise ValueError('unexpected MotionAGFormer-XS 3DHP configuration')
    return cfg


class AGFAdapter(torch.nn.Module):
    """Expose the original AGFormer representation to the prior memory code."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def head(self):
        return self.model.head

    def get_representation(self, x):
        return self.model(x, return_rep=True)

    def forward(self, x):
        return self.model(x)


def make_train_data(cfg):
    """Keep official 27/9 clips and group adjacent, disjoint clips by source."""
    dataset = MPI3DHP(cfg, train=True)
    raw = np.load(DATA_ROOT / 'data_train_3dhp.npz', allow_pickle=True)['data'].item()
    groups = []
    source_descriptors = []
    offset = 0
    for seq, value in raw.items():
        for cam, anim in value[0].items():
            n_frames = len(anim['data_2d'])
            count = (n_frames + cfg.stride - 1) // cfg.stride
            source_descriptors.append({'seq': str(seq), 'camera': str(cam),
                                       'frames': n_frames, 'clips': count, 'offset': offset})
            for phase in range(3):
                ids = list(range(offset + phase, offset + count, 3))
                groups.extend([ids[i:i+3] for i in range(0, len(ids), 3)])
            offset += count
    del raw
    flat = [i for group in groups for i in group]
    if (offset != len(dataset) or len(flat) != len(dataset) or
            sorted(flat) != list(range(len(dataset)))):
        raise ValueError('source group index does not cover the original training clips exactly once')
    if len(source_descriptors) != 128 or len(dataset) != 117024 or len(groups) != 39144:
        raise ValueError(f'unexpected source/clip/group counts: {len(source_descriptors)}, {len(dataset)}, {len(groups)}')
    return dataset, groups, source_descriptors


def load_bucket(dataset, groups, ids, epoch, device):
    """Stack groups of one length, using one consistent official flip per group."""
    lengths = {len(groups[i]) for i in ids}
    if len(lengths) != 1:
        raise ValueError('bucket contains different group lengths')
    indexes = np.asarray([groups[i] for i in ids], dtype=np.int64)
    x = np.asarray(dataset.poses_2d[indexes], dtype=np.float32)
    y = np.asarray(dataset.normalized_poses3d[indexes], dtype=np.float32)
    for k, group_id in enumerate(ids):
        flipped = random.Random(SEED + 1000003 * epoch + group_id).random() > 0.5
        if flipped:
            x[k] = flip_data(x[k], dataset.left_joints, dataset.right_joints)
            y[k] = flip_data(y[k], dataset.left_joints, dataset.right_joints)
    # Official dataset items are one 27-frame clip; group and batch are added here.
    x = torch.from_numpy(x.transpose(1, 0, 2, 3, 4).copy()).to(device, non_blocking=True)
    y = torch.from_numpy(y.transpose(1, 0, 2, 3, 4).copy()).to(device, non_blocking=True)
    return x, y


def clip_loss(prediction, target, cfg):
    return (loss_mpjpe(prediction, target) + cfg.lambda_scale * n_mpjpe(prediction, target) +
            cfg.lambda_3d_velocity * loss_velocity(prediction, target))
