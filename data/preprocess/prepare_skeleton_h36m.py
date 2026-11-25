"""Prepare separate H36M detector subsets using upstream DataReaderH36M.

Run from repository root:
python -m data.preprocess.prepare_skeleton_h36m --dt-file h36m_cpn_cam_source.pkl --subset H36M-27-CPN
The pickle must use the upstream schema and matching joint order, split, and
camera/source metadata. This script does not generate real detector predictions.
"""
import argparse
from pathlib import Path
import pickle
import random
import numpy as np
from data.reader.h36m import DataReaderH36M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt-file", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--data-root", default="data/motion3d")
    p.add_argument("--n-frames", type=int, default=27)
    args = p.parse_args()
    root = Path(args.data_root)
    destination = root / args.subset
    if destination.exists():
        raise FileExistsError(f"refusing to mix with existing clips: {destination}")
    if args.n_frames < 3:
        raise ValueError("n-frames must be at least 3")
    random.seed(0)
    np.random.seed(0)
    reader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                            data_stride_train=args.n_frames // 3,
                            data_stride_test=args.n_frames,
                            dt_root=str(root), dt_file=args.dt_file)
    train_x, test_x, train_y, test_y = reader.get_sliced_data()
    for split,x,y in (("train",train_x,train_y),("test",test_x,test_y)):
        if x.shape != y.shape or x.shape[-2:] != (17,3):
            raise ValueError(f"unexpected {split} shape: {x.shape}, {y.shape}")
        folder = destination / split
        folder.mkdir(parents=True,exist_ok=False)
        for i,(a,b) in enumerate(zip(x,y)):
            with (folder / f"{i:08d}.pkl").open("wb") as f:
                pickle.dump({"data_input":a.astype(np.float32),"data_label":b.astype(np.float32)},f)
        print(f"{split}: {len(x)} clips -> {folder}")


if __name__ == "__main__":
    main()
