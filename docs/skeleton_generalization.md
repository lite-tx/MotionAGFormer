# SkeletonAGFormer-XS — skeleton representation generalization

## Research question

Can one **set of weights** lift 2D pose sequences under an unseen detector
and an unseen skeleton layout, without detector-specific adapters or retraining?
Detector shift (same joint definition, different observations) and skeleton
shift (joint count/order/definition/topology) are independent axes.
A shape-compatible forward pass is not evidence of accurate zero-shot lifting.

This branch is independent from memory experiments. Based on upstream
MotionAGFormer-XS (12 blocks, width 64, 8 heads, 27 frames, representation 512).
Primary reference: [AnyTop, SIGGRAPH 2025](https://arxiv.org/abs/2502.17327).
We adapt skeleton-conditioned per-joint enrichment, query/key-relative
topological attention and skeleton augmentation. We retain the AGFormer
attention/GCN branches, temporal blocks and adaptive fusion. No diffusion,
no pretrained T5, and no per-detector conversion plugin.

## Implemented architecture

1. Source input: x [B,T,J,2 or 3], a SkeletonSpec with unique joint names,
   parent indices (-1 for root), schematic anatomy reference xyz for each joint.
   Layout metadata are known at inference. No sample 3D GT is supplied.
2. Joint embedding: shared linear on xy/confidence + a shared MLP on reference
   xyz; replaces the fixed learned index table. It is geometry-conditioned,
   not AnyTop's full text/semantic encoder. Two-coordinate input gets conf=1.
3. Spatial attention: vanilla qk + query/key relative graph-distance and
   directed relation embeddings (self,parent,child,ancestor,descendant,other).
4. Spatial GCN: build normalized adjacency from current parent tree; use
   channel LayerNorm instead of fixed node-index BatchNorm. Temporal attention
   and temporal GCN operate on the current number of joints, fixed T=27.
5. Target query readout: cross-attend from requested target anatomy to
   source features, with geometry bias and exact semantic-name skip for
   observed joints. The primary target remains the dataset's 17-joint
   skeleton, including unobserved joints, for identical evaluation targets.
   Passing target_skeleton=source gives native variable-joint output.
6. The same set of weights is used for all layouts. A minibatch uses one
   layout (bucket actual heterogeneous samples by layout); no padding or
   arbitrary unknown joint metadata inference is implemented.

Rest positions are schematic reference anatomy in a shared coordinate
system (anatomical left positive x, y up, z anterior), not measured 3D poses.
Custom schemas must use consistent names and reference conventions.
MPI uses the repository H36M_TO_MPI index mapping; check original dataset
semantics before any cross-dataset claim.

## Data and training

Use the original upstream H36M-27 SH preparation first (see root README).
To prepare a separate CPN source, obtain the upstream-compatible detector
pickle, then from repo root:

    python -m data.preprocess.prepare_skeleton_h36m --dt-file h36m_cpn_cam_source.pkl --subset H36M-27-CPN

The input pickle needs the same 2D/3D camera/source schema as DataReaderH36M.
It does not generate CPN detections. Keep each detector in a separate subset;
do not overwrite SH clips. Source-balanced sampling is used when multiple
training subsets are supplied. The single evaluation subset and matching
metadata must agree (override --test-subset and --dt-file together).

Train on SH only (tests isolate topology from detector shift):

    python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml --new-checkpoint checkpoint/skeleton-xs-sh

Train on SH and CPN (only if both datasets exist):

    python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml --train-subsets H36M-27 H36M-27-CPN --new-checkpoint checkpoint/skeleton-xs-multi

Train a matched architecture without topology augmentation:

    python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml --disable-topology-aug --new-checkpoint checkpoint/skeleton-xs-noaug

Baseline with original architecture and same evaluation flip policy:

    python train.py --config configs/h36m/MotionAGFormer-xsmall.yaml --no-eval-flip --new-checkpoint checkpoint/original-xs-sh

Training keeps source J=17 25% of batches. Others randomly drop up to 3
from an explicit training-only joint pool, reconnect children to retained
ancestors, insert edge midpoints from training-only proximal edges and
permute node order with metadata. Synthetic midpoints interpolate observed
2D coordinates and use min endpoint confidence, not independent observations.
Root is retained. The target and original XS losses (MPJPE, scale, velocity)
remain fixed; original zero-weight bone/angle losses are not part of the
v1 objective. Noise augmentation is optional (default off); no detector ID.

## Held-out evaluation

Run every layout with exactly the same checkpoint:

    python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml --eval-only -c checkpoint/skeleton-xs-sh --checkpoint-file best_epoch.pth.tr --skeleton-layout holdout_sparse

Change layout to canonical, permuted (order-invariance sanity check),
holdout_dense (21 joints, new distal midpoint edges), holdout_mixed
(17 joints, different joint set), or holdout_sparse (13 joints, unseen elbow
deletions). These exact structures are not selected during training; the
permutation sanity check is not a held-out structure.
H36M uses original official-style evaluator (MPJPE/P-MPJPE), and no test-time
flip for the topology runs. Compare the original XS with the same layout
flag and its own config/checkpoint: an evaluation-only fixed-layout
projection keeps observed joints and interpolates missing 2D points from
available reference neighbors; it never uses hidden observations or 3D GT.
Use --no-eval-flip for the canonical baseline too. This projection is a
baseline, not part of our model.

To assess detector generalization (SH-only checkpoint on CPN):

    python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml --eval-only -c checkpoint/skeleton-xs-sh --checkpoint-file best_epoch.pth.tr --test-subset H36M-27-CPN --dt-file h36m_cpn_cam_source.pkl --skeleton-layout holdout_sparse

For a genuinely unseen detector, prepare its real detections and do not
include them in --train-subsets or checkpoint selection. Evaluate the
factorial grid: seen/unseen detector x seen/unseen layout. Report absolute
MPJPE, relative degradation from canonical, parameters and compute. Also
compare original XS trained with the same source mix and the no-augmentation
ablation. Use at least three seeds and consistent data splits/checkpoint
selection before making conclusions.

MPI integration:

    python train_3dhp.py --config configs/mpi/SkeletonAGFormer-xs.yaml --new-checkpoint mpi-checkpoint/skeleton-xs

The MPI script preserves upstream evaluation/TTA conventions. Its held-out
layout mode supports SkeletonAGFormer, not the fixed-layout baseline yet.
The original two datasets both expose 17 joints: merely testing across them
does not establish new-topology generalization.

## Native external layouts

    from model.skeleton import H36M, SkeletonSpec
    from model.SkeletonAGFormer import SkeletonAGFormer
    model = SkeletonAGFormer().eval()
    # Load the trained checkpoint before making accuracy claims.
    # Given x with shape [B,27,J,2 or 3] and a matching spec:
    pred = model(x, skeleton=spec, target_skeleton=H36M)
    native = model(x, skeleton=spec, target_skeleton=spec)

For actual externally detected keypoints, use the generic NPZ evaluator
with a known skeleton definition and paired normalized 3D ground truth:

    python evaluate_native_skeleton.py --config configs/h36m/SkeletonAGFormer-xs.yaml --checkpoint checkpoint/skeleton-xs-sh/best_epoch.pth.tr --data external.npz --source-skeleton external.json

NPZ: x [N,T,J,2/3], y [N,T,K,3]; optional mm_per_unit scalar or [N,T].
JSON: names, parents, rest arrays. Optional --target-skeleton supplies a
different target; for fair primary experiments keep the canonical target.
Report normalized units unless a validated mm conversion is supplied.
This diagnostic is not the official H36M/3DHP protocol; data and coordinate
alignment must be verified. Native-output accuracy on novel targets also
needs training/evaluation with those target definitions; the fixed-target
training does not establish it.

## Tests and limits

    python -m unittest discover -s tests -p test_skeleton_generalization.py

Tests cover topology permutation, graph distances, layout separation,
variable J and target J, fixed-baseline observation restriction,
gradient flow, checkpoint round-trip and a full-size XS forward.

Not an arbitrary unlabeled-node system: a skeleton schema is required.
Not an arbitrary temporal-length system: T remains fixed.
No measured MPJPE or GPU training results are claimed. The first milestone
is preserving canonical accuracy while reducing degradation on the synthetic
holdouts. Synthetic holdouts are only a controlled proxy; independent real
layouts and detections are required for publication-grade generalization.
