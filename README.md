# SkeletonAGFormer-XS: Detector- and Skeleton-Representation Generalization

> **Research branch built on MotionAGFormer-XS.** The goal is to train one 3D human pose lifting model whose weights can be reused across different 2D pose detectors and different skeleton representations, without a detector-specific adapter, a per-skeleton model, or test-time fine-tuning.

This is an experimental research prototype. The architecture and evaluation pipeline are under development; **no benchmark accuracy or zero-shot generalization result is claimed yet**.

- **Base model:** [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer), XS configuration
- **Primary architectural reference:** [AnyTop: Character Animation Diffusion with Any Topology](https://anytop2025.github.io/Anytop-page/) (SIGGRAPH 2025)
- **Implementation:** [`model/SkeletonAGFormer.py`](model/SkeletonAGFormer.py)
- **Research design and experiment instructions:** [`docs/skeleton_generalization.md`](docs/skeleton_generalization.md)

---

## 1. Research motivation

A 3D human pose estimator should recover the underlying 3D human motion, not merely learn the conventions of one particular 2D detector or one fixed way of describing a human skeleton.

In common 2D-to-3D pose lifting benchmarks, the model is usually trained with a fixed joint set, a fixed joint order, and a predefined skeletal graph. In deployment, however, the 2D observations may come from different detectors, and the input skeleton may use a different set of joints or a different topology. A model tied to one input convention can lose accuracy or require an explicit conversion stage and additional training.

This branch investigates a more general question:

**Can a single MotionAGFormer-derived model learn reusable human-motion representations while being explicitly conditioned on the structure of the input skeleton?**

The intention is not to build a collection of detector/skeleton adapters. We want the model itself to accept the input representation, reason about its structure, and produce a useful 3D pose with the same shared weights.

### 1.1 Two independent generalization axes

We separate two sources of distribution shift because they are related but not interchangeable.

| Axis | What changes? | Examples | What should remain consistent? |
|---|---|---|---|
| **Detector generalization** | The observation and error distribution | Stacked Hourglass (SH), CPN, a previously unseen detector | Joint semantics and skeleton definition |
| **Skeleton-representation generalization** | The representation of the body | Joint count, selected joints, joint ordering, parent-child relations and topology | The underlying human pose and motion |

For detector generalization, the same semantic joint can have different localization errors, confidence values, and temporal noise depending on the detector.

For skeleton-representation generalization, the input may contain a different number of joints, omit some joints, add intermediate joints, or arrange the same semantic joints in a different order and graph. The model must not assume that joint index (i) always has one fixed meaning.

The long-term objective is to handle both axes together: a detector and a skeleton layout that were not present during training should still be usable by the same model.

### 1.2 A further distinction: body morphology

Different skeleton layouts are not the same problem as different human body proportions.

- **Representation change:** a different set of joints or graph describes the same underlying body.
- **Morphology change:** limb lengths, torso-to-leg ratio, or other anatomical proportions differ between people.
- **Observation change:** a detector measures the same joints with a different error distribution.

These factors may interact, but they should be measured separately. The current prototype conditions on a supplied reference skeleton; it does **not** yet establish robust generalization to arbitrary subject-specific body proportions.

### 1.3 What counts as success?

A model accepting tensors with different values of (J) is only shape-compatible. That alone is not evidence of generalization.

We want to demonstrate that:

1. One set of trained weights works on the canonical training skeleton.
2. The same weights work on skeleton layouts withheld from topology augmentation.
3. Performance degrades less under a layout shift than a fixed-skeleton MotionAGFormer baseline.
4. Detector shift can be measured independently, then combined with layout shift.
5. No target-layout-specific retraining or detector-specific conversion plugin is needed for the primary inference path.

---

## 2. Research approach

### 2.1 Why MotionAGFormer-XS?

We use MotionAGFormer-XS as the starting point so that the experiment remains computationally manageable and the effect of the architectural changes can be isolated.

We retain its core spatiotemporal design:

- Parallel spatial-temporal attention and graph branches.
- Temporal attention and temporal graph processing.
- Adaptive fusion between the two branches.
- The XS-scale configuration: 12 blocks, feature width 64, 8 attention heads, 27 input frames, and a 512-dimensional motion representation.

The main changes are concentrated in the parts that assume a fixed skeleton: joint encoding, spatial graph construction, spatial interaction, and output-joint readout.

### 2.2 Why AnyTop?

AnyTop is the primary reference because it studies motion modeling across different skeletal topologies. We borrow the general principle of making the skeleton structure explicit to the network, rather than assuming one fixed node layout.

We do **not** copy its diffusion architecture. Instead, we adapt topology-conditioned ideas to the existing MotionAGFormer-XS spatiotemporal lifting architecture.

The intended design is:

| Component | Fixed-skeleton assumption in the baseline | Direction in this branch |
|---|---|---|
| Joint embedding | Learned positional embedding indexed by a fixed joint ID | Shared feature projection plus an embedding of supplied anatomical reference geometry |
| Spatial attention | Attention over joints without explicit topology metadata | Relative graph-distance and directed joint-relation embeddings |
| Spatial GCN | One predefined adjacency matrix | Adjacency constructed from the supplied skeleton tree |
| GCN normalization | Normalization tied to a fixed node count | Feature/channel normalization independent of joint count |
| Temporal modeling | Existing temporal attention and graph operations | Retained, with the current joint dimension treated dynamically |
| Output | Per-index prediction for the fixed joint set | Target-skeleton queries that read from the observed source joints |

### 2.3 Explicit skeleton metadata

The model receives a `SkeletonSpec` together with the 2D pose sequence. A specification contains:

- `names`: semantic name of each joint;
- `parents`: parent index for each joint, with `-1` for the root;
- `rest`: reference 3D coordinates for the joints in a shared coordinate convention.

This metadata lets the model distinguish, for example, an elbow from a knee even when their tensor positions change. Parent-child relations and graph distances describe how joints are connected.

The reference coordinates currently represent a **schematic reference anatomy**, not measured 3D ground truth for each frame or each subject. A custom skeleton must supply a consistent schema and coordinate convention.

This is a skeleton-conditioned model, not an automatic skeleton-discovery system: it does not infer arbitrary joint names or topology from an unlabelled tensor.

### 2.4 Source skeleton and target skeleton

The source skeleton describes what the 2D detector observed. The target skeleton describes which 3D joints the model should predict.

For the primary benchmark, the default target is the canonical 17-joint skeleton. This keeps the 3D supervision and MPJPE target consistent when the source representation changes or omits joints.

The model also exposes a native-output path in which the requested target skeleton can match the source skeleton. However, the current training configurations supervise the canonical target. **Native-output accuracy on arbitrary new target layouts is therefore not established by this branch yet.**

---

## 3. Topology-aware training

A model trained only on one 17-joint layout has little reason to become robust to other layouts. We therefore introduce controlled skeleton transformations during training.

The current training augmentation includes:

1. Keep the canonical layout for a portion of batches (25% in the current sampler).
2. Randomly remove joints from an explicitly defined training-only pool.
3. Reconnect a retained joint to its nearest retained ancestor when an intermediate joint is removed.
4. Insert synthetic intermediate joints on selected training-only edges.
5. Randomly permute joint order while applying the same permutation to all skeleton metadata.

The synthetic inserted 2D joint is interpolated from its two observed endpoints; its confidence is conservatively derived from the endpoints. It is not an independent detector observation.

### 3.1 Hold out topology changes

We reserve different structural changes for evaluation. In the current controlled tests:

| Evaluation layout | Input joints | Purpose |
|---|---:|---|
| Canonical | 17 | Standard reference result |
| Permuted | 17 | Check that joint order itself is not meaningful |
| Holdout sparse | 13 | Test unseen elbow removal together with a changed reduced layout |
| Holdout dense | 21 | Test intermediate joints on edges not used for training augmentation |
| Holdout mixed | 17 | Test a different combination of removed and inserted joints |

The permutation test is an order-equivariance sanity check, not by itself evidence of unseen-topology generalization.

Synthetic layouts are a controlled first step. They do not replace evaluation on independently collected datasets with genuinely different skeleton definitions.

---

## 4. Experimental plan

We treat detector shift and skeleton shift as separate experimental variables, then evaluate their combination.

| Experiment | Detector used for training | Detector used for evaluation | Skeleton at evaluation |
|---|---|---|---|
| A: canonical baseline | SH | SH | Canonical |
| B: topology generalization | SH | SH | Held-out layout |
| C: detector generalization | SH | CPN / unseen detector | Canonical |
| D: joint generalization | SH | CPN / unseen detector | Held-out layout |
| E: multi-source training | SH + CPN | SH / CPN / unseen detector | Canonical and held-out layouts |

For a fair comparison, use the same train/test split, target skeleton, evaluation protocol, and test-time augmentation policy wherever applicable.

Recommended comparisons and ablations:

- Original MotionAGFormer-XS with the same detector data.
- SkeletonAGFormer-XS with topology augmentation disabled.
- SkeletonAGFormer-XS with topology augmentation enabled.
- Single-detector versus multi-detector training.
- Canonical versus held-out skeleton layouts.
- At least three random seeds for final claims.

Report absolute MPJPE (and the dataset's standard secondary metrics), the degradation relative to canonical evaluation, parameter count, and computational cost. Select checkpoints using only the designated validation data; held-out topology and unseen detector test sets must not influence model selection.

**Important dataset note:** Human3.6M and MPI-INF-3DHP commonly use 17-joint benchmark representations after preprocessing. Evaluating on both datasets is useful for cross-dataset robustness, but does not by itself demonstrate generalization to a different joint count or topology.

---

## 5. Current implementation

| File | Responsibility |
|---|---|
| `model/SkeletonAGFormer.py` | Topology-conditioned MotionAGFormer architecture |
| `model/skeleton.py` | Skeleton schemas, graph metadata, layout transforms and train/test layouts |
| `utils/learning.py` | Model factory integration |
| `train.py` | H36M training, source-balanced detector subsets and held-out-layout evaluation |
| `train_3dhp.py` | MPI-INF-3DHP integration |
| `data/preprocess/prepare_skeleton_h36m.py` | Prepare a separate H36M detector subset from an upstream-compatible detection pickle |
| `utils/skeleton_experiment.py` | Fixed-layout baseline projection for controlled comparisons |
| `evaluate_native_skeleton.py` | Generic diagnostic evaluator for supplied skeleton schemas and paired 3D labels |
| `tests/test_skeleton_generalization.py` | Architecture, shape, permutation, gradient and checkpoint regression tests |
| `docs/skeleton_generalization.md` | Detailed implementation notes and experiment commands |

### Current scope and limitations

- A skeleton schema must be supplied at inference.
- The temporal input length remains fixed by the model configuration (27 frames for XS).
- The primary supervised target remains the canonical dataset skeleton.
- Reference anatomy coordinates are schematic; arbitrary subject-specific morphology generalization is not yet demonstrated.
- Synthetic topology augmentation does not create real detector observations.
- Multi-detector training requires separately prepared, correctly aligned detector datasets and matching evaluation metadata.
- No formal accuracy result, training result, or publication-level zero-shot claim is made until the experiments have been run and checked.

---

## 6. Setup and usage

The environment and original dataset preparation follow the upstream MotionAGFormer implementation. See the [upstream repository](https://github.com/TaatiTeam/MotionAGFormer) and the original instructions below. The original model configurations remain available for baselines.

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the original Human3.6M SH data as described in the upstream README. The XS model uses 27-frame clips.

### Train SkeletonAGFormer-XS on SH

```bash
python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml \
  --new-checkpoint checkpoint/skeleton-xs-sh
```

### Prepare a separate CPN subset

Obtain an upstream-compatible H36M detection pickle containing the expected camera/source metadata and 2D/3D fields. Then run from the repository root:

```bash
python -m data.preprocess.prepare_skeleton_h36m \
  --dt-file h36m_cpn_cam_source.pkl \
  --subset H36M-27-CPN
```

This script slices an existing detector pickle; it does not run CPN or generate detections. Keep detector subsets separate and do not overwrite the SH data.

### Train with SH and CPN

Only run this after both subsets have been prepared and checked:

```bash
python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml \
  --train-subsets H36M-27 H36M-27-CPN \
  --new-checkpoint checkpoint/skeleton-xs-multi
```

When multiple subsets are supplied, the training loader samples the detector subsets with equal total probability.

### Evaluate a held-out skeleton

Use the same trained checkpoint for every layout:

```bash
python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml \
  --eval-only --checkpoint checkpoint/skeleton-xs-sh \
  --checkpoint-file best_epoch.pth.tr --skeleton-layout holdout_sparse
```

Available layouts: `canonical`, `permuted`, `holdout_sparse`, `holdout_dense`, and `holdout_mixed`.

### Evaluate detector shift

For example, evaluate an SH-trained checkpoint on the separately prepared CPN subset:

```bash
python train.py --config configs/h36m/SkeletonAGFormer-xs.yaml \
  --eval-only --checkpoint checkpoint/skeleton-xs-sh \
  --checkpoint-file best_epoch.pth.tr \
  --test-subset H36M-27-CPN --dt-file h36m_cpn_cam_source.pkl
```

The `--test-subset` and `--dt-file` arguments must refer to the same detector source and matching metadata. For the combined detector/layout experiment, add `--skeleton-layout holdout_sparse`.

### Run regression tests

```bash
python -m unittest discover -s tests -p test_skeleton_generalization.py
```

### MPI-INF-3DHP

The branch also provides an XS configuration and training entry point:

```bash
python train_3dhp.py --config configs/mpi/SkeletonAGFormer-xs.yaml \
  --new-checkpoint mpi-checkpoint/skeleton-xs
```

See [the detailed research and evaluation notes](docs/skeleton_generalization.md) for custom `SkeletonSpec` inputs, the native-layout diagnostic evaluator, baseline comparisons, and caveats.

---

## 7. Related work

- **AnyTop (SIGGRAPH 2025):** [Project page](https://anytop2025.github.io/Anytop-page/) — primary inspiration for topology-conditioned motion modeling and topology variation.
- **REVE:** [Paper](https://arxiv.org/abs/2510.21585) — relevant ideas for spatially grounded encoding under different sensor layouts.
- **MetaMorph:** [Project page](https://metamorph-iclr.github.io/site/) — relevant ideas for shared Transformer representations across different robot morphologies.
- **MotionAGFormer:** [Paper](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html) — base spatiotemporal architecture.

These works motivate the design; they do not establish that the present implementation achieves the intended generalization. Novelty and performance claims must be based on a systematic review and completed experiments.

---

## 8. Upstream MotionAGFormer

This branch is based on the official [MotionAGFormer implementation](https://github.com/TaatiTeam/MotionAGFormer), whose original paper is:

> S. Mehraban, V. Adeli, and B. Taati, “MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network,” WACV 2024.

The original training configurations, evaluation code, pretrained weights, demo, acknowledgements, and citation information are retained in the repository. Use the original `MotionAGFormer-*` configurations when reproducing the fixed-skeleton baseline.
