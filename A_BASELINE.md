# A memory evolution: verified starting point

This branch starts from the original MotionAGFormer `master` commit
`4756fd1eb7cc73f0e991f091ff2280e030ab85f3` and preserves the verified A
experiment before adopting further memory techniques.

## Baseline to preserve

| Item | Recorded setting/result |
| --- | --- |
| Model | MotionAGFormer-XS + incremental F/R/H memory + H-conditioned query |
| Dataset/input | MPI-INF-3DHP, GT 2D, 27 frames, 17 joints |
| Seed | 20261111 |
| Endpoint | Epoch150 last, 183600 optimizer updates |
| MPJPE | **18.21779441833496 mm** on all2875 valid test targets |
| Recheck | Independently repeated2026-11-24; difference0.0mm |
| Query change | `q = key_proj(F) + W_H LayerNorm_noaffine(H_old)` |
| Added query parameters | 4096, bias-free64x64 projection, zero initialized |
| Current-window supervision | `0.75 * native(main) + 0.25 * native(head(F))` |
| Arithmetic | FP32, TF32 disabled |

The checkpoint SHA256 is
`085863d38f5c45b74577761731b4d337bd0148ae1e3d9010998ba8ef3754edc5`.
Dataset files and checkpoint binaries remain in the experiment archive, outside Git.

## Source and evidence

- [`research/frh_a_baseline/frozen/`](research/frh_a_baseline/frozen/): all14
  original Python files plus the frozen-reference manifest, copied without edits.
- [`research/frh_a_baseline/baseline.json`](research/frh_a_baseline/baseline.json):
  settings, identities and SHA256 values for every preserved source file.
- [`research/frh_a_baseline/results/epoch150.json`](research/frh_a_baseline/results/epoch150.json):
  original full-test result and per-video scores.
- [`research/frh_a_baseline/results/recheck.json`](research/frh_a_baseline/results/recheck.json):
  independent full-test result and exact agreement with the original evaluation.

This first branch commit archives the working experiment source as recorded.
The frozen code retains its original server/data paths and experiment imports;
it is not yet integrated into the repository's ordinary training CLI. That
integration is a separate next step and should first establish equivalence
against the recorded checkpoint. Keep `frozen/` intact and implement later
changes outside that snapshot.

## Research direction

以已经验证的A为起点，逐项评估新技术，同时保留原A的结果和可回退版本。
每项改动应注明假说、实际改变的路径、训练预算及评价口径，并与匹配的A比较。
本轮成绩是选定种子的固定150轮末轮，不是全程best，也不是已证明的协同机制。

本分支独立于`memory-v1`的三层记忆设计。后续可以研究其中值得迁移的技术，
先保持比较条件可核对，再判断是否保留改动。创建本分支没有启动新训练。
