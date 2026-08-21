# SegEarth-OV SimFeatUp Phase-A installation gate (2026-08-22)

## Verdict

**STOP / NO-GO before installation.** The isolated SegEarth-OV runtime has no
CUDA Toolkit compiler (`nvcc`) and no `/usr/local/cuda*` directory. Official
SimFeatUp defines `adaptive_conv_cuda_impl` as a PyTorch `CUDAExtension`, so
the missing compiler blocks a controlled build of the official `AdaptiveConv`
dependency.

No repository clone, package installation, download, runtime upgrade,
adapter/source change, smoke test, formal inference, semantic-GT read, or
metric computation was performed in this phase.

## Frozen pre-install evidence

| Item | Value |
|---|---|
| Gate log | `/data/second_open_vocabulary_storage/external_baselines/segearth_ov/runtime_py39/simfeatup_gate_20260822T000001Z.log` |
| Gate-log SHA-256 | `a213fa4da50db5d72c100035df3776127d3a36958b8d5cc7ab49515fb88e02b1` |
| SegEarth-OV commit | `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| Isolated runtime | Python 3.9; torch `2.1.2+cu118`; MMCV `2.1.0`; MMEngine `0.10.4`; MMSeg `1.2.2`; NumPy `1.26.4` |
| Runtime integrity | `pip check`: no broken requirements |
| GPU | RTX 3090; no compute process before the gate |
| Build tools | GCC 9.4 present; `nvcc` absent; `/usr/local/cuda*` absent |

The fixed SegEarth README directs users to install
[`likyoo/SimFeatUp`](https://github.com/likyoo/SimFeatUp), but does not pin a
revision. For a future reproducible installation, the proposed source is the
official repository commit `78a0ba70b1d6ea7283684a88c98ce338af4593ca`: the
last official SimFeatUp commit at or before the frozen SegEarth checkout date.
Its `setup.py` defines the required CUDA extension. This source was **not**
cloned or installed in the current phase.

## Required new authorization before proceeding

Only an environment-scoped CUDA Toolkit provision, followed by a fixed-commit,
`--no-deps` SimFeatUp build inside `runtime_py39`, can reopen the next gate.
That future action must first verify that it cannot alter torch, MMCV, MMSeg,
or NumPy; it must then pass `AdaptiveConv` import and one frozen-tile,
prediction-only smoke test before any new formal run.
