# RemoteCLIP Vaihingen complete partial-support evidence

Date: 2026-08-21
Status: completed; scientific evidence=true

## Frozen controlled run

This is the artifact-complete replacement for the earlier cache-only audit. It
uses the separate, non-overwriting 3090v2 run
`/home/zhongsz/second_open_vocabulary/outputs/baselines/remoteclip_complete/vaihingen_v0/run_20260821T070310Z_ab6e429b`.

Only the encoder was replaced by RemoteCLIP. SAM3 candidates, prompts,
registered support subsets, visual-prototype construction, C2, CTP-v1,
`alpha`, FusionCanvas, evaluator, data split, and GT-isolation protocol were
frozen. The manifest records prediction sealing before GT access.

| Item | Value |
|---|---|
| Run status | completed; scientific=true |
| Score dtype | float32 |
| Full-support semantic maps | 15 |
| Partial-support semantic maps | 375 |
| Total semantic maps | 390 |
| Score archives | 27 |
| Partial-support confusion matrices | 390 |
| Per-area metric rows | 390 |
| Pixel-accounting rows | 390 |
| Registered subset-metric rows | 78 |
| Bootstrap records | 25 (area clusters, seed 42, 5,000 resamples) |
| Source code commit | `3564ea7` |

## Immutable provenance

| Artifact | SHA-256 |
|---|---|
| Run manifest | `56eec9...30ff` |
| Prediction manifest | `ae53f2...8584` |
| Metrics | `93fc2b...7cb7` |
| Bootstrap | `e05513...4b38` |
| Artifact hash inventory | `d62a1f...fe4e` |
| Run report | `b2bfee...2d0c` |

The RemoteCLIP checkpoint is `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4`.
The frozen SAM3 candidate aggregate is
`c06b9b8008566a1ff3ce748ab624154272c7cae32964451d4fb1d962d86b2da8`.

## Results available for paper statistics

The run retains, for every registered partial-support subset and test area,
OA, Macro-F1, mIoU, S/U/H F1, S/U/H IoU, the confusion matrix, and valid,
assigned, conflict-ignore, and uncovered pixel accounting. It also retains
area-cluster bootstrap records for the partial-support and full-support
comparisons. These are formal outputs, not float16 reconstructions.

The full-support metrics exactly reproduce the prior sealed RemoteCLIP run:

| Method | OA | Macro-F1 | mIoU |
|---|---:|---:|---:|
| Text-only | 0.143945203 | 0.131737719 | 0.078768029 |
| C2 normalized | 0.487243649 | 0.554823030 | 0.405053193 |
| CTP-v1 | 0.498766426 | 0.567815872 | 0.414507073 |

For all statistical reporting, use the attached per-subset/per-area records and
their stored bootstrap result rather than inferring uncertainty from aggregate
means. This report intentionally does not duplicate a large numeric table;
the sealed run manifest and CSV/JSON assets are the authority.

## Claim boundary

The evidence supports that, within the frozen Vaihingen protocol, CTP-v1
continues to operate with RemoteCLIP and is evaluated with complete
partial-support accounting. It does not alone prove a universal,
backbone-independent claim, nor establish a numerical RemoteCLIP-versus-OpenAI
ranking without a matched OpenAI comparison.

No first-paper file, CTP-v1 component, SAM3 output, or pre-existing experiment
output was modified. No training, adapter, prompt tuning, or calibration was
performed.
