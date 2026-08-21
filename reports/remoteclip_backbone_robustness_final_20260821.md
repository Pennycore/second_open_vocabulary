# Final RemoteCLIP backbone-robustness evidence

Date: 2026-08-21

## Controlled question and answer

Question: does the frozen CTP-v1 procedure remain effective after replacing
the original encoder with a remote-sensing vision-language encoder?

Answer supported by this experiment: **yes, within the frozen Vaihingen
protocol and the tested RemoteCLIP ViT-B/32 implementation.** The evidence is
not a claim of universal backbone independence.

The completed artifact-complete run is
`run_20260821T070310Z_ab6e429b` on 3090v2. It has float32 scores, prediction
sealing before GT access, 390 semantic maps, 390 confusion matrices, 390
per-area metric/accounting records, and registered partial-support bootstrap
assets. It reproduces the previously sealed full-support metrics.

| RemoteCLIP method | OA | Macro-F1 | mIoU |
|---|---:|---:|---:|
| Text-only | 0.143945203 | 0.131737719 | 0.078768029 |
| C2 normalized | 0.487243649 | 0.554823030 | 0.405053193 |
| CTP-v1 | 0.498766426 | 0.567815872 | 0.414507073 |

CTP-v1 minus C2 is +0.0115228 OA, +0.0129928 Macro-F1, and +0.0094539
mIoU. The previously recorded five-area full-support bootstrap (seed 42,
5,000 repeats) gave percentile 95% CIs of [0.0069954, 0.0162585],
[0.0084988, 0.0157350], and [0.0033819, 0.0119766], respectively. Four of
five held-out areas have a positive mIoU direction; the area-level
heterogeneity must be stated.

The complete run now also preserves the registered k=2/3/4 partial-support
per-area metrics, S/U/H metrics, valid/assigned/conflict-ignore/uncovered
accounting, and bootstrap outputs. Those records replace the prior limitation
that only aggregate partial metrics were available.

## Required paper wording

Use: “Under the same frozen SAM3 candidates, prompts, support subsets,
prototype construction, CTP-v1 formula, FusionCanvas, and evaluation protocol,
CTP-v1 retained a positive controlled effect with RemoteCLIP on Vaihingen.”

Do not use: “CTP is universally backbone-independent,” “RemoteCLIP is better
than OpenAI CLIP,” or any direct cross-backbone ranking without an explicitly
matched OpenAI result.

## Integrity boundary

No CTP-v1/SCC/Guard/C2/prompt/alpha/prototype/FusionCanvas/SAM3/split change,
no training, and no first-paper modification occurred in this phase.
