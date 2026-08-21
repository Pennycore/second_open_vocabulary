# Backbone-robustness evidence summary — 2026-08-21

## What is supported

The frozen Vaihingen pipeline was executed with RemoteCLIP ViT-B/32 as a
backbone replacement, using the same SAM3 candidate cache, prompts, registered
support subsets, visual-prototype construction, C2/CTP-v1 formulas, `alpha`,
FusionCanvas, and evaluator as the controlled protocol. The completed source
run is `run_20260820T152937Z_1afc6939`; its checkpoint SHA-256 is
`60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4`.

| Method | OA | Macro F1 | mIoU |
|---|---:|---:|---:|
| RemoteCLIP Text-only | 0.143945203 | 0.131737719 | 0.078768029 |
| RemoteCLIP C2 normalized | 0.487243649 | 0.554823030 | 0.405053193 |
| RemoteCLIP CTP-v1 | 0.498766426 | 0.567815872 | 0.414507073 |

For CTP-v1 minus C2, the five-area cluster bootstrap (seed 42, 5,000 repeats)
gives +0.0115228 OA [0.0069954, 0.0162585], +0.0129928 Macro F1
[0.0084988, 0.0157350], and +0.0094539 mIoU [0.0033819, 0.0119766].

The k=2/3/4 registered partial-support aggregates are documented in
`reports/remoteclip_vaihingen_partial_support_20260821.md`. Across all three
support sizes, RemoteCLIP C2 records zero U-IoU and H-IoU, while RemoteCLIP
CTP-v1 retains non-zero U-IoU/H-IoU.

## Claim boundary

The evidence supports: **within this frozen Vaihingen protocol, CTP-v1 remains
effective after replacing the original CLIP encoder with RemoteCLIP.** It does
not support saying that RemoteCLIP is numerically superior or inferior to
OpenAI CLIP unless a matched OpenAI baseline is cited alongside it. The
partial-support source artifacts do not support partial per-area confidence
intervals.

## External-baseline decision

SegEarth-OV was audited but not run. It is a protocol-different dense OVSS
pipeline that does not share SAM3 candidates, FusionCanvas, support subsets, or
partial-support definitions. The current reproducibility/fairness decision is
**NO-GO**; see `reports/segearth_ov_feasibility_audit_20260821.md`. It must not
be placed in the controlled backbone-replacement table or used as a CTP plug-in.

## Integrity statement

This phase did not alter CTP-v1, SCC, Guard, C2, prompts, prototypes, `alpha`,
FusionCanvas, SAM3, data splits, first-paper files, or existing output runs.
No training, adapter, prompt tuning, or SegEarth-OV inference was performed.
