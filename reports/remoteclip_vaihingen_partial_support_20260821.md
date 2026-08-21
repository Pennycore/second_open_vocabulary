# RemoteCLIP Vaihingen partial-support evidence — cache-only audit

Date: 2026-08-21
Status: completed post-processing audit; no new model inference

## Frozen scope

This audit reuses only the sealed RemoteCLIP Vaihingen source run
`/home/zhongsz/second_open_vocabulary/outputs/baselines/remoteclip/vaihingen_v0/run_20260820T152937Z_1afc6939`.
The corrected cache-only evaluator ran in the separate non-overwriting directory
`/home/zhongsz/second_open_vocabulary/outputs/baselines/remoteclip/vaihingen_partial_cache_audit/run_20260821T044500Z_35155e8`.

No SAM3 candidates, RemoteCLIP features, predictions, prompts, support subsets,
visual prototypes, C2, CTP-v1, FusionCanvas, `alpha`, or evaluator definitions
were changed. No first-paper file was accessed for modification. No training or
new GPU inference was performed. The audit code revision is
`35155e8b78cde1928bbe0ca5d7f80cc7d586eeba`.

Source evidence bindings: candidate aggregate `c06b9b8008566a1ff3ce748ab624154272c7cae32964451d4fb1d962d86b2da8`;
RemoteCLIP checkpoint `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4`;
source prediction / feature / partial-metric manifest prefixes `4205d9f8`,
`6dba7bc3`, and `9515b6dd` respectively.

## Registered partial-support aggregate results

Each value is the recorded mean over the pre-registered subsets at that support
size. `S`, `U`, and `H` denote seen, unseen, and harmonic metrics.

| k | Method | S-F1 | U-F1 | H-F1 | S-IoU | U-IoU | H-IoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | Text-only | 0.131738 | 0.131738 | 0.081527 | 0.078768 | 0.078768 | 0.047250 |
| 2 | C2 normalized | 0.493463 | 0.000000 | 0.000000 | 0.341186 | 0.000000 | 0.000000 |
| 2 | CTP-v1 | 0.380447 | 0.130995 | 0.138907 | 0.271940 | 0.077499 | 0.082318 |
| 3 | Text-only | — | — | 0.081527 | — | — | 0.047250 |
| 3 | C2 normalized | 0.539868 | — | 0.000000 | 0.389064 | — | 0.000000 |
| 3 | CTP-v1 | 0.457721 | 0.120651 | 0.155257 | 0.328907 | 0.070205 | 0.092869 |
| 4 | Text-only | — | — | 0.060824 | — | — | 0.035052 |
| 4 | C2 normalized | 0.554188 | — | 0.000000 | 0.403987 | — | 0.000000 |
| 4 | CTP-v1 | 0.535899 | 0.134927 | 0.174585 | 0.386458 | 0.079197 | 0.108198 |

Under the frozen protocol, C2 has zero recorded unseen and harmonic IoU for
all three support sizes; CTP-v1 retains non-zero U-IoU and H-IoU. This is a
descriptive result, not a basis for tuning any frozen component.

## Full-support area-cluster bootstrap

The audit bootstrapped the five held-out test areas as clusters, with seed 42
and 5,000 resamples. Values are CTP-v1 minus C2 normalized. The following
intervals are percentile 95% confidence intervals, conditional on this fixed
five-area test set and its frozen predictions.

| Metric | Delta | 95% CI |
|---|---:|---|
| OA | +0.0115228 | [+0.0069954, +0.0162585] |
| Macro F1 | +0.0129928 | [+0.0084988, +0.0157350] |
| mIoU | +0.0094539 | [+0.0033819, +0.0119766] |

The held-out-area mIoU direction is positive in four of five areas; area 34 is
negative. This heterogeneity should remain visible in any paper claim.

## Availability boundary

The sealed source run has aggregate partial-support records, but does not retain
partial-support semantic maps or per-area partial confusion matrices. Its
float16 feature cache cannot be used to recreate the original float32 semantic
maps exactly. Therefore partial-support per-area bootstrap intervals, per-subset
standard deviations, and partial pixel accounting are **unavailable**; they
have not been fabricated or estimated. The aggregate table above is the maximum
supported evidence from the immutable source artifacts.

## Interpretation boundary

This supports the bounded claim that the frozen CTP-v1 mechanism remains
operational with RemoteCLIP and avoids the recorded C2 unseen/harmonic collapse
on these registered Vaihingen support subsets. It does not establish a numerical
OpenAI-CLIP-versus-RemoteCLIP ranking without matched OpenAI evidence, and it
does not generalize beyond this protocol.
