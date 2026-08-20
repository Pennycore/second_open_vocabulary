# RemoteCLIP Vaihingen baseline — completed run record

Date: 2026-08-20
Evidence class: controlled backbone-replacement baseline
Status: completed; immutable new output only

## Scope and frozen protocol

This record covers the completed 3090v2 run:

`/home/zhongsz/second_open_vocabulary/outputs/baselines/remoteclip/vaihingen_v0/run_20260820T152937Z_1afc6939`

RemoteCLIP ViT-B/32 replaced only the OpenAI CLIP encoder. The SAM3 candidate
pipeline, prompts, support-subset definitions, visual-prototype construction,
`alpha`, C2 normalization, CTP-v1, FusionCanvas, and evaluation protocol were
frozen. No training, adapter, prompt tuning, calibration, DINO, or
multi-prototype method was added. No OpenAI CLIP features or scores were read.

The runner implementation was introduced by code commit `7fbbaaa`. The run
uses candidate cache run `run_20260820T145952Z_5beba872`, whose aggregate
SHA-256 is
`c06b9b8008566a1ff3ce748ab624154272c7cae32964451d4fb1d962d86b2da8`.

| Input | SHA-256 |
|---|---|
| SAM3 checkpoint | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| RemoteCLIP ViT-B/32 checkpoint | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` |
| candidate cache aggregate | `c06b9b8008566a1ff3ce748ab624154272c7cae32964451d4fb1d962d86b2da8` |

## Full-support results

| Method | OA | Macro F1 | mIoU |
|---|---:|---:|---:|
| RemoteCLIP Text-only | 0.143945203 | 0.131737719 | 0.078768029 |
| RemoteCLIP C2 normalized | 0.487243649 | 0.554823030 | 0.405053193 |
| RemoteCLIP CTP-v1 | 0.498766426 | 0.567815872 | 0.414507073 |

Under this frozen protocol, CTP-v1 improves over RemoteCLIP C2 by 0.011523
OA, 0.012993 Macro F1, and 0.009454 mIoU. This within-RemoteCLIP comparison
does not establish a numerical comparison with OpenAI CLIP without separately
matched OpenAI evidence.

## Partial-support results

Each row is the recorded mean over the frozen subsets for the stated support
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

The recorded result shows C2's zero unseen/harmonic outcomes for k=2/3/4,
whereas CTP-v1 retains non-zero unseen and harmonic performance. This is a
descriptive outcome of the frozen protocol; it is not a rationale to retune
CTP-v1.

## Output evidence inventory

The remote run manifest records output-file hashes. The following verified
prefixes are retained here for efficient cross-checking:

| Artifact | SHA-256 prefix |
|---|---|
| predictions | `4205d9` |
| RemoteCLIP features | `6dba7bc3` |
| candidate records | `f7e42c02` |
| full-support metrics | `34b13570` |
| partial-support metrics | `9515b6dd` |
| run report | `2e7e3428` |
| run manifest | `f8459b8e` |

## Interpretation boundary

This completed run supports the limited claim that CTP-v1 remains operational
and improves the recorded full-support metrics over C2 within a RemoteCLIP
backbone replacement under the same frozen Vaihingen pipeline. It does not
license a claim that RemoteCLIP is better or worse than OpenAI CLIP unless the
matched OpenAI baseline is cited separately. SegEarth-OV remains
feasibility-only, and no LoveDA artifact was modified or rerun.
