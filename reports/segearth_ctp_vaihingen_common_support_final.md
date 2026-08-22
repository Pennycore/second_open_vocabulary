# SegEarth-OV / CTP-v1 Vaihingen frozen common-support comparison

Date: 2026-08-22  
Status: completed offline evaluation; external-protocol comparison only

## A. Asset audit

The formal OpenAI-CLIP CTP-v1 maps were recovered unchanged from
`research_archive/artifacts/remote_2080ti_workspace_outputs_20260820.tar.gz`
(archive SHA-256 `c39019fe5169aaa74ae66fe1745d154671ec5dfbeabe1da1524c6ca9234590d5`).
The archive manifest and all five CTP map bytes were verified before transfer
to the 3090 recovery directory; details are in
`reports/ctp_segearth_common_support_asset_audit.md`.

| Binding | Verified identity |
|---|---|
| Formal CTP manifest | `b064984cd2a3baf7f70835ec8a8c8d767477066223ad7874ddcbfaeab51b0309` |
| Frozen CTP config | `788f1962d497022fbd5cacd7b63eaedddecd0343104aa726ee80afcdf1b37430` |
| SegEarth official commit | `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| SegEarth prediction manifest / aggregate | `f1f8f4c7…0ade264` / `c6bd7bbc…d0f1ab` |
| Frozen tile / Omega manifests | `2a0fab56…69b634` / `2a41f582…23a25f` |
| Areas and classes | `[11,15,28,30,34]`; impervious surface, building, low vegetation, tree, car; clutter GT ignored |

The offline evaluator (`32b86f8`) re-hashed every CTP, SegEarth, and Omega
input before opening GT and again after scoring.  It did not import model
runtimes, use the GPU, run inference, modify frozen assets, or write maps.

## B. Primary fixed `Omega_candidate` comparison

Every one of the same 15,893,365 frozen Omega pixels is evaluated for both
methods.  A CTP `255` is an error: it remains in the denominator, adds a FN to
its GT class, and contributes no semantic TP/FP.

| Method | OA | Macro F1 | mIoU |
|---|---:|---:|---:|
| SegEarth-OV | 0.683463 | 0.533888 | 0.407631 |
| CTP-v1 | 0.483556 | 0.505763 | 0.359686 |
| CTP − SegEarth | -0.199908 | -0.028125 | -0.047945 |

This primary strict comparison therefore falls under **SegEarth stronger on
common support**.  It does not establish an overall ranking, because the two
whole methods retain different backbones, proposal/coverage mechanisms, and
fusion pipelines.

## C. Mutual-valid diagnostic

The diagnostic set is `Omega_candidate ∩ (CTP != 255) ∩ (SegEarth predicts a
five-class semantic label)`: 12,846,384 pixels, or 80.8286% of fixed Omega.

| Method | OA | Macro F1 | mIoU |
|---|---:|---:|---:|
| SegEarth-OV | 0.647717 | 0.516967 | 0.388222 |
| CTP-v1 | 0.593741 | 0.569860 | 0.427663 |
| CTP − SegEarth | -0.053975 | +0.052893 | +0.039441 |

The diagnostic changes the Macro-F1/mIoU direction but not OA.  It is not the
primary result because it excludes CTP abstentions.

## D. Per-class result (IoU; fixed Omega strict)

| Class | SegEarth-OV | CTP-v1 | CTP − SegEarth |
|---|---:|---:|---:|
| impervious surface | 0.227917 | 0.413083 | +0.185167 |
| building | 0.725695 | 0.463595 | -0.262100 |
| low vegetation | 0.230039 | 0.075802 | -0.154237 |
| tree | 0.709757 | 0.305610 | -0.404147 |
| car | 0.144746 | 0.540338 | +0.395592 |

CTP is higher for impervious surface and car.  SegEarth is higher for building,
low vegetation, and tree.  These are descriptive class-wise observations, not
a causal explanation of either pipeline.

## E. Per-area strict fixed-Omega result

| Area | SegEarth mIoU | CTP mIoU | Δ mIoU | CTP abstention |
|---:|---:|---:|---:|---:|
| 11 | 0.424232 | 0.369394 | -0.054839 | 16.066% |
| 15 | 0.378133 | 0.280028 | -0.098105 | 21.018% |
| 28 | 0.368815 | 0.410974 | +0.042159 | 17.636% |
| 30 | 0.431850 | 0.416459 | -0.015390 | 17.770% |
| 34 | 0.410132 | 0.343969 | -0.066163 | 20.348% |

CTP has higher strict mIoU on 1/5 areas (area 28).  SegEarth has higher strict
mIoU on 4/5; no individual area reverses the pooled OA direction.

## F. Area-cluster bootstrap

Cluster unit = five test areas; seed = 42; repeats = 5,000.  Values are
CTP-v1 minus SegEarth-OV and are descriptive only.

| Metric | Point | Bootstrap mean | 95% interval |
|---|---:|---:|---:|
| ΔOA | -0.199908 | -0.199036 | [-0.294694, -0.123541] |
| ΔMacro F1 | -0.028125 | -0.026604 | [-0.071539, +0.030647] |
| ΔmIoU | -0.047945 | -0.045975 | [-0.086970, +0.006876] |

With only five independent areas, these intervals must not be described as
large-sample statistical significance tests.

## G. Fairness and abstention accounting

CTP has 2,947,811 `255` abstentions on fixed Omega (18.5474%).  They were
scored as strict errors, not removed.  SegEarth has 108,379 non-five-class
predictions (all clutter here; 0 ignore) and these likewise remain errors on
five-class GT pixels.  Thus CTP does not obtain its primary comparison result
by excluding its own difficult pixels.  Its conditional semantic-only
diagnostic improvement is disclosed together with its 80.8286% coverage.

## H. Interpretation and protocol disclosure

SegEarth-OV uses official OpenAI CLIP ViT-B/16, SimFeatUp, dense logits, its
own prompt list and mean-logit tile stitch.  CTP-v1 uses OpenAI CLIP ViT-B/32,
frozen SAM3 candidates, region semantics and FusionCanvas.  Consequently this
is a **protocol-different external whole-method baseline**, not a controlled
backbone replacement or a partial-support comparison.

For reference only, SegEarth whole-image five-class performance is
OA/Macro-F1/mIoU = 0.560106/0.449311/0.318387.  CTP's own frozen protocol has
different candidate coverage and must not be numerically ranked against that
whole-image row.  The fixed-Omega table above is the sole common-support
semantic comparison.

## I. TGRS decision

1. **SegEarth-OV closure:** Yes for a reproducible Vaihingen external baseline
   and transparent frozen common-support evidence; it is not a controlled CTP
   comparison.
2. **ReAttnCLIP:** Not required to close this minimum external-baseline chain.
   It may add method diversity later, but should not be started automatically.
3. **Potsdam SegEarth-OV:** **NOT READY.** Its frozen CTP geometry/class-mapping
   gate remains unresolved; do not run it from this result alone.
4. **Paper-writing readiness:** Baseline experimentation can move into writing
   with the bounded claim that SegEarth is stronger under strict common-support
   semantics, while CTP targets the distinct partial-support vocabulary-bias
   setting.  Do not claim that CTP outperforms SegEarth overall.

## Output provenance

Canonical evaluation run:

`/data/second_open_vocabulary_storage/outputs/external_baselines/ctp_segearth_common_support/run_20260822T055130Z_0e07becd/`

| Output | SHA-256 |
|---|---|
| Manifest | `12d76420abc9a383f725b216354ac1903d1035d3139cc74b95375e83705bf109` |
| Fixed metrics CSV | `3db529ca27b28c2d450785337b6cec00e8137b006d5ca6cb7b6c3b59d2501a9c` |
| Mutual-valid CSV | `5820102e31c7fba0fdf1e07b8c5a875bdf773dab4128e42e9ff2d51acd6525f5` |
| Per-area CSV | `2876a2119df135bfc770f25ab21cf5e9ca1440eb45c405fa0c19559150d4a2c4` |
| Per-class CSV | `55756cd7f4a8e383667a01bc55099d8b7623fe300b65980c5cca5f332b516e7c` |
| Bootstrap JSON | `617c5785b1865c0cd93cf608e89fc6f3decc897a6836fdf2473729b3d24ea0ff` |
