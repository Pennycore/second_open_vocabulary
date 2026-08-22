# CTP-v1.1 development-only tuning：Phase A 资产与隔离审计

日期：2026-08-22
范围：仅为 `CTP-v1.1-tuning-candidate` 的 development-only 参数选择建立输入资产、代码隔离和停止门禁。本报告不包含任何 registered-test 预测、GT 内容、指标或比较数字。

## 1. 结论

**资产审计：GO。** 现有正式 Vaihingen 划分提供了固定的 11 个 development areas；每个 area 都有一张全幅 RGB、一个官方 GT 文件和一个冻结的 SAM3 candidate cache。无需重新运行 SAM3。

**grid-search 执行门禁：暂为 NO-GO。** 目前没有 development-only 的 OpenAI CLIP region-feature/score cache，且现有 pixel runner 明确绑定 registered test areas、固定 `alpha=0.5` 和固定 FusionCanvas conflict margin `0.03`。在生成一次 dev feature cache 和实现带 test-id hard-block 的 dev-only runner 前，不能启动 grid。

本结论来自路径、文件名、文件数量、哈希和源码静态审查；没有打开任何 registered-test map/GT/metric，也没有运行模型或 GPU 任务。

## 2. 划分冻结

完整的官方 GT area inventory 为：

`[1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]`。

项目已有正式 split（`configs/vaihingen_scc_protocol_v0.json` 与 `src/ov_probe/vaihingen_blind.py`）如下：

| Role | Area IDs | Count |
|---|---|---:|
| Registered test（永久排除） | `[11, 15, 28, 30, 34]` | 5 |
| Development pool（沿用既有 train split） | `[1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]` | 11 |

未按性能、场景难度或类别覆盖对上述 11 个 development areas 进行二次筛选。调参只允许读取和写入该 development pool 的资产；Potsdam、RemoteCLIP、SegEarth 及所有 registered-test artifacts 均不属于本轮输入。

## 3. Development asset inventory

基准路径：

- 全幅 RGB：`/home/zhongsz/second_open_vocabulary/inputs/vaihingen/images/`
- 官方 development GT（仅供未来 evaluate 阶段）：`/home/zhongsz/second_open_vocabulary/inputs/vaihingen/labels/`
- 冻结 candidates：`/home/zhongsz/second_open_vocabulary/outputs/proposals/vaihingen_sam3_v0/run_20260820T145952Z_5beba872/candidates/`
- candidate-run manifest SHA-256：`072e21b28c3243afccd8ab85d2aaa96d19ed283f836743291a9d86288515da6f`

候选目录有 16 个 `.npz` / `.json` 配对，覆盖全部官方 GT areas；development subset 的 11 个 `.npz` 均存在。候选由先前的 frozen SAM3 pipeline 产生，弱监督来源是 all-positive image-level assumption 和 candidate 内的 SAM3 source class id；本轮不得重新生成、筛选或以 GT 修正它们。

| Dev area | Full RGB | 896px patch pairs | Candidate cache | Existing OpenAI region feature / score cache | GT for future dev evaluation | RGB SHA-256 | Candidate SHA-256 |
|---:|---:|---:|---|---|---|---|---|
| 1 | 1 | 54 | yes | **no** | yes, locked until evaluate | `656e4f35…cdef0e3d` | `8ca897df…4606e8d2` |
| 3 | 1 | 60 | yes | **no** | yes, locked until evaluate | `d1687208…b0fb7821e` | `4fecfab4…b21115cb` |
| 5 | 1 | 48 | yes | **no** | yes, locked until evaluate | `14d2ce0b…679295fa7` | `94bdf139…a6b063f77` |
| 7 | 1 | 48 | yes | **no** | yes, locked until evaluate | `e0e2eda8…3709fe572` | `19c33db0…a4b6c42bc5` |
| 13 | 1 | 80 | yes | **no** | yes, locked until evaluate | `82279beb…0b5a7b8fe` | `ea639182…2b9a4a8c` |
| 17 | 1 | 32 | yes | **no** | yes, locked until evaluate | `42eb3ad9…511d089bb` | `635d8eb4…01502cdfd` |
| 21 | 1 | 48 | yes | **no** | yes, locked until evaluate | `701185b4…3149d3cac` | `044a0848…38bf0d2be` |
| 23 | 1 | 48 | yes | **no** | yes, locked until evaluate | `05293bf0…841e373359` | `231ecf03…e03843e20` |
| 26 | 1 | 50 | yes | **no** | yes, locked until evaluate | `01bd8467…6f62ed2f7` | `d8d1c461…814e23a10` |
| 32 | 1 | 48 | yes | **no** | yes, locked until evaluate | `d52bd258…bf5362946` | `06d837a1…4e5672ba2` |
| 37 | 1 | 36 | yes | **no** | yes, locked until evaluate | `681887e9…216be9907` | `ab5d8c67…ae599a3e2` |

Patch counts are a file-existence inventory for the incoming 896px paired image/label layout, not an evaluation split and not a replacement for the frozen full-area candidate geometry. The full-area RGB/candidate pair is the only valid input geometry for the planned FusionCanvas workflow.

No development-specific OpenAI CLIP feature, text-embedding, anchored-score, or prototype cache was found in the active project storage. Existing historical prediction/score outputs are registered-test assets and must not be read or reused by this tuning process.

## 4. Permitted future feature/prototype construction

One new, one-time development-only feature cache is necessary. It may be generated only after a manifest records the 11 IDs and the input hashes above. Required frozen inputs are:

| Input | Required frozen setting |
|---|---|
| Region geometry | The 11 listed candidate `.npz` files; no SAM3 invocation |
| Image source | Matching full-area RGB files in `inputs/vaihingen/images/` |
| Crop/preprocessing | Existing `ov_probe.vaihingen_blind._crop_view`: context ratio `0.25`, minimum crop `48`, background retain `0.25`; OpenCLIP transform of `ViT-B-32-quickgelu` |
| Encoder | OpenAI CLIP ViT-B/32 quick-GELU, OpenCLIP `3.3.0`, 512-D, checkpoint SHA-256 `9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e` |
| Feature rule | L2-normalized float32 region features; stable row mapping `(area_id, candidate_index)` |
| Text | Existing five-class Vaihingen vocabulary and frozen eight Group-A templates; no prompt edits |
| Visual prototypes | Development candidate features grouped only by frozen SAM3 source class id, L2-per-row → arithmetic mean → L2; no GT, candidate filtering, or weak-label revision |

Relevant frozen code/protocol hashes in the local repository at this audit:

- `configs/vaihingen_scc_protocol_v0.json`: `e4079363232dc18ccc439eb0080cdb1a471d6f9cd49a76a0992965bdcc21064d`
- `configs/pixel_ovss_protocol_v0.json`: `8fd0f020bd13ede994c99320ab0b025b68b4b825e3ec1c07eb7c06c593d5c358`
- `configs/ctp_v1_frozen.json`: `788f1962d497022fbd5cacd7b63eaedddecd0343104aa726ee80afcdf1b37430`
- `src/ov_probe/vaihingen_blind.py`: `603ae6356e5c79c90d53b7a50cc8416fd3d145abb4ef1dadf5eba9173869365a`

The cache generation phase must have `label_dir = null`, must reject all non-development area IDs, and must write its manifest before any development GT is read.

## 5. Development GT boundary

Development GT exists for all 11 development areas, but is locked during candidate loading, feature generation, text/prototype construction, scoring, FusionCanvas rendering and prediction-manifest creation.

After prediction/config/hash verification, it may be used **only** for the pre-registered development metrics: OA, Macro F1, mIoU, per-class IoU, and S/U/H-F1 / S/U/H-IoU for the complete fixed grid. It must never be used to construct visual prototypes, alter SAM3 candidates, amend weak labels, change prompts, select crops, or introduce any parameter outside `alpha` and `tau_conflict`.

## 6. Static runner/FusionCanvas audit

The current implementation is not safe to invoke for development tuning as-is.

1. `scripts/run_pixel_ovss_vaihingen.py` imports `TEST_AREAS`, filters records with `split == "test"`, and writes a manifest field named `test_areas`. It has no `--areas` input and no rejection guard against registered test IDs. This runner is test-bound, not development-capable.
2. Its prediction path loads precomputed `anchored_scores`; hence it cannot recompute anchors for a new scalar alpha. `method_score_matrices` receives `anchored` as an already constructed input.
3. `FusionCanvas.conflict_margin` defaults to module constant `0.03`; `assemble_semantic_map` creates `FusionCanvas(height, width)` without an externally supplied threshold. A `tau_conflict` grid therefore cannot be represented without a narrowly scoped parameter pass-through.
4. `scripts/pixel_partial_support.py` is likewise registered-test bound. It cannot be reused for development evaluation.

Required minimal implementation boundary before Phase B:

- a separate dev-only runner/output namespace, not an edit that changes historical CTP-v1 artifacts;
- a fixed development manifest and explicit fail-closed rejection of `{11,15,28,30,34}` in inputs, records, output names and label paths;
- `alpha` passed only into normalized anchor construction `Normalize((1-alpha)t + alpha v)`;
- `tau_conflict` passed only into `FusionCanvas` construction; no other decision rule changes;
- prediction phase without GT paths, immutable manifest/hash seal, then a separate dev-evaluate phase;
- no code path that scans, opens, or reports registered-test outputs while selecting a configuration.

`src/ov_probe/pixel_ovss.py` SHA-256 is `9f85c9443f4bdab6164fb872c40853dbb9532a9eee4dfa1389a3940d5c67c82f`; `scripts/run_pixel_ovss_vaihingen.py` SHA-256 is `13a153a21b0ab53b1d2469695806207dec4b877d177d96bd31c2c6febd4f9745`.

## 7. Known-test disclosure and non-leakage record

The research record already contains historical Vaihingen, Potsdam, RemoteCLIP and SegEarth test work; the researcher therefore has prior awareness that such test results exist. This development-only process must be described as **post-hoc development-only performance optimization with frozen test-evaluation rules**, not prospectively blind tuning.

For this Phase-A audit specifically, no registered-test RGB/map, semantic prediction, GT, metric JSON/CSV, common-support artifact, or external-baseline result was opened, parsed, hashed, or used as a selection criterion. No test numerical value is recorded here.

## 8. Gate and minimum next artifacts

| Gate | Status | Reason |
|---|---|---|
| Fixed development split | GO | Existing official 11/5 split is explicit and image-disjoint. |
| Frozen candidate reuse | GO | All 11 development caches exist; SAM3 rerun is unnecessary and prohibited. |
| Development GT availability | GO, locked | All 11 labels exist; access deferred until prediction seal. |
| Reusable development region features | NO | No active dev-only cache was found. |
| Existing runner safe for dev-only grid | NO | Test-bound selection and fixed alpha/tau prevent safe use. |
| Phase-B grid execution | **NO-GO until the two items below are completed** | Prevents accidental registered-test access or implicit method changes. |

Minimum files to create in the next approved phase (none created by this audit):

1. a development ID/input manifest with the 11 IDs and full SHA-256 values;
2. one frozen preprocessing-compatible dev feature cache plus stable row manifest;
3. a new dev-only runner/config that hard-rejects registered test IDs and exposes only the two pre-registered scalars; and
4. separate prediction and development-evaluation manifests beneath a new `outputs/ctp_tuning_v1_1/` namespace.

No grid, model run, prediction, GT read, or remote write was performed in Phase A.

## 9. Phase-B implementation status and asset-binding blocker

Local-only Phase-B scaffolding now exists in `configs/ctp_v1_1_tuning_protocol.json`, `configs/ctp_v1_1_tuning_template.yaml`, `src/ov_probe/ctp_v11_tuning.py`, and `scripts/run_ctp_v11_dev_tuning.py`. It is intentionally a metadata-only preflight at this point: it validates the exact IDs/grid/protocol, rejects test/external path markers, and does not load a model, cache, candidate, image, map, or GT.

Static provenance is sufficient to bind the intended weak-supervision/text procedure: `src/ov_probe/vaihingen_blind.py` defines the same five-class Group-A text construction, crop/preprocess rule and SAM3-source-label visual-prototype construction, while `configs/vaihingen_scc_protocol_v0.json` binds the required OpenAI checkpoint hash. The candidate files are present as recorded above and carry the frozen SAM3 source class assignment needed for future weak prototypes.

The exact local checkpoint has now been bound at `C:\Users\28457\Desktop\exp_code\openai_clip_ckpt\open_clip_pytorch_model.bin`: size `605,225,782` bytes and SHA-256 `9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e`. The historical YAML still names an old `/home/undergr/...` location, which is not evidence that the current 3090 project can read the file. A future deployment configuration must therefore supply a readable server-side copy with that same hash; this local Phase-B implementation did not transfer or access it remotely.

The other required runtime asset is the new manifest-bound development feature cache; no such cache exists yet. Neither missing asset is synthesized or downloaded by the scaffolding.
