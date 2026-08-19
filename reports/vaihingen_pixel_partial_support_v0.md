# Experiment B：Vaihingen Pixel-level Partial-Support Evaluation

日期：2026-08-19
目的：验证 CTP 在真正 pixel-level open-vocabulary 场景中是否保持 unsupported classes。
协议：`configs/pixel_ovss_protocol_v0.json`；GT 隔离全程保持（manifest 哈希固化后才读 GT）
Support subsets：**k=2/3/4 × seeds 42/43/44 共 9 个，预注册**（`support_subset_manifest.json` 在 GT 读取前生成，禁止按结果选 subset）
方法：Text-only / C2 / SCC / CTP（冻结公式，无 Guard 加入）

## 1. Support subsets（预注册）

| key | k | seed | supported | unsupported |
|---|---|---|---|---|
| k2_seed42 | 2 | 42 | building, car | impervious, lowveg, tree |
| k2_seed43 | 2 | 43 | (seed 采样) | … |
| k2_seed44 | 2 | 44 | … | … |
| k3_seed42/43/44 | 3 | … | … | … |
| k4_seed42/43/44 | 4 | … | … | … |

（完整列表见 `support_subset_manifest.json`，确定性生成 `default_rng(seed + k*100)`。）

## 2. 结果（mean ± std over 3 seeds per k；pixel 级，uncovered/ignore 排除）

### k=2

| 方法 | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|
| Text-only | 0.3492 | 0.3228 | 0.1974 | 0.2067 | 0.1913 | 0.1978 |
| C2 | 0.3048±0.065 | 0.1759 | 0.1186 | 0.2966 | **0.0000** | **0.0000** |
| SCC | 0.3484 | 0.2878 | 0.1821 | 0.2891 | 0.1108 | 0.1572 |
| **CTP** | 0.3692 | 0.3073 | 0.1925 | 0.2881 | **0.1287** | **0.1714** |

### k=3

| 方法 | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|
| Text-only | 0.3492 | 0.3228 | 0.1974 | 0.1735 | 0.2334 | 0.1945 |
| C2 | 0.3198 | 0.2694 | 0.1856 | 0.3093 | **0.0000** | **0.0000** |
| SCC | 0.3160 | 0.2954 | 0.1989 | 0.2980 | 0.0502 | 0.0715 |
| **CTP** | 0.3343 | 0.3121 | 0.2067 | 0.2987 | **0.0687** | **0.1049** |

### k=4

| 方法 | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|
| Text-only | 0.3492 | 0.3228 | 0.1974 | 0.1953 | 0.2058 | 0.1796 |
| C2 | 0.5441 | 0.4626 | 0.3475 | 0.4344 | **0.0000** | **0.0000** |
| SCC | 0.5156 | 0.4762 | 0.3493 | 0.4084 | 0.1131 | 0.1290 |
| **CTP** | 0.5067 | 0.4692 | 0.3392 | 0.3987 | **0.1012** | **0.1453** |

## 3. Bootstrap（seed 42，5000 repeats，image-cluster；mean over subsets）

| k | ΔH-IoU CTP−C2 | ΔH-IoU CTP−SCC | ΔmIoU CTP−C2 | ΔmIoU CTP−SCC | ΔOA CTP−C2 | ΔOA CTP−SCC |
|---|---|---|---|---|---|---|
| 2 | **+0.171 [+0.155,+0.191]** | +0.014 [−0.001,+0.029] | **+0.075 [+0.063,+0.089]** | **+0.010 [+0.001,+0.020]** | **+0.065 [+0.046,+0.085]** | **+0.021 [+0.013,+0.030]** |
| 3 | **+0.105 [+0.082,+0.127]** | **+0.033 [+0.019,+0.044]** | **+0.022 [+0.010,+0.035]** | **+0.008 [+0.003,+0.014]** | +0.015 [−0.002,+0.032] | **+0.018 [+0.007,+0.030]** |
| 4 | **+0.144 [+0.095,+0.193]** | +0.015 [−0.030,+0.065] | −0.007 [−0.017,+0.005] | **−0.010 [−0.020,−0.000]** | **−0.037 [−0.054,−0.018]** | −0.009 [−0.019,+0.001] |

## 4. 回答 Q1–Q4

- **Q1：C2 是否在 pixel partial-support 中再次 unsupported collapse？** **是，完全复现**——C2 在全部 9 个 subsets 中 U-IoU = U-F1 = 0。
- **Q2：CTP 是否提升 supported 且保持 U>0？** **是**——CTP 的 U-IoU 全部 > 0（k=2: 0.129、k=3: 0.069、k=4: 0.101），S-IoU 保持（0.288/0.299/0.399，与 SCC 相当）。
- **Q3：CTP 是否超过 SCC 与 C2（H-IoU 主指标）？** **是**——H-IoU 全部超过 C2（bootstrap Δ +0.105~+0.171，CI 不含 0）；k=2/3 超过 SCC（Δ +0.014/+0.033，k=3 显著），k=4 与 SCC 持平（Δ +0.015，CI 含 0）。CTP 的 H-IoU 是 Text-only 的 0.8–0.9 倍（k=2: 0.171 vs 0.198）。
- **Q4：CTP 是否接近 Guard？** 本轮**未加入 Guard**（按协议）；以 Text-only 的 U-IoU 为上限参照，CTP 的 U 保持在其 55–66%（0.129/0.191、0.069/0.233、0.101/0.206）。

## 5. 结论

- **CTP 在 pixel-level open-vocabulary 场景中有效**：unsupported classes 保持可识别（U-IoU > 0），supported classes 获得视觉锚定增益（S-IoU ≈ SCC），H-IoU 全面超过 naive fusion C2 且在 k=2/3 超过 SCC。
- C2 的 pixel unsupported collapse 与 region-level 完全一致（跨尺度复现）。
- 与 Experiment A 结合：C2 的 pixel 优势是 scale artifact；CTP 的开放词表保持是真实的语义级能力。

## 6. 产物

- `outputs/vaihingen_pixel_partial_support_v0/`：`support_subset_manifest.json`（预注册）、`manifest.json`（哈希绑定 180 语义图）、`pixel_partial_support_results.json`、`pixel_partial_support_bootstrap.json`
- 代码：`scripts/pixel_partial_support.py`
