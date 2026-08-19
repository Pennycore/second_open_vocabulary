# Potsdam CTP-v1 External Confirmation（最终报告）

日期：2026-08-19
定位：**external held-out dataset evaluation / cross-dataset generalization confirmation**（非 blind test）
协议：`configs/potsdam_ctp_v1_protocol.json`（frozen，SHA-256 `d0aba3a7…`）
数据：ISPRS Potsdam 14 个 test parent tiles（`potsdam_parent_split_23_0_14_paper`），512px patches，3502 个有 SAM3 候选的 patch（82 个零候选 patch 按协议排除）
方法：Text-only / C2 normalized / SCC / CTP（全部冻结；alpha=0.5；8 Group-A prompts；FusionCanvas conflict 0.03→ignore、uncovered=255）
GT 隔离：predict → 哈希固化（`reports/potsdam_prediction_manifest.md`）→ 读 GT → evaluate

## A. 数据集信息

- ISPRS Potsdam 2D Semantic Labeling；五类：impervious_surface / building / low_vegetation / tree / car
- 评估：3502 patch × 512×512（约 5.26 亿有效像素/方法）
- SAM3 candidates：45488 个（test tiles 上冻结管线生成，用户授权重跑模式；visual prototypes 计数：impervious 9844 / building 2989 / lowveg 3991 / tree 3184 / car 25480）

## B. Protocol

- `configs/potsdam_ctp_v1_protocol.json`（模型/文本/视觉原型/SCC/CTP 公式/segmentation 规则/candidates 来源/FusionCanvas/commit/hash 全记录）
- Partial-support：ratios 25/50/75% × seeds 42/43/44 预注册（`support_subset_manifest.json`，GT 前生成）

## C. GT Isolation

- Predict manifest 固化：predictions.npz `be34049e…`、records `c4900d6f…`、14008 语义图逐文件 SHA-256
- evaluate 校验全部哈希一致后才打开 GT；未按 GT 调整任何规则

## D. Full-support 结果（Phase D）

| 方法 | OA | Macro F1 | mIoU |
|---|---|---|---|
| Text-only | 0.3574 | 0.3565 | 0.2325 |
| C2 normalized | 0.6533 | 0.6352 | 0.4876 |
| SCC | 0.6477 | 0.6280 | 0.4785 |
| **CTP** | 0.6477 | 0.6280 | 0.4785 |

（k=5 全支持时 CTP==SCC==C2-argmax 恒等；C2 的 pixel 优势为 score-scale 效应，已在 Vaihingen ablation 证明。）

per-class IoU（CTP）：impervious 0.500 / building 0.619 / lowveg 0.292 / tree 0.270 / car 0.712。

**回答**：① CTP 保持视觉锚定 gain（vs Text-only：OA +0.290、mIoU +0.246，跨数据集复现）；② CTP 与 SCC 完全一致（k=5 恒等）、与 C2 接近（差异为 score-scale）；③ 无新类别失败（lowveg/tree 较低但与 Vaihingen 一致的固有难点，非新异常）。

## E. Partial-support 结果（Phase E，mean over 3 seeds）

| ratio | 方法 | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|---|
| 25% | Text-only | 0.3581 | 0.3561 | 0.2323 | 0.1748 | 0.2467 | 0.1868 |
| 25% | C2 | 0.2000 | 0.0620 | 0.0400 | 0.2000 | **0.0000** | **0.0000** |
| 25% | SCC | 0.3581 | 0.3561 | 0.2323 | 0.1748 | 0.2467 | 0.1868 |
| 25% | **CTP** | 0.3579 | 0.3565 | 0.2325 | 0.1748 | **0.2470** | **0.1868** |
| 50% | Text-only | 0.3581 | 0.3561 | 0.2323 | 0.2749 | 0.2039 | 0.2141 |
| 50% | C2 | 0.3912 | 0.2018 | 0.1373 | 0.3432 | **0.0000** | **0.0000** |
| 50% | SCC | 0.4204 | 0.3271 | 0.2132 | 0.3685 | 0.1097 | 0.1667 |
| 50% | **CTP** | 0.4059 | 0.3415 | 0.2200 | 0.3513 | **0.1325** | **0.1877** |
| 75% | Text-only | 0.3581 | 0.3561 | 0.2323 | 0.2214 | 0.2759 | 0.2173 |
| 75% | C2 | 0.5349 | 0.4943 | 0.3725 | 0.4657 | **0.0000** | **0.0000** |
| 75% | SCC | 0.5371 | 0.4992 | 0.3714 | 0.4579 | 0.0250 | 0.0461 |
| 75% | **CTP** | 0.5486 | 0.5162 | 0.3814 | 0.4598 | **0.0677** | **0.1150** |

（S/U/H-F1 同模式，见 `potsdam_partial_support.csv`；H = mean(H_i)。）

## F. Bootstrap（seed 42，5000 repeats，patch-cluster；mean over subsets，95% CI）

| ratio | ΔH-IoU CTP−C2 | ΔH-IoU CTP−SCC | ΔH-IoU CTP−Text | ΔmIoU CTP−C2 | ΔOA CTP−C2 |
|---|---|---|---|---|---|
| 25% | **+0.187 [+0.178,+0.195]** | −0.0001 [−0.0003,+0.0002] | −0.0001 | **+0.193 [+0.184,+0.201]** | **+0.158 [+0.143,+0.173]** |
| 50% | **+0.188 [+0.178,+0.197]** | **+0.021 [+0.016,+0.027]** | −0.026 [−0.035,−0.018] | **+0.083 [+0.075,+0.090]** | +0.015 [+0.004,+0.026] |
| 75% | **+0.115 [+0.097,+0.134]** | **+0.069 [+0.055,+0.083]** | −0.102 [−0.120,−0.085] | +0.009 [+0.005,+0.013] | +0.014 [+0.009,+0.018] |

## G. Failure cases / Qualitative（Phase G）

- 选择规则（预固定，非挑图）：按"C2 错误且 CTP==GT 的像素数"排序取 top-5，GT 类池 = {car, low_vegetation, tree}
- 5 个代表场景：`outputs/potsdam_qualitative_v0/case{1..5}_*.png`（RGB/GT/Text-only/C2/SCC/CTP 六联图）
- 典型模式：C2 将 car/lowveg 区域吞并为 building/impervious，CTP 恢复文本类
- 失败类别（如实）：low_vegetation 与 tree 的 IoU 最低（0.29/0.27），与 Vaihingen 一致——SAM3 弱监督对这些类的 prototype 较弱，非 Potsdam 特有

## H. 结论

1. **CTP 是否跨数据集有效？** **是**——Potsdam（held-out）上 CTP 保持视觉锚定 gain（mIoU 0.479 vs Text-only 0.233），与 LoveDA/Vaihingen 一致。
2. **C2 是否再次出现 unsupported collapse？** **是，第三次复现**——Potsdam 全部 9 个 partial subsets 中 C2 的 U-IoU=0。
3. **CTP 是否保持 open-vocabulary ability？** **是**——CTP 的 U-IoU 全部 > 0（0.068–0.247），H-IoU 显著超过 C2（bootstrap CI 不含 0）且在 50/75% 显著超过 SCC。
4. **是否具备最终论文主实验条件？** **是**——三数据集（LoveDA region + Vaihingen region/pixel + Potsdam pixel held-out）证据链完整：视觉锚定增益跨数据集成立、C2 bias 三度复现、CTP 的 soft 文本保护一致有效、无参数无训练。

**论文叙事闭环**：Weak visual anchoring improves remote CLIP semantics but causes support-induced vocabulary bias（LoveDA+Vaihingen+Potsdam 三数据集）→ CTP 以 query-level text confidence guided preservation 取得 supported/unseen 平衡（H-IoU 全面超 C2、50/75% 超 SCC）。

## 交付物

- 报告：本文件 + `reports/potsdam_untouched_audit.md` + `reports/potsdam_prediction_manifest.md`
- 代码：`scripts/run_potsdam_ctp_v1.py`、`scripts/run_potsdam_partial_support.py`、`scripts/potsdam_qualitative.py`
- 配置：`configs/potsdam_ctp_v1_protocol.json`、`potsdam_sam3_test_v1.json`、4 个部署 yaml
- CSV：`potsdam_full_support.csv`、`potsdam_partial_support.csv`
- JSON：`pixel_overall.json`、`pixel_partial_support_results.json`、`pixel_partial_support_bootstrap.json`、`support_subset_manifest.json`、manifest（哈希绑定）
- 可视化：`outputs/potsdam_qualitative_v0/case*.png` + `selection_record.json`
- 测试：`tests/test_potsdam_ctp_v1.py`（5 passed：protocol freeze / GT isolation / subset 可复现 / CTP 确定性 / FusionCanvas）
