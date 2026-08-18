# Phase C–F：Calibration 候选评估 + 全支持对比 + 开放词表权衡 + P0 Bootstrap

运行日期：2026-08-18（服务器 luo-W360-E20，2× RTX 2080 Ti）
协议：`configs/loveda_blind_gt_protocol_v0.json`（冻结，未修改）
数据：冻结 P0 预测（`predictions.npz`，SHA-256 `95a2a765…`）+ LoveDA Train GT，**无重新预测、无训练、无网格搜索**
输出（`outputs/loveda_blind_gt_v0/run_20260818_001/`）：`calibration_loco.csv`、`calibration_loco_summary.json`、`calibration_fully_supported.csv`、`calibration_fully_supported_per_class.csv`、`calibration_bootstrap_summary.json`、`calibration_bootstrap_repeats.npz`

## 0. 实验纪律（Phase G 应用）

- LoveDA heldout（411 图 / 1104 labeled regions）自 P1 发现问题起标记为 **development-after-P0**。
- P0 原始方法结果保留其 blind 确认性意义（预测已固化，未触碰）。
- C1/C2 在本数据上的所有结果仅作为 **method-development evidence**，不称 blind test。
- 仅允许预先定义的 C1/C2 两种修改；所有失败结果如实保留。

## 1. 候选定义（Phase C，全部 training-free，alpha 固定 0.5）

- **C0（原始方法 baseline）**：supported `S_c = 0.5·T_c + 0.5·V_c`；unsupported 按当前原始实现。
- **C1（Support-Aware Text Fallback）**：supported `S_c = 0.5·T_c + 0.5·V_c`；unsupported `S_c = T_c`（完整文本）。
- **C0 与 C1 实际一致**：Phase A 审计证明当前原始实现即 C1（unsupported 列被替换为完整 `text_scores` 列），故按用户指令不重复创建，C0≡C1，报告中两者数字相同。
- **C2（Normalized Prototype Calibration）**：supported 类 `p_c = L2(0.5·t_c + 0.5·v_c)`，`S_c = cosine(x, p_c)`；unsupported `p_c = t_c`，`S_c = cosine(x, t_c)`。由单位原型性质，`S_c = (0.5·T_c + 0.5·V_c)/‖0.5·t_c+0.5·v_c‖`，可从冻结产物精确计算，无新参数。

实测原型间关系（冻结 prototypes）：‖0.5t+0.5v‖ ≈ 0.813–0.821（各 class），text–visual prototype cosine ≈ 0.32–0.35。

## 2. Phase D：六轮 LOCO 对比（S / U / H）

S = 六轮 supported macro F1 均值；U = 六轮 unsupported 类 F1 均值；H = 2SU/(S+U)。

| 方法 | S (supported) | U (unsupported) | H | 每 fold 全六类 Macro F1 范围 |
|---|---|---|---|---|
| Text-only baseline | 0.4572 | 0.4572 | 0.4572 | 0.4572（fold 无关） |
| C0 ≡ C1 | 0.6146 | **0.0000** | **0.0000** | 0.4792–0.5575 |
| C2 | 0.6182 | **0.0000** | **0.0000** | 0.4815–0.5575 |

每 fold 明细（C1 / C2 的 S、U、H、全类 Macro F1、Accuracy、Macro IoU）：

| unsupported | C1 S | C1 U | C1 H | C1 macroF1 | C2 S | C2 U | C2 H | C2 macroF1 |
|---|---|---|---|---|---|---|---|---|
| building | 0.5750 | 0.0000 | 0.0000 | 0.4792 | 0.5939 | 0.0000 | 0.0000 | 0.4815 |
| road | 0.6298 | 0.0000 | 0.0000 | 0.5248 | 0.6301 | 0.0000 | 0.0000 | 0.5244 |
| water | 0.6068 | 0.0000 | 0.0000 | 0.5056 | 0.5985 | 0.0000 | 0.0000 | 0.5056 |
| barren | 0.6689 | 0.0000 | 0.0000 | 0.5575 | 0.6744 | 0.0000 | 0.0000 | 0.5575 |
| forest | 0.6293 | 0.0000 | 0.0000 | 0.5244 | 0.6322 | 0.0000 | 0.0000 | 0.5244 |
| agriculture | 0.5778 | 0.0000 | 0.0000 | 0.4815 | 0.5799 | 0.0000 | 0.0000 | 0.4815 |

（完整 accuracy/macro_iou/per-class F1/IoU/confusion matrix 见 `calibration_loco.csv` 与 `evaluate_manifest.json`。）

**结论：C1 与 C2 均无法恢复 unsupported 类（六轮 U=0）。** C2 的 supported 性能略优于 C1（S 0.6182 vs 0.6146），但对 unsupported 无任何恢复。按用户协议，不再堆模块，返回诊断。

## 3. Phase E：fully-supported 对比（六类全有 visual prototype）

| 方法 | Accuracy | Macro F1 | Macro IoU |
|---|---|---|---|
| Text-only | 0.5317 | 0.4572 | 0.3160 |
| Visual-only | 0.5933 | 0.5794 | 0.4200 |
| C0 ≡ C1 (fused) | **0.6721** | 0.6327 | 0.4714 |
| **C2** | 0.6703 | **0.6404** | **0.4784** |

Per-class（C2 vs C0/C1，F1 / IoU）：

| 类别 | C0/C1 F1 | C2 F1 | C0/C1 IoU | C2 IoU |
|---|---|---|---|---|
| building | 0.7441 | 0.7294 | 0.5922 | 0.5731 |
| road | 0.6207 | 0.6411 | 0.4503 | 0.4714 |
| water | 0.6338 | 0.6532 | 0.4639 | 0.4854 |
| barren | 0.4225 | 0.4572 | 0.2680 | 0.2973 |
| forest | 0.6273 | 0.6421 | 0.4570 | 0.4721 |
| agriculture | 0.7478 | 0.7196 | 0.5970 | 0.5710 |

**结论：C2 不牺牲 fully-supported 性能**——Macro F1 +0.0077（0.6327→0.6404）、Macro IoU +0.0070，Accuracy 略降 0.0018（0.6721→0.6703）。即开放词表修复（如果有效）不以 P0 增益为代价；但 C2 本身未能修复开放词表。

## 4. Phase F：P0 image-cluster bootstrap（seed 42，5000 repeats，cluster=image_id）

对原始 P0 冻结预测做 image-level cluster bootstrap（有放回采样 heldout image_id，聚合其 labeled regions），计算三种方法及 Delta：

| Delta | point estimate | bootstrap mean | 95% CI |
|---|---|---|---|
| Δ Macro F1 (fused − text) | +0.1755 | +0.1755 | **[+0.1370, +0.2132]** |
| Δ Macro IoU (fused − text) | +0.1554 | +0.1554 | **[+0.1205, +0.1903]** |
| Δ Accuracy (fused − text) | +0.1404 | +0.1405 | **[+0.1083, +0.1735]** |

单方法 bootstrap 分布（Macro F1）：text-only 0.4558 [0.4240, 0.4870]；visual-only 0.5780 [0.5448, 0.6117]；fused 0.6312 [0.5968, 0.6654]。

**结论：P0 的 text+visual 提升在 image-cluster bootstrap 下统计显著**（三个 Delta 的 95% CI 均不包含 0，下界 +0.108~+0.137），确认性意义保留。

## 5. 综合回答（核心问题）

> Can weak visual prototypes calibrate remote-sensing CLIP representations while preserving unsupported text-defined categories?

- **校准增益**：成立（P0 blind + bootstrap 显著；C2 在全支持下 Macro F1/IoU 略优于 C0/C1 且不牺牲性能）。
- **开放词表保持**：**不成立**。C1（文本回退）与 C2（原型归一化）两个预定义 training-free 候选在六轮 LOCO 中 unsupported F1 均为 0。根因（Phase B）：supported 类的融合分数被 V 分量系统性抬升（margin 恒正 0.28–0.35），unsupported 类停留在文本尺度，跨尺度 argmax 必然压制后者；C1 只修 unsupported 自身尺度，C2 归一化后 supported 仍获 V boost。
- **结论**：当前 0.5/0.5 静态融合框架内，两个预设候选都无法同时满足"保留锚定增益 + 保持开放词表"。按用户协议，不继续堆模块；是否进入 query-adaptive / text-guarded calibration 由用户决定。

## 6. 文件清单

- 报告：本文件 + `loveda_p1_score_audit_20260818.md` + `loveda_p1_unsupported_diagnosis_20260818.md`
- 脚本：`scripts/run_loveda_calibration.py`、`scripts/run_loveda_p1_diagnosis.py`
- CSV/JSON：`p1_diagnosis_summary.csv`、`p1_diagnosis_per_region.csv`、`calibration_loco.csv`、`calibration_loco_summary.json`、`calibration_fully_supported.csv`、`calibration_fully_supported_per_class.csv`、`calibration_bootstrap_summary.json`、`calibration_bootstrap_repeats.npz`
