# Phase I：Phase B 指标一致性审计

审计日期：2026-08-18
对象：`reports/loveda_p1_unsupported_diagnosis_20260818.md` 中的六类 unsupported 统计
结论：**发现真实索引 bug，Phase B 部分数字作废，本文给出修正值**

## 1. 用户指出的矛盾（属实）

road fold 原报告：
- text 本身错误率 B = 0.983 → text-only recall ≈ 0.017 → F1 上界 ≈ 0.033
- 同时报告 text-only unsupported F1 = 0.4720

二者不可能同时成立。water（B=0.989 vs F1=0.3675）、forest（B=0.988 vs F1=0.5232）、agriculture（B=0.949 vs F1=0.7123）同样矛盾。

## 2. 根因：诊断脚本 row 索引 bug

`scripts/run_loveda_p1_diagnosis.py`（旧版）中：

```python
pos = {int(row): i for i, row in enumerate(hold_indices.tolist())}   # row_index -> heldout 位置 (0..1226)
...
r = pos[row_idx]
T_gt = float(text_scores[r, u_idx])          # 用 heldout 位置索引 6000 行全量数组 —— BUG
pred_text_name = _CLASSES[int(text_pred[r])] # 同上 —— BUG
```

`text_scores` / `text_pred` 是**按冻结缓存 row_index 索引的 6000 行全量数组**，而 `pos[row_idx]` 返回的是 heldout 记录在 heldout 列表中的位置（0..1226）。用位置去索引全量数组取到了**错误的行**，导致：
- B（text 错误率）、A、C、destination、margin 全部算错；
- 只有 `f1_stats` 内部用 `text_pred[hold_labeled]`（row_index 索引，正确），所以 text_unsup_f1 是对的。

**修正**：`pos = {row: row}`（恒等映射），且 text 预测改用冻结的 `text_only` 数组（float32 时代产物，与 P0 完全一致，避免 float16 重算 argmax 的边界差异）。修正后 A==TP_text、recall=TP/n_gt、F1 上界全部通过。

## 3. 修正后的六类逐类整数 count 与指标

（输出：`outputs/loveda_blind_gt_v0/run_20260818_001/phaseI_metric_audit.csv`、`p1_diagnosis_summary.csv`）

| class | n_gt | TP_t | FP_t | FN_t | prec_t | rec_t | f1_t | 2r/(1+r) 上界 | TP_f | FP_f | FN_f | prec_f | rec_f | f1_f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| building | 212 | 173 | 219 | 39 | 0.4413 | 0.8160 | 0.5728 | 0.8987 | 189 | 107 | 23 | 0.6385 | 0.8915 | 0.7441 |
| road | 173 | 80 | 87 | 93 | 0.4790 | 0.4624 | 0.4706 | 0.6324 | 117 | 87 | 56 | 0.5735 | 0.6763 | 0.6207 |
| water | 186 | 43 | 5 | 143 | 0.8958 | 0.2312 | 0.3675 | 0.3755 | 90 | 8 | 96 | 0.9184 | 0.4839 | 0.6338 |
| barren | 91 | 7 | 58 | 84 | 0.1077 | 0.0769 | 0.0897 | 0.1429 | 30 | 21 | 61 | 0.5882 | 0.3297 | 0.4225 |
| forest | 167 | 80 | 56 | 87 | 0.5882 | 0.4790 | 0.5281 | 0.6478 | 101 | 54 | 66 | 0.6516 | 0.6048 | 0.6273 |
| agriculture | 275 | 204 | 92 | 71 | 0.6892 | 0.7418 | 0.7145 | 0.8518 | 215 | 85 | 60 | 0.7167 | 0.7818 | 0.7478 |

（t = text-only 全局预测；f = fused 0.5/0.5 全局预测。LOCO fold 下 fusion unsupported TP 全为 0。）

**一致性验证（全部通过）**：
- `recall_t == TP_t / n_gt`：六类全等 ✓
- `f1_t <= 2*recall_t/(1+recall_t)`：六类全等 ✓（water 0.3675 ≤ 0.3755 最紧）
- `A_count == TP_text`（LOCO 每 fold，fusion unsupported TP=0 时）：六 fold 全等 ✓（building 173、road 80、water 43、barren 7、forest 80、agriculture 204）

## 4. 修正后的 LOCO 诊断（每 fold，GT=unsupported 区域）

| fold | n_gt | A: text对但fusion错 | B: text本身错 | C: text top1被抢 | margin mean | text unsup F1 | fusion unsup F1 |
|---|---|---|---|---|---|---|---|
| building | 212 | 173 (0.816) | 39 (0.184) | 173 (0.816) | +0.2817 | 0.5728 | 0.0000 |
| road | 173 | 80 (0.462) | 93 (0.538) | 80 (0.462) | +0.2943 | 0.4706 | 0.0000 |
| water | 186 | 43 (0.231) | 143 (0.769) | 43 (0.231) | +0.3114 | 0.3675 | 0.0000 |
| barren | 91 | 7 (0.077) | 84 (0.923) | 7 (0.077) | +0.3169 | 0.0897 | 0.0000 |
| forest | 167 | 80 (0.479) | 87 (0.521) | 80 (0.479) | +0.2982 | 0.5281 | 0.0000 |
| agriculture | 275 | 204 (0.742) | 71 (0.258) | 204 (0.742) | +0.2933 | 0.7145 | 0.0000 |

## 5. 修正后的结论

- **旧结论"road/water/barren/forest/agriculture 的 text 分支本身弱（B≥92%）"错误**。修正后只有 barren 是 text 本身很弱（recall 0.077）；road 0.462、water 0.231、forest 0.479、agriculture 0.742 的 text-only recall 均非平凡。
- **六个 fold 的 fusion unsupported TP 全为 0（U=0 结论不变）**，且每 fold 满足 `A == TP_text`：即 **text-only 每识别对 1 个 unsupported region，fusion 就翻转 1 个**。unsupported collapse 的本质是：supported 类的 0.5/0.5 融合分数被 V 分量系统性抬升（margin 恒正 +0.28~+0.32，std <0.03），跨尺度 argmax 无条件压制文本尺度上的 unsupported 类。
- building/agriculture 是"text 能力强却被 anchors 压制"的典型（A=0.816/0.742）；barren 是"text 能力弱"的典型（A=0.077）。
- 对后续方法（SCC、Guard）的含义：恢复 unsupported 的关键是**修正 supported 类的分数尺度偏移**（Phase K 的 b(x) centering）或**显式保护 text 决策**（Phase L 的 guard），而不是给 unsupported 类补分数。

## 6. 审计产物

- `outputs/loveda_blind_gt_v0/run_20260818_001/phaseI_metric_audit.csv`
- `outputs/loveda_blind_gt_v0/run_20260818_001/p1_diagnosis_summary.csv`（修正版）
- `outputs/loveda_blind_gt_v0/run_20260818_001/p1_diagnosis_per_region.csv`（修正版）
- 修复脚本：`scripts/run_loveda_p1_diagnosis.py`、新增 `scripts/run_loveda_phaseI_audit.py`

审计完全通过，允许进入 Phase J。
