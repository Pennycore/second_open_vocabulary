# Phase F：P0 原始结果的 image-cluster bootstrap 统计置信区间

运行日期：2026-08-18
对象：原始 P0 blind GT 结果（冻结预测，未重新训练、未重新预测）
方法：以 `image_id` 为 cluster 单位的 bootstrap（非 region-level 独立 bootstrap）
参数：seed = 42，repeats = 5000
输出：`outputs/loveda_blind_gt_v0/run_20260818_001/calibration_bootstrap_summary.json`、`calibration_bootstrap_repeats.npz`

## 1. 流程

1. 使用 P0 冻结预测（`predictions.npz`，SHA-256 `95a2a765…`）与已固化的 region GT（1104 labeled regions，来自 411 heldout images）。
2. 每次 repeat：对 heldout image_id 集合有放回采样（样本数 = 图像数），收集被采样图像的全部 labeled regions，重新计算 Text-only、Visual-only、Text+Visual 的 Accuracy / Macro F1 / Macro IoU。
3. 5000 次重复后，取 Delta 指标（fused − text）的分布，报告 point estimate、bootstrap mean、2.5%/97.5% 分位数 CI。

## 2. 结果

| Delta 指标 | point estimate | bootstrap mean | 95% CI |
|---|---|---|---|
| Δ Macro F1 | +0.1755 | +0.1755 | [+0.1370, +0.2132] |
| Δ Macro IoU | +0.1554 | +0.1554 | [+0.1205, +0.1903] |
| Δ Accuracy | +0.1404 | +0.1405 | [+0.1083, +0.1735] |

单方法 Macro F1 bootstrap 分布：

| 方法 | bootstrap mean | 95% CI |
|---|---|---|
| Text-only | 0.4558 | [0.4240, 0.4870] |
| Visual-only | 0.5780 | [0.5448, 0.6117] |
| Text+Visual | 0.6312 | [0.5968, 0.6654] |

## 3. 结论

- 三个 Delta 的 95% cluster-bootstrap CI 均**不包含 0**（下界 +0.108~+0.137），text+visual 相对 text-only 的提升在图像层面聚类后依然统计显著。
- 原 P0 结果（blind GT、预测先于 GT 读取）的确认性意义保留：ΔMacro F1 点估计 +0.1755，其 95% CI 上界 0.2132 下界 0.1370，即使按最保守解释也远大于 0。
- 本分析未触碰任何预测或 GT；纯统计推断。
