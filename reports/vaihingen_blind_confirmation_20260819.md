# Vaihingen 外部 Blind Confirmation：SCC-v1 跨数据集验证

运行日期：2026-08-19（服务器 luo-W360-E20，2× RTX 2080 Ti）
数据集：ISPRS Vaihingen 2D Semantic Labeling（官方分发，**untouched_external_confirmation**，审计见 `reports/vaihingen_untouched_audit_20260818.md`）
协议：`configs/vaihingen_scc_protocol_v0.json`（SHA-256 `e4079363…`，frozen_pre_result）
预测固化：`reports/vaihingen_blind_prediction_manifest.md`（predictions.npz `96be715f…`、subset_predictions.npz `c5358675…`、subset_manifest `ebc2f477…`）
输出：`outputs/vaihingen_blind_scc_v1/`

## 0. Blind 纪律执行记录

1. 官方 16 个 GT area 上传服务器，构建 flat 布局（`Vaihingen_main_v1/`，train 11 / test 5 areas）。
2. SAM3 弱监督（用户授权重跑）：all-positive image-level 假设（无 GT 派生弱标签），`--save-candidates` 生成 20470 个候选区域（train 13038 / test 7432），弱标签 = SAM3 class id。
3. Predict 阶段：OpenAI CLIP 编码全部区域（train+test），train-area SAM3 weak labels 构建 visual prototypes（building 1807 / car 4366 / impervious_surface 983 / low_vegetation 48 / tree 5834），text prototypes 用 Vaihingen 词汇表 + 8 个 Group-A 模板。
4. 全部预测 + 32 个 support subsets（SCC/Guard）+ 配置哈希固化，**之后才读取 GT**。
5. Evaluate 校验全部哈希一致后解锁 GT，region GT = 候选 mask 内 5 类颜色多数投票（clutter 为 ignore）。

## 1. Overall（fully-supported，k=5，test 7432 regions 全部 labeled）

| 方法 | OA | Macro F1 | mIoU |
|---|---|---|---|
| Text-only | 0.4059 | 0.3453 | 0.2358 |
| Visual-only | 0.8293 | 0.6437 | 0.5437 |
| C2 | 0.8416 | 0.6537 | 0.5547 |
| **SCC（k=5）** | 0.8416 | 0.6537 | 0.5547 |
| Text-Top1 Guard（k=5） | 0.8416 | 0.6537 | 0.5547 |

（k=5 全支持时 SCC ≡ C2 ≡ Guard——SCC 的 k=6 恒等性质，delta=0 已由 bootstrap 确认。）

**Vaihingen 上视觉锚定的增益比 LoveDA 更大**：OA +0.436、Macro F1 +0.308、mIoU +0.319（vs LoveDA 的 +0.140/+0.176/+0.155）。SCC-v1 冻结公式跨数据集直接生效，无任何调整。

## 2. Partial-support（k=1..4，mean over subsets；H 聚合 = mean(H_i)）

| k | 方法 | S-F1 | U-F1 | H-F1 | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|---|
| 1 | text-only | 0.3453 | 0.3453 | 0.2842 | 0.2358 | 0.2358 | 0.1794 |
| 1 | C2 | 0.2992 | **0.0000** | 0.0000 | 0.2265 | **0.0000** | 0.0000 |
| 1 | SCC | 0.3453 | 0.3453 | 0.2842 | 0.2358 | 0.2358 | 0.1794 |
| 1 | Guard | 0.4162 | 0.4060 | 0.3575 | 0.3017 | 0.2938 | 0.2404 |
| 2 | text-only | 0.3453 | 0.3453 | 0.3038 | 0.2358 | 0.2358 | 0.1948 |
| 2 | C2 | 0.4544 | **0.0000** | 0.0000 | 0.3651 | **0.0000** | 0.0000 |
| 2 | SCC | 0.5551 | 0.2944 | 0.3548 | 0.4372 | 0.2120 | 0.2532 |
| 2 | Guard | 0.4849 | 0.4613 | 0.4331 | 0.3672 | 0.3509 | 0.3138 |
| 3 | text-only | 0.3453 | 0.3453 | 0.3038 | 0.2358 | 0.2358 | 0.1948 |
| 3 | C2 | 0.5598 | **0.0000** | 0.0000 | 0.4624 | **0.0000** | 0.0000 |
| 3 | SCC | 0.5968 | 0.2365 | 0.2832 | 0.4857 | 0.1817 | 0.2092 |
| 3 | Guard | 0.5471 | 0.5117 | 0.4861 | 0.4311 | 0.4072 | 0.3723 |
| 4 | text-only | 0.3453 | 0.3453 | 0.2842 | 0.2358 | 0.2358 | 0.1794 |
| 4 | C2 | 0.6186 | **0.0000** | 0.0000 | 0.5208 | **0.0000** | 0.0000 |
| 4 | SCC | 0.6297 | 0.2073 | 0.2025 | 0.5250 | 0.1674 | 0.1474 |
| 4 | Guard | 0.6033 | 0.5573 | 0.4960 | 0.4936 | 0.4627 | 0.3900 |

注：k=1 时 SCC 数学上恒等于 text-only（唯一 supported 类的 centering b(x)=A_s−T_s 使 S_s=T_s，与 LoveDA 观察一致）。

## 3. 外部确认核心科学问题（协议 Q1–Q4）

### Q1：C2 在新数据集上是否仍产生 supported-class bias？

**是，且完全复现**：C2 在 Vaihingen 全部 k=1..4 subsets 中 unsupported U-F1 = U-IoU = 0。这与 LoveDA 完全一致——**supported-class bias 是 0.5/0.5 late fusion 的跨数据集固有属性**，与数据域无关。

### Q2：SCC 是否仍能提高 supported 并保持 meaningful unsupported？

**部分成立，但强度弱于 LoveDA**：
- supported：SCC 的 S-F1 随 k 单调上升（k=2: 0.555 → k=4: 0.630），高于 text-only（0.345）——提高成立。
- unsupported：SCC 的 U-F1 为 0.207–0.345（k=2..4），**非零但明显低于 text-only 的 0.345 水平，且随 k 增大而下降**（k=4 时 0.207）。在 LoveDA 上 SCC 的 U 稳定在 0.45–0.46（≈ text-only），Vaihingen 上未完全保持。

### Q3：SCC 的 H 是否稳定超过 Text-only 和 C2？

**部分**：H-F1 vs C2 全部为正（k=2: +0.117、k=3: +0.096、k=4: +0.065，95% CI 不含 0）；H-F1 vs Text-only 在 k=2 约持平（−0.0003，CI 含 0）、k=3 −0.022、k=4 −0.046（CI 不含 0，显著为负）。**Guard 的 H 全面最高**（k=2..4: 0.433/0.486/0.496），SCC 的 H（0.355/0.283/0.203）低于 Guard 且随 k 下降。

### Q4：support-induced common shift 是否跨数据集复现？

**机制复现但幅度/方向有差异**：Vaihingen 上 SCC 的 U 下降说明 `b(x)=mean_{supported}[A_c−T_c]` 的 centering 在 Vaihingen 上**过度**——supported 类被压低后，unsupported 类（保持 T_c）反而相对上升有限，而 supported 类内 competition 变化导致 unsupported 类 recall 受损。不修改 SCC-v1（冻结纪律），如实记录。

## 4. image-cluster bootstrap（seed 42，5000 repeats，cluster=area/image_id）

### k=5 全支持（SCC ≡ C2，delta=0 为数学恒等）

| Delta | point | bootstrap mean | 95% CI |
|---|---|---|---|
| SCC vs Text-only ΔMacro F1 | +0.308 | +0.309 | [+0.296, +0.321] |
| SCC vs Text-only ΔmIoU | +0.319 | +0.319 | [+0.305, +0.335] |
| SCC vs Text-only ΔOA | +0.436 | +0.436 | [+0.404, +0.475] |
| SCC vs C2（全部） | 0.0000 | 0.0000 | [0, 0] |

### partial-support（k=2..4，mean over subsets of per-subset bootstrap）

| k | ΔMacroF1 SCC−C2 | ΔMacroF1 SCC−Text | ΔH-F1 SCC−C2 | ΔH-F1 SCC−Text |
|---|---|---|---|---|
| 2 | +0.048 [+0.035,+0.060] | +0.017 [+0.006,+0.031] | +0.117 [+0.100,+0.133] | −0.0003 [−0.017,+0.017] |
| 3 | +0.021 [+0.013,+0.029] | +0.023 [+0.010,+0.039] | +0.096 [+0.079,+0.112] | −0.022 [−0.040,−0.002] |
| 4 | +0.007 [+0.003,+0.011] | +0.037 [+0.023,+0.054] | +0.065 [+0.052,+0.078] | −0.046 [−0.064,−0.024] |

（负结果如实保留：SCC 的 H-F1 vs Text-only 在 k=3/4 显著为负。）

## 5. 结论（协议 A–J 汇报格式）

- **A**：LoveDA 冻结指标见 `reports/scc_v1_freeze_record_20260818.md`（k=6：SCC OA 0.6703 / Macro F1 0.6404 / mIoU 0.4784；k=1..5 S/U/H 齐全）。
- **B**：SCC-v1 frozen commit `f5d7d91`，配置 `configs/scc_v1_frozen.json`（协议 SHA-256 `e4079363…`）。
- **C**：**Vaihingen 真正 untouched**（三项检查全否；官方 16 area GT 为评估依据；remote/ 预处理标签来源存疑未采用）。
- **D**：外部 overall：Text-only 0.406/0.345/0.236；Visual-only 0.829/0.644/0.544；C2 0.842/0.654/0.555；SCC 0.842/0.654/0.555。
- **E**：partial-support S/U/H-F1 与 S/U/H-IoU 见上表（k=1..4）。
- **F**：**C2 在 Vaihingen 上再次出现 unsupported collapse（U=0 全部 k）——跨数据集复现。**
- **G**：**SCC 部分恢复 unsupported（U 0.207–0.345 > 0），但未达到 text-only 水平且随 coverage 下降；Guard 完全恢复（U 0.406–0.557 ≈ text-only）。**
- **H**：95% cluster-bootstrap CI 见上表；SCC vs C2 的 Macro F1/H-F1 改善显著（CI 不含 0），SCC vs Text-only 的 H-F1 在 k=3/4 显著为负。
- **I**：**失败/异常**：(1) low_vegetation 类 SAM3 候选极少（train 48 个），其 visual prototype 弱；(2) SCC 在 Vaihingen 的 unsupported 保持弱于 LoveDA——support-shift 的 centering 跨数据集不完全一致；(3) k=1 时 SCC≡text-only 为数学性质。
- **J**：**进入 pixel-level OVSS 主实验的证据**：视觉锚定（C2/SCC/Guard 全支持）在 Vaihingen 上增益显著（OA +0.436，bootstrap CI 不含 0），C2 的 supported-bias 跨数据集复现，SCC 部分缓解、Guard 完全缓解（但 Guard 是 hard 规则）。SCC-v1 的跨数据集 unsupported 保持弱于预期——**方法选择需在 SCC 与 Guard 间权衡（soft centering vs hard text guard），建议由用户决策**；证据充分到可支持冻结任一候选进入 pixel-level 评估。

## 6. 文件

- 报告：本文件 + `reports/vaihingen_blind_prediction_manifest.md`
- 协议/配置：`configs/vaihingen_scc_protocol_v0.json`、`configs/vaihingen_blind_v0.yaml`、`configs/vaihingen_blind_evaluate_v0.yaml`、`configs/vaihingen_sam3_v0.json`
- 代码：`src/ov_probe/vaihingen_blind.py`、`scripts/run_vaihingen_blind.py`
- 测试：`tests/test_vaihingen_blind.py`（7 passed）
- CSV/JSON：`vaihingen_overall.csv`、`vaihingen_subset_metrics.csv`（SCC/Guard）、`vaihingen_subset_metrics_c2_text.csv`（C2/text）、`vaihingen_bootstrap_summary.json`、`vaihingen_partial_bootstrap.json`、`evaluate_manifest.json`、`manifest.json`、`subset_manifest.json`
