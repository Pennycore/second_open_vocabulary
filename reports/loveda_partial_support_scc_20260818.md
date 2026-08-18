# Partial-Support Open-Vocabulary Calibration：Phase I–O 完整报告

运行日期：2026-08-18（服务器 luo-W360-E20，2× RTX 2080 Ti）
协议：`configs/loveda_blind_gt_protocol_v0.json`（冻结）+ 用户 Phase I–O 协议
数据：冻结 P0 预测（`predictions.npz`，SHA-256 `95a2a765…`）+ LoveDA Train GT
纪律：LoveDA heldout = **development-after-P0**；本阶段所有结果均为 method-development evidence，非 blind test；仅允许 SCC 与 Text-Top1 Guard 两个预定义候选；无训练、无参数搜索、无新模块。

---

## Phase I：指标一致性审计（通过）

**用户指出的矛盾属实**：road fold 原报告 B=0.983（text 错误率）与 text unsup F1=0.4720 不可能同时成立。

**根因**：`run_loveda_p1_diagnosis.py`（旧版）中 `pos = {row_index: heldout_position}`，然后用 heldout 位置（0..1226）去索引 **6000 行全量预测数组**（按 row_index 索引）——A/B/C/margin/destination 全部取错行。只有 `f1_stats` 内部用 `text_pred[hold_labeled]`（row_index 索引）是对的。

**修复**：`pos` 改为恒等映射（row_index 直接索引），text 预测改用冻结的 `text_only` 数组。修复后逐类整数 count（`phaseI_metric_audit.csv`）：

| class | n_gt | TP_t | FP_t | FN_t | prec_t | rec_t | f1_t | 上界 2r/(1+r) | TP_f | FP_f | FN_f | prec_f | rec_f | f1_f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| building | 212 | 173 | 219 | 39 | 0.4413 | 0.8160 | 0.5728 | 0.8987 | 189 | 107 | 23 | 0.6385 | 0.8915 | 0.7441 |
| road | 173 | 80 | 87 | 93 | 0.4790 | 0.4624 | 0.4706 | 0.6324 | 117 | 87 | 56 | 0.5735 | 0.6763 | 0.6207 |
| water | 186 | 43 | 5 | 143 | 0.8958 | 0.2312 | 0.3675 | 0.3755 | 90 | 8 | 96 | 0.9184 | 0.4839 | 0.6338 |
| barren | 91 | 7 | 58 | 84 | 0.1077 | 0.0769 | 0.0897 | 0.1429 | 30 | 21 | 61 | 0.5882 | 0.3297 | 0.4225 |
| forest | 167 | 80 | 56 | 87 | 0.5882 | 0.4790 | 0.5281 | 0.6478 | 101 | 54 | 66 | 0.6516 | 0.6048 | 0.6273 |
| agriculture | 275 | 204 | 92 | 71 | 0.6892 | 0.7418 | 0.7145 | 0.8518 | 215 | 85 | 60 | 0.7167 | 0.7818 | 0.7478 |

验证：`recall_t == TP_t/n_gt` 六类全等 ✓；`f1_t <= 2r/(1+r)` 六类全等 ✓；LOCO 每 fold `A == TP_text`（fusion unsupported TP=0 时）六 fold 全等 ✓。

**修正后的关键结论**：旧报告"road/water/forest/agriculture 的 text 分支本身弱（B≥92%）"**错误**。修正后仅 barren 是 text 真正弱（recall 0.077）；road 0.462、forest 0.479、agriculture 0.742、building 0.816 的 text recall 均非平凡。**fusion unsupported TP=0 的结论不变**（U=0 仍成立），且 `A==TP_text` 证明"text 每识别对 1 个 unsupported region，fusion 就翻转 1 个"——collapse 的本质是 supported 类 0.5/0.5 融合分数被 V 分量系统性抬升（margin 恒正 +0.28~+0.32）。

---

## Phase J：64 个 support subsets 穷举基准（Text-only / C1 / C2）

枚举全部 2^6=64 个视觉支持子集。按支持数 k 汇总（完整表见 `partial_support_by_k.csv`）：

| k | 方法 | S_mean | U_mean | H_mean |
|---|---|---|---|---|
| 1 | text_only | 0.4572 | 0.4572 | 0.4206 |
| 1 | C1 / C2 | 0.2826 | **0.0000** | 0.0000 |
| 2 | text_only | 0.4572 | 0.4572 | 0.4380 |
| 2 | C1 / C2 | 0.4439 / 0.4473 | **0.0000** | 0.0000 |
| 3 | text_only | 0.4572 | 0.4572 | 0.4406 |
| 3 | C1 / C2 | 0.5310 / 0.5334 | **0.0000** | 0.0000 |
| 4 | text_only | 0.4572 | 0.4572 | 0.4380 |
| 4 | C1 / C2 | 0.5831 / 0.5856 | **0.0000** | 0.0000 |
| 5 | text_only | 0.4572 | 0.4572 | 0.4206 |
| 5 | C1 / C2 | 0.6146 / 0.6182 | **0.0000** | 0.0000 |

**C2 的 collapse pattern**：在**每一个** 1≤k≤5 的 support subset 中，unsupported 类的 macro F1 均为 0——C2 的 U 不随 coverage 变化，恒为 0。collapse 与 support coverage 大小无关，只要存在 supported 类，其 V 分量 boost 就压过所有 unsupported 类。k=0（纯文本）与 k=6（全支持）边界行为正确。

---

## Phase K：SCC（Support-Centered Calibration）

公式（固定 alpha=0.5，arithmetic mean，无温度/阈值/beta）：
- `A_c(x) = (0.5·T_c + 0.5·V_c)/‖0.5·t_c+0.5·v_c‖`（C2 anchored score）
- `b(x) = mean_{c∈Supported}[A_c(x) − T_c(x)]`（support-induced mean shift）
- supported：`S_c(x) = A_c(x) − b(x)`；unsupported：`S_c(x) = T_c(x)`
- k=0：全为 T_c；k=6：所有类减同一 b(x)，argmax 与 C2 一致

**自动测试（单元测试 + 真实数据校验）**：
- `tests/test_loveda_partial_support.py`：5 tests passed（k=0 恒等、k=6 恒等、unsupported 列保文本分数、supported 列减同一常数、degenerate norm fail-closed）
- 真实数据：**SCC k=0 == Text-only: True；SCC k=6 == C2: True** ✓

---

## Phase L：Text-Top1 Guard（诊断 baseline，非最终方法）

规则：text-only top1 属于 unsupported 集合 → 保留 text top1；否则用 C2 竞争。无阈值。

---

## Phase M：统一比较（64 subsets）

### S / U / H（k=1~5，mean over subsets；完整表见 `partial_support_by_k.csv`）

| k | 方法 | S_mean | U_mean | H_mean | H_std |
|---|---|---|---|---|---|
| 2 | SCC | 0.5407 | 0.4483 | **0.4730** | 0.0480 |
| 2 | Guard | 0.4957 | 0.4572 | 0.4572 | 0.0472 |
| 3 | SCC | 0.5839 | 0.4475 | **0.4930** | 0.0533 |
| 3 | Guard | 0.5354 | 0.4572 | 0.4782 | 0.0359 |
| 4 | SCC | 0.6087 | 0.4508 | **0.5013** | 0.0871 |
| 4 | Guard | 0.5722 | 0.4572 | 0.4911 | 0.0690 |
| 5 | SCC | 0.6275 | 0.4605 | **0.5012** | 0.1488 |
| 5 | Guard | 0.6070 | 0.4572 | 0.4870 | 0.1573 |

（k=1 时 SCC/Guard ≈ text-only：SCC S=0.4568/U=0.4568/H=0.4205，Guard H=0.4206——单 supported 类时增益有限但 unsupported 不 collapse。）

### Phase M 五个问题的回答

1. **C2 的 U 是否随 support coverage 增大而系统性下降？** 是，但方式是"一步到底"：任意 k≥1 的 subset 中 U 恒为 0，与 coverage 大小无关。
2. **Text-Top1 Guard 是否恢复 text-only U？** **是**。Guard 在所有 64 个 subset 中 U 恒等于 text-only 的 U（0.4572 附近，mean U 0.4398–0.5071 by class），同时 S 随 k 增长（k=5 S=0.6070 vs text 0.4572）。
3. **SCC 是否无需 hard guard 恢复 U？** **是**。SCC 在全部 64 个 subset 中 unsupported F1 > 0（k=1~5 U_mean 0.4475–0.4605），无任何 hard 规则。
4. **SCC 的 H 是否稳定超过 Text-only 和 C2？** **是**。k=2~5：SCC H=0.4730/0.4930/0.5013/0.5012，全部高于同 k 的 text-only H（0.4206–0.4406）与 Guard H（0.4572–0.4911）；C2 H=0。SCC 的 H 在 k=4、5 超过 Guard。
5. **是否存在无论怎样 calibration 都持续 collapse 的类？** **否**。per-class（作为 unsupported，n=32 subsets each）：SCC U mean building 0.4857 / road 0.4613 / water 0.4094 / barren 0.3873 / forest 0.4616 / agriculture 0.4973，全部 > 0。barren 最低（text 本身最弱 0.3860→SCC 0.3873），但未 collapse 到 0。

### k=6 全支持（fully-supported）对照

| 方法 | Accuracy | Macro F1 | Macro IoU |
|---|---|---|---|
| Text-only | 0.5317 | 0.4572 | 0.3160 |
| C1 | 0.6721 | 0.6327 | 0.4714 |
| C2 | 0.6703 | 0.6404 | 0.4784 |
| **SCC** | 0.6703 | **0.6404** | **0.4784** |
| Guard | 0.6703 | 0.6404 | 0.4784 |

（k=6 时 SCC 与 C2 完全一致——自动测试保证；Guard 在无 unsupported 时退化为 C2。）

---

## Phase N：方法选择纪律（遵守）

- LoveDA heldout 全程按 development-after-P0 处理；本报告所有 SCC/Guard 数字为 method-development evidence。
- 未搜索 alpha/threshold/centering 公式变体；只实现了协议预定义的 SCC（arithmetic mean）与 Guard 两个候选。
- 所有失败结果保留：C1/C2 在 partial-support 下 U=0 的完整 64-subset 记录在 `partial_support_all_subsets.csv` 中。

## Phase O：交付物

- 报告：本文件 + `reports/loveda_phaseI_metric_consistency_audit_20260818.md`
- 脚本：`scripts/run_loveda_phaseI_audit.py`、`scripts/run_loveda_partial_support.py`（修正版 `run_loveda_p1_diagnosis.py`）
- 模块：`src/ov_probe/loveda_partial_support.py`
- 测试：`tests/test_loveda_partial_support.py`（5 passed）
- CSV/JSON：`phaseI_metric_audit.csv`、`p1_diagnosis_{summary,per_region}.csv`（修正版）、`partial_support_all_subsets.csv`、`partial_support_by_k.csv`、`partial_support_checks.json`
- git commit + push

## 结论（对应协议最终汇报 A–G）

- **A**：Phase B 矛盾根因 = 诊断脚本用 heldout 位置索引 6000 行全量数组（索引 bug），已修复并验证。
- **B**：修正后六类 text-only precision/recall/F1 见表（road rec 0.4624 f1 0.4706 等），与 F1 上界一致。
- **C**：C2 在全部 63 个含 supported 的 subset 中 U=0；collapse 与 coverage 无关，只要存在任一 supported 类。
- **D**：SCC：k=1 H=0.4205 → k=5 H=0.5012；U 保持 0.4475–0.4605。
- **E**：Guard：k=1 H=0.4206 → k=5 H=0.4870；U 恒等于 text-only（0.4572 附近）。
- **F**：SCC k=0 == Text-only ✓，k=6 == C2 ✓（单元测试 + 真实数据双重验证）。
- **G**：**是**——SCC 在 k=1~5 同时提高 supported performance（S 从 0.457 单调升至 0.6275）并保持 meaningful unsupported performance（U≈0.45–0.46，无 hard guard）。SCC 的 H 在 k≥2 稳定超过 text-only 与 C2，k≥4 超过 Guard。

已按 Phase O 停止，不进入 Potsdam；方法是否冻结由你决定。
