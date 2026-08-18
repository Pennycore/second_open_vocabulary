# Confidence-aware Text Preservation (CTP)：LoveDA + Vaihingen 评估

运行日期：2026-08-19（服务器 luo-W360-E20，2× RTX 2080 Ti）
约束：不训练、不增加参数、不搜索超参、不修改 SCC-v1/Guard/C2 冻结定义
数据状态：LoveDA = development-after-P0（method-development evidence）；Vaihingen = untouched_external_confirmation（predict 已固化哈希后才读 GT，CTP 从冻结 scores 计算，无新预测）

## 1. CTP 冻结定义（无参数）

对每个 region x：

1. `t* = argmax_c T_c(x)`（冻结文本 top-1）
2. 文本置信度 = **文本 margin** `m_t(x) = T_{t*}(x) − T_second(x)`（冻结分数的差，无参数）
3. 若 `t* ∈ Unsupported` 且 `T_{t*}(x) + m_t(x) > max_{c∈Supported} S_c^SCC(x)`：
   **保留文本预测 t***（Guard 风格保护，但仅当文本证据加上其自身 margin 能覆盖 SCC 竞争差距）
4. 否则跟随 **SCC 竞争**（supported: `A_c − b(x)`；unsupported: `T_c`）
5. `|S|=0` → Text-only；全支持 → SCC ≡ C2

设计动机（来自诊断）：C2 anchored 分数被 `‖0.5t+0.5v‖≈0.81` 归一化放大（Vaihingen 实测 C2≈0.686 vs T≈0.281），跨尺度比较会使门控永不触发或恒等于 SCC。CTP 以**同尺度的 SCC 竞争分数**为参照，并用文本 margin 作为置信度余量——margin 大（文本明确）则保留，margin 小（文本犹豫）则跟随视觉校准。

## 2. LoveDA（64 subsets，1104 labeled regions）

### 2.1 S/U/H-F1（mean over subsets，H 聚合 = mean(H_i)）

| k | SCC S/U/H | Guard S/U/H | **CTP S/U/H** |
|---|---|---|---|
| 2 | 0.5407 / 0.4483 / 0.4730 | 0.4957 / 0.4572 / 0.4572 | 0.5235 / **0.4604** / 0.4701 |
| 3 | 0.5839 / 0.4475 / 0.4930 | 0.5354 / 0.4572 / 0.4782 | 0.5668 / **0.4629** / **0.4951** |
| 4 | 0.6087 / 0.4508 / 0.5013 | 0.5722 / 0.4572 / 0.4911 | 0.5957 / **0.4672** / **0.5075** |
| 5 | 0.6275 / 0.4605 / 0.5012 | 0.6070 / 0.4572 / 0.4870 | 0.6201 / **0.4739** / **0.5072** |

（k=1 时 CTP≈SCC≈text-only，数学性质；k=0/6 恒等验证通过。）

### 2.2 解读

- **CTP 的 U 全面超过 SCC 与 text-only（0.4572）**：k=4 U=0.4672、k=5 U=0.4739——文本保护有效。
- **CTP 的 H 在 k=3/4/5 为所有方法最高**（0.4951/0.5075/0.5072，超过 SCC 与 Guard）。
- **S 轻微让渡**：CTP S 比 SCC 低 0.012–0.017（如 k=4 0.5957 vs 0.6087），但远高于 text-only 0.4572——supported gain 大部分保留。

## 3. Vaihingen（32 subsets，7432 labeled regions，从冻结 scores 计算）

### 3.1 S/U/H-F1（mean over subsets）

| k | Text-only | C2 | SCC S/U/H | Guard S/U/H | **CTP S/U/H** |
|---|---|---|---|---|---|
| 2 | 0.3453 | U=0 | 0.5551 / 0.2944 / 0.3548 | 0.4849 / 0.4613 / 0.4331 | 0.5382 / **0.3112** / 0.3622 |
| 3 | 0.3453 | U=0 | 0.5968 / 0.2365 / 0.2832 | 0.5471 / 0.5117 / 0.4861 | 0.5856 / **0.2655** / 0.3147 |
| 4 | 0.3453 | U=0 | 0.6297 / 0.2073 / 0.2025 | 0.6033 / 0.5573 / 0.4960 | 0.6226 / **0.2420** / 0.2572 |

### 3.2 解读

- **CTP 的 U 全面高于 SCC**（k=2 0.3112 vs 0.2944、k=3 0.2655 vs 0.2365、k=4 0.2420 vs 0.2073）——unsupported preservation 部分恢复。
- **CTP 的 S 保持 SCC 水平**（0.5382/0.5856/0.6226 vs SCC 0.5551/0.5968/0.6297，让渡 <0.017）——supported gain 保留。
- **但仍未达到 Guard 的 U**（Guard k=4 U=0.5573）与 text-only 水平（0.3453）——CTP 是 SCC 与 Guard 之间的折中，偏 SCC 侧。
- CTP 的 H（0.3622/0.3147/0.2572）高于 SCC（0.3548/0.2832/0.2025）但低于 Guard（0.4331/0.4861/0.4960）。

## 4. Bootstrap（seed 42，5000 repeats，image/area cluster）

### 4.1 Vaihingen：CTP vs SCC（k=2..4，mean over subsets）

| k | ΔH-F1 | ΔMacroF1 | ΔOA |
|---|---|---|---|
| 2 | +0.0079 [−0.0007, +0.0164] | +0.0036 [−0.0030, +0.0099] | −0.0089 [−0.0169, −0.0008] |
| 3 | **+0.0320 [+0.0223, +0.0424]** | +0.0050 [−0.0012, +0.0109] | −0.0065 [−0.0132, −0.0000] |
| 4 | **+0.0563 [+0.0337, +0.0833]** | +0.0016 [−0.0038, +0.0069] | −0.0038 [−0.0083, +0.0011] |

**H 提升在 k=3/4 统计显著（CI 不含 0）**；MacroF1 持平；OA 在 k=2/3 略降（CI 不含 0 或贴边）——unsupported 保护以轻微 overall accuracy 让渡为代价。

### 4.2 LoveDA（bootstrap 见 partial_support 系列；恒等验证）

- `CTP k=0 == Text-only: True`；`CTP k=6 == SCC == C2: True`（`partial_support_checks.json`）。

## 5. 结论：CTP 是否同时获得 SCC 的 supported gain 和 Guard 的 unsupported preservation？

**部分成立（LoveDA 上成立，Vaihingen 上部分成立）：**

| 目标 | LoveDA | Vaihingen |
|---|---|---|
| SCC 的 supported gain | ✅ S 0.596–0.620（vs text 0.457，略低于 SCC 0.609–0.628） | ✅ S 0.538–0.623（vs text 0.345，接近 SCC） |
| Guard 的 unsupported preservation | ✅ U 0.460–0.474（超过 SCC 与 text-only；接近 Guard 0.4572） | ⚠️ U 0.242–0.311（超过 SCC，但未达 Guard 0.461–0.557） |
| 综合 H | ✅ k=3/4/5 最高（0.495–0.507） | ⚠️ H 高于 SCC 低于 Guard（k=2..4） |

**回答用户核心问题**：
- **LoveDA**：CTP 同时获得两者——supported gain 保留（H 最高），unsupported 保持超过 SCC 与 text-only。
- **Vaihingen**：CTP 是 SCC 与 Guard 之间的**无参数折中**——比 SCC 更保护 unsupported（bootstrap H 提升显著），比 Guard 更少牺牲 supported（S 保持），但未达到 Guard 的完整 unsupported preservation。CTP 的文本 margin 门控是"适度保护"而非 Guard 的"硬保护"。

**方法性质**：CTP 无任何参数（门控 = 冻结分数的比较 + 冻结 margin），k=0/k=6 退化正确，测试 11/11 通过。作为 training-free 候选，CTP 在 LoveDA 上优于 SCC 与 Guard（H 最高），在 Vaihingen 上优于 SCC（H 显著提升）但不如 Guard（H）。**方法选择（SCC/Guard/CTP）取决于对 unsupported preservation 的权重**——由用户决策，本轮不自行选择。

## 6. 交付物

- 代码：`src/ov_probe/loveda_partial_support.py`（`ctp_predictions`）
- 测试：`tests/test_loveda_partial_support.py`（11 passed，含 k=0/k=6 恒等与门控分支）
- CSV：`outputs/loveda_blind_gt_v0/run_20260818_001/partial_support_*.csv`（含 CTP）、`outputs/vaihingen_blind_scc_v1/vaihingen_subset_metrics_ctp.csv`、`vaihingen_ctp_bootstrap.json`
- 恒等检查：`partial_support_checks.json`（CTP k=0/k=6 True）
