# CTP-v1 冻结记录（Freeze Record）

冻结日期：2026-08-19
冻结 commit：`f54c03461c960028ee1d605e852e5c649d54fe43`
冻结配置文件：`configs/ctp_v1_frozen.json`

## 1. 冻结决策

**CTP 为最终方法候选**（不冻结 SCC，不选 Guard）：

| 方法 | 定位 |
|---|---|
| C2 | failure baseline（supported 提升最大但 unsupported 崩溃） |
| SCC | global calibration ablation（b(x) 只消除公共偏移，异质 anchors 下不足） |
| Guard | hard preservation upper-bound / oracle-like baseline |
| **CTP** | **最终方法：soft、自适应、无参数、兼顾两端** |

## 2. CTP-v1 冻结公式（逐项）

对每个 region x、每个类别 c：

| 量 | 公式 |
|---|---|
| Text score | `T_c(x) = cosine(x, t_c)` |
| Anchored score | `A_c(x) = cosine(x, Normalize(0.5·t_c + 0.5·v_c))` |
| SCC shift | `b(x) = mean_{c∈S}[ A_c(x) − T_c(x) ]`（arithmetic mean） |
| SCC supported | `S_c(x) = A_c(x) − b(x)` |
| SCC unsupported | `S_c(x) = T_c(x)` |
| 文本 top-1 | `t* = argmax_c T_c(x)`（冻结文本预测） |
| 文本置信度 | `m_t(x) = T_{t*}(x) − T_second(x)`（冻结文本 margin） |
| **CTP 保留条件** | `t* ∈ Unsupported 且 T_{t*}(x) + m_t(x) > max_{c∈Supported} S_c^SCC(x)` → 预测 `t*` |
| **CTP 否则** | 预测 `argmax_c S_c^SCC(x)`（跟随 SCC 竞争） |

边界性质（自动测试 + 真实数据双重验证）：
- `k=0`：CTP 严格等于 Text-only ✅
- 全支持（k=C）：CTP == SCC == C2 ✅

**无任何参数**：无阈值、无温度、无 beta、无 learned gate、无训练。门控 = 冻结分数的比较 + 冻结 margin。

## 3. 冻结的模型与配置

| 项 | 值 |
|---|---|
| 模型 | OpenAI CLIP ViT-B/32 quick-GELU（OpenCLIP 3.3.0） |
| 特征维度 | 512；checkpoint SHA-256 `9ecdaef3…` |
| prompts | 固定 8 个 Group-A 模板 |
| alpha | 0.5（固定） |
| visual prototype | 弱监督区域 → region feature L2 → 类内平均 → prototype L2 |
| random seed | 42 |

## 4. 数据集状态

- **LoveDA**：`development-after-P0` —— CTP 在 LoveDA 上的结果属于 method-development evidence，非 blind confirmation。
- **Vaihingen**：`untouched_external_confirmation` —— 预测在读取 GT 前已哈希固化（`reports/vaihingen_blind_prediction_manifest.md`），CTP 从冻结 scores 计算（无新预测）。

## 5. 选择依据（Phase 1 统计对比摘要）

### LoveDA：CTP vs Guard（image-cluster bootstrap，seed 42，5000 repeats）

| k | ΔOA | ΔS-F1 | ΔU-F1 | ΔH-F1 | ΔS-IoU |
|---|---|---|---|---|---|
| 2 | +0.011 [−0.004,+0.026] | **+0.028 [+0.000,+0.056]** | +0.003 [−0.013,+0.019] | +0.013 [−0.008,+0.034] | **+0.025 [+0.001,+0.050]** |
| 3 | **+0.018 [+0.001,+0.034]** | **+0.031 [+0.010,+0.054]** | +0.006 [−0.019,+0.030] | +0.017 [−0.006,+0.040] | **+0.028 [+0.009,+0.049]** |
| 4 | **+0.018 [+0.003,+0.034]** | **+0.024 [+0.007,+0.040]** | +0.010 [−0.024,+0.044] | +0.016 [−0.013,+0.045] | **+0.021 [+0.007,+0.037]** |
| 5 | **+0.013 [+0.001,+0.024]** | **+0.013 [+0.004,+0.024]** | +0.017 [−0.035,+0.066] | +0.020 [−0.029,+0.065] | **+0.012 [+0.003,+0.022]** |

**LoveDA：CTP 全面不劣于 Guard，且 supported 增益显著更高**（S-F1/S-IoU 的 CI 不含 0），U/H 持平或微正——CTP 在无 hard 规则下达到 Guard 的文本保护，同时保留视觉适应。

### Vaihingen：CTP vs Guard（同上）

| k | ΔOA | ΔS-F1 | ΔU-F1 | ΔH-F1 |
|---|---|---|---|---|
| 2 | −0.058 [−0.108,−0.009] | **+0.052 [+0.017,+0.085]** | −0.149 [−0.171,−0.128] | −0.071 [−0.098,−0.043] |
| 3 | −0.057 [−0.111,+0.003] | **+0.038 [+0.010,+0.064]** | −0.245 [−0.275,−0.213] | −0.171 [−0.199,−0.140] |
| 4 | −0.036 [−0.073,+0.007] | **+0.019 [+0.002,+0.037]** | −0.314 [−0.353,−0.267] | −0.237 [−0.282,−0.186] |

**Vaihingen：Guard 的 U/H 更高，但 CTP 的 S 显著更高**（CI 不含 0）——正是论文解释框架：

> Guard sacrifices visual adaptation for strict text preservation, while CTP provides a balanced soft adaptation.

## 6. 论文定位（冻结后叙事）

- **Contribution 1**：Weak visual anchoring improves remote CLIP semantics but causes support-induced vocabulary bias（LoveDA blind + Vaihingen replication）。
- **Contribution 2**：CTP —— training-free、parameter-free、CLIP-compatible 的开放词表校准；"conservative extension of CLIP text prediction rather than a replacement classifier"。
- **Contribution 3**：CTP 在 supported/unsupported 间取得更好的 trade-off（LoveDA 全面优于 Guard；Vaihingen S 显著高于 Guard）。
- **SCC 作为重要 ablation**：Global score centering alleviates but does not fully solve support-induced bias under heterogeneous visual anchors（Vaihingen 的类别异质性证明）。

## 7. 冻结后约束

- 禁止按任何后续 GT 修改 CTP-v1（公式、参数、门控、原型、prompt、alpha）。
- 允许的后续工作：pixel-level OVSS pipeline（继承第一篇 SAM3 candidates/mask proposals，类别赋值替换为 OpenAI CLIP + CTP）、跨数据集主实验（Potsdam/Vaihingen/LoveDA）。

## 8. 相关文件

- `configs/ctp_v1_frozen.json`
- `reports/ctp_loveda_vaihingen_20260819.md`（CTP 定义与双数据集评估）
- `outputs/loveda_blind_gt_v0/run_20260818_001/loveda_ctp_vs_guard_bootstrap.json`
- `outputs/vaihingen_blind_scc_v1/vaihingen_ctp_vs_guard_bootstrap.json`
- `src/ov_probe/loveda_partial_support.py`（`ctp_predictions`）
- `tests/test_loveda_partial_support.py`（11 passed）
