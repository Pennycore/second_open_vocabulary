# SCC-v1 冻结记录（Freeze Record）

冻结日期：2026-08-18
冻结 commit：`f5d7d917ab9c1fadeb755f1bf14edd435d7e1fa1`
冻结配置文件：`configs/scc_v1_frozen.json`

## 1. 冻结状态声明

从本记录起，SCC-v1 视为 **SCC-v1-frozen**：

- **禁止**再根据任何后续数据集 GT 修改 SCC-v1（公式、参数、聚合方式、原型构造、prompt、alpha）。
- 本记录之后的所有外部数据集评估均为 SCC-v1 的确认性测试，而非方法开发。

## 2. 冻结的 SCC 公式（逐项）

对每个类别 c：

| 量 | 公式 |
|---|---|
| Text score | `T_c(x) = cosine(x, t_c)` |
| Anchored prototype | `p_c = Normalize(0.5·t_c + 0.5·v_c)` |
| Anchored score | `A_c(x) = cosine(x, p_c)` |
| Support-induced shift | `b(x) = mean_{c∈S}[ A_c(x) − T_c(x) ]`（arithmetic mean） |
| supported 最终分数 | `S_c(x) = A_c(x) − b(x)` |
| unsupported 最终分数 | `S_c(x) = T_c(x)` |

特殊情况：
- `|S| = 0`：SCC **严格退化为 Text-only**（冻结预测数组直接复用，已自动测试验证）。
- 全类 supported：所有类减同一 `b(x)`，argmax **严格等价于 C2**（已自动测试验证）。

禁止修改项：mean→median、beta、confidence threshold、temperature、k-dependent rule、hard gating、alpha、prompt、prototype construction。

## 3. 冻结的模型与配置

| 项 | 值 |
|---|---|
| 模型 | OpenAI CLIP ViT-B/32 quick-GELU（OpenCLIP 3.3.0，timm 转换权重） |
| 特征维度 | 512 |
| checkpoint SHA-256 | `9ecdaef3…` |
| prompts | 固定 8 个 Group-A 模板（`configs/scc_v1_frozen.json` 内完整列表） |
| alpha | 0.5（固定） |
| visual prototype | 弱监督区域 → region feature L2 → 类内平均 → prototype L2 |
| region features | 冻结 OpenAI CLIP 缓存（6000×512 float16） |
| random seed | 42 |
| 类别词汇表 | building, road, water, barren, forest, agriculture |

## 4. LoveDA 状态

**LoveDA 属于 `development-after-P0`**：

- LoveDA heldout 用于发现 P1（unsupported collapse）并开发 SCC；
- 因此 **SCC 在 LoveDA 上的全部结果属于 method-development evidence**；
- LoveDA 结果**不是** SCC 的独立 blind confirmation；
- SCC-v1 的独立确认必须来自未参与方法开发的外部数据集。

## 5. LoveDA 冻结前最终指标（Phase P 审计，`phaseP_final_metrics.csv` / `phaseP_metrics_by_k.csv`）

### 5.1 fully-supported（k=6，六类全有 visual prototype）

| 方法 | OA | Macro F1 | mIoU |
|---|---|---|---|
| Text-only | 0.5317 | 0.4572 | 0.3160 |
| C1 | 0.6730 | 0.6333 | 0.4722 |
| C2 | 0.6703 | 0.6404 | 0.4784 |
| **SCC** | 0.6703 | 0.6404 | 0.4784 |
| Text-Top1 Guard | 0.6703 | 0.6404 | 0.4784 |

（k=6 时 SCC ≡ C2 ≡ Guard，OA/Macro F1/mIoU 完全一致——严格恒等已自动测试验证。）

### 5.2 partial-support（k=1~5，SCC；H 聚合方式 = mean over subsets of per-subset H，即 mean(H_i)，非 H(mean S, mean U)）

| k | S-F1 | U-F1 | H-F1 | S-IoU | U-IoU | H-IoU |
|---|---|---|---|---|---|---|
| 1 | 0.4568 | 0.4568 | 0.4205 | 0.3154 | 0.3154 | 0.2831 |
| 2 | 0.5407 | 0.4483 | 0.4730 | 0.3869 | 0.3086 | 0.3258 |
| 3 | 0.5839 | 0.4475 | 0.4930 | 0.4260 | 0.3084 | 0.3439 |
| 4 | 0.6087 | 0.4508 | 0.5013 | 0.4491 | 0.3120 | 0.3519 |
| 5 | 0.6275 | 0.4605 | 0.5012 | 0.4664 | 0.3192 | 0.3500 |

（数值来自 `phaseP_metrics_by_k.csv`；完整 64-subset 逐项见 `phaseP_final_metrics.csv`。）

### 5.3 严格恒等检查（自动测试 + 真实数据）

- `SCC k=0 == Text-only`：**True**（macro_f1 0.457215 == 0.457215，OA 0.531703 == 0.531703）
- `SCC k=6 == C2`：**True**（macro_f1 0.640386 == 0.640386，OA 0.670290 == 0.670290）

## 6. 冻结后约束

- 不允许根据 Vaihingen 或其他任何数据集 GT 修改 SCC-v1。
- 不允许 alpha/prompt/threshold/beta/temperature 搜索。
- 不允许根据外部结果挑选 support subsets。
- 外部数据集结果若与 LoveDA 不一致，如实记录，不修改方法。

## 7. 相关文件

- `configs/scc_v1_frozen.json`（冻结配置，含全部字段）
- `configs/loveda_blind_gt_protocol_v0.json`（SHA-256 `f46c433e…`，LoveDA 协议）
- `outputs/loveda_blind_gt_v0/run_20260818_001/phaseP_final_metrics.csv`（64-subset 全指标）
- `outputs/loveda_blind_gt_v0/run_20260818_001/phaseP_metrics_by_k.csv`（k 汇总）
- `src/ov_probe/loveda_partial_support.py`（SCC 实现，commit `f5d7d91`）
