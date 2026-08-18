# LoveDA Blind GT 验证：P0 真盲评估 + P1 Leave-One-Class-Out

运行日期：2026-08-18（服务器 luo-W360-E20，2× RTX 2080 Ti，172.18.56.240）
项目根：C:/Users/28457/Desktop/open_vocabulary
服务器 workspace：/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c
输出目录：`outputs/loveda_blind_gt_v0/run_20260818_001/`（本地归档副本）

## 0. 协议锁（与用户冻结配置逐项对应）

| 项目 | 冻结值 |
|---|---|
| 模型 | OpenAI CLIP ViT-B/32 quick-GELU（OpenCLIP 3.3.0，checkpoint SHA-256 `9ecdaef3…`） |
| 区域特征 | 既有 OpenAI CLIP 特征缓存（6000×512 float16，`ac75c0be…`），不重新编码 |
| 文本 prototype | 固定 8 个 Group-A 模板 → 逐模板 L2 → 类内平均 → 再 L2 |
| 视觉 prototype | SAM3 source weak label（仅 development 分区）→ 逐 region L2 → 类内平均 → 再 L2 |
| 融合 | score = 0.5·cosine(text) + 0.5·cosine(visual)，alpha/prompt/threshold 均不调整 |
| 划分 | 按原始 image_id（sha256("42:image_id") 排序，前 1647 张 development / 后 411 张 heldout），image-disjoint，无图同时进入 support 与 test |
| blind 保证 | predict 阶段配置强制拒绝 GT 目录；evaluate 阶段先校验 predict manifest 与预测产物哈希，之后才打开 GT PNG |

约束全部冻结：无训练、无 adapter、无新网络、无 alpha/prompt/threshold 搜索、无模型选择、不覆盖。

## 1. P0：真盲 LoveDA GT 评估

流程严格按用户 Step 1–6：加载冻结特征 → SAM3 weak label 构建 visual prototype → 计算三种 score → 保存全部 prediction（`predictions.npz`，SHA-256 `95a2a765…`）→ 此后才读取 LoveDA Train pixel GT → 计算真实类别指标。

GT 关联规则（冻结于 `configs/loveda_blind_gt_protocol_v0.json`）：以 SAM3 候选 mask 内像素对 6 类 GT 颜色多数投票；background (0,0,0) 与 ignore (255,0,255) 不投票；mask 内无 6 类像素的 region 记为 unlabeled 并披露。

### 1.1 heldout 整体指标（1104 labeled / 1227 全部，123 unlabeled）

| 方法 | Accuracy | Macro F1 | Macro IoU |
|---|---|---|---|
| Text-only CLIP | 0.5317 | 0.4572 | 0.3160 |
| Visual-only prototype | 0.5933 | 0.5794 | 0.4200 |
| **Text + Visual（0.5/0.5）** | **0.6721** | **0.6327** | **0.4714** |

### 1.2 Per-class F1 / IoU（fused）

| 类别 | F1 | IoU |
|---|---|---|
| building | 0.7441 | 0.5922 |
| road | 0.6207 | 0.4503 |
| water | 0.6338 | 0.4639 |
| barren | 0.4225 | 0.2680 |
| forest | 0.6273 | 0.4570 |
| agriculture | 0.7478 | 0.5970 |

### 1.3 结论（P0）

- **text+visual 在真 GT 上显著超过 text-only**：Accuracy +14.0 点（0.532→0.672），Macro F1 +17.5 点（0.457→0.633），Macro IoU +15.5 点（0.316→0.471）。
- fused 也超过 visual-only（Accuracy +7.9，F1 +5.3），说明文本与视觉锚点互补而非简单叠加。
- 此前 43.8%→59.3% 的“提升”是预测对 SAM3 弱标签的 agreement；本实验证明该提升在 **真实 GT** 上仍然存在（甚至幅度相当：GT macro-F1 0.633 ≈ 弱标签 agreement 0.593），即提升来自真实语义改善，不是弱标签循环验证。
- 123/1227（10.0%）region 因 mask 内无 6 类 GT 像素被披露为 unlabeled（多为 mask 落在背景/ignore 区域），已排除出指标。

## 2. P1：Leave-One-Class-Out 开放词表验证

规则（冻结）：每轮将一类设为 unsupported（其 visual prototype 禁用），其余 5 类保留 0.5/0.5 融合；unsupported 类的 fused score 退化为纯文本 score（其视觉分量缺失）；预测仍覆盖完整 6 类词表。六类轮流作为 unsupported class。

| unsupported | supported macro F1 | supported macro IoU | unsupported F1 | unsupported IoU | 全部 macro F1 |
|---|---|---|---|---|---|
| building | 0.5750 | 0.4123 | 0.0000 | 0.0000 | 0.4792 |
| road | 0.6298 | 0.4708 | 0.0000 | 0.0000 | 0.5248 |
| water | 0.6068 | 0.4466 | 0.0000 | 0.0000 | 0.5056 |
| barren | 0.6689 | 0.5062 | 0.0000 | 0.0000 | 0.5575 |
| forest | 0.6293 | 0.4676 | 0.0000 | 0.0000 | 0.5244 |
| agriculture | 0.5778 | 0.4245 | 0.0000 | 0.0000 | 0.4815 |

对照：text-only macro F1 = 0.4572；full anchor macro F1 = 0.6327。

### 2.1 结论（P1）

- **视觉锚点显著增强已有（supported）类别**：每轮 5 个 supported 类的 macro F1（0.575–0.669）都高于 text-only 全类基线（0.457）。
- **但 unsupported 类 F1 = 0.0（六类全部）**：当某类没有 visual prototype 时，融合预测从未选中该类——文本分支无法单独挽回被弃锚的类。这说明当前 0.5/0.5 融合的开放词表能力是**受限的**：它增强已锚定类别，但会把预测强有力地拉向有锚点的类，未锚定类在 argmax 下被完全压制。
- 这是本实验最重要的负面发现：**视觉锚定不是“只增不减”**。它提升 supported 类，却以破坏未锚定类的可识别性为代价（unsupported recall = 0），与“保持 CLIP 开放词表能力”的目标存在张力。
- 解释：fused score 中 visual 分量对 supported 类始终为正且同类匹配更高，unsupported 类仅剩文本证据，在 6 类 argmax 中系统性落败。

## 3. 文件与可复现性

- 协议：`configs/loveda_blind_gt_protocol_v0.json`（frozen_pre_result，SHA-256 `f46c433e…`）
- 部署配置（服务器）：`configs/loveda_blind_gt_v0.2080ti.local.yaml`（predict，无 GT 路径）/ `…evaluate.yaml`（evaluate，指向 LoveDA Train labels）
- runner：`scripts/run_loveda_blind_gt.py --phase predict|evaluate [--support-fraction F --support-seed S --output-subdir TAG]`
- 模块：`src/ov_probe/loveda_blind_gt.py`
- 测试：`tests/test_loveda_blind_gt.py`（7 passed，含索引对齐守卫）
- 每个 run 记录：seed、support fraction/seed、visual prototype 每类计数、prompt 模板、模型版本与 checkpoint SHA、文本 token SHA、设备、约束、输入哈希
- CSV：`metrics.csv`、`per_class_f1.csv`、`confusion_matrix_<method>.csv`、`leave_one_out_summary.csv`

## 4. 后续（P2/P3）

- P2（prototype 稳定性）：25/50/75/100% support 子集 × seed 42/43/44 已在服务器运行，结果汇总见稳定性报告。
- P3（VOC）：维持 image-level mAP sanity check 定位，不作为 segmentation 结果。

*本报告所有数字来自 blind 流程：predict 阶段（`manifest.json` + `predictions.npz`）先于任何 GT 读取；evaluate 阶段校验预测哈希后才读取 GT。*
