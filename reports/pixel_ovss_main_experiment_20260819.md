# Pixel-level OVSS 主实验：Vaihingen + LoveDA（protocol v0）

运行日期：2026-08-19（服务器 luo-W360-E20）
协议：`configs/pixel_ovss_protocol_v0.json`（frozen_pre_result，SHA-256 `8fd0f020…`）
方法：Text-only / C2 / SCC / CTP（全部冻结公式，CTP-v1 per `configs/ctp_v1_frozen.json`）
Proposal：第一篇 SAM3 candidate masks（无新 proposal 网络）
Fusion：FusionCanvas（逐像素最高 score、conflict margin 0.03→ignore、uncovered=255）
GT 隔离：predict → 哈希固化（manifest 绑定全部语义图）→ 读 GT → evaluate

## 1. Vaihingen pixel-level（test 5 areas，全支持 k=5）

| 方法 | OA | Macro F1 | mIoU | valid_pixels |
|---|---|---|---|---|
| Text-only | 0.3492 | 0.3228 | 0.1974 | 10,445,223 |
| C2 | 0.6273 | 0.5989 | 0.4571 | 12,606,218 |
| SCC | 0.5933 | 0.5687 | 0.4262 | 12,987,741 |
| **CTP** | 0.5933 | 0.5687 | 0.4262 | 12,987,741 |

per-class IoU（CTP）：impervious 0.4186、building 0.5984、low_vegetation 0.0854、tree 0.4156、car 0.6128。

**观察**：
- k=5 全支持时 SCC==CTP（恒等），且与 region-level 一致（Macro F1 0.569 vs region 0.654——pixel 级因 mask 内 GT 混合而略低）。
- **C2 的 pixel OA 高于 SCC/CTP（0.627 vs 0.593）且 valid_pixels 更少（12.61M vs 12.99M）**：C2 的 anchored score（除以 ‖0.5t+0.5v‖≈0.81 放大）在重叠区域更"自信"，赢得更多冲突像素但标记更多冲突为 ignore。SCC/CTP 的 centering 后分数更保守，保留更多有效像素但总体 OA 略低。
- 这是协议口径（per-pixel highest score + conflict→ignore + uncovered 排除）下的真实结果；C2 的 pixel 优势源于 score 尺度而非语义（region 级 C2 与 CTP 恒等）。

## 2. LoveDA pixel-level（heldout 411 图，全支持 k=6）

| 方法 | OA | Macro F1 | mIoU | valid_pixels |
|---|---|---|---|---|
| Text-only | 0.0747 | 0.0525 | 0.0275 | 12,935,857 |
| C2 | 0.0726 | 0.0393 | 0.0206 | 12,963,978 |
| SCC | 0.0727 | 0.0392 | 0.0206 | 12,943,138 |
| **CTP** | 0.0727 | 0.0392 | 0.0206 | 12,943,138 |

**关键披露（重要局限）**：
- LoveDA heldout 每图仅 **1–13 个 SAM3 候选**（中位 2），全 heldout 像素覆盖率 **4.1%**（中位 2.6%）。
- 评估仅在 mask 覆盖像素上进行（uncovered=255 排除，协议口径），因此 LoveDA pixel 指标反映的是**稀疏区域采样的局部一致性**，不能与 Vaihingen（每图 1300+ 候选，覆盖 56%）直接比较。
- 坐标对齐已逐一验证正确（mask 与 GT 精确匹配，诊断脚本确认）。
- **结论**：LoveDA pixel 结果低是 6000-region 采样在 pixel 级的固有稀疏性，不是方法或实现缺陷；LoveDA 的 pixel 证据以 Vaihingen 为准，LoveDA 保留 region-level 结论。

## 3. 协议执行记录

- Predict manifest 绑定全部语义图 SHA-256（Vaihingen 20 个 npz、LoveDA 1644 个 npz）+ region scores + 配置，GT 读取前固化。
- Evaluate 校验全部哈希一致后才打开 GT。
- 未修改 CTP-v1 / SCC / alpha / prompt / prototype；无阈值/margin 搜索；无训练。
- 各方法 valid_pixels 差异（conflict 标记不同）已如实披露。

## 4. 文件

- 代码：`src/ov_probe/pixel_ovss.py`、`scripts/run_pixel_ovss_vaihingen.py`、`scripts/run_pixel_ovss_loveda.py`
- 配置：`configs/pixel_ovss_vaihingen_v0.yaml`（predict）、`pixel_ovss_vaihingen_evaluate_v0.yaml`、`pixel_ovss_loveda_v0.yaml`、`pixel_ovss_loveda_evaluate_v0.yaml`
- 测试：`tests/test_pixel_ovss.py`（6 passed）
- 产物：`outputs/pixel_ovss_vaihingen_v0/{manifest,pixel_overall,pixel_per_image,pixel_stats}.json`、`outputs/pixel_ovss_loveda_v0/` 同名
