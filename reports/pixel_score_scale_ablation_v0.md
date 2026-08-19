# Experiment A：Pixel Score-Scale Ablation（Vaihingen）

日期：2026-08-19
目的：解释主实验中 C2（mIoU 0.4571）> SCC/CTP（0.4262）的差异来源——score magnitude 还是语义。
协议：`configs/pixel_ovss_protocol_v0.json`（固定 candidates/FusionCanvas/conflict 0.03/uncovered 255/GT 隔离）
输入：冻结 region scores（`predictions.npz`）+ SAM3 candidate masks；无新预测、无参数修改。

## 1. 方法

| 版本 | score 表示 | 说明 |
|---|---|---|
| A1 C2 raw | `s_c = 0.5·T_c + 0.5·V_c` | 未归一化 late fusion |
| A2 C2 normalized | `s_c = cos(x, Norm(0.5·t_c + 0.5·v_c))` | 归一化原型分数（= anchored，主实验 C2） |
| A3 SCC/CTP | `S_c^SCC`（frozen） | centering 后分数（k=5 时 CTP==SCC） |

其余完全冻结（FusionCanvas、conflict margin 0.03、uncovered=255、GT 隔离、评估口径）。

## 2. 结果（Vaihingen pixel 全支持 k=5）

| 版本 | OA | Macro F1 | mIoU | assigned px | uncovered px |
|---|---|---|---|---|---|
| A1 C2 raw | 0.5747 | 0.5531 | **0.4073** | 12,429,288 | 10,774,506 |
| A2 C2 normalized | 0.6273 | 0.5989 | **0.4571** | 12,620,180 | 10,583,614 |
| A3 SCC/CTP | 0.5933 | 0.5687 | **0.4262** | 13,001,718 | 10,202,076 |

per-class IoU：

| 类 | A1 raw | A2 norm | A3 CTP |
|---|---|---|---|
| building | 0.5047 | 0.6254 | 0.5984 |
| car | 0.5983 | 0.6249 | 0.6128 |
| impervious_surface | 0.4217 | 0.4528 | 0.4186 |
| low_vegetation | 0.0823 | 0.0996 | 0.0854 |
| tree | 0.4294 | 0.4828 | 0.4156 |

## 3. 结论（回答用户问题）

1. **归一化对 C2 影响巨大**：A1 raw（0.4073）→ A2 normalized（0.4571），mIoU +0.050——**C2 的 pixel 优势很大一部分来自 score normalization**（除以 ‖0.5t+0.5v‖≈0.81 将 raw 分数放大并改变竞争尺度）。raw C2 甚至低于 CTP。
2. **A2 vs A3 的差异不是 scale artifact**：两者都是归一化原型分数（同尺度），差异来自 SCC 的 b(x) centering（A2 不减 b，A3 减 b）——这是**语义级差异**：centering 改变 supported/unsupported 竞争。A2 的 mIoU 0.4571 > A3 0.4262，说明在 pixel 竞争中**不 centering 的 anchored 分数更优**（因为 centering 压低 supported 分数，使重叠区域冲突标记更多、胜者更保守）。
3. **对论文的定位**：主实验中 C2 的 pixel 优势是 (a) normalization 带来的尺度效应为主，(b) SCC centering 的语义代价为辅。region 级 C2==CTP（argmax 恒等）不矛盾——pixel 差异来自分数值（非 argmax）进入 FusionCanvas 竞争。

## 4. 产物

- `outputs/pixel_score_scale_ablation_v0/{manifest,pixel_overall,pixel_stats}.json` + 15 个语义图 npz
- 代码：`scripts/pixel_score_scale_ablation.py`
- CSV：见 `pixel_overall.json`（overall + per-class + pixel stats aggregate）
