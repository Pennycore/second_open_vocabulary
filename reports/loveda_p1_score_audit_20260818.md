# Phase A：P1 Leave-One-Class-Out Score 实现审计

审计日期：2026-08-18
审计对象：`src/ov_probe/loveda_blind_gt.py` → `run_loveda_blind_gt_evaluate` 中的 leave-one-class-out 分支
审计目的：确认类别 c 无 visual prototype 时最终 score 的精确代码路径与数学公式，检查是否存在 `S_c = 0.5*T_c + 0` 的 scale unfairness。

## 1. 代码路径（逐行）

文件：`src/ov_probe/loveda_blind_gt.py`，`run_loveda_blind_gt_evaluate`，第 476–489 行：

```python
# Leave-one-class-out variants reuse the frozen scores: the unsupported class's
# visual prototype is dropped, so its fused score degenerates to the text score
# while the other five classes keep the fixed 0.5/0.5 fusion.
leave_one_out: dict[str, dict[str, Any]] = {}
if has_scores:
    for unsupported in _CLASSES:
        unsupported_index = label_index[unsupported]
        loo_scores = 0.5 * text_scores + 0.5 * visual_scores      # (1) 全 6 类 late fusion
        loo_scores[:, unsupported_index] = text_scores[:, unsupported_index]  # (2) unsupported 列替换
        loo_pred = np.argmax(loo_scores, axis=1).astype(np.int64) # (3) 全 6 类 argmax
```

其中 `text_scores`、`visual_scores` 来自 predict 阶段持久化的 `predictions.npz`（第 294–296 行）：

```python
text_scores = regions @ _normalize(text).T      # T_c = cosine(region, text_prototype_c)
visual_scores = regions @ _normalize(visual).T  # V_c = cosine(region, visual_prototype_c)
fused_scores = 0.5 * text_scores + 0.5 * visual_scores
```

`text`（文本 prototype）与 `visual`（SAM3 弱监督 visual prototype）均为 L2 归一化后的 6×512 矩阵。

## 2. 数学公式（精确）

设 x 为 L2 归一化 region feature，t_c、v_c 分别为 L2 归一化的文本与视觉 prototype。

**supported 类 c'（有视觉支持）：**

```
S_c' = 0.5 * T_c' + 0.5 * V_c'
T_c' = cosine(x, t_c') = x·t_c'
V_c' = cosine(x, v_c') = x·v_c'
```

**unsupported 类 c（无视觉 prototype）：**

```
S_c = T_c = x·t_c          （完整 text score，第 (2) 行直接覆盖）
```

**判定：** `pred = argmax_c S_c`，vocabulary 恒为完整 6 类。

## 3. 审计结论

| 检查项 | 结果 |
|---|---|
| 是否存在 `S_c = 0.5*T_c + 0`？ | **否**。第 (2) 行将 unsupported 列整体替换为 `text_scores` 列，即完整文本分数 |
| unsupported 类保留的文本比例 | **100%**（`S_c = T_c`），不是 50% |
| supported 类公式 | `S_c = 0.5*T_c + 0.5*V_c`（与 P0 fused 完全一致） |
| 是否与 C1（Support-Aware Text Fallback）一致 | **是**。当前实现即 C1 的定义：unsupported 类完整保留 OpenAI CLIP text classifier |
| C0 与 C1 是否一致 | **是**。C0 定义中"unsupported 按当前原始实现执行"，而当前原始实现即为 C1 公式，故 C0 ≡ C1，无需重复创建方法（按用户指令在报告中明确说明） |

## 4. 对 P1 unsupported F1=0 的初步判定

unsupported F1=0 **不是** score-scale unfairness 造成的（unsupported 类拥有 100% 文本分数，尺度没有被人为缩小）。

因此 P1 的 unsupported collapse 只能来自两类原因，需由 Phase B 诊断区分：

1. **文本分支本身能力不足**：对某些类别（如 water/barren），CLIP text 分支的 T_gt 在 6 类文本竞争中本身就排不到 top-1（text-only F1 已很低）。
2. **跨类别 score competition 偏移**：supported 类的 fused 分数 `0.5*T + 0.5*V` 因 V 分量获得系统性 boost，即使 T_gt 是文本 top-1，`S_win(supported) > S_gt` 仍可能成立，把预测从 unsupported 类抢走。

Phase B 将直接测量 margin = S_win − S_gt 的分布与 A/B/C 三类混淆统计来区分这两种机制。

## 5. 附带确认（P0/P1 产物完整性）

- 审计基于的预测产物：`outputs/loveda_blind_gt_v0/run_20260818_001/predictions.npz`（SHA-256 `95a2a765…`，与 evaluate manifest 绑定校验通过）
- evaluate 阶段先校验 predict manifest 与 predictions 哈希后才读 GT，blind 纪律未被破坏
- 未修改任何 P0/P1 预测；本审计只读代码与产物
