# Phase B：Unsupported Collapse 根因诊断

诊断日期：2026-08-18
数据源：P0/P1 冻结预测 `outputs/loveda_blind_gt_v0/run_20260818_001/predictions.npz`（SHA-256 `95a2a765…`）+ LoveDA Train GT
输出：`p1_diagnosis_per_region.csv`（1227 行 heldout 逐区域）、`p1_diagnosis_summary.csv`

## 1. 诊断对象

对六轮 Leave-One-Class-Out（C1 实现，即当前原始实现），仅对 **GT=unsupported 类**的 heldout regions 统计：

- `T_gt`：该区域 GT 类的 text-only score
- `S_gt`：该区域 GT 类的最终融合 score（C1 下 = T_gt，完整文本分数，见 Phase A 审计）
- `pred_supported`：融合后的获胜类（必为 supported 类，因 unsupported F1=0）
- `S_win`：获胜 supported 类的最终 score
- `margin = S_win − S_gt`
- `pred_text`：text-only 预测；`text_correct`：text-only 是否正确

## 2. 六轮汇总

| fold (unsupported) | n (GT 区域) | A: text 对但融合错 | B: text 本身错 | C: text top-1 被抢 | margin mean | margin median | text-only unsup F1 | fusion unsup F1 |
|---|---|---|---|---|---|---|---|---|
| building | 212 | **0.844** | 0.156 | **0.844** | +0.2799 | +0.2814 | 0.5747 | 0.0000 |
| road | 173 | 0.017 | 0.983 | 0.017 | +0.3380 | +0.3405 | 0.4720 | 0.0000 |
| water | 186 | 0.011 | 0.989 | 0.011 | +0.3437 | +0.3439 | 0.3675 | 0.0000 |
| barren | 91 | 0.077 | 0.923 | 0.077 | +0.3358 | +0.3380 | 0.0909 | 0.0000 |
| forest | 167 | 0.012 | 0.988 | 0.012 | +0.3474 | +0.3499 | 0.5232 | 0.0000 |
| agriculture | 275 | 0.051 | 0.949 | 0.051 | +0.3382 | +0.3417 | 0.7123 | 0.0000 |

（A = text-only 正确但融合后错误的比例；B = text-only 本身错误的比例；C = text-only top-1 为 unsupported 类但融合后被 supported 抢走的比例。C 与 A 数值相同：GT=unsupported 时 text 正确 ⟺ text top-1=unsupported。）

## 3. 关键回答

> unsupported=0 到底是因为 CLIP text branch 本身无法识别这些类别，还是因为 supported visual anchors 改变了跨类别 score competition？

**两种机制都存在，且可分三类：**

1. **building（A=C=84.4%）：anchor 竞争是主因。** text 分支对 GT=building 的 212 个区域中 179 个（84.4%）能给出正确 top-1，但融合后这 179 个全部被 supported 类抢走（destination：road 103、water 56、barren 11、forest 27、agriculture 15，building=0）。C1 已保留 100% 文本分数（Phase A 证明无 scale unfairness），仍被抢走——说明 **supported 类的 V 分量系统性抬高其融合分数**（margin 恒正，mean +0.28），使 unsupported 类即使文本证据充分也无法在 argmax 中胜出。

2. **road/water/barren/forest/agriculture（B≥92%）：text 分支本身弱是主因。** 这些类 text-only 自身错误率 92–99%（barren 的 text-only unsup F1 仅 0.09）。但注意：即使如此，text-only 对这些类仍有一定正确预测（F1 0.09–0.71），融合后连这些也被清零（F1→0）——anchor 竞争是"补刀"。

3. **margin 恒正且集中（mean +0.28~+0.35，std <0.03）**：所有 fold 中获胜 supported 类分数平均高出 GT 类分数 0.28–0.35（相似度尺度），且方差极小——这是 supported 类获得 V 分量系统性 boost 的直接证据，不是随机竞争噪声。

## 4. 结论

- Phase A 已排除 `S_c = 0.5*T_c + 0` 的 scale unfairness（unsupported 类保留 100% 文本分数）。
- 本诊断进一步证明：**即使 unsupported 类保留完整文本分类器，融合 argmax 仍系统性偏向 supported 类**，因为 supported 类的 0.5/0.5 融合分数天然高于纯文本分数。
- 因此 unsupported collapse 的本质是 **score competition 层面的系统偏差**：visual anchor 的加入改变了所有 supported 类的分数分布（整体上移），而 unsupported 类留在文本尺度上，跨尺度 argmax 必然压制后者。
- building 是"text 能力存在却被 anchors 压制"的最强证据（84.4% 被抢），其余五类则是"text 能力弱 + anchors 补刀"。

## 5. 对 C1/C2 设计的预示

- C1（unsupported 保留 T_c）只修复了"unsupported 类自己的分数尺度"，没有修复"supported 类分数被 V 抬升"这一竞争源头，因此 U 仍为 0。
- C2（原型空间归一化）把 supported 分数也归一化到单位原型余弦尺度，可能缓解尺度差异——是否有效由 Phase C/D 实测决定。
- 若 C1/C2 均无效，说明需要 query-adaptive / text-guarded 机制（用户协议已预留），本阶段不实施。
