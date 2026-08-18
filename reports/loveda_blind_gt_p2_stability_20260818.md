# P2：Prototype 构造稳定性分析（轻量）

运行日期：2026-08-18（服务器 luo-W360-E20，2× RTX 2080 Ti）
输出目录（服务器）：`outputs/loveda_blind_gt_v0/stability_frac{0.25,0.50,0.75}_seed{42,43,44}/`
协议：`configs/loveda_blind_gt_protocol_v0.json`（与 P0 相同冻结协议）

## 1. 设计（按用户协议，不引入复杂方法）

只测试 class-mean prototype 对 support 子集规模的敏感性：

- support 子集：development 分区内按类随机采样 25% / 50% / 75% / 100%（每类等比例，`_subsample_support`，seed 42/43/44）
- 固定：prompt（Group-A 8 模板）、alpha（0.5/0.5）、特征缓存（6000×512 冻结）、test set（411 张 image-disjoint heldout 图，1104 labeled regions）
- 视觉 prototype 构造不变：逐 region L2 → 类内平均 → 再 L2
- 每个组合独立 predict → evaluate（blind 流程与 P0 相同）

## 2. 结果（fused text+visual，heldout 1104 labeled regions）

| support 比例 | Accuracy mean±std | Macro F1 mean±std |
|---|---|---|
| 25%（3 seeds） | 0.6694 ± 0.0055 | 0.6293 ± 0.0072 |
| 50%（3 seeds） | 0.6703 ± 0.0024 | 0.6325 ± 0.0011 |
| 75%（3 seeds） | 0.6745 ± 0.0032 | 0.6366 ± 0.0040 |
| 100%（1 run） | 0.6721 | 0.6327 |

逐组合明细（服务器 evaluate_manifest，全部校验通过）：

| run | Accuracy | Macro F1 |
|---|---|---|
| frac0.25_seed42 | 0.6721 | 0.6319 |
| frac0.25_seed43 | 0.6630 | 0.6212 |
| frac0.25_seed44 | 0.6730 | 0.6349 |
| frac0.50_seed42 | 0.6721 | 0.6335 |
| frac0.50_seed43 | 0.6712 | 0.6325 |
| frac0.50_seed44 | 0.6676 | 0.6314 |
| frac0.75_seed42 | 0.6775 | 0.6401 |
| frac0.75_seed43 | 0.6712 | 0.6322 |
| frac0.75_seed44 | 0.6748 | 0.6375 |
| 1.00（P0 主 run） | 0.6721 | 0.6327 |

## 3. 结论

- 即使只保留 25% 的 support 区域（每类约 200 个 region），fused macro F1 仍为 0.6293±0.0072，与 100% 的 0.6327 几乎无差（≤0.004 差距）。
- 所有 9 个随机子集的指标波动极小（std ≤ 0.007），证明 visual prototype **不是偶然依赖某批区域**，class-mean 构造对该数据规模是稳定的。
- 该稳定性为 P0 结论提供支持：text+visual 相对 text-only 的提升（+0.175 macro F1）远超 support 采样引入的波动（±0.007），不是采样运气。
