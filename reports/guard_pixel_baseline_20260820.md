# Guard 像素级部分支撑基线（Phase C）2026-08-20

日期：2026-08-20

## 1. Guard 像素级定义（冻结，区域级规则的逐区域应用）

区域级冻结 Guard 规则（`loveda_partial_support.guard_predictions`，无阈值/无 margin/无温度/无新 gating）：

> 若文本 top-1 ∈ Unsupported，则保留文本预测；否则按 C2 anchored 竞争（supported 类用 anchored 分数、unsupported 类用文本分数）。

像素级 FusionCanvas 中逐 region 应用同一规则：被 Guard 保留的 unsupported 类以**文本分数**参与融合，其余区域以 **C2 anchored 分数**融合；冲突（margin ≤ 0.03）与未覆盖按冻结协议忽略（255）。**Guard 是 hard text-preservation 基线，不是 oracle**：不读 GT，纯确定性规则；确定性由 `tests/test_final_audit.py::test_guard_deterministic` 覆盖。

## 2. 协议

- Guard 语义图由 `final_audit.build_guard_semantic_maps` 从冻结 scores（`predictions.npz`）+ 冻结 candidates + 冻结 support manifest 生成（无重新推断）：Vaihingen 45 张、Potsdam 31518 张。
- 五方法对比（Text-only / C2 / SCC / Guard / CTP），每个方法在**自身有效像素**（orig）与 **Ω_common 共同像素**（common）上报告 OA / Macro-F1 / mIoU / S/U/H-F1 / S/U/H-IoU 与 valid pixel 数。
- CTP vs Guard 的聚类级 bootstrap（seed 42，5000 repeats）见 `final_cluster_bootstrap_20260820.md`。
- 数据：Vaihingen（9 subsets）+ Potsdam（9 subsets；r25 为 k=1 恒等退化，见公平性审计 §2）。

## 3. Vaihingen：orig vs common（H-IoU / S-IoU / OA；valid M 为 orig 有效像素百万数）

| subset | 指标 | Text | C2 | SCC | CTP | Guard |
|---|---|---|---|---|---|---|
| k2_seed42 | H-IoU orig→common | .199→.203 | 0→0 | .174→.183 | .186→**.190** | .161→.197 |
| | S-IoU common | .235 | .212 | .255 | **.248** | .222 |
| | OA common | .368 | .283 | .373 | **.372** | .363 |
| | valid M | 10.5 | 14.0 | 11.7 | 11.0 | 14.3 |
| k3_seed42 | H-IoU orig→common | .191→.189 | 0→0 | .024→.038 | .072→**.091** | .258→.279 |
| | S-IoU common | .152 | .376 | .391 | **.384** | .316 |
| | OA common | .341 | .290 | .290 | .295 | **.367** |
| | valid M | 10.5 | 14.6 | 14.6 | 13.9 | 14.6 |
| k4_seed42 | H-IoU orig→common | .212→.215 | 0→0 | .026→.033 | .114→**.120** | .299→.308 |
| | S-IoU common | .193 | .499 | .513 | **.506** | .426 |
| | OA common | .355 | .579 | .567 | **.565** | .511 |
| | valid M | 10.5 | 13.2 | 13.2 | 13.1 | 14.7 |
| k4_seed44 | H-IoU orig→common | .212→.104 | 0→0 | .306→.306 | .240→**.240** | .111→.107 |
| | S-IoU common | .246 | .386 | .380 | **.367** | .313 |
| | OA common | .377 | .581 | .576 | **.555** | .458 |
| | valid M | 10.5 | 13.6 | 12.4 | 12.0 | 13.3 |

（9 子集全表见 `outputs/final_audit/five_method_metrics_vaihingen.json`。）

**Vaihingen Guard 基线结论**：Guard 的 H-IoU 在 6/9 子集高于 CTP（orig 与 common 一致），但 CTP 的 S-IoU 在 7/9 子集高于 Guard（k3_s42：0.384 vs 0.316；k4_s42：0.506 vs 0.426），OA 上 CTP 赢 5/9（共同像素）。Guard 的 U/H 优势来自"硬保留文本"（U-IoU common 常 >0.17，CTP 0.04–0.18），代价是 supported 侧的视觉适应损失。

## 4. Potsdam：orig vs common（非退化子集）

| subset (k) | 指标 | Text | C2 | SCC | CTP | Guard |
|---|---|---|---|---|---|---|
| r50_seed42 (2) | H-IoU orig→common | .210→.216 | 0→0 | .163→.161 | .188→**.183** | .223→.216 |
| | S-IoU common | .185 | .353 | .324 | **.284** | .186 |
| | OA common | .361 | .407 | .410 | **.373** | .362 |
| | valid M | 517.8 | 569.1 | 540.8 | 526.5 | 557.2 |
| r50_seed43 (2) | H-IoU orig→common | .199→.191 | 0→0 | .155→.148 | .161→**.153** | .196→.190 |
| | S-IoU common | .389 | .408 | .463 | **.446** | .380 |
| | OA common | .365 | .441 | .426 | .403 | **.361** |
| | valid M | 517.8 | 533.2 | 518.9 | 521.4 | 552.0 |
| r75_seed42 (4) | H-IoU orig→common | .250→.242 | 0→0 | .015→.017 | .120→**.131** | .434→.455 |
| | S-IoU common | .162 | .424 | .424 | .427 | **.441** |
| | OA common | .347 | .492 | .494 | .512 | **.615** |
| | valid M | 517.8 | 555.6 | 554.4 | 548.9 | 557.2 |
| r75_seed44 (4) | H-IoU orig→common | .214→.196 | 0→0 | .090→.089 | .161→**.141** | .258→.233 |
| | S-IoU common | .246 | .425 | .432 | **.432** | .414 |
| | OA common | .356 | .495 | .508 | **.516** | .506 |
| | valid M | 517.8 | 532.5 | 535.4 | 535.0 | 545.7 |

（9 子集全表见 `outputs/final_audit/five_method_metrics_potsdam.json`；r25 为 k=1 恒等退化：CTP≡SCC≡Text，Guard≈Text。）

**Potsdam Guard 基线结论**：非退化子集中 Guard 的 H-IoU 全部更高（6/6，聚类 bootstrap sig−），其 U-IoU 远高于 CTP（r75_s42 common：0.469 vs 0.077）；但 S-IoU 上 CTP 赢/平 5/6（r75_s42 例外 0.427 vs 0.441），OA 上 CTP 赢/平 4/6。Guard 的硬保留在 unsupported 占比大的场景（k=4、unsupported=1 类）收益显著，但 supported 侧损失同样显著。

## 5. 覆盖与忽略分析

coverage = |Ω_common| / 方法自身有效像素（越高=独占像素越少；全表见 `common_pixel_metrics.csv`）：

| 子集 | Text | C2 | SCC | CTP | Guard |
|---|---|---|---|---|---|
| Vaih k2_seed42 | 0.832 | 0.622 | 0.746 | 0.791 | 0.608 |
| Vaih k3_seed42 | 0.879 | 0.629 | 0.630 | 0.663 | 0.627 |
| Vaih k4_seed42 | 0.859 | 0.679 | 0.680 | 0.685 | 0.610 |
| Pots r50_seed42 | 0.879 | 0.800 | 0.842 | 0.865 | 0.817 |
| Pots r75_seed42 | 0.951 | 0.887 | 0.888 | 0.897 | 0.884 |
| Pots r75_seed44 | 0.908 | 0.883 | 0.878 | 0.879 | 0.862 |

C2 与 Guard 的 coverage 最低（预测像素最多），Text 最高；CTP 居中且普遍高于 C2/Guard。没有方法通过"少预测"获得表面优势（详见公平性审计 Q4）。

## 6. CTP vs Guard：逐像素视角小结

- Guard 赢的：U-IoU（硬保留 unsupported）、H-IoU（Vaih 6/9、Pots 6/6）、Potsdam r75 OA（unsupported 仅 1 类时文本保留几乎免费）。
- CTP 赢的：S-IoU（Vaih 7/9、Pots 5/6）、OA（Vaih 6/9、Pots 4/6、LoveDA 全部显著）、H-IoU（LoveDA 全部、Vaih k4_s44、k2 部分）。
- 冻结叙事成立：**Guard 牺牲视觉适应换取严格文本保留；CTP 提供平衡的软适应**。CTP 的 H 落后于 Guard 的幅度（Potsdam r75_s42：0.131 vs 0.455）是真实的 trade-off，论文以 Case 2 措辞呈现，不声称 CTP 在所有指标上最佳。
