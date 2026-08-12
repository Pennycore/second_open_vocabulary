# OV-WSSS Stage 0 完整报告：RemoteCLIP 视觉—文本对齐

日期：2026-08-12  
随机种子：42  
区域正式运行代码版本：`ec1a661841161e167674b450608291eb490785d4`  
原型正式运行代码来源：运行时工作树（发生于本仓库首次 Git commit 之前，未记录可追溯 commit；数值产物已复核）  
状态：E0.1–E0.5 已完成；E0.6 按协议跳过

## 1. 实验目标与边界

本阶段检验第一篇 LoveDA Train 产出的 RemoteCLIP 类级视觉原型和候选区域特征，能否在同一 RemoteCLIP 文本空间中稳定匹配类别概念。

本实验不是开放词汇分割系统，不生成新伪标签，不重新运行 SAM3，不训练 student，不微调 RemoteCLIP，也不评价真实 unseen class 的像素级分割性能。

## 2. 输入与来源

| 项目 | 本次使用内容 |
|---|---|
| 数据集与 split | LoveDA Train，2522 张图像 |
| 前景类别 | building、road、water、barren、forest、agriculture |
| Main-v1 | 6 × 512 单原型，float32 |
| Main-v2 | 12 × 512 多原型，每类 K=2，float32 |
| 区域特征 | 270,641 个第一篇候选区域，`region_features` 为 512 维 float16 |
| 正式区域样本 | 按 SAM3 source class、seed=42 reservoir sampling，每类 1000，共 6000 |
| 文本编码器 | RemoteCLIP ViT-B-32 |
| checkpoint SHA-256 | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` |
| 文本缓存 | 219 个预注册唯一文本字符串，219 × 512 float32 |
| 弱标签 | SAM3 candidate source class；candidate mask 内 CAM mean top-1 |

区域探针没有使用 region cache 中已有的 `predicted_class_ids` 作为标签，因为它本身来自 RemoteCLIP 文本预测，会形成循环自证。CAM 标签由 candidate mask 与 CAM cache 重算，独立于 stored RemoteCLIP prediction；SAM3 标签来自 candidate cache。CAM 与 SAM3 仍共享上游 active image tags 约束，因此三方一致率不是三个统计独立标注器的共识。每个区域通过 `(image_id, candidate_index)`、candidate pair SHA-256 和严格索引检查对齐。

## 3. 合规说明

- 本 Stage 0 没有直接读取 pixel GT、LoveDA Val、Potsdam Val 或 E2 oracle。
- 第一篇上游图像级弱标签并非天然标签，而是由 LoveDA Train pixel mask 按“类别至少 16 像素”规则模拟得到；因此不能声称整个上游完全不含 GT 派生信息。
- 未修改、移动、删除或覆盖第一篇项目文件；正式区域运行审计了 12,610 个逐图 cache 文件，共 13,493,038,064 bytes（2522 × region JSON/NPZ、candidate JSON/NPZ、CAM NPZ；不含 checkpoint、文本缓存、协议文件及 region 辅助文件）。
- 输入 stat inventory 在运行前后均为 `725d28da97aa72570cb59690aec660fdfc8d322bf9f17f88ff634ffc032301db`；该摘要基于逐图 cache 的路径、大小和 mtime，不是所有文件内容的聚合 SHA-256。
- 视觉与文本特征在余弦比较前均执行 L2 normalization；正式样本输入范数均值为 1.0000008，标准差为 0.0000985。
- Group A 的 8 个模板、Group B 的 3 个模板、全部别名和 15 个干扰词在运行前冻结；没有依据结果删模板或调阈值。
- 服务器测试为 20/20 passed；每次运行各自进入第二篇项目中唯一、不覆盖的 run 目录，正式区域结果为 `run_20260812_005`。

## 4. E0.1 单原型—文本对齐

| Prompt | 词表 | Top-1 | Top-3 | 平均正确排名 | 平均正确相似度 | 平均 margin |
|---|---|---:|---:|---:|---:|---:|
| A | closed | 1.000 | 1.000 | 1.000 | 0.3274 | 0.0188 |
| A | expanded | 1.000 | 1.000 | 1.000 | 0.3274 | 0.0136 |
| B | closed | 1.000 | 1.000 | 1.000 | 0.3257 | 0.0136 |
| B | expanded | 0.833 | 1.000 | 1.333 | 0.3257 | 0.0100 |

Group A 在 6 类 closed 与 21 类 expanded 词表中均为 6/6 rank-1，但 expanded 平均 margin 下降约 27%。Group B expanded 中 building 被 sand 超过，正确类别排第 3。类级对齐存在，但 margin 整体偏小。

## 5. E0.2 多原型—文本对齐

| Prompt | 词表 | 聚合 | Top-1 | Top-3 | 平均正确排名 | 平均 margin |
|---|---|---|---:|---:|---:|---:|
| A | closed | max / mean / LSE | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 | 0.0227 / 0.0194 / 0.0201 |
| A | expanded | max / mean / LSE | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 | 0.0170 / 0.0135 / 0.0141 |
| B | closed | max / mean / LSE | 1.000 / 1.000 / 1.000 | 1.000 | 1.000 | 0.0169 / 0.0135 / 0.0140 |
| B | expanded | max / mean / LSE | 1.000 / 0.833 / 0.833 | 1.000 | 1.000 / 1.333 / 1.167 | 0.0131 / 0.0101 / 0.0106 |

12 个子原型中 11 个在 Group A closed 下最近文本为所属类别。例外是 `barren_0`，它最近的是 building；barren 的两个子原型只有 1/2 正确对齐。类级聚合可恢复 barren，但这说明部分视觉子簇更像外观模式或混合区域，不能默认每个子原型都是可独立命名的语义概念。

本探针中 max 的平均 margin 较大，但样本只有 6 类、每类 K=2，不能据此声称 max 在其他数据和 K 上普遍最优。

## 6. E0.3 Prompt 稳定性

| Prompt 组 | 单 prompt 数 | 平均正确相似度 | 相似度标准差 | 平均 margin | 单 prompt Top-1 |
|---|---:|---:|---:|---:|---:|
| A | 48 | 0.3136 | 0.0218 | 0.0050 | 0.625 |
| B | 66 | 0.3025 | 0.0282 | -0.0111 | 0.333 |

在类级探针中，Group A 的预注册 ensemble 明显比许多单模板稳定。`a remote sensing image of {class}` 的模板平均最好，但多个 aerial/satellite 模板明显更差，因此不能概括为“遥感措辞系统性更优”。Group B 的别名并未稳定改善结果，且 building 在 expanded 词表发生翻转。

## 7. E0.4 预注册类别级错误分析

- barren：Group A expanded 单原型仍为 rank-1，但对最强错误类 building 的 margin 只有 0.0043；多原型中的 `barren_0` 最近文本也是 building，说明其问题不只来自 agriculture/road/sand 词义竞争。
- forest：对 wetland 和 agriculture 的 Group A expanded 相似度差分别约为 0.0173 和 0.0183，存在相近语义竞争。
- agriculture：Group A 较稳定；Group B expanded 的最强错误类变为 barren，margin 仅 0.0079。
- building：Group A expanded 仍正确；Group B expanded 被 sand 超过。其预注册 residential/industrial 干扰虽非最强错误，类别措辞仍明显影响决策。
- road：Group A expanded 单原型 margin 为 0.0089，多原型 max 的 margin 仅 0.0018；预注册 railway/parking lot/bridge 均形成竞争。
- water：Group A expanded 对 wetland 的 margin 为 0.0263，在六类中相对稳定。

这些是运行前固定的错误对分析，并非事后只挑选有利案例；区域级对应错误见下一节的平衡样本结果。

## 8. E0.5 正式候选区域级探针

### 8.1 总体结果

下表基于 seed=42、按 SAM3 source class 每类 1000 的平衡 reservoir 样本，而不是 270,641 个候选的自然类别分布。一致率是对第一篇 Train-only 弱标签的一致性，不是真实分类准确率。样本来自嵌套于 2522 张图像的区域，只运行一个随机种子，未估计图像级置信区间。

| Prompt | 词表 | N | CAM–text | SAM3–text | CAM–SAM3 | 三方一致 | 平均 margin | 归一化熵 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | closed | 6000 | 0.4907 | 0.5367 | 0.6757 | 0.4097 | 0.0159 | 0.9790 |
| A | expanded | 6000 | 0.3923 | 0.4273 | 0.6757 | 0.3307 | 0.0135 | 0.9711 |
| B | closed | 6000 | 0.4790 | 0.5132 | 0.6757 | 0.3935 | 0.0127 | 0.9809 |
| B | expanded | 6000 | 0.3945 | 0.4205 | 0.6757 | 0.3272 | 0.0116 | 0.9748 |

类级原型的强 rank-1 结果没有直接转化为同等可靠的单区域匹配。closed 词表下，文本与 SAM3 source 的一致率约为 51%–54%；加入 15 个干扰词后降至约 42%–43%。margin 小且熵高，说明大批区域的文本决策并不锐利。

Group A/B 的 closed 区域预测仅 72.8% 相同，expanded 为 75.5%，说明别名和模板选择会实质改变约四分之一的区域决策。

### 8.2 Expanded 词表压力

- Group A 有 23.77% 区域被分到 15 个干扰类；Group B 为 22.65%。
- Group A 最常见干扰类为 ship 640、sand 295、wetland 225、railway 107、residential area 79。
- closed 到 expanded 保持同一预测的比例仅为 Group A 76.23%、Group B 77.35%。

在该平衡样本上，expanded 词表暴露的不是一次整体崩溃，而是幅度较大的区域级开放词表脆弱性。

### 8.3 按 SAM3 source 的预注册类别分析

每类恰好 1000 个区域。以下为 Group A：

| 类别 | closed 一致率 | expanded 一致率 | expanded 干扰类比例 | 主要 expanded 预测 |
|---|---:|---:|---:|---|
| building | 0.883 | 0.609 | 0.366 | building 609；ship 262；sand 67 |
| road | 0.477 | 0.354 | 0.221 | road 354；building 264；agriculture 81；railway 80 |
| water | 0.680 | 0.536 | 0.194 | water 536；agriculture 116；ship 91 |
| barren | 0.225 | 0.199 | 0.109 | agriculture 218；building 211；barren 199 |
| forest | 0.389 | 0.328 | 0.252 | forest 328；agriculture 179；ship 135 |
| agriculture | 0.566 | 0.538 | 0.284 | agriculture 538；wetland 111；ship 110 |

预注册困难类中，barren 最弱，并且主要被分到 agriculture 和 building；forest 也明显混入 agriculture。agriculture 相对较好，但 wetland/ship 干扰较强。相对稳定类中，building closed 很强，却在 expanded 下大量流向 ship/sand；road 易混到 building/railway；water 仍是较稳定类别之一。

Group B 对 barren/forest 有局部改善（expanded 分别为 0.295/0.425），但 building 降至 0.386；这进一步说明别名并非统一增益。

## 9. E0.6 跨数据集诊断

未执行。当前没有 LoveDA/Potsdam 同源、Train-only、可严格映射的原型束；没有强行映射语义不同的类别。

## 10. 当前证据支持什么

1. Main-v1/Main-v2 的类级视觉原型与同一 RemoteCLIP 文本空间存在清晰但低 margin 的类别对应。
2. Group A 的类级原型在固定 21 类词表中没有整体崩溃；多原型聚合也保持稳定。
3. RemoteCLIP 可以保留为下一阶段候选 backbone 或语义锚点。
4. 在 seed=42 的每类平衡区域样本上，类级平均原型掩盖了明显的区域异质性；直接把单区域 text top-1 当成可靠伪标签风险较高。
5. 后续若继续使用多原型，应考虑 class-level 与 sub-prototype-level 双层匹配及可靠性权重，而不是给每个子簇强行命名。

## 11. 当前证据不能支持什么

- 已实现 open-vocabulary segmentation；
- 能定位或分割真实 unseen class；
- 弱标签一致率等于真实分类或像素分割准确率；
- RemoteCLIP 优于 CLIP、SigLIP 或其他编码器；
- max 聚合普遍最优；
- 当前协议是完全无 GT 派生信息的天然弱监督设定。
- 270,641 个自然候选分布上的总体一致率、跨 seed 不确定性或图像级置信区间。

## 12. 决策与下一阶段最小建议

决策：**有条件保留 RemoteCLIP，但不进入直接 text-only 伪标签生成，也不启动完整 OV-WSSS 训练。**

类级结果接近“情况 A”，但 seed=42 平衡区域样本出现明显“情况 C”信号：expanded 一致率只有约 39%–43%，约 23% 区域被干扰词接管，且 margin 偏小。因此下一阶段最小实验应是冻结编码器的公平比较，而非训练分割网络：

1. 固定本次 6000 个 `(image_id, candidate_index)`、同一 crop/mask view、同一 Group A/B 与 closed/expanded 词表；
2. 在不训练的条件下比较 RemoteCLIP 与一个通用 CLIP；只有现有权重可合法复用时再加入 SigLIP；
3. 预先固定指标，报告 CAM/SAM3 一致率、三方一致率、margin、熵、干扰词率及逐类错误；
4. 同时测试 class prototype anchor + region-text score 的可靠性加权，但不依据 Val 调阈值；
5. 只有冻结编码器比较证明区域级信号足够稳定后，再设计 seen/unseen benchmark 和像素级方法。

本轮在 Stage 0 停止，不执行上述下一阶段。

## 13. 运行审计与产物

原型正式运行：`outputs/ov_probe_v1/run_20260812_001/`  
区域正式运行本地只读备份：`outputs/server_region_probe_v0/formal_ec1a661_run_20260812_005/`

区域正式运行门禁：

- `status=completed`
- `run_mode=formal_native_region`
- `scientific_evidence=true`
- `registered_formal_scope=true`
- image count = 2522
- candidate count = 270,641
- source counts = building 88,737；road 48,614；water 17,279；barren 11,574；forest 25,802；agriculture 78,635
- selected counts = 每类 1000，总计 6000

关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `summary_metrics.json` | `ce5b2080358be3f280a0c9b4078dae1731d01f2fac9f23b14ddbeebe4b94a230` |
| `validated_region_input.json` | `8ab19d48599864148234f79f57dc109dc08ee14a6e0623f59cd7b80011a2072e` |
| `input_manifest.json` | `fbd7e24812eda182334a1aa74ad9973d5372276472347e73aef1e5c8f8edbf67` |
| `region_level_results.json` | `8e2dd7f66855c1add346df416b77870997f844f5171f2f3b254ee477fe27ad4a` |
| `selected_region_records.jsonl` | `c7e5104cea0e0de85a711967982788f972d966f0053538fc05ab894181d6c5b7` |
| `region_weak_agreement.png` | `3d675d5bd3ab2d5f92232a9dfb786e41b72b733bf1f766382d3a700debfff5af` |

Pilot 审计：`run_001` 为 synthetic dry run；`run_002` 暴露旧版 evidence 标记问题并保留、排除于科学结论；`run_003` 为修复后的 5 图 non-scientific pilot；`run_004` 为 50 图 non-scientific pilot；只有 `run_005` 通过完整注册门禁。

本报告正文已脱敏，可进入公开仓库。原始 resolved config、manifest 和运行产物仍含机器绝对路径，保持在 Git-ignored 私有 outputs；若作为投稿补充材料发布，必须先生成脱敏副本并重新记录哈希。
