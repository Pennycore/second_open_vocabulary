# Potsdam Untouched Audit（第二篇 CTP-v1 外部确认前）

审计日期：2026-08-19
目的：确认 Potsdam 未参与第二篇 CTP-v1 的任何方法开发，可作为 cross-dataset generalization confirmation 数据集。
定位：**external held-out dataset evaluation**（非 blind test——第一篇研究已接触 Potsdam）。

## 1. 审计项

### 1.1 Potsdam 是否用于第二篇方法开发？

| 检查项 | 结论 |
|---|---|
| CTP 设计 | 否——CTP 在 LoveDA（development-after-P0）上设计，Vaihingen 上验证；git 历史（`git log --all`）**零 Potsdam 提交** |
| alpha 选择 | 否——alpha=0.5 冻结于 CTP-v1 freeze record（`f54c034`），早于任何 Potsdam 接触 |
| prompt 修改 | 否——8 个 Group-A 模板在 LoveDA 冻结 |
| prototype 修改 | 否——visual prototype 构造规则在 LoveDA/Vaihingen 冻结 |
| support subset 选择 | 否——Vaihingen subsets 用 seeds 42/43/44 预注册；Potsdam subsets 本阶段将同样预注册（GT 前） |
| threshold 调整 | 否——无阈值（CTP margin 门控为冻结公式） |

### 1.2 Potsdam GT 是否曾用于方法选择/参数调整/qualitative selection？

| 检查项 | 结论 |
|---|---|
| 第二篇方法选择 | 否——第二篇 git 历史无 Potsdam；Potsdam 仅在第一篇工程中作为候选提及（`encoder_compare_protocol_v0.json`、`pixel_ovss_protocol_v0.json` 等协议文本中列为"未来候选"） |
| 参数调整 | 否 |
| qualitative selection | 否——第二篇从未读取 Potsdam GT 做任何可视化选择 |

### 1.3 第一篇工程中的 Potsdam 接触（不影响第二篇定位）

- 第一篇有 Potsdam SAM3 配置（`potsdam_server_prompt4.json` 等）与 candidates（`manual4_candidates_full_train_v1` 等），但**仅覆盖 train/val tiles**。
- **14 个 test tiles（`potsdam_parent_split_23_0_14_paper`）无任何现成 candidates、无任何第二篇结果**——构成真正的 held-out 评估集。
- 用户此前对 Vaihingen 的授权模式（"你可以重跑"）同样适用于 Potsdam test tiles 的冻结 SAM3 proposal 生成。

## 2. 结论

- **Potsdam 未参与第二篇 CTP-v1 的任何方法开发**（三项检查全否）。
- 定位：**external held-out dataset evaluation / cross-dataset generalization confirmation**。
- 评估集：`potsdam_parent_split_23_0_14_paper` 的 **14 个 test parent tiles**（`Postdam_patches_512_full` 中对应 patches，含图像+GT+image_level_labels 弱监督来源）。
- Proposal 来源：与 Vaihingen 完全相同的**冻结 SAM3 管线**（同 config `potsdam_server_prompt4.json` 风格、同 checkpoint、同 prompting）在 test patches 上生成（用户授权重跑模式）。
- 若审计发现问题，本报告将标记并停止；当前审计通过，允许进入 Phase B。

## 3. 后续冻结

- `configs/potsdam_ctp_v1_protocol.json`（Phase B）
- GT 隔离预测 → 哈希固化 → 读 GT（Phase C）
