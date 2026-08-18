# Vaihingen Blind Prediction Manifest

生成时间：2026-08-19（服务器 luo-W360-E20）
状态：**本 manifest 生成前未读取任何 Vaihingen GT**；GT 仅在 evaluate 阶段校验以下哈希后解锁。

## 1. 冻结的预测产物哈希（SHA-256）

| 产物 | 哈希 |
|---|---|
| `predictions.npz` | `96be715f5796a7877985c6a2a2a644acb28c3c1adc74ef395fcea5786712f39a` |
| `subset_predictions.npz`（32 个 support subsets × SCC/Guard） | `c5358675827d1a5b772233968f4d9a67da82a4170c8eaa15182cd49af44ed051` |
| `subset_manifest.json` | `ebc2f477a7b9bd76505ca26c5c997acf826d1169223eeb56291d7d516c32689d` |
| `records.jsonl`（20470 条：train 13038 / test 7432） | predict manifest `records.sha256` |

（evaluate 阶段逐一校验这些哈希与 predict manifest 一致后才允许读取 GT；任何不一致立即失败。）

## 2. 冻结配置哈希

| 配置 | SHA-256 |
|---|---|
| `configs/vaihingen_scc_protocol_v0.json` | `e4079363232dc18ccc439eb0080cdb1a471d6f9cd49a76a0992965bdcc21064d` |
| `configs/vaihingen_sam3_v0.json`（SAM3 弱监督） | 与 protocol 内 `sam3` 字段一致 |

## 3. 实验设置（冻结）

- 数据集：ISPRS Vaihingen 官方 16 个 GT area（`untouched_external_confirmation`，审计见 `reports/vaihingen_untouched_audit_20260818.md`）
- Train areas（prototype support，SAM3 weak labels）：1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37（11 个）
- Test areas（盲测评估）：11, 15, 28, 30, 34（5 个）
- 类别词汇表：impervious_surface, building, low_vegetation, tree, car（clutter 为 ignore）
- 弱监督：SAM3 prompt-driven candidates，all-positive image-level 假设（无 GT 派生弱标签）
- 文本：OpenAI CLIP ViT-B/32 quick-GELU + 8 个 Group-A 模板（类名替换为 Vaihingen 词汇）
- alpha = 0.5；SCC-v1 冻结公式（`configs/scc_v1_frozen.json`）
- Partial-support：全部 2^5 = 32 个 subsets（预注册 bitmask 枚举，无 GT 选择）
- 视觉 prototype 计数（train areas）：building 1807、car 4366、impervious_surface 983、low_vegetation 48、tree 5834

## 4. blind 声明

- 本 manifest 生成前，predict 阶段仅读取：图像、SAM3 candidates、冻结 checkpoint、SCC-v1 配置。
- 未读取任何 GT 像素、未使用 GT 派生统计、未使用 GT 选择 subsets。
- evaluate 阶段将校验上述哈希，全部一致后才打开 `Vaihingen_main_v1/labels/`。

## 5. 方法清单（evaluate 将输出）

- Text-only / Visual-only / C2 / SCC (k=5) / Text-Top1 Guard (k=5)
- 32 个 partial-support subsets × SCC / Guard
- OA / Macro F1 / mIoU；per-class P/R/F1/IoU；confusion matrix；S/U/H-F1；S/U/H-IoU
- image-cluster bootstrap（seed 42、5000 repeats）：SCC vs Text-only、SCC vs C2 的 Delta 95% CI
