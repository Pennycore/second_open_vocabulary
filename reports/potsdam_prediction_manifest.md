# Potsdam Blind Prediction Manifest

生成时间：2026-08-19（服务器 luo-W360-E20）
状态：**本 manifest 生成前未读取任何 Potsdam GT**；GT 仅在 evaluate 阶段校验以下哈希后解锁。

## 1. 冻结的预测产物哈希（SHA-256）

| 产物 | 哈希 |
|---|---|
| `predictions.npz`（45488 区域 scores + prototypes） | `be34049e6aadcd5763bf7034c182758980cc9895d08eaef3b50df3420b4f829e` |
| `records.jsonl`（3502 个有候选的 patch） | `c4900d6f94d6ae536017b1c163a301c0317cced4d3152e292cf2d63d3047359f` |
| 14008 个语义图（4 方法 × 3502 图） | manifest `artifacts`（逐文件 SHA-256，evaluate 逐一校验） |
| 文本 token | `0fd3b7d2…` |

（3502 = 3584 test patches 中有 82 个 SAM3 零候选 patch 被协议排除：无 proposal → 无预测 → uncovered=ignore，不参与评估。）

## 2. 冻结配置哈希

| 配置 | SHA-256 |
|---|---|
| `configs/potsdam_ctp_v1_protocol.json` | `d0aba3a7657e4e44a0d1e67c6d807c5334e4eff9433b19ecb915a53cf85196b5` |
| SAM3 配置 `potsdam_sam3_test_v1.json` | 与 protocol `segmentation.candidate_masks` 一致 |

## 3. 实验设置（冻结）

- 数据集：ISPRS Potsdam 14 个 test parent tiles（`potsdam_parent_split_23_0_14_paper`），`Postdam_patches_512_full` 512px patches
- 定位：**external held-out dataset evaluation**（非 blind test；untouched audit 见 `reports/potsdam_untouched_audit.md`）
- 弱监督：SAM3 prompt-driven candidates（all-positive image-level 假设，无 GT 派生标签）；visual prototypes 计数：impervious 9844 / building 2989 / lowveg 3991 / tree 3184 / car 25480
- 方法：Text-only / C2 normalized / SCC / CTP（全部冻结公式，alpha=0.5，8 Group-A prompts）
- Fusion：FusionCanvas（conflict 0.03 → ignore；uncovered = 255）
- Partial-support：ratios 25/50/75% × seeds 42/43/44 预注册（`support_subset_manifest.json`，GT 前生成）

## 4. blind 声明

- predict 阶段仅读取：Potsdam 图像、SAM3 candidates、冻结 checkpoint、冻结配置。
- 未读取任何 GT、未使用 GT 派生统计、未按 GT 选择 subsets。
- evaluate 将校验 predictions.npz + records + 全部 14008 语义图哈希一致后才打开 GT。

## 5. 状态记录

- Full-support predict：✅ 完成（manifest 已固化）
- Full-support evaluate：✅ 完成（`pixel_overall.json`：CTP OA 0.6477 / MacroF1 0.6280 / mIoU 0.4785）
- Partial-support manifest：✅ 预注册（9 subsets）
- Partial-support predict：✅ 完成（126072 语义图）
- **Partial-support evaluate：⏳ 服务器后台运行中**（预计 2026-08-19 晚完成；进程 nohup 独立，不受本机关机影响）
