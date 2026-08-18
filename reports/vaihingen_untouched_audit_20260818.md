# Phase R（更新）：Vaihingen 数据存在性审计与资源盘点

审计日期：2026-08-18（第二次审计，用户提供本地数据后）
用户提供路径：`C:\Users\28457\Desktop\remote_dataset`

## 1. 数据存在性：✅ 已确认存在（两部分）

### 1.1 官方 ISPRS Vaihingen（权威 GT）

`C:\Users\28457\Desktop\remote_dataset\ISPRS_semantic_labeling_Vaihingen\`

| 内容 | 数量 | 说明 |
|---|---|---|
| `top/`（RGB 航拍图） | 33 张 area（1–38 中的 33 个） | `top_mosaic_09cm_area*.tif`，原始分辨率约 1919×2569 |
| `gts_for_participants/`（官方 GT） | **16 张 area**：1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37 | RGB 编码 6 类（实测 area1：蓝=building、白=impervious、绿=tree、青=lowveg、黄=car、红=clutter 仅 2px） |
| `dsm/`（数字表面模型） | 16 张（与 GT area 对应） | 未计划使用 |
| 压缩包 | `ISPRS_semantic_labeling_Vaihingen.zip`（879MB） | 官方分发 |

### 1.2 第三方预处理布局（RemoteCLIP benchmark 风格）

`C:\Users\28457\Desktop\remote_dataset\remote\`

- `remote/Vaihingen/Vaihingen/{images,labels}/`：**1198 个 512×512 图块**（覆盖全部 33 个 area），标签为**单通道 uint8，值 0–4（5 类，无 clutter）**
- `remote/partitions/Vaihingen/`：`val.txt`（612 行）、`all/labeled.txt`（586 行）、`1-2 / 1-4 / 1-8 / 1-16` 各含 `labeled.txt / unlabeled.txt`（标注比例划分）
- 同布局还包含 iSAID、LoveDA、MER、MSL、Postdam 及 partitions 下 ade20k/cityscapes/coco/DFC22/GID-15/pascal 等

## 2. 关键发现：remote 预处理版本标签来源存疑，不可作为盲测 GT

- `remote/Vaihingen/Vaihingen/labels/` 覆盖**全部 33 个 area**，其中包括**无官方 GT 的 area**（如 area2 有 367 个 patch、area24/29/33/35/38 等）。
- `remote/partitions/Vaihingen/val.txt` 的验证集包含非官方 GT area：**2,4,6,24,29,33,35,38**。
- 官方仅 16 个 area 有 GT。remote 版本中这些额外 area 的"标签"来源**无法在本项目内验证**（可能是第三方基准的伪标签/重标注/其他来源）。
- 标签仅 5 类（0–4），与官方 6 类（含 clutter）编码不一致，且映射规则未知。

**结论：`remote/` 预处理版本不能作为 blind confirmation 的权威 GT。** 盲测 GT 只能使用官方 `gts_for_participants/`（16 个 area，RGB 6 类编码，可验证）。

## 3. untouched 状态：✅ 三项检查均为否（可标记 untouched_external_confirmation）

| 问题 | 结论 |
|---|---|
| Vaihingen 是否曾存在于本项目/被本项目读取？ | 否——git 全历史无引用（仅本审计报告），项目代码/配置/报告无任何 Vaihingen 引用，服务器第一篇/第二篇/experiment 目录无 Vaihingen 产物 |
| 是否曾用 Vaihingen GT 调整过 prompt/alpha/prototype/SCC/threshold/region selection/segmentation rule？ | 否 |
| 是否曾看过 SCC/C1/C2 在 Vaihingen 上的指标？ | 否 |

**标记：`untouched_external_confirmation`**（以官方 16 area GT 为评估依据的前提下）。

## 4. 资源盘点（协议 Phase S 要求汇报）

| 资源 | 状态 |
|---|---|
| region proposal source | **无**——Vaihingen 从未运行 SAM3/任何 proposal 生成；第一篇工程无 Vaihingen 候选区域缓存；协议禁止重新运行 SAM3 |
| weak semantic source | **无**——无现成 image-level 弱标签；remote/partitions 的 labeled/unlabeled 是"有/无 GT 的 patch 划分"（semi-supervised 用），不是语义弱标签 |
| image-level labels | 无现成文件（Postdam 有 `image_level_labels_*.csv`，Vaihingen 无） |
| existing cached candidates | **无**——无 Vaihingen 的 OpenAI CLIP 特征缓存、无候选区域缓存、无原型 |
| 权威 GT | 官方 16 个 area（RGB 6 类）✅ |
| 图像 | 官方 33 张 top 图 ✅（其中 16 张有 GT 可评估） |

## 5. 协议约束下的障碍（必须由用户决策）

按协议 Phase S："如果 Vaihingen 没有现成弱监督区域来源：**不要使用 GT 构造 visual prototype。也不要自行重新设计复杂 weak supervision。先汇报当前能获得的……再决定如何建立公平 protocol。**"

当前障碍：
1. **无弱监督区域来源**：Vaihingen 无 SAM3 缓存；禁止重跑 SAM3；禁止用 GT 构造 prototype；禁止自行设计 weak supervision。
2. 评估可用图像仅 16 张（官方 GT 覆盖），且类别词汇表（impervious/building/lowveg/tree/car/clutter）与 LoveDA 6 类（building/road/water/barren/forest/agriculture）**不同**——SCC-v1 冻结的 8 个 Group-A prompts 针对 LoveDA 类名设计，外部确认的 class vocabulary 需要用户明确批准（例如：使用 Vaihingen 自身 5/6 类 + 相同 prompt 模板结构，或指定类名映射）。

## 6. 等待用户决策（不自行选择）

可选路径（供用户选择，不自行执行）：

- **A**：用户批准使用官方 16 个 area GT + Vaihingen 自身类别词汇表（impervious_surface/building/low_vegetation/tree/car，prompt 模板结构不变、类名替换为 Vaihingen 类别），并明确 weak visual support 来源（若批准，则需说明 prototype 的弱监督区域从何而来——例如允许使用与 LoveDA 相同的 SAM3 弱监督管线但**不重跑 SAM3** 则不可行，需要用户提供或批准替代 proposal 来源）。
- **B**：用户提供 Vaihingen 的现成 region proposals / 特征缓存 / 弱标签（如服务器上其他工具生成的缓存）。
- **C**：用户改选其他数据集（需重新审计 untouched）。

在获得明确指示前，不执行任何 Vaihingen 实验。
