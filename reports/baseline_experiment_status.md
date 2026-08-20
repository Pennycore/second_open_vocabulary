# Baseline 实验资产状态与冻结边界

审计日期：2026-08-20（仅资产、协议与可复现性审计；未运行模型、SAM3、训练或下载）。  
审计范围：本地第二篇工程 `C:\Users\28457\Desktop\open_vocabulary`，以及双 2080 Ti 上的只读工作区和第一篇工程中的既有资产。  
本文件是后续 baseline 的前置门禁，不是新的实验结果。除明确标为 `COMPLETED` 的既有输出外，RemoteCLIP 的下列矩阵均为 `PREPARED / NOT RUN`。

## 0. Potsdam RemoteCLIP partial-support attempt（2026-08-20，stopped）

为补充已完成 `run_20260820_002` 的 partial-support 指标，新增了仅使用既有 RemoteCLIP `predictions.npz`、`records.jsonl`、冻结的 CTP-v1 配置、Potsdam support-subset manifest 以及第一篇已落盘 candidate masks 的离线 evaluator。预检已通过：RemoteCLIP 预测缓存、records 行号与 candidate 顺序/元数据、候选目录 SHA-256、support-manifest SHA-256、RemoteCLIP protocol SHA-256 和 CTP-v1 frozen SHA-256 均成功绑定；预检完成后才允许读取 GT。此尝试没有加载模型、没有读取 OpenAI CLIP 特征、没有重跑 SAM3，也没有改变 prompt、alpha、prototype construction、FusionCanvas 或 CTP-v1。

该 evaluator 对每个 candidate-bearing image 顺序重建 `9` 个预注册 support subset × `4` 个方法的 FusionCanvas。由于未实现跨 subset/method 的画布聚合复用，完整运行 20 分 21 秒后仍未产生指标；按性能门限已主动停止。目录 `outputs/baselines/remoteclip/potsdam_v0/partial_run_20260820_001/` **仅包含 pre-GT validation manifest**，没有 `metrics.json`、`metrics.csv`、`report.md` 或 partial semantic maps，故它是 stopped attempt，**不是实验结果，不得进入表格、比较或结论**。相关源码 `scripts/evaluate_remoteclip_potsdam_partial.py` 与单测 `tests/test_remoteclip_potsdam_partial.py` 仅作为可追溯的 attempted-but-stopped 实现保存。

已完成的 full-support `run_20260820_002` 及其已登记数值完全不变。Vaihingen RemoteCLIP 仍为 **PREPARED / NOT RUN**；SegEarth-OV 仍仅处于 feasibility audit 状态，均未因本次停止尝试而改变。

## 1. 已登记的数据集、方法与冻结对象

| 类别 | 已有对象 | 审计状态 |
|---|---|---|
| 数据集 | LoveDA、ISPRS Vaihingen、ISPRS Potsdam | 三者均已有已登记的研究输出与报告；各自的数据来源、GT 隔离状态和 split 不能互换。|
| 对照/方法 | Text-only、Visual-only、C2 normalized、SCC、Guard、CTP-v1 | 已在既有研究记录中出现；其角色不能在 baseline 阶段重新定义。|
| 候选区域 | 第一篇工程的 SAM3 candidate pipeline 与已落盘 candidate masks | **冻结**；禁止重跑 SAM3、重新采样候选或以 GT 改候选。|
| 语义协议 | OpenAI CLIP ViT-B/32 quick-GELU 协议、8 个 Group-A prompts、特征 L2 规范 | 对原 OpenAI 主实验冻结；RemoteCLIP replacement 必须保留同一 prompt 文本与同一规范，而不是混用特征空间。|
| 组合规则 | C2、SCC、Guard、CTP-v1、`alpha=0.5`、视觉原型构造 | **冻结**。权威定义在 `configs/ctp_v1_frozen.json`（CTP freeze commit `f54c03461c960028ee1d605e852e5c649d54fe43`，该配置已由既有测试绑定 SHA-256 `788f1962d497022fbd5cacd7b63eaedddecd0343104aa726ee80afcdf1b37430`）。|
| 像素融合与评估 | FusionCanvas、support subset manifests、GT-isolation、OA / Macro F1 / mIoU、S/U/H 指标 | **冻结**；可复用评价和融合接口，但不得通过它们改变任何方法。|

CTP-v1 的固定决策是：以冻结文本分数、固定视觉锚点和 SCC 分数竞争为基础；仅当冻结文本 top-1 为 unsupported 且其冻结 margin 条件满足时保留文本 top-1。它没有阈值、temperature、beta、学习门控或训练。后续 RemoteCLIP baseline 不得改变该定义。

## 2. 已有实验资产与可复用边界

### 已存在、可作为审计或输入资产的输出

| 数据集/范围 | 已见输出或记录 | 可复用内容 | 不可据此宣称 |
|---|---|---|---|
| LoveDA | `outputs/loveda_blind_gt_v0/`、OpenAI feature cache / visual-anchor 记录、partial-support 与 bootstrap 文件 | 注册的 support manifests、候选关联、冻结评价代码和已生成的报告 | 用 OpenAI 分数、预测或 semantic maps 代替 RemoteCLIP 的分数/预测。|
| Vaihingen | `outputs/vaihingen_blind_scc_v1/`、`outputs/pixel_ovss_vaihingen_v0/`、`outputs/vaihingen_pixel_partial_support_v0/` | 已冻结的 candidates、FusionCanvas、partial-support manifest、预测锁定/GT 隔离模式、评估口径 | 把 OpenAI 特征、类别分数或 semantic maps 作为 RemoteCLIP baseline 的数值输入。|
| Potsdam | `outputs/potsdam_ctp_v1_v0/`、`outputs/potsdam_ctp_v1_partial_v0/`、`outputs/final_audit/` 与对应 reports | 同上，尤其是已有的 prediction-manifest 和外部确认审计模式 | 直接复用原 OpenAI CTP 分数而称为 RemoteCLIP 结果。|

因此，“已有 prediction / scores / masks 可以复用”需要精确理解：**候选 mask、支持集清单、FusionCanvas 定义、评价协议和已锁定的完整性清单可复用；OpenAI CLIP 的 embedding、text vectors、class scores、class ids 与语义图不可跨 backbone 复用。** RemoteCLIP 必须从相同的候选 crop / mask 重新得到自身特征、文本特征、分数和预测；这不是重新运行 SAM3。

`research_archive/experiment_registry.csv` 与 `research_archive/source_commit_map.csv` 已登记 LoveDA、Vaihingen、Potsdam 的既有输出及其历史提交。但早期运行存在“精确 per-run commit unknown”“server_uncommitted_workspace”记录，故不得把档案中的每个历史文件都自动视作可复现的 RemoteCLIP 输入。

## 3. 本地 / 服务器来源差异（必须在执行前解决）

本地工程当前审计到的 HEAD 是 `66431b3`（归档索引）；双 2080 Ti 的隔离工作区 HEAD 是 `51ec14981ecf746fd8ef15bba534ef664a20c7ae`；第一篇工程 HEAD 是 `718574dc69f1ea3cc4d19d51fccc18638fb69ce5`。三者不同。

这意味着后续不能只写“在服务器运行”。每次 baseline 必须在一个新且隔离的工作区中记录：代码 commit、所有协议/配置 SHA-256、源候选 manifest SHA-256、checkpoint SHA-256、环境版本、输出清单及输入路径。不得覆盖既有 run，也不得修改第一篇工程。若某个 Vaihingen/Potsdam candidate 或 support manifest 只存在于服务器，则应先做只读 hash 对照，再决定是否将其作为允许读取的固定输入。

## 4. RemoteCLIP replacement：可行性审计

### 已核实事实

| 字段 | 审计事实 |
|---|---|
| 模型身份 | `RemoteCLIP ViT-B-32` / OpenCLIP `ViT-B-32`。|
| checkpoint | 仅在第一篇工程中核实到：`/home/undergr/Sheungzhen_project_1/checkpoints/RemoteCLIP-ViT-B-32.pt`。本次只读检查确认存在，大小 605,208,421 bytes。|
| checkpoint SHA-256 | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4`。|
| embedding 维度 | 512。由 `configs/remoteclip_bridge_protocol_v0.json` 与已登记 LoveDA bridge 协议声明。|
| 预处理 | OpenCLIP `3.3.0` 的 `create_model_and_transforms('ViT-B-32', pretrained=None)` eval transform；候选区域使用固定的 context 与 mask-emphasized 两视图，各自 L2、均值、再 L2。|
| 既有 feature cache | 第一篇 LoveDA `candidate_region_remoteclip_loveda_main_v1` / 导出的 `encoder_compare_pack_v0` 有 6,000×512 的注册 reference features；RemoteCLIP bridge 只验证该 LoveDA pack 的环境再现，角色为非评价 gate。|
| 现有结果 | 未发现可证明符合本阶段相同 CTP-v1 协议的 Vaihingen 或 Potsdam RemoteCLIP Text-only / C2 / CTP 特征缓存、预测、语义图或指标。状态必须是 **NOT RUN**。|

### 必须遵守的相容性约束

1. 不能混合 RemoteCLIP 与 OpenAI CLIP 的任何 feature、text prototype 或 visual prototype。二者是不同 embedding space；这是既有 `configs/architecture_v1.json` 的明确边界。
2. 候选 masks、候选 crop 规则、8 个 prompt 模板的字符串、support subset、视觉原型构造、`alpha=0.5`、C2/SCC/Guard/CTP 公式和 FusionCanvas 必须逐项保持不变。
3. 不得为 RemoteCLIP 改 prompt、alpha、threshold、margin、temperature、beta 或 prototype construction；不得按 RemoteCLIP 结果改 CTP-v1。
4. 需要验证 RemoteCLIP 的 image encoder、text encoder、tokenizer、preprocess 和 checkpoint 是同一模型身份。现有 `RemoteCLIPTextEncoder` 会在 checkpoint/model key 不匹配时停止；正式执行应记录实际加载检查结果，而不是仅记录模型名。
5. 任何新缓存和结果只能写入新的 `outputs/baselines/remoteclip/...` run 目录；不得复写已有 `outputs/*`。

### 最小冻结矩阵（尚未执行）

| 数据集 | 方法 | full-support 指标 | partial-support 指标 | 状态 |
|---|---|---|---|---|
| Vaihingen | RemoteCLIP Text-only | OA、Macro F1、mIoU | S-F1、U-F1、H-F1；S-IoU、U-IoU、H-IoU | PREPARED / NOT RUN |
| Vaihingen | RemoteCLIP C2 normalized | 同上 | 同上 | PREPARED / NOT RUN |
| Vaihingen | RemoteCLIP CTP-v1 | 同上 | 同上 | PREPARED / NOT RUN |
| Potsdam | RemoteCLIP Text-only | 同上 | 同上 | PREPARED / NOT RUN |
| Potsdam | RemoteCLIP C2 normalized | 同上 | 同上 | PREPARED / NOT RUN |
| Potsdam | RemoteCLIP CTP-v1 | 同上 | 同上 | PREPARED / NOT RUN |
| Vaihingen、Potsdam | RemoteCLIP SCC | 同上 | 同上 | OPTIONAL；仅在前三项完整且计算预算允许时执行 |

`Visual-only` 与 `Guard` 是既有冻结对照资产，**不是**本阶段 RemoteCLIP 最小矩阵的隐含新增运行。若被纳入，必须另行写清其预注册原因，不得悄然扩大矩阵。

## 5. 执行前门禁与保存格式

在任何 GPU 推断前，执行者必须先完成以下只读门禁并停止于任一失败项：

1. 对 Vaihingen 与 Potsdam 分别对齐 candidate record order、mask hash、image list、support manifest、类别顺序和 ignore-label 规则。
2. 验证 checkpoint hash、OpenCLIP version、tokenizer、preprocess 表示及输出维度为 512；确认没有加载 OpenAI feature/text cache。
3. 冻结一次性配置并写入其 SHA-256，明确 `overwrite=false` 和 GT-isolation 顺序（先预测锁定，后打开 GT 评价）。
4. 每一个输出记录必须有 dataset、backbone、method、code commit、config hash、input hashes、checkpoint hash、指标和异常项。

建议的非覆盖目录是：

```text
outputs/baselines/
└── remoteclip/
    ├── features/<dataset>/<run-id>/
    ├── predictions/<dataset>/<run-id>/
    ├── metrics.csv
    └── report.md
```

结论：RemoteCLIP 已具备**资产级可行性**（已知 checkpoint、512-D 空间、可复现的 LoveDA bridge 规则），但尚未具备**Vaihingen/Potsdam CTP baseline 已完成**的证据。当前正确状态是“准备完成、等待冻结的实现与一次性受控运行”，不是“已有 robustness 结果”。

## 6. 执行后状态（2026-08-20）

Potsdam 的 RemoteCLIP full-support baseline 已完成，受控输出位于远端隔离工作区的 `outputs/baselines/remoteclip/potsdam_v0/run_20260820_002/`。该 run 使用既有 Potsdam test patches 与冻结的 SAM3 candidates；共包含 3,502 个 candidate-bearing images、45,488 个 regions。RemoteCLIP Text-only、C2、SCC 与 CTP-v1 均已完成；数值见同一 run 的 `metrics.csv` 与 `report.md`。预测阶段先完成并冻结清单，随后才读取 GT；GT phase 后 manifest 标记为 `scientific_evidence=true`。

这不改变已冻结的协议：没有重跑 SAM3、训练或调整 prompt、alpha、prototype construction、FusionCanvas 或 CTP-v1。该 run 不读取或复用 OpenAI CLIP features/scores；OpenAI CLIP 仅作为 `report.md` 中明确标记的受控、不同-backbone 对照。

Vaihingen RemoteCLIP 仍为 **PREPARED / NOT RUN**，不得将 Potsdam 结果外推为 Vaihingen 结果。SegEarth-OV 仍仅完成 feasibility audit，尚未作为可直接混合报告的外部 baseline。首次 `run_20260820_001` 在模型加载前因执行层摘要门禁格式不一致而停止，目录为空，不构成实验结果，也不得纳入任何表格或结论。
