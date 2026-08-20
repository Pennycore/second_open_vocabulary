# SegEarth-OV 外部 OVSS baseline 可行性审计

审计日期：2026-08-20。  
审计类型：代码、文档、数据布局和公平比较边界的可行性审计。未 clone、未下载权重、未安装依赖、未运行任何第三方模型，也未读取/变更本项目 GT。

## 1. 范围与证据状态

本文把下列事实分成两类：

- **Web-verified**：本次只读查阅官方仓库页面或其官方 README / dataset document 能直接支持的陈述。
- **Local-unverified**：在本地工程和双 2080 Ti 只读检查中尚未验证，或必须在固定第三方 commit / 权重落地后才能验证的事项。它们绝不是已完成的 baseline 结果。

官方入口：

- [SegEarth-OV（likyoo/SegEarth-OV）](https://github.com/likyoo/SegEarth-OV) — 官方仓库将其标为 CVPR 2025 Oral、training-free remote-sensing OVSS，并公开代码。
- [SegEarth-OV 数据准备说明](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md) — 官方给出 LoveDA、Potsdam、Vaihingen 的预期目录与 converter 路径。
- [SegEarth-OV requirements.txt](https://github.com/likyoo/SegEarth-OV/blob/main/requirements.txt) — 官方依赖版本清单。
- [SegEarth-OV-2（earth-insights/SegEarth-OV-2）](https://github.com/earth-insights/SegEarth-OV-2) — 是扩展到 SAR 的不同仓库；不能与原 SegEarth-OV 直接等同。

## 2. 官方代码、依赖和数据支持

| 审计项 | 状态 | 证据 / 解释 |
|---|---|---|
| 官方代码可获得 | **Web-verified: YES** | 原始 SegEarth-OV 官方 GitHub 仓库公开，README 明确给出 clone、`demo.py`、`eval.py`、`eval_all.py`。|
| 是否训练-free | **Web-verified: YES（作者声明）** | 官方 README / abstract 将其描述为 training-free，并描述 SimFeatUp 与 patch-token global-bias subtraction。此处只记录作者声明，不把它扩展解释为“共享本项目 pipeline”。|
| 官方数据配置：LoveDA | **Web-verified: YES** | README 将 LoveDA 列为 semantic-segmentation dataset；数据文档要求 `data/loveDA/img_dir/{train,val,test}` 和 `ann_dir/{train,val}`，并提供 converter。|
| 官方数据配置：Potsdam | **Web-verified: YES** | 数据文档列出 Potsdam，期望 `data/potsdam/img_dir/{train,val}`、`ann_dir/{train,val}`；默认 converter 产生 patch 训练/验证布局。|
| 官方数据配置：Vaihingen | **Web-verified: YES** | 数据文档列出 Vaihingen，期望 `data/vaihingen/img_dir/{train,val}`、`ann_dir/{train,val}`；默认 converter 产生 patch 训练/验证布局。|
| 基本软件栈 | **Web-verified: required, local-unverified: installed** | 官方 README 建议 Python 3.9 与 SimFeatUp；requirements 固定或列出 torch 2.1.2、torchvision 0.16.2、mmcv 2.1.0、mmengine 0.10.4、mmsegmentation 1.2.2、timm 1.0.9、transformers 4.44.2 等。双 2080 Ti 的现有工程环境为项目自己的 torch/OpenCLIP 环境，尚未证明能与此栈共存。|
| 官方 pretrained checkpoint 的身份与可得性 | **UNRESOLVED** | 本次审计到的原 SegEarth-OV README 没有给出一个可验证的、对应原光学方法的 checkpoint 文件 URL、文件 SHA-256、revision 或许可。不得把“代码可得”写成“权重已验证可得”。|
| 本地/服务器已有 SegEarth-OV clone 或权重 | **Local-unverified / not found in scoped audit** | 本次只读审计没有发现本项目工作区内的第三方 clone、已登记的 SegEarth-OV config、原方法 checkpoint 或既有预测。未做广泛磁盘扫描，故措辞是 scoped audit 中未发现，而非全盘不存在。|

### SegEarth-OV-2 的边界

**Web-verified**：SegEarth-OV-2 README 在 2025-08-26 宣布代码和 `AlignEarth` 权重，并将其描述为面向 SAR 的扩展；安装说明要求下载 AlignEarth SAR ViT-B/16 权重。  
**结论**：AlignEarth 权重不是原始光学 SegEarth-OV baseline 的可替代 checkpoint。除非论文明确做 SAR 扩展比较并独立冻结任务、数据和指标，本阶段不下载、不使用也不报告它。

## 3. 输入、输出与当前评估协议的适配性

| 接口 | 可行性判断 | 当前状态 / 必须验证项 |
|---|---|---|
| 图像输入 | **Potentially compatible** | 官方有三数据集目录与 converter；需在固定官方 commit 下逐项确认本项目图像位深、通道、patch 尺寸、tile/overlap、split image-id 与其 converter 输出一致。|
| GT 输入 | **Potentially compatible, not yet proven** | 两边都使用 LoveDA/Vaihingen/Potsdam，但“同数据集名”不足以证明同一 GT 编码、ignore 类、边界处理或 test split。必须比对源文件 hash、类 id、void/eroded-label 规则。|
| 类别与 prompts | **Unresolved** | 需对照官方每个 `cfg_DATASET.py` 的 class order / vocabulary 与本项目冻结 class order；禁止事后为对齐结果改本项目 prompt 或 RemoteCLIP/CTP 规则。|
| 输出预测 | **Potentially convertible** | 官方 README 表明 `eval.py` 运行评估并保存 `results.xlsx`，但本次未 checkout 固定 commit，故尚未确认它保存 raw per-image mask、dtype、class id、ignore id 或仅保存 aggregate metric。公平复评需要 raw predictions（或可由原始 logits 无歧义重建），而不只是 `results.xlsx`。|
| 本项目评价 | **Potentially compatible** | 当前协议要求 OA、Macro F1、mIoU 和 partial-support 的 S/U/H F1/IoU，及特定 ignore/FusionCanvas 口径。外部方法若不能输出对齐的 per-image class maps，就只能报告其官方指标，不能伪装为本项目 controlled comparison。|

## 4. 公平比较的两种合法模式

### A. Controlled comparison（要求严格满足）

只有在以下条件全部满足时，才能把结果与 OpenAI CLIP / RemoteCLIP / CTP 放入同一“controlled”表：

1. 相同数据版本、相同 image-id / tile split、相同 GT 和 ignore 处理；
2. 相同冻结 SAM3 candidate masks 与相同 FusionCanvas；
3. 使用相同 support subset manifests 和同一 OA / Macro F1 / mIoU / S/U/H 计算器；
4. 仅替换规定的 semantic encoder / baseline 模块，并保存每项输入、代码与 config hash。

原始 SegEarth-OV 的 SimFeatUp、patch-token operation 与其自身输出路径看起来是**不同的 dense prediction/proposal pipeline**。除非其代码在固定版本上能明确接到相同 SAM3 candidates + FusionCanvas，默认不满足 controlled comparison 条件。

### B. External method comparison（当前更现实的预期）

若执行原始 SegEarth-OV，应作为外部方法单列：明确它的 backbone、SimFeatUp、patch-level/dense proposal 路径、图像预处理、官方 split 和评价实现与本项目不同。只在 dataset、class mapping、GT 和指标实现被逐项验证相同后，才可并列比较结果；仍须在 caption/正文说明 **protocol difference / proposal difference / backbone difference**。不得将这种比较描述为“only-backbone replacement”。

## 5. 明确阻塞项

1. **原方法 checkpoint identity**：需取得官方来源、文件名、SHA-256、许可和对应 git commit；当前没有可审计的原始光学权重证据。
2. **第三方代码锁定**：需 clone 到独立目录并记录 commit / submodule / SimFeatUp revision；本阶段尚未执行。
3. **独立环境**：当前项目 torch/OpenCLIP 与官方建议的 Python 3.9 / torch 2.1.2 / mmcv-mmseg stack 未验证相容；不可在主项目环境中就地安装或升级。
4. **数据转换可逆性**：Vaihingen、Potsdam converter 会 patch 数据；需确认 converter 不改变当前评估的有效像素、边界、类 id 或 split。若会改变，则只能 external comparison。
5. **raw prediction export**：必须确认可导出每张（或可映射至每个原始 tile）的 class-id map 和 ignore handling；仅 aggregate Excel 不足以复核公平性。
6. **类别映射与 background/void**：官方 cfg 的 class order、背景/杂类处理与本项目协议尚未在固定源码中核查，不能假定一致。
7. **不可混合的定义**：SegEarth-OV-2 / AlignEarth 是 SAR extension；其权重和结果不可替代原始 optical SegEarth-OV baseline。

## 6. 建议的“仅审计”下一步（不自动执行）

1. 在服务器新建独立、不可写入主工程的第三方审计目录，clone 官方原 SegEarth-OV 到固定 commit；不下载权重、不运行代码。
2. 读取并 hash `requirements.txt`、dataset converters、`configs/cfg_loveDA.py` / `cfg_potsdam.py` / `cfg_vaihingen.py`、评估器和保存路径，制作类表、split、ignore 与输出字段对照。
3. 单独核验官方原始 optical checkpoint 是否存在且可合法获取；记录 URL、发布日期、文件 hash、许可证、backbone identity 和对应 commit。若无，则将“可执行 baseline”状态保持 BLOCKED，而不是以 SegEarth-OV-2 代替。
4. 仅在上述审计全部通过后，冻结比较类型：若共享候选/FusionCanvas 不可能，则预注册为 external-method comparison；先运行 Vaihingen、Potsdam，后续才考虑 LoveDA。

## 7. 审计结论

SegEarth-OV 的**官方代码和三个目标数据集的配置支持均已 web-verified**，所以它值得进入外部 baseline 的可行性队列；但其原始 optical checkpoint、当前机器的兼容环境、完整类别/split/ignore 对齐、raw prediction export 和与 SAM3/FusionCanvas 的可控共享尚未证明。故当前结论是：

> **Feasible to audit further; not yet approved to run, and not yet eligible for a controlled comparison.**

在上述阻塞项关闭前，不产生 SegEarth-OV 指标，不与 CTP/RemoteCLIP 的 controlled 表混写，也不下载或运行 SegEarth-OV-2 的 AlignEarth 权重。
