# SegEarth-OV 可复现性与公平性审计（2026-08-21）

**范围。** 本文是第二篇论文 frozen CTP-v1 的外部 OVSS baseline 审计。仅检查公开的官方资料、固定源码 revision 与本项目的已登记协议；**未 clone、未下载、未安装依赖、未转换数据、未运行 SegEarth-OV，也未读取或修改 CTP/SAM3/GT 输出。** 因此，所有“可运行”均须区分为源码层面的可行性与本机/3090 已验证可运行性。

**本次结论：NO-GO（当前不得启动 SegEarth-OV baseline）。** 官方代码和随仓库发布的 SimFeatUp 权重足以支持后续的、隔离环境中的 external-method baseline；但当前尚未满足可复现部署和与 CTP 同一评估支持集的绑定条件。特别是，原方法不共享 SAM3/FusionCanvas，也没有 partial-support 或视觉 support score。因此它不能进入 controlled comparison，不能做 partial-support 对比，亦不能作为 CTP plug-in 的候选。

## 1. 固定身份与证据等级

| 项目 | 状态 | 可核验事实 |
|---|---|---|
| 官方代码 | **已核验：可得** | 作者官方仓库为 [likyoo/SegEarth-OV](https://github.com/likyoo/SegEarth-OV)，项目页也直接链接该仓库和 CVPR 2025 论文。[官方仓库](https://github.com/likyoo/SegEarth-OV) |
| 固定 revision | **已核验** | 2026-08-21 对 `main` 执行 `git ls-remote`：`HEAD = 3e22a969b32c6d751bdbba64a88a0b670e630f55`。审计时 `refs/tags/*` 无返回，GitHub Releases 为 0；后续若执行，必须 pin 此 commit，不得仅记录 `main`。|
| 论文/方法身份 | **已核验** | CVPR 2025 论文为 *SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images*，DOI `10.1109/CVPR52734.2025.00986`。[CVF open-access paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf) |
| 运行依赖 | **已核验，尚未部署** | 官方清单指定/列出 Python 3.9、torch 2.1.2、torchvision 0.16.2、mmcv 2.1.0、mmengine 0.10.4、mmsegmentation 1.2.2、numpy 2.0.0、timm 1.0.9、transformers 4.44.2 等。[requirements](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/requirements.txt) |
| 项目环境兼容性 | **未核验** | 当前第二篇项目的 RemoteCLIP 环境与官方 Python/mmcv/mmseg/torch 组合不同；不得在现有环境中就地升级。后续必须建立独立环境并记录 `pip freeze`/CUDA/GPU。|

### 权重身份

官方固定源码树包含 `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt`；在上述 commit 的 Git blob id 为 `d40f00838331331faa04a926ea5991f5203f91a8`，API tree 报告大小为 5,690,776 bytes。默认配置正是引用该文件作为 `feature_up_cfg.model_path`。[默认配置](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/base_config.py)

这证明 SimFeatUp checkpoint **随固定源码可获得**，但 `d40f...` 是 Git blob id，不是下载文件的 SHA-256；正式运行前仍须对落盘文件计算并保存 SHA-256。默认视觉语言模型则由 `open_clip.create_model('ViT-B/16', pretrained='openai')` 解析，官方未在该仓库提供该 OpenAI CLIP asset 的独立 URL 或 SHA-256。[segmentor source](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/segearth_segmentor.py)

所以“training-free”只能准确表述为：**官方 evaluation 没有 target-dataset training step，使用已训练好的 CLIP 与已发布的 SimFeatUp。** 这不是“没有任何学习过的权重”；项目页明确说明 SimFeatUp 有训练过程，但在推理时使用它。[官方项目页](https://likyoo.github.io/SegEarth-OV/)

## 2. 原方法实际推理接口

固定 `base_config.py` 的默认模式是：OpenAI CLIP `ViT-B/16`、`model_type='SegEarth'`、`feature_up=True`、`cls_token_lambda=-0.3`、slide crop/stride 由 segmentor 默认值 `224/112` 控制。图像被 MMSeg 预处理（均值 `[122.771,116.746,104.094]`、标准差 `[68.501,70.323,70.323]`、BGR-to-RGB），随后做滑窗特征提取、SimFeatUp、text-query similarity、CLS-token subtraction、softmax 和 argmax。[默认配置](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/base_config.py)；[segmentor source](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/segearth_segmentor.py)

它输出：

- dense per-class `seg_logits`（`PixelData`）；
- dense `pred_sem_seg` class-id map；
- 官方 `eval.py` 通过 MMSeg `IoUMetric` 汇总，结果表保存 `aAcc`、`mIoU`、`mAcc` 至 `results.xlsx`。[eval.py](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/eval.py)；[result writer](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/utils.py)

因此，OA 可对应官方 `aAcc`，mIoU 可直接取得；**Macro-F1 不是官方 aggregate 输出字段**。只有在不改动其模型核心、从 `pred_sem_seg` 导出逐图 map，并以冻结的本项目 confusion-matrix evaluator 重新评分时，才可得到 Macro-F1。此导出/评分应被单列为 external-evaluation wrapper，不应误称为官方 `results.xlsx` 指标。

## 3. 数据集、类别与标注协议

| 数据集 | 官方原生配置 | 类别 / background / ignore | 与当前冻结 CTP 的关系 |
|---|---|---|---|
| Vaihingen | **有** `cfg_vaihingen.py`，官方转换器以 512 patch、256 stride 生成图像/标注；官方 val areas 为 `6,24,35,16,14,22,10,4,2,20,8,31,33,27,38,12,29`。[config](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/cfg_vaihingen.py)；[converter](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/tools/dataset_converters/vaihingen.py) | 官方 MMSeg `ISPRSDataset` 把 source label `0` 视为 ignore，并 `reduce_zero_label=True`；六个可评分类为 impervious/building/low-vegetation/tree/car/clutter。[MMSeg v1.2.2 source](https://raw.githubusercontent.com/open-mmlab/mmsegmentation/v1.2.2/mmseg/datasets/isprs.py) | 当前 CTP/RemoteCLIP 固定测试 areas 为 `11,15,28,30,34`，不是官方 SegEarth val split；并且 CTP 是五类，red clutter 作为 ignore。故**不能直接套用官方 Vaihingen config 后声称同协议比较**。 |
| Potsdam | **有** `cfg_potsdam.py`；官方 parent-tile val list 正好与本项目冻结的 14 个 parent tile 名称一致，但官方 converter 以 512/256 overlapping patches，当前 CTP 使用自己冻结的 512-patch/candidate/FusionCanvas protocol。[config](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/cfg_potsdam.py)；[converter](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/tools/dataset_converters/potsdam.py) | 官方 prompt file 的 6 类为 road/parking-lot、building、low vegetation、tree、car、clutter/background；本项目将 impervious surface、building、low vegetation、tree、car 作为五类，red clutter ignore。 [official classes](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/cls_potsdam.txt) | **最有希望做 external full-support 对比**，但仍须在运行前逐一验证原始 parent-tile、patch reconstruction、GT color/id、five-class evaluator 与 coverage。不是 controlled comparison。 |
| LoveDA | **有** `cfg_loveda.py` 与 converter；官方用 dataset `val`，`reduce_zero_label=True`，将 no-data label 0 ignore，六个评分前景类为 building/road/water/barren/forest/agriculture。[config](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/configs/cfg_loveda.py)；[converter](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/tools/dataset_converters/loveda.py) | 当前 CTP 是 LoveDA train 内按 `sha256(42:image_id)` 划分 1647 development / 411 heldout 的 region-level protocol，非官方 val dense-pixel split。 | 只有“同一数据集名称”，没有同一 split/evaluation unit；当前阶段不应运行，也不能并列数值比较。 |

**类别映射的硬门。** SegEarth-OV 的 Vaihingen/Potsdam 官方词表含 clutter，而当前 CTP 是五类且把 clutter GT 置 ignore、并把 SAM3 uncovered/conflict 置 `255` ignore。SegEarth-OV 是稠密预测，预测 clutter 不能在事后静默改成 ignore，否则会让其避免本应承担的错误。因此任何 external wrapper 都必须预注册：

1. GT 只保留当前五类的有效像素；
2. SegEarth 的 clutter 预测在这些有效 GT 像素上计为错误，而不是删掉；
3. CTP 仅使用其已经冻结的 valid/common-pixel 口径；
4. 必须报告两种方法的 valid/ignored/coverage 像素数，不能仅比较单个 mIoU。

该规则只是一项未来可审计的**评分绑定条件**，不是修改 CTP、SegEarth 的核心算法或 prompt。

## 4. 可比性结论

### Controlled comparison：NO

SegEarth-OV 不使用本项目的 SAM3 candidates、visual prototypes、C2/SCC/CTP score construction 或 FusionCanvas。它以 CLIP patch feature、SimFeatUp、global-bias subtraction 和滑窗 dense logits 形成自己的完整 pipeline。两者的 proposal、resolution、coverage/uncovered 处理、backbone、postprocessing 都不同。

因此 SegEarth-OV 绝不能放入“OpenAI CLIP / RemoteCLIP / CTP only-backbone replacement”的 controlled 表，不能以“只换 backbone”解释，亦不能用其论文里的 mIoU 直接宣称胜过/不如 CTP。

### External full-support comparison：条件可行，当前尚未执行

若未来满足以下 preflight，Potsdam 可作为优先的 external full-support baseline；Vaihingen 只能标为“official core inference on the frozen five-area CTP test subset”，不能标成官方 SegEarth validation result：

1. 在独立、专属目录和独立 Python 环境固定官方 commit；不污染当前 RemoteCLIP 环境；
2. 验证 SimFeatUp 文件的 SHA-256、实际 OpenAI CLIP cache asset 的 URL/SHA-256、依赖 lock 与 GPU runtime；
3. 对每个 parent/image 保存输入 SHA-256、patch坐标/重建逻辑、官方 config hash 与 prediction-map hash；
4. 对照当前冻结的五类 GT map、ignore 规则与 split，先运行小型**无 GT 推理 preflight**，再由已登记 evaluator 计算 OA/Macro-F1/mIoU 和 coverage；
5. 结果表标题显式标记 `external method comparison — protocol-different`，且不与 CTP controlled table 混合。

这些条件尚未全部满足，故本审计不给出运行命令，也不建议本轮启动 inference。

## 5. Partial-support 与 CTP plug-in 审计

### Partial-support：NO，protocol-incompatible

SegEarth-OV 的 `name_path` 是固定 text-query class list；源码中不存在 support subset、support manifest、visual prototype、supported/unsupported score competition 或等价输入。通过删除/重排其 class list 来模拟 k=2/3/4 会改变预测词表，而不是“仅限制 visual/support information”，与冻结的 CTP partial-support 定义不同。

所以不得计算或声称 SegEarth-OV 的 S-IoU、U-IoU、H-IoU；应在论文中写明：**“Partial-support comparison is not protocol-compatible.”**

### CTP plug-in：NO，数学/协议均不匹配

冻结 CTP-v1 需要同一类别上的 text score `T_c`、由视觉 support 构造的 anchored score `A_c`，以及 supported/unsupported mask。SegEarth-OV 的公开推理仅产生从 image feature 到 text-query 的 dense semantic logits，外加 CLS-token global-bias subtraction；没有 CTP 所需的外部 visual support score 或 support partition。[segmentor source](https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/segearth_segmentor.py)

为它人工新增 prototype/support score 或重写 logits 将是新方法，违反本轮冻结要求。因此不生成 `segearth_ctp_plugin_plan.md`，也不进行 plug-in 实验。

## 6. 当前停止条件与下一阶段门槛

| 问题 | 审计结论 |
|---|---|
| 官方代码可用？ | **是**，固定 commit 已记录。 |
| 官方权重可用？ | **部分是**：SimFeatUp checkpoint 位于固定源码；OpenAI CLIP 实际 asset 尚无本地 SHA-256/落盘证据。 |
| Vaihingen 可直接按当前 CTP 协议跑？ | **否**：官方 val split 与当前五个 CTP test areas 不同，且 six-class/coverage 定义不同。 |
| Potsdam 可成为 external baseline？ | **条件是**：parent-tile split 具有明显交集，但需完成输入/label/patch/coverage binding 后才能运行。 |
| LoveDA 应现在跑？ | **否**：split 与 evaluation unit 不同。 |
| 属于 target-dataset training-free inference？ | **是（限定含义）**：官方 evaluation 无 target training；但含预训练的 SimFeatUp。 |
| 能公平复用 CTP 的 OA/Macro-F1/mIoU？ | **尚未证明**；只能通过预测图导出 + 五类、coverage 显式的 external wrapper 达成。 |
| Partial-support 是否可公平实现？ | **否。** |
| SegEarth-OV + CTP 是否合理？ | **否。** |

**下一阶段建议（仅分析，不执行）。** 先完成本轮 RemoteCLIP 的 partial-support/area-bootstrap 证据。之后若仍需要 external baseline，应先做一个仅审计的、isolated deployment preflight：固定 commit 与两类实际权重 SHA-256，验证 Potsdam 14 parent tile 的输入/GT/patch-coordinate 对齐，并冻结 external-evaluation wrapper 的五类/coverage manifest。只有该 preflight 成功后，才可一次性运行官方 SegEarth-OV core 的 Potsdam full-support external baseline；Vaihingen 和 LoveDA 不应抢先运行。

## 7. 审计完整性检查

- 本文件仅新增审计文档；未生成 `outputs/`、prediction、feature、dataset 或模型文件。
- 未编辑 CTP-v1、SCC、Guard、C2、alpha、prompts、support subsets、SAM3 candidates、FusionCanvas 或任一第一篇项目文件。
- 所有外部方法事实均链接至官方仓库、固定 revision 的 raw source、官方项目页或 CVF 论文；未以第三方 fork、SegEarth-OV-2/3 或文献表格替代原始 optical SegEarth-OV 证据。
