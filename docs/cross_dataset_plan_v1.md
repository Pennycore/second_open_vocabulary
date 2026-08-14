# 跨数据集 OV-WSSS 计划 v1

第二篇不再以 LoveDA 六类为模型结构边界。核心方法必须同时支持遥感与自然图像，数据集差异只通过 adapter 和 registry 注入。

## 基准顺序

1. **VOC 2012 + SBD train_noval**：第一自然图像基准。VOC分割任务是20个前景类加背景；SBD扩充时必须排除VOC2012 val。
2. **LoveDA**：遥感域迁移验证，保留现有证据和六折协议。
3. **COCO 2014**：80类大规模压力测试，仅在VOC无训练门禁通过后启动。

## 核心架构

```text
dataset adapter
  ├─ image + image-level tags
  ├─ class ontology / prompt nouns
  ├─ split manifest
  └─ background / ignore semantics
            │
proposal provider（SAM3只是一个实现）
            │
frozen VLM image tower + matched text tower
            │
dataset-independent region-text assignment
            │
dataset adapter semantic fusion / evaluation
            │
optional student（门禁通过后才训练）
```

核心代码禁止依赖LoveDA图像命名、固定1024尺寸、六类张量、类别ID连续或SAM3缓存字段。

## VOC执行边界

- 初始下载只包括VOC2012 train/val与SBD所需文件，全部放在第二篇服务器目录。
- 下载文件绑定官方MD5与SHA-256；VOC主压缩包使用公开镜像加速，但最终字节必须通过VOC官方MD5。
- 解包进入全新staging目录，拒绝绝对路径、`..`、反斜杠、链接、设备文件和已有输出覆盖；通过split审计后再原子改名。
- 弱训练不直接读取训练像素掩码；若图像标签由标注派生，必须披露。
- VOC train的初始图像级标签优先取自官方 `ImageSets/Main/<class>_train.txt`；该分类split未覆盖的313张segmentation-train新增图像，仅从同一官方包的XML目标类别字段补齐。分类标签`0`（only difficult）和XML difficult目标都按“类别存在”处理，且不读取分割掩码。
- VOC2012 val不进入SBD增强训练。
- 先运行固定小样本的RemoteCLIP桥接和无训练region-text探针，再决定是否生成全量候选。

当前执行状态（2026-08-14）：VOC2012、SBD和官方 `train_noval.txt` 均已通过注册MD5并安全解包。审计得到VOC图像17,125张、分割train/val为1,464/1,449；SBD图像和类别MAT各11,355份，原始train/val为8,498/2,857，官方 `train_noval` 为5,623且与VOC2012 val交集为0。安全准备器、标签adapter及其协议在3090环境通过40项测试。整个准备阶段未读取像素标注值、运行SAM3、生成候选或启动训练。

这里的5,623张官方 `train_noval` 不得写成常见的10,582张VOC `train_aug`。若后续采用10,582张增强训练池，必须另行冻结“全部SBD标注图像减VOC val”的构造规则、ID清单和哈希，并作为不同split报告。当前先用VOC segmentation train的1,464张做低成本、无训练sanity check。

无训练sanity check已完成：在同一1,464张VOC train整图和同一冻结prompt下，OpenAI CLIP macro AP为77.81%，RemoteCLIP为52.98%；OpenAI CLIP减RemoteCLIP的配对bootstrap差值为+24.62pp，95% CI [+22.46,+26.80]pp。该结果只支持“自然图像编码器优先CLIP”的域内选择，不是分割性能。LoveDA继续保留RemoteCLIP，核心pipeline通过encoder registry共享。

## COCO执行边界

COCO不是原生语义分割标签格式。正式评估前必须冻结实例到语义标签的转换，明确crowd区域和不同类别实例重叠的处理。COCO下载与候选生成需要单独的存储和算力批准，不能随VOC自动启动。
