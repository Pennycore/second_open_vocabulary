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
- 弱训练不直接读取训练像素掩码；若图像标签由标注派生，必须披露。
- VOC2012 val不进入SBD增强训练。
- 先运行固定小样本的RemoteCLIP桥接和无训练region-text探针，再决定是否生成全量候选。

## COCO执行边界

COCO不是原生语义分割标签格式。正式评估前必须冻结实例到语义标签的转换，明确crowd区域和不同类别实例重叠的处理。COCO下载与候选生成需要单独的存储和算力批准，不能随VOC自动启动。

