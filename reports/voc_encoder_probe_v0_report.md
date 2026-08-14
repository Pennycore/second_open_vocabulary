# VOC2012 无训练双编码器 sanity check

状态：completed；正式运行目录（3090）：`outputs/voc_encoder_probe_v0/run_20260814_001`。

## 设计边界

- 数据：VOC2012 segmentation train，共1,464张；未使用VOC val。
- 监督：官方图像级类别。1,151张来自 `ImageSets/Main/*_train.txt`，313张仅从XML的目标类别字段补齐。
- 未读取 `SegmentationClass` 像素值，未运行SAM/SAM3、候选生成或训练。
- 两个模型分别使用自己的匹配图像塔、文本塔和官方预处理；不跨编码器复用特征。
- 该实验只检验自然图像整图语义匹配，不是分割精度或开放词表分割证据。

## 结果

| 模型 | 20类 macro AP | 平均 recall@真实标签数 |
|---|---:|---:|
| RemoteCLIP ViT-B/32 | 52.98% | 59.88% |
| OpenAI CLIP ViT-B/32 quick-GELU | 77.81% | 81.50% |

主对比为 OpenAI CLIP 减 RemoteCLIP：配对image bootstrap均值 **+24.62 pp**，95% CI **[+22.46, +26.80] pp**，2,000次重采样，seed=42。20个类别的AP均由OpenAI CLIP更高。

## 决策

不把RemoteCLIP全局替换为CLIP。第二篇采用dataset-independent核心和encoder registry：自然图像基准默认OpenAI CLIP，遥感基准保留RemoteCLIP；proposal、region-text assignment、fusion和评估接口保持共享。这个结论与LoveDA区域弱标签诊断中RemoteCLIP更强并不冲突，反而支持域匹配编码器的设计。

下一步仍不运行SAM3。先冻结VOC proposal-provider小样本协议与可替换接口；若继续遵守“不重新运行SAM3”，工程门禁使用非SAM3的轻量class-agnostic proposal baseline，不能把它的结果伪装成最终方法。

## 复现锚点

- code commit: `b415a504d0d3919917ed54539c7019ea355eb0fc`
- protocol SHA-256: `8706b86ba390c071bdbf125697ecc991ed569dc219a5ed58bef7ac9144fcee29`
- summary SHA-256: `c98a3d677a289e63efb22574ba3a93132b7fc31712d3e7c6596f354d8c5bf14b`
- encoder outputs SHA-256: `39147d2371219bee3d993ffa7285b4f5e17cf182df30cd29002d330da4bd1bcf`

VOC官方开发包说明分类任务的 `-1/1/0` 分别是negative/positive/only-difficult，并允许训练时自行处理difficult；本实验预注册为类别存在。[VOC2012 development kit](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)
