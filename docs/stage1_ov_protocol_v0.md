# Stage 1 OV-WSSS 协议 v0

## 决策

Stage 0 的预注册成对比较支持继续使用 RemoteCLIP ViT-B/32。OpenAI CLIP ViT-B/32 quick-GELU 保留为冻结基线，不替换主编码器。

当前不启动 student 训练，也不把现有 6000 个平衡区域称为严格的 unseen benchmark。该像素包按 SAM3 来源弱类别每类采样 1000 个区域，适合编码器成对诊断，但其抽样已受弱类别条件影响。

## 类别轮换

LoveDA 只有六个前景类，因此采用六折 leave-one-class-out：每折留出一个类别，其余五类为 seen。所有六折必须报告，不能观察结果后只选择有利类别。

## 第一阶段：无训练代理实验

第一阶段只测试冻结 RemoteCLIP 的 held-out 文本匹配能力：

1. 候选集合必须按图像 ID 冻结或使用全部候选，不能按 VLM 分数、预测、margin、pixel GT 或 held-out 正确性选样本。
2. 每折只允许用 seen 类构建视觉原型或统计量。
3. unseen 类不提供视觉原型、校准样本或阈值调节。
4. 文本词表包含六个类和 Stage 0 已冻结的干扰词。
5. 统计推断以图像为聚类单位，并汇总六折结果。

这仍然只是一项弱标签代理诊断，不能证明真实 unseen 分类或分割。

## 第二阶段：最终像素级评价

方法与全部超参数冻结后，才允许使用 LoveDA Val pixel GT 做一次最终评价。Val 不能用于 prompt、阈值、fold、checkpoint 或方法选择。最终必须报告全部六折的 seen mIoU、unseen mIoU、调和平均、逐类 IoU 和折间离散程度。

## 当前阻塞项

3090 上现有 6000 区域包不满足严格 held-out 抽样要求。下一项允许执行的工程任务，是在不运行 SAM3 的前提下，从既有 Train 图像和候选缓存建立一个不按类别平衡、不按模型预测筛选的不可变候选包；其规则和哈希必须先冻结，再产生任何 held-out 结果。

在该数据契约完成前，不运行 Stage 1 结果实验，不训练 student，也不读取 LoveDA Val pixel GT。

