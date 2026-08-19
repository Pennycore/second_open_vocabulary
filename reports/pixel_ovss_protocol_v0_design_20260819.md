# Pixel-level OVSS Protocol v0（设计文档）

日期：2026-08-19
状态：frozen_pre_result（**在读取任何 pixel GT 前冻结**）
配置文件：`configs/pixel_ovss_protocol_v0.json`

## 1. 目标

将已验证的 region-level CTP-v1 校准转化为 **open-vocabulary remote sensing semantic segmentation**，不改方法、不设计新 proposal 网络、不边看 GT 边调规则。

## 2. 四要素（协议要求逐项明确）

### 2.1 region/mask proposal 来源（冻结）

- **继承第一篇 SAM3 candidate masks**（用户授权 Vaihingen 重跑已完成的产物；LoveDA 用第一篇既有 candidate caches）。
- 禁止：新 proposal 网络、GT 派生 proposals、超出现有冻结资产的 SAM3 rerun。

### 2.2 semantic assignment（替换为 OpenAI CLIP + CTP）

- 每 region：OpenAI CLIP ViT-B/32 quick-GELU 特征 → 冻结 prototypes → 按方法分数 argmax。
- 方法：Text-only / C2 / SCC / CTP（CTP 引用 `configs/ctp_v1_frozen.json`，alpha=0.5，8 Group-A prompts，类名数据集特定）。

### 2.3 mask fusion（预定义，参考第一篇 FusionCanvas）

- overlap：逐像素取最高 region score；平局按 class id 序；冲突 margin 0.03。
- uncovered：`uncovered_label = 255`（ignore）。
- background：词表内无独立 background 类；未覆盖=ignore；Vaihingen clutter（红）= GT ignore。
- ignore index：255。
- semantic map：uint8 每图语义图（region class × mask 填充）。

### 2.4 evaluation protocol（GT 隔离）

```
prediction 完成 → 预测/配置/support manifest 哈希固化 → 读取 GT → evaluation
```

- predict 阶段禁止访问 GT/统计/调规则；evaluate 阶段先校验哈希一致才解锁 GT。
- 指标：OA、mIoU、Macro F1、S-IoU、U-IoU、H-IoU、per-class IoU、confusion matrix。
- partial-support：LoveDA 2^6=64 / Vaihingen 2^5=32 subsets 枚举（或 25/50/75%×seed42/43/44 预注册）；subsets 在读 GT 前保存。
- bootstrap：seed 42、5000 repeats、image cluster；CTP vs Text-only/C2/SCC/Guard 的 Delta 95% CI。

## 3. Pipeline 阶段

1. 加载冻结 SAM3 candidate masks（test 图）
2. OpenAI CLIP 编码 region crops（冻结规则）
3. 逐 region 计算 T_c / A_c / S_c^SCC / CTP 决策
4. 逐方法分配 region class
5. FusionCanvas 组装语义图（overlap/uncovered/ignore 冻结）
6. 持久化语义图 + scores + 配置 + 哈希
7. 读 GT
8. pixel 指标 + partial-support S/U/H + cluster bootstrap

## 4. 主实验设计（Phase 6 预留）

| 方法 | 定位 |
|---|---|
| Text-only segmentation | 开放词表下限 |
| C2 segmentation | naive fusion（supported 提升 / unsupported 崩溃） |
| SCC segmentation | global centering ablation |
| **CTP segmentation** | **最终方法** |

指标：OA、mIoU、Macro F1、S-IoU、U-IoU、H-IoU、per-class IoU、confusion matrix；GT 隔离与哈希固化同 region-level 纪律。

## 5. 论文实验结构（Phase 7 预留）

- **Experiment 1**：Weak visual anchoring effectiveness —— Text-only < Visual anchor（LoveDA blind + Vaihingen）。
- **Experiment 2**：Naive fusion failure —— C2 提升 supported 但破坏 unsupported vocabulary。
- **Experiment 3**：Calibration comparison —— SCC / Guard / CTP 的 supported/unseen trade-off（CTP 更优）。
- **Experiment 4**：Cross-dataset confirmation —— LoveDA + Vaihingen（region-level，已完成）。
- **Experiment 5**：Pixel-level OVSS —— CTP 提升 open-vocabulary semantic segmentation（本协议执行）。

## 6. 可执行性结论

✅ **可执行**：所有冻结资产在位（Vaihingen 16 pixel maps + 16 candidate caches；LoveDA 第一篇 candidate caches；CTP-v1 frozen；FusionCanvas 实现于第一篇）。下一阶段按本协议实现 pipeline（继承第一篇 mask 工具，类别赋值替换为 OpenAI CLIP + CTP），先冻结 pixel protocol 哈希，再执行主实验。

## 7. 禁止项

训练 / adapter / learned gating / 新 proposal 网络 / alpha / prompt / threshold / margin 调参 / temperature / beta / GT 前读 GT / 看 GT 调规则 / RemoteCLIP / DINO / multi-prototype / overwrite。
