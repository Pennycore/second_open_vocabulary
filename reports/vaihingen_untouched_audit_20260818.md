# Phase R：Vaihingen untouched 状态审计

审计日期：2026-08-18
审计范围：本地项目（`C:\Users\28457\Desktop\open_vocabulary`）、服务器（172.18.56.240：`/home/undergr/` 全盘）、Git 历史、第一篇项目

## 1. 审计结果

| 问题 | 结论 |
|---|---|
| Vaihingen 是否已存在于当前项目？ | **否（无现成数据）** |
| 是否曾用 Vaihingen GT 调整过 prompt/alpha/prototype/SCC/threshold/region selection/segmentation rule？ | **否（无数据即无使用）** |
| 是否曾看过 SCC/C1/C2 在 Vaihingen 上的指标？ | **否** |

### 详细证据

1. **本地项目**：`C:\Users\28457\Desktop\open_vocabulary` 内无任何 `vaih*`/`isprs*` 文件或目录；git 全历史（`git log --all`）无 Vaihingen/ISPRS 相关提交。
2. **服务器 remote_dataset**：`/home/undergr/remote_dataset/` 仅有 `LoveDA_incoming / LoveDA_main_v1 / LoveDA_raw / Postdam / Postdam_patches_256_paper / Postdam_patches_512 / Postdam_patches_512_full`——**无 Vaihingen 目录**。
3. **服务器全盘搜索**（`find /home/undergr -iname "*vaihingen*"`）：仅命中第三方参考仓库 `experiment/SegEarth-OV-3-main/configs/` 下的两个**配置文件**（`cfg_vaihingen.py`、`cls_vaihingen.txt`，类别列表 road/building/grass/tree/car/clutter），其 `data_root='data/Vaihingen'` 指向的目录**不存在**（`SegEarth-OV-3-main/data/` 无此目录）。这是开源的 SegEarth-OV 第三方实现（related_work），不是本项目数据，也从未运行。
4. **第一篇项目**（`Sheungzhen_project_1`）：无 Vaihingen。
5. **本地桌面其他项目**：`exp_code` 下仅有 mmsegmentation 单元测试的 `pseudo_vaihingen_dataset`（合成伪数据，非真实 Vaihingen）。

## 2. 结论（按用户协议）

- Vaihingen **无现成数据**，且从未参与本论文方法开发（三项检查均为否）。
- 按协议："如果项目中没有现成 Vaihingen 数据或该数据此前已经用于本论文方法开发，则**不要自行替换数据集，先汇报实际情况并停止**。"
- 因此：**停止执行 Phase S/T 外部盲测**，不自行替换为 Potsdam 或其他数据集。

## 3. 供决策的现状盘点（不自行选择）

若用户决定更换外部确认数据集，当前可获得的候选资源（未参与 SCC 开发）：

| 候选 | 服务器路径 | 备注 |
|---|---|---|
| Potsdam（ISPRS 2D） | `/home/undergr/remote_dataset/Postdam`（9.0G）、`Postdam_patches_512_full`（7.2G）、`Postdam_patches_256_paper`（3.8G）、`Postdam_patches_512`（200M） | 有 GT label（`5_Labels_all/*_label.tif`）；是否触碰过需另行审计其"已参与第一篇方法开发"的历史 |
| LoveDA（外部子集） | 已全部用于本论文开发 | 不满足 untouched |
| VOC 2012 | 已用于 sanity check | 非遥感 |

**注意**：Potsdam 在第一篇论文工程中有对应工作（`Postdam_patches_256_paper` 命名暗示），若选用 Potsdam 必须另行审计其是否"已参与本论文方法开发"（第一篇 vs 第二篇的独立性）；该判断留给用户，本阶段不执行。

## 4. 已完成的冻结前置工作（不受 Vaihingen 缺失影响）

- Phase P：LoveDA 冻结前最终指标审计（OA/Macro F1/mIoU + S/U/H-F1 + S/U/H-IoU，k=0..6 汇总，mean(H_i) 聚合已注明）✅
- Phase Q：`configs/scc_v1_frozen.json` + `reports/scc_v1_freeze_record_20260818.md` ✅
- SCC-v1 冻结 commit：`f5d7d91`（待本阶段提交后更新）

## 5. 下一步（等待用户决定）

1. 由用户提供 Vaihingen 数据（官方 ISPRS Vaihingen 2D 需授权下载）；或
2. 用户明确批准改用 Potsdam（需先完成 Potsdam untouched 审计）；或
3. 用户决定其他数据集。

在获得明确指示前，不执行任何外部 blind 实验。
