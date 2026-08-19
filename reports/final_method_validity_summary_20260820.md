# 最终方法有效性总结（Review-Defense 结题）2026-08-20

日期：2026-08-20
前置：CTP-v1 冻结 commit `f54c034`（`configs/ctp_v1_frozen.json` SHA-256 `788f1962…`）；本报告只做审计与分类，**不修改任何方法**。

## 1. 结论分类（Phase E）

依据规则逐条核对（全部证据见同目录三份审计报告）：

| 判据 | 证据 | 结果 |
|---|---|---|
| Guard H-IoU 显著更高？ | Vaihingen 聚类 bootstrap 6/9 sig−；Potsdam 非退化 6/6 sig−；LoveDA 相反（CTP ≥ Guard 聚类占比 ≈98%） | 部分成立（像素级 Vaih/Potsdam 成立，区域级 LoveDA 不成立） |
| CTP S-IoU 明显更高？ | Vaih common 7/9、Pots 5/6、LoveDA 全部 k（S-F1/S-IoU CI 正） | **成立** |
| CTP OA/mIoU 明显更高？ | LoveDA 全部显著正（ΔOA 0.05–0.26）；Vaih common OA 5/9；Pots 非退化 4/6 | 多数成立 |

**分类：Case 2** —— Guard 的 H-IoU 显著更高（像素级数据集），但 CTP 的 S-IoU/OA/mIoU 在多数子集明显更高。CTP **保持主方法**，但论文**不得声称 CTP 在所有指标上最佳**，措辞为 **soft adaptation–preservation trade-off**。

## 2. 措辞规范（Phase F）

- 一律使用 **"alleviates support-induced vocabulary bias"**（缓解），**绝不使用 "eliminates"**（消除）。
- 推荐表述（论文结论用）：
  > CTP prevents catastrophic suppression of unsupported categories while retaining a substantial portion of the gains introduced by weak visual anchoring.
- Guard 定位（与冻结记录一致）：**hard text-preservation baseline / upper bound**（U/H 上界），不是方法候选、不是 oracle（不读 GT）。

## 3. 证据链摘要（A–D）

1. **共同像素公平性（A）**：Ω_common 上 CTP vs C2 的 H-IoU 优势 18/18 子集成立（C2=0）；CTP vs SCC 优势 Vaih 8/9、Pots 9/9；C2 的 OA 领先在共同像素上缩小或消失（k4_s42：+0.044→+0.014）；无方法通过忽略像素获得表面优势（C2/Guard coverage 最低）。
2. **正确聚类 bootstrap（B）**：image_id / area / parent-tile 聚类（seed 42，5000 repeats）：CTP vs C2 三数据集全子集显著正；LoveDA CTP≥Guard 聚类占比 ≈98%；Vaih/Potsdam Guard H 更高但 CTP S 更高；方向一致性逐聚类记录（`per_cluster_deltas.csv`）。
3. **Guard 像素级基线（C）**：冻结规则逐 region 应用，五方法 orig+common 全表；Guard U/H 上界、CTP S/OA 折中，trade-off 量化。
4. **定性（D）**：Targeted Recovery（既有 5 例，目录未动）+ Representative（seed 42 随机 4 例，7 面板含 Guard）+ Failure（CTP 失败而 Guard 恢复的 top-3，规则预注册）；`outputs/final_audit/qualitative/selection_record_final.json`。

## 4. 最终问题回答（Phase I）

1. **CTP 是否仍是最终方法？** **是**。Case 2 分类下 CTP 保持主方法：它在全部 92 个子集对比（56 LoveDA + 18 Vaih + 18 Potsdam）中显著优于 C2（failure baseline，H-IoU 与 OA 全正），在像素级显著优于 SCC（Potsdam 6/6 显著；Vaihingen 6/9 显著、点估计 8/9 为正），在 LoveDA 上全面不劣于 Guard；Guard 仅在其 U/H 维度占优。
2. **Guard 的定位**：**hard text-preservation baseline / upper bound**（U/H 维度），作为对照基线写入论文，不作为候选方法（它在 S/OA 维度系统性落后于 CTP）。
3. **措辞 "preserve" 还是 "alleviate"？** **alleviate**（缓解）。像素级证据显示 unsupported 类别仍被部分抑制（如 Potsdam r75_s42 CTP U-IoU 0.077），"preserve/eliminate" 均过度承诺；用 §2 推荐句。
4. **论文写作前剩余缺口**：
   - (a) 像素级 LoveDA partial-support 未执行（本次 LoveDA 仅区域级）；若论文主实验含 LoveDA 像素级，需按冻结协议补跑（预测阶段可先做，GT 隔离不变）。
   - (b) C2 全支持 OA 高于 CTP 的现象（Potsdam full-support：0.6533 vs 0.6477）已由 score-scale 消融（A1/A2/A3）解释，论文需引用该消融；本次共同像素审计显示部分支撑下 C2 的 OA 领先缩小或消失，可一并引用。
   - (c) Failure cases 提示 CTP 在大面积 unsupported 场景（Potsdam r50/r75）H-IoU 落后 Guard 0.1–0.3，论文 Limitations 需如实讨论，并给出 trade-off 分析（S 侧收益 vs H 侧损失）。
   - (d) 指标聚合口径：H 为逐子集 H 的均值（非 H(均值 S, 均值 U)），论文需与方法部分一致。
   - (e) 计算开销：CTP 仅为冻结分数比较（无新推断），与 SCC 同量级，可在论文中声明。

## 5. 交付物清单

- 审计代码：`src/ov_probe/final_audit.py`、`scripts/run_final_audit.py`、`scripts/run_loveda_cluster_bootstrap.py`、`scripts/final_qualitative.py`
- 测试：`tests/test_final_audit.py`（8 passed：共同像素交集正确性、共同 mask 一致性、parent-tile/area 聚类映射、Guard 确定性、support manifest 未变、冻结 CTP hash 未变、预测阶段无 GT 配置）
- 数据（服务器 `outputs/final_audit/`）：`common_pixel_metrics_{vaihingen,potsdam}.json`、`five_method_metrics_{vaihingen,potsdam}.json`、`cluster_bootstrap_{vaihingen,potsdam,loveda}.json`、`qualitative/`（7 面板 × 7 图 + 3 份 selection record）
- CSV：`outputs/final_audit/common_pixel_metrics.csv`、`guard_pixel_partial_support.csv`、`cluster_bootstrap_summary.csv`、`per_cluster_deltas.csv`
- 定性记录：`reports/qualitative_selection_record_final.json`（副本，原始在服务器 `outputs/final_audit/qualitative/`）
- 报告：本目录 4 份 md
