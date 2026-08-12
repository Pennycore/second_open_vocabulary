from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import (  # noqa: E402
    FeatureBundle,
    build_input_manifest,
    create_run_dir,
    environment_text,
    inspect_configured_inputs,
    load_config,
    load_feature_bundle,
    load_region_bundle,
    seed_everything,
    write_json,
    write_yaml,
)
from ov_probe.metrics import l2_normalize  # noqa: E402
from ov_probe.probe import (  # noqa: E402
    predeclared_error_analysis,
    run_multi_probe,
    run_region_probe,
    run_single_probe,
)
from ov_probe.prompts import (  # noqa: E402
    HashTextEncoder,
    CachedTextEncoder,
    RemoteCLIPTextEncoder,
    build_prompt_bank,
    encode_prompt_group,
)
from ov_probe.visualization import (  # noqa: E402
    save_heatmap,
    save_margin_plot,
    save_prompt_stability,
    save_rank_plot,
)


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ov_probe")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", mode="x", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def synthetic_bundles(cfg: dict[str, Any], bank: dict[str, Any]) -> tuple[FeatureBundle, FeatureBundle, FeatureBundle, HashTextEncoder]:
    seed = int(cfg["experiment"]["seed"])
    rng = np.random.default_rng(seed)
    encoder = HashTextEncoder(int(cfg["model"]["feature_dim"]), seed)
    class_names, text, _ = encode_prompt_group(encoder, bank, "A", "closed")
    single = l2_normalize(text + rng.normal(0, 0.015, text.shape).astype(np.float32))
    multi_rows, multi_names, proto_ids, sizes = [], [], [], []
    region_rows, cam, sam3 = [], [], []
    for index, name in enumerate(class_names):
        for sub_index, noise in enumerate((0.012, 0.025, 0.05)):
            multi_rows.append(text[index] + rng.normal(0, noise, text[index].shape))
            multi_names.append(name)
            proto_ids.append(f"{name}_{sub_index}")
            sizes.append(40 + index * 5 + sub_index)
        for region_index in range(5):
            region_rows.append(text[index] + rng.normal(0, 0.04, text[index].shape))
            cam.append(name)
            sam3.append(name if region_index != 4 else class_names[(index + 1) % len(class_names)])
    metadata = {"provenance": {"synthetic": True, "scientific_evidence": False}}
    return (
        FeatureBundle(single, class_names, metadata),
        FeatureBundle(l2_normalize(np.asarray(multi_rows)), multi_names, metadata, prototype_ids=proto_ids, cluster_sizes=np.asarray(sizes)),
        FeatureBundle(l2_normalize(np.asarray(region_rows)), ["unknown"] * len(region_rows), metadata, cam_labels=cam, sam3_source_labels=sam3),
        encoder,
    )


def prompt_stability_summary(frame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (group, vocabulary), subset in frame.groupby(["prompt_group", "vocabulary"]):
        result.setdefault(str(group), {})[str(vocabulary)] = {}
        for name, values in subset.groupby("visual_class", sort=False):
            result[str(group)][str(vocabulary)][str(name)] = {
                "correct_similarity_mean": float(values["correct_similarity"].mean()),
                "correct_similarity_std": float(values["correct_similarity"].std(ddof=0)),
                "correct_similarity_min": float(values["correct_similarity"].min()),
                "correct_similarity_max": float(values["correct_similarity"].max()),
                "rank_min": int(values["correct_rank"].min()),
                "rank_max": int(values["correct_rank"].max()),
                "margin_mean": float(values["positive_negative_margin"].mean()),
                "margin_std": float(values["positive_negative_margin"].std(ddof=0)),
            }
    return result


def write_project_report(path: Path, run_dir: Path, status: str, cfg: dict[str, Any], summary: dict[str, Any], blocked_reasons: list[str]) -> None:
    classes = ", ".join(item["name"] for item in cfg["data"]["classes"])
    if status == "completed":
        scientific = summary.get("scientific_evidence", False)
        if scientific:
            single = summary["single"]["A"]["closed"]
            result_text = (
                f"Group A closed-vocabulary Top-1={single.get('top_1_accuracy', 'n/a'):.3f}, "
                f"mean rank={single['mean_correct_rank']:.3f}, mean margin={single['mean_positive_negative_margin']:.3f}."
            )
        else:
            result_text = "Only a synthetic pipeline dry run completed; its metrics are not scientific evidence."
    else:
        result_text = "The real Stage 0 probe is blocked; no alignment metric has been fabricated."
    missing = "\n".join(f"- {item}" for item in blocked_reasons) or "- None"
    text = f"""# OV-WSSS Stage 0 报告

## 1. 实验目标

检验第一篇论文已有的 LoveDA Train-only RemoteCLIP 视觉原型/区域特征与 RemoteCLIP 文本空间是否具备可靠类别对应关系。该探针不等于开放词汇语义分割。

## 2. 数据、split 与输入

- 数据集：{cfg['data']['dataset_name']}
- 预注册类别：{classes}
- split：train
- 第一篇候选项目（只读）：`{cfg['paths'].get('source_project_root')}`
- RemoteCLIP checkpoint（只读）：`{cfg['paths'].get('remoteclip_checkpoint')}`
- Main-v1 单原型：`{cfg['paths'].get('single_prototype_file')}`
- Main-v2 多原型：`{cfg['paths'].get('multi_prototype_file')}`
- 区域特征缓存：`{cfg['paths'].get('region_feature_cache')}`
- 运行目录：`{run_dir}`
- 状态：`{status}`

严格未使用 Val 或 pixel GT；未运行 SAM3、未生成伪标签、未训练 student、未微调 RemoteCLIP。

## 3. 单原型、扩展词表与 Prompt 结果

{result_text}

Group A、Group B、closed vocabulary 与 expanded vocabulary 均已在配置和 prompt bank 中完整预注册。正式输入缺失时，不报告任何伪造数值。

## 4. 多原型与区域级结果

正式 Main-v2 多原型或区域缓存缺失时对应实验跳过。区域级结果即使可用，也只衡量与 Train-only 弱标签的一致性，不是真实分类准确率。

跨数据集 E0.6 因没有 LoveDA/Potsdam 的合法 Train-only 同源原型和严格类别映射而跳过，未强行映射语义不同的类别。

## 5. 预注册困难类别错误分析

已固定分析 barren、forest、agriculture，同时保留 building、road、water 的指定干扰类；不会根据 Val 结果事后改词表或只选有利案例。

## 6. 缺失项/阻塞原因

{missing}

## 7. 当前证据支持与不支持的结论

当前证据至多支持“已有视觉表示能否在固定文本候选中匹配正确类别”的判断。它不支持以下声明：

- 已实现 open-vocabulary segmentation；
- 能分割真实 unseen classes；
- Train-only 弱标签一致率等价于像素级分割性能。

## 8. RemoteCLIP 决策与下一步最小实验

若真实 Stage 0 尚未完成，则不能判断 RemoteCLIP 是否值得进入下一阶段。应先提供带完整来源元数据的 Main-v1/Main-v2 或区域缓存，并在相同区域、相同 prompt、相同协议下完成本探针。若多数类排名稳定、margin 为正且扩展词表不崩溃，下一阶段才考虑 region-level text-guided semantic assignment；若整体较弱，再以相同协议比较 CLIP/SigLIP 等候选。此次不执行下一阶段。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def execute(cfg: dict[str, Any], dry_run: bool, run_dir: Path, logger: logging.Logger) -> tuple[str, dict[str, Any], list[str]]:
    bank = build_prompt_bank(cfg)
    write_json(run_dir / "prompt_bank.json", bank)
    write_json(run_dir / "class_mapping.json", {item["name"]: item["id"] for item in cfg["data"]["classes"]})
    blocked: list[str] = []
    if dry_run:
        single, multi, region, encoder = synthetic_bundles(cfg, bank)
        logger.info("Using explicitly synthetic 512-D features; results are pipeline validation only.")
    else:
        readiness = inspect_configured_inputs(cfg)
        if not readiness["ready"]:
            blocked = [f"{key}: {value['status']} ({value.get('path')})" for key, value in readiness["items"].items() if value["status"] != "ready"]
            return "blocked", {"scientific_evidence": False, "input_readiness": readiness}, blocked
        single = load_feature_bundle(cfg["paths"]["single_prototype_file"], cfg, "single")
        multi = load_feature_bundle(cfg["paths"]["multi_prototype_file"], cfg, "multi") if cfg["paths"].get("multi_prototype_file") else None
        region = load_region_bundle(
            cfg["paths"]["region_feature_cache"],
            cfg["paths"].get("weak_region_label_file"),
            cfg,
        ) if cfg["paths"].get("region_feature_cache") else None
        text_cache = cfg["paths"].get("text_feature_cache")
        encoder = CachedTextEncoder(text_cache, cfg) if text_cache else RemoteCLIPTextEncoder(cfg)
    single_result = run_single_probe(single, encoder, bank, cfg)
    single_result["long_frame"].to_csv(run_dir / "single_prototype_similarity.csv", index=False, mode="x")
    primary_scores, primary_candidates = single_result["matrices"]["A_closed"]
    primary_detail = single_result["details"]["A_closed"]
    save_heatmap(primary_scores, bank["class_names"], primary_candidates, run_dir / "single_prototype_text_heatmap.png", "Single prototype–text similarity (Group A, closed)")
    save_margin_plot(primary_detail, run_dir / "positive_negative_margin.png")
    save_rank_plot(primary_detail, run_dir / "per_class_rank.png")
    stability = single_result["prompt_stability"]
    save_prompt_stability(
        stability[(stability["vocabulary"] == "closed") & (stability["prompt_group"] == "A")].rename(columns={"prompt_index": "template_index"}),
        run_dir / "prompt_stability.png",
    )
    summary: dict[str, Any] = {
        "status": "completed",
        "scientific_evidence": not dry_run,
        "synthetic_dry_run": dry_run,
        "single": single_result["summary"],
        "prompt_stability": prompt_stability_summary(stability),
        "class_error_analysis": predeclared_error_analysis(single_result, cfg),
        "interpretation_limits": [
            "Prototype-text alignment is not open-vocabulary segmentation.",
            "Expanded-vocabulary ranking is not unseen-class segmentation.",
            "Weak-label agreement is not pixel-level segmentation accuracy.",
        ],
    }
    if multi is not None:
        multi_result = run_multi_probe(multi, encoder, bank, cfg)
        multi_result["frame"].to_csv(run_dir / "multi_prototype_similarity.csv", index=False, mode="x")
        summary["multi"] = multi_result["summary"]
        summary["subprototype_analysis"] = multi_result["subprototype_analysis"]
        if multi_result["primary_matrix"] is not None:
            save_heatmap(multi_result["primary_matrix"], bank["class_names"], bank["class_names"], run_dir / "multi_prototype_text_heatmap.png", "Multi-prototype mean aggregation (Group A, closed)")
    else:
        blocked.append("Main-v2 multi-prototype file not configured; E0.2 skipped.")
    if region is not None:
        region_result = run_region_probe(region, encoder, bank, cfg)
        write_json(run_dir / "region_level_results.json", region_result)
        summary["region"] = {
            "sampling": region_result["sampling"],
            **{
                group: {
                    vocab: {key: value for key, value in metrics.items() if key != "per_region"}
                    for vocab, metrics in region_result[group].items()
                }
                for group in ("A", "B")
            },
        }
    else:
        blocked.append("Region feature cache not configured; E0.5 skipped.")
    write_json(run_dir / "summary_metrics.json", summary)
    return "completed", summary, blocked


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OV-WSSS Stage 0 probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic features only to validate the software pipeline")
    args = parser.parse_args()
    cfg = load_config(args.config, PROJECT_ROOT)
    seed_everything(int(cfg["experiment"]["seed"]))
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    logger = configure_logging(run_dir)
    write_yaml(run_dir / "config_resolved.yaml", cfg)
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    manifest = build_input_manifest(cfg)
    manifest["run_mode"] = "synthetic_dry_run" if args.dry_run else "formal"
    write_json(run_dir / "input_manifest.json", manifest)
    logger.info("Allocated non-overwriting run directory: %s", run_dir)
    try:
        status, summary, blocked = execute(cfg, args.dry_run, run_dir, logger)
        if status == "blocked":
            write_json(run_dir / "summary_metrics.json", {"status": status, **summary, "blocked_reasons": blocked})
            logger.error("Formal run blocked: %s", "; ".join(blocked))
        else:
            logger.info("Probe pipeline completed. scientific_evidence=%s", summary["scientific_evidence"])
        if not args.dry_run:
            report = PROJECT_ROOT / "reports" / f"{cfg['experiment']['name']}_report.md"
            if report.exists():
                logger.warning("Project report already exists and was not overwritten: %s", report)
            else:
                write_project_report(report, run_dir, status, cfg, summary, blocked)
        print(json.dumps({"status": status, "run_dir": str(run_dir), "blocked_reasons": blocked}, ensure_ascii=False))
        return 0 if status == "completed" else 2
    except Exception as exc:
        logger.error("Probe failed: %s", exc)
        logger.debug(traceback.format_exc())
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        if not (run_dir / "summary_metrics.json").exists():
            write_json(run_dir / "summary_metrics.json", failure)
        print(json.dumps({"status": "failed", "run_dir": str(run_dir), **failure}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
