from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .encoder_bridge import _iter_pixel_views
from .io import InputValidationError, sha256_file, write_json


def load_analysis_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Analysis config must be a mapping with overwrite=false.")
    for key, value in cfg["paths"].items():
        candidate = Path(str(value))
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise InputValidationError(f"Analysis path must remain inside the project root: {key}") from exc
        cfg["paths"][key] = str(resolved)
    if Path(cfg["paths"]["output_root"]).parent != (root / "outputs").resolve():
        raise InputValidationError("Analysis output must be directly under project outputs/.")
    protocol_path = Path(cfg["paths"]["protocol_file"])
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol["path"] = str(protocol_path)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_analysis_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != expected_commit:
        raise InputValidationError("Analysis code commit differs from the approved commit.")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    if dirty.strip():
        raise InputValidationError("Tracked analysis worktree must be clean.")
    actual = sha256_file(root / "configs" / "encoder_compare_analysis_protocol_v0.json")
    if actual != expected_protocol_sha256:
        raise InputValidationError("Analysis protocol differs from the approved SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def classify_pair(source: str, remote: str, clip: str) -> str:
    remote_correct = remote == source
    clip_correct = clip == source
    if remote_correct and clip_correct:
        return "both_correct"
    if remote_correct:
        return "remoteclip_only_correct"
    if clip_correct:
        return "openai_clip_only_correct"
    return "both_wrong_same" if remote == clip else "both_wrong_different"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stable_examples(
    records: list[dict[str, Any]], categories: np.ndarray, protocol: dict[str, Any]
) -> list[int]:
    selected: list[int] = []
    seed = int(protocol["seed"])
    count = int(protocol["example_selection"]["per_class_category"])
    for class_name in protocol["classes"]:
        for category in protocol["example_selection"]["categories"]:
            candidates = []
            for index, (record, actual_category) in enumerate(zip(records, categories)):
                if record["sam3_source_label"] != class_name or actual_category != category:
                    continue
                identity = f"{seed}:{record['image_id']}:{record['candidate_index']}:{category}"
                candidates.append((hashlib.sha256(identity.encode()).hexdigest(), index))
            selected.extend(index for _, index in sorted(candidates)[:count])
    return selected


def _selected_views(package: Path, selected: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rows, contexts, _ in _iter_pixel_views(package):
        for row, context in zip(rows, contexts):
            index = int(row)
            if index not in selected:
                continue
            # Re-read the corresponding packed mask directly through the iterator's
            # masked view: pixels changed by masking identify background except for zeros.
            # The exact boolean mask is loaded below from the shard for clean plotting.
            result[index] = (context, np.ones(context.shape[:2], dtype=bool))
    # Replace placeholder masks with exact packed masks.
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    for shard_name in sorted(name for name in manifest["artifacts"] if name.startswith("shards/")):
        with np.load(package / shard_name, allow_pickle=False) as archive:
            rows = np.asarray(archive["row_indices"], dtype=np.int32)
            shapes = np.asarray(archive["crop_shapes"], dtype=np.int32)
            offsets = np.asarray(archive["crop_mask_offsets"], dtype=np.int64)
            bits = np.asarray(archive["crop_mask_bits"], dtype=np.uint8)
        for local, row in enumerate(rows):
            index = int(row)
            if index not in selected:
                continue
            height, width = (int(value) for value in shapes[local])
            mask = np.unpackbits(
                bits[int(offsets[local]) : int(offsets[local + 1])],
                bitorder="little",
                count=height * width,
            ).reshape(height, width).astype(bool)
            result[index] = (result[index][0], mask)
    if set(result) != selected:
        raise InputValidationError("Could not recover every registered qualitative example.")
    return result


def run_descriptive_analysis(cfg: dict[str, Any], protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    package = Path(cfg["paths"]["pixel_pack"])
    comparison = Path(cfg["paths"]["comparison_run"])
    registered = protocol["registered_inputs"]
    checks = {
        "pixel_pack_manifest_sha256": sha256_file(package / "manifest.json"),
        "pixel_pack_records_sha256": sha256_file(package / "records.jsonl"),
        "comparison_summary_sha256": sha256_file(comparison / "summary.json"),
        "comparison_outputs_sha256": sha256_file(comparison / "encoder_outputs.npz"),
    }
    for key, actual in checks.items():
        if actual != registered[key]:
            raise InputValidationError(f"Registered descriptive-analysis input changed: {key}")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest["bundle_id"] != registered["bundle_id"]:
        raise InputValidationError("Pixel-pack bundle differs from analysis registration.")
    records = [json.loads(line) for line in (package / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    group = protocol["primary_group"]
    with np.load(comparison / "encoder_outputs.npz", allow_pickle=False) as archive:
        remote = np.asarray(archive[f"predictions_remoteclip_{group}"])
        clip = np.asarray(archive[f"predictions_openai_clip_{group}"])
    source = np.asarray([row["sam3_source_label"] for row in records])
    categories = np.asarray([classify_pair(s, r, c) for s, r, c in zip(source, remote, clip)])
    classes = list(protocol["classes"])
    category_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for model_name, prediction in (("remoteclip", remote), ("openai_clip", clip)):
        matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for actual_index, actual in enumerate(classes):
            for predicted_index, predicted in enumerate(classes):
                matrix[actual_index, predicted_index] = int(np.sum((source == actual) & (prediction == predicted)))
                confusion_rows.append(
                    {"model": model_name, "source_class": actual, "predicted_class": predicted, "count": int(matrix[actual_index, predicted_index])}
                )
        matrices[model_name] = matrix
    for class_name in classes:
        class_mask = source == class_name
        remote_rate = float(np.mean(remote[class_mask] == class_name))
        clip_rate = float(np.mean(clip[class_mask] == class_name))
        per_class_rows.append(
            {"class": class_name, "remoteclip": remote_rate, "openai_clip": clip_rate, "clip_minus_remoteclip": clip_rate - remote_rate}
        )
        counts = Counter(categories[class_mask])
        for category in protocol["comparison_categories"]:
            category_rows.append({"class": class_name, "category": category, "count": int(counts[category])})
    _write_csv(output_dir / "primary_confusion_counts.csv", ["model", "source_class", "predicted_class", "count"], confusion_rows)
    _write_csv(output_dir / "per_class_primary_deltas.csv", ["class", "remoteclip", "openai_clip", "clip_minus_remoteclip"], per_class_rows)
    _write_csv(output_dir / "paired_outcome_categories.csv", ["class", "category", "count"], category_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for axis, model_name in zip(axes, ("remoteclip", "openai_clip")):
        normalized = matrices[model_name] / matrices[model_name].sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(model_name.replace("_", " ").title() + " — Group A closed")
        axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Text prediction")
        axis.set_ylabel("SAM3 source weak label")
        for row in range(len(classes)):
            for column in range(len(classes)):
                axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, label="Row proportion", shrink=0.85)
    figure.savefig(output_dir / "primary_confusion_heatmaps.png", dpi=180)
    plt.close(figure)

    example_indices = _stable_examples(records, categories, protocol)
    views = _selected_views(package, set(example_indices))
    example_rows = []
    figure, axes = plt.subplots(len(classes), 2, figsize=(8, 18), constrained_layout=True)
    for class_index, class_name in enumerate(classes):
        for category_index, category in enumerate(protocol["example_selection"]["categories"]):
            axis = axes[class_index, category_index]
            matches = [index for index in example_indices if source[index] == class_name and categories[index] == category]
            if not matches:
                axis.axis("off")
                axis.set_title(f"{class_name}: no {category}")
                continue
            index = matches[0]
            context, mask = views[index]
            axis.imshow(context)
            axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=0.8)
            axis.axis("off")
            axis.set_title(f"{class_name} | R={remote[index]} | C={clip[index]}\n{category}", fontsize=9)
            record = records[index]
            example_rows.append(
                {"row_index": index, "image_id": record["image_id"], "candidate_index": record["candidate_index"], "source_class": class_name, "cam_label": record["cam_label"], "remoteclip_prediction": str(remote[index]), "openai_clip_prediction": str(clip[index]), "category": category}
            )
    figure.savefig(output_dir / "deterministic_disagreement_examples.png", dpi=160)
    plt.close(figure)
    _write_csv(
        output_dir / "deterministic_example_records.csv",
        ["row_index", "image_id", "candidate_index", "source_class", "cam_label", "remoteclip_prediction", "openai_clip_prediction", "category"],
        example_rows,
    )
    overall = Counter(categories)
    result = {
        "status": "completed",
        "scientific_evidence": False,
        "analysis_role": protocol["role"],
        "input_sha256": checks,
        "bundle_id": manifest["bundle_id"],
        "record_count": len(records),
        "primary_group": group,
        "overall_paired_categories": {name: int(overall[name]) for name in protocol["comparison_categories"]},
        "per_class": per_class_rows,
        "example_selection": protocol["example_selection"],
        "claims": protocol["claims"],
    }
    write_json(output_dir / "analysis_summary.json", result)
    return result


def render_analysis_report(result: dict[str, Any], comparison_summary: dict[str, Any]) -> str:
    primary = comparison_summary["primary_endpoint"]["result"]
    per_class_lines = "\n".join(
        f"| {row['class']} | {row['remoteclip']:.3f} | {row['openai_clip']:.3f} | {row['clip_minus_remoteclip']:+.3f} |"
        for row in result["per_class"]
    )
    categories = result["overall_paired_categories"]
    return f"""# RemoteCLIP 与 OpenAI CLIP 区域对比：描述性错误分析

## 结论摘要

预注册主比较已经判定 RemoteCLIP 更优：OpenAI CLIP 减 RemoteCLIP 的宏平均 SAM3 来源弱标签一致率差为 {primary['mean_delta']:+.4f}，按图像聚类的 95% bootstrap 区间为 [{primary['ci95'][0]:+.4f}, {primary['ci95'][1]:+.4f}]。

本报告是结果产生后的描述性分析，不是新的确认性检验。指标是与弱标签的一致性，不是真实分类准确率，也不是开放词汇分割性能。

## 逐类主指标

| 类别 | RemoteCLIP | OpenAI CLIP | CLIP − RemoteCLIP |
|---|---:|---:|---:|
{per_class_lines}

RemoteCLIP 的主要优势来自 water、barren、road 和 building；OpenAI CLIP 在 agriculture 与 forest 上更高。这表明差异不是所有类别一致平移，后续方法不能只看整体平均值。

## 成对结果构成

- 两者都正确：{categories['both_correct']}
- 仅 RemoteCLIP 正确：{categories['remoteclip_only_correct']}
- 仅 OpenAI CLIP 正确：{categories['openai_clip_only_correct']}
- 两者同样错误：{categories['both_wrong_same']}
- 两者错误且预测不同：{categories['both_wrong_different']}

## 图表和明细

- `primary_confusion_heatmaps.png`：Group A 闭集行归一化混淆矩阵。
- `deterministic_disagreement_examples.png`：按冻结哈希规则抽取的分歧区域，不按置信度或视觉效果挑选。
- `primary_confusion_counts.csv`：完整混淆计数。
- `per_class_primary_deltas.csv`：逐类配对差值。
- `paired_outcome_categories.csv`：每类成对结果构成。
- `deterministic_example_records.csv`：案例身份和弱标签来源。

## 下一步决策

1. 保留 RemoteCLIP 作为第二篇的主视觉语言编码器，OpenAI CLIP 作为已冻结基线。
2. 不因本结果立即训练 student；先注册 seen/unseen 类别轮换协议。
3. 针对 barren、forest、agriculture 设计区域可靠性模块时，必须使用图像级分组或独立划分，不能在这 6000 条上设计并回报同样本结果。
4. 最终像素 GT 只用于冻结方案后的评价，不用于 prompt、阈值或类别划分选择。
"""

