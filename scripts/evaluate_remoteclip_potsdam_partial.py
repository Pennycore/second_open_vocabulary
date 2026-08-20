"""Offline RemoteCLIP Potsdam partial-support evaluation.

This evaluator intentionally never loads a model or OpenAI CLIP feature.  It uses
only the immutable RemoteCLIP score cache/records from a completed full-support
run, the pre-registered Potsdam support manifest, and existing first-project
candidate masks.  Predictions are reconstructed in memory and are never saved
as new semantic-map files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ov_probe.io import InputValidationError, sha256_file  # noqa: E402
from ov_probe.pixel_ovss import (  # noqa: E402
    IGNORE_INDEX,
    assemble_semantic_map,
    load_candidate_masks,
    method_predictions,
    method_score_matrices,
)
from ov_probe.remoteclip_potsdam_baseline import CLASSES, COLORS, METHODS, directory_sha256  # noqa: E402

EXPECTED_SUPPORT_MANIFEST_SHA256 = "79b80bd646d15750fdd76bdbd44f32b9606782764798c50612a6065b5d88138d"
EXPECTED_SUBSET_KEYS = tuple(
    f"r{ratio}_seed{seed}" for ratio in (25, 50, 75) for seed in (42, 43, 44)
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_subsets(path: Path) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise InputValidationError("Support manifest hash differs from the frozen Potsdam manifest.")
    subsets = json.loads(path.read_text(encoding="utf-8"))
    if tuple(sorted(subsets)) != tuple(sorted(EXPECTED_SUBSET_KEYS)):
        raise InputValidationError("Support manifest subset keys differ from the frozen Potsdam protocol.")
    for key, info in subsets.items():
        supported, unsupported = info.get("supported"), info.get("unsupported")
        if not isinstance(supported, list) or not isinstance(unsupported, list):
            raise InputValidationError(f"Malformed support subset: {key}")
        if sorted(supported + unsupported) != sorted(CLASSES) or set(supported) & set(unsupported):
            raise InputValidationError(f"Support/unsupported partition is invalid: {key}")
        expected_k = {"r25": 1, "r50": 2, "r75": 4}[key.split("_")[0]]
        if len(supported) != expected_k or int(info.get("k", -1)) != expected_k:
            raise InputValidationError(f"Support subset cardinality is invalid: {key}")
    return subsets


def _load_cache(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.npz"
    records_path = run_dir / "records.jsonl"
    metrics_path = run_dir / "metrics.json"
    for path in (manifest_path, predictions_path, records_path, metrics_path):
        if not path.is_file():
            raise InputValidationError(f"Completed RemoteCLIP run is missing: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("phase") != "evaluate":
        raise InputValidationError("RemoteCLIP source run is not a completed evaluated run.")
    if manifest.get("methods") != METHODS or manifest.get("record_count", 0) <= 0:
        raise InputValidationError("RemoteCLIP source method matrix or record count is invalid.")
    if sha256_file(predictions_path) != manifest.get("predictions_npz_sha256"):
        raise InputValidationError("RemoteCLIP predictions cache differs from its completed-run manifest.")
    if sha256_file(metrics_path) != manifest.get("metrics_sha256"):
        raise InputValidationError("RemoteCLIP full metrics differ from its completed-run manifest.")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    with np.load(predictions_path, allow_pickle=False) as archive:
        needed = {"features", "text_scores", "visual_scores", "text_prototypes", "visual_prototypes", "text_pred"}
        if set(archive.files) != needed:
            raise InputValidationError("RemoteCLIP prediction cache arrays differ from the frozen cache schema.")
        cache = {key: archive[key].astype(np.float32 if key != "text_pred" else np.int64) for key in archive.files}
    n = len(records)
    if n != int(manifest["record_count"]) or cache["text_scores"].shape != (n, len(CLASSES)):
        raise InputValidationError("RemoteCLIP record/cache cardinality mismatch.")
    if cache["visual_scores"].shape != cache["text_scores"].shape or cache["text_pred"].shape != (n,):
        raise InputValidationError("RemoteCLIP cached score dimensions are invalid.")
    if cache["text_prototypes"].shape != cache["visual_prototypes"].shape or cache["text_prototypes"].shape[0] != len(CLASSES):
        raise InputValidationError("RemoteCLIP prototype dimensions are invalid.")
    if not all(np.isfinite(value).all() for key, value in cache.items() if key != "text_pred"):
        raise InputValidationError("RemoteCLIP cache has non-finite values.")
    # CTP's frozen text top-1 is the value saved by the original float32
    # prediction run.  `text_scores` was subsequently compressed to float16,
    # so near ties may legitimately change a recomputed argmax.  Recomputing
    # it here would alter the frozen CTP decision input.
    if np.any(cache["text_pred"] < 0) or np.any(cache["text_pred"] >= len(CLASSES)):
        raise InputValidationError("RemoteCLIP cached text predictions are out of class range.")
    return manifest, records, cache


def _validate_candidate_binding(
    records: list[dict[str, Any]], candidates_dir: Path, source_manifest: dict[str, Any]
) -> OrderedDict[str, tuple[tuple[int, int], list[dict[str, Any]], list[dict[str, Any]]]]:
    actual_hash, actual_count = directory_sha256(candidates_dir, "*.npz")
    expected = source_manifest.get("candidates", {})
    if actual_hash != expected.get("sha256") or actual_count != expected.get("count"):
        raise InputValidationError("First-project candidate cache differs from the completed RemoteCLIP run.")
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, record in enumerate(records):
        if int(record.get("row_index", -1)) != index:
            raise InputValidationError("RemoteCLIP records do not have contiguous cache row indices.")
        grouped.setdefault(str(record.get("image_id")), []).append(record)
    bound: OrderedDict[str, tuple[tuple[int, int], list[dict[str, Any]], list[dict[str, Any]]]] = OrderedDict()
    for image_id, image_records in grouped.items():
        shape, regions = load_candidate_masks(candidates_dir, image_id)
        if len(regions) != len(image_records):
            raise InputValidationError(f"Candidate count differs for {image_id}.")
        for candidate_index, (record, region) in enumerate(zip(image_records, regions)):
            if int(record.get("candidate_index", -1)) != candidate_index:
                raise InputValidationError(f"Candidate order differs for {image_id}.")
            fields = ("x0", "y0", "class_name")
            if any(record.get(field if field != "class_name" else "sam3_source_label") != region[field] for field in fields):
                raise InputValidationError(f"Candidate metadata differs for {image_id}.")
        bound[image_id] = (shape, regions, image_records)
    return bound


def _aggregate(matrix: np.ndarray) -> dict[str, Any]:
    per_iou, per_f1 = {}, {}
    for index, name in enumerate(CLASSES):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    valid = int(matrix.sum())
    return {
        "OA": float(np.trace(matrix) / valid) if valid else 0.0,
        "macro_f1": float(np.mean(list(per_f1.values()))),
        "mIoU": float(np.mean(list(per_iou.values()))),
        "per_class_f1": per_f1,
        "per_class_iou": per_iou,
        "confusion_matrix": matrix.tolist(),
        "valid_pixels": valid,
    }


def _metrics_for_subset(matrix: np.ndarray, supported: list[str], unsupported: list[str]) -> dict[str, Any]:
    metrics = _aggregate(matrix)
    s_f1 = float(np.mean([metrics["per_class_f1"][name] for name in supported]))
    u_f1 = float(np.mean([metrics["per_class_f1"][name] for name in unsupported]))
    s_iou = float(np.mean([metrics["per_class_iou"][name] for name in supported]))
    u_iou = float(np.mean([metrics["per_class_iou"][name] for name in unsupported]))
    metrics.update({
        "S_F1": s_f1, "U_F1": u_f1, "H_F1": 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 else 0.0,
        "S_IoU": s_iou, "U_IoU": u_iou, "H_IoU": 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou else 0.0,
    })
    return metrics


def _gt_from_label(path: Path) -> np.ndarray:
    import tifffile
    image = tifffile.imread(path)
    rgb = image[:, :, :3] if image.ndim == 3 else image
    gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for index, name in enumerate(CLASSES):
        gt[np.all(rgb == np.asarray(COLORS[name], dtype=np.uint8), axis=-1)] = index
    return gt


def _write_csv(path: Path, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    fields = ["subset", "method", "OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU", "valid_pixels"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subset, methods in results.items():
            for method, row in methods.items():
                writer.writerow({key: row.get(key) if key not in {"subset", "method"} else (subset if key == "subset" else method) for key in fields})


def _write_report(path: Path, results: dict[str, dict[str, dict[str, Any]]], full_metrics: dict[str, Any]) -> None:
    lines = [
        "# RemoteCLIP Potsdam partial-support evaluation",
        "",
        "This is an offline reconstruction from the completed RemoteCLIP score cache. No model, SAM3 candidate generation, OpenAI CLIP feature, prompt, alpha, or prototype construction was run or changed.",
        "",
        "## Full-support metrics (completed source run)",
        "",
        "| Method | OA | Macro F1 | mIoU |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = full_metrics[method]
        lines.append(f"| {method} | {row['OA']:.6f} | {row['macro_f1']:.6f} | {row['mIoU']:.6f} |")
    lines.extend(["", "## Partial-support metrics", "", "See `metrics.csv` and `metrics.json` for all pre-registered r25/r50/r75 × seed 42/43/44 subsets.", ""])
    for subset, methods in results.items():
        lines.extend([f"### {subset}", "", "| Method | S-F1 | U-F1 | H-F1 | S-IoU | U-IoU | H-IoU |", "|---|---:|---:|---:|---:|---:|---:|"])
        for method in METHODS:
            row = methods[method]
            lines.append(f"| {method} | {row['S_F1']:.6f} | {row['U_F1']:.6f} | {row['H_F1']:.6f} | {row['S_IoU']:.6f} | {row['U_IoU']:.6f} | {row['H_IoU']:.6f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline frozen RemoteCLIP Potsdam partial-support evaluator.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--support-manifest", required=True, type=Path)
    parser.add_argument("--candidates-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--ctp-config", required=True, type=Path)
    parser.add_argument("--remoteclip-protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise InputValidationError("Output directory already exists; overwrite is prohibited.")

    source_manifest, records, cache = _load_cache(args.run_dir)
    protocol_hash = _sha256_bytes(args.remoteclip_protocol)
    ctp_hash = _sha256_bytes(args.ctp_config)
    if protocol_hash != source_manifest.get("protocol_sha256"):
        raise InputValidationError("Current RemoteCLIP protocol differs from the completed source run.")
    if json.loads(args.ctp_config.read_text(encoding="utf-8")).get("status") != "frozen":
        raise InputValidationError("CTP configuration is not frozen.")
    subsets = _load_subsets(args.support_manifest)
    bound = _validate_candidate_binding(records, args.candidates_dir, source_manifest["sources"])

    text_scores, visual_scores = cache["text_scores"], cache["visual_scores"]
    prototype_mix = 0.5 * cache["text_prototypes"] + 0.5 * cache["visual_prototypes"]
    normalizers = np.linalg.norm(prototype_mix, axis=1)
    if np.any(normalizers <= 1e-8):
        raise InputValidationError("Frozen RemoteCLIP C2 prototype has zero norm.")
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
    score_sets, prediction_sets = {}, {}
    for key, info in subsets.items():
        support_mask = np.asarray([name in info["supported"] for name in CLASSES], dtype=bool)
        score_sets[key] = method_score_matrices(text_scores, visual_scores, anchored, support_mask, cache["text_pred"])
        prediction_sets[key] = method_predictions(score_sets[key], cache["text_pred"], support_mask)

    # All cache, frozen-formula, candidate and support bindings are now complete.
    # Only after this point is it permitted to read ground-truth label files.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pre_gt = {
        "format_version": 1, "phase": "pre_gt_validation", "status": "completed", "gt_read": False,
        "source_run": str(args.run_dir.resolve()), "source_run_manifest_sha256": sha256_file(args.run_dir / "manifest.json"),
        "predictions_npz_sha256": sha256_file(args.run_dir / "predictions.npz"), "records_jsonl_sha256": sha256_file(args.run_dir / "records.jsonl"),
        "support_manifest_sha256": sha256_file(args.support_manifest), "ctp_v1_frozen_sha256": ctp_hash,
        "remoteclip_protocol_sha256": protocol_hash, "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "candidate_sources": source_manifest["sources"]["candidates"], "record_count": len(records), "image_count": len(bound),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluator_sha256": sha256_file(Path(__file__)), "semantic_maps_saved": False,
        "float16_text_score_argmax_mismatch_count": int(np.count_nonzero(cache["text_pred"] != np.argmax(text_scores, axis=1))),
        "text_pred_policy": "use the completed run's stored float32-era text top-1; do not recompute it from float16-compressed scores",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(pre_gt, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    matrices = {key: {method: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64) for method in METHODS} for key in subsets}
    for image_id, (shape, regions, image_records) in bound.items():
        gt = _gt_from_label(args.labels_dir / f"{image_id}_label.tif")
        if tuple(gt.shape) != tuple(shape):
            raise InputValidationError(f"GT/candidate image shape differs for {image_id}.")
        rows = np.asarray([int(record["row_index"]) for record in image_records], dtype=np.int64)
        for key in subsets:
            for method in METHODS:
                prediction = prediction_sets[key][method][rows]
                scores = score_sets[key][method][rows, prediction]
                semantic, _ = assemble_semantic_map(shape, regions, prediction, scores, CLASSES)
                valid = (semantic != IGNORE_INDEX) & (gt != IGNORE_INDEX)
                codes = gt[valid].astype(np.int64) * len(CLASSES) + semantic[valid].astype(np.int64)
                matrices[key][method] += np.bincount(codes, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))

    full_metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    results = {
        key: {method: _metrics_for_subset(matrices[key][method], info["supported"], info["unsupported"]) for method in METHODS}
        for key, info in subsets.items()
    }
    metrics = {"full_support": full_metrics, "partial_support": results, "support_subsets": subsets}
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    _write_csv(args.output_dir / "metrics.csv", results)
    _write_report(args.output_dir / "report.md", results, full_metrics)
    final_manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    final_manifest.update({
        "phase": "evaluate", "status": "completed", "gt_read": True,
        "metrics_sha256": sha256_file(metrics_path), "metrics_csv_sha256": sha256_file(args.output_dir / "metrics.csv"),
        "report_sha256": sha256_file(args.output_dir / "report.md"),
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
