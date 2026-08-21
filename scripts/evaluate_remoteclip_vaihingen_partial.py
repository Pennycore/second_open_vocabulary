"""Cache-only partial-support audit for the completed RemoteCLIP Vaihingen run.

The evaluator is intentionally incapable of loading RemoteCLIP, SAM3, or source
imagery.  It reconstructs the already-registered k=2/3/4 support decisions from
the completed run's immutable score cache and its saved ``text_pred`` vector,
then applies the unchanged FusionCanvas to the immutable candidate cache.  It
never writes into the source run and saves no reconstructed semantic maps.

GT is opened only after source-run, cache, candidate, protocol, and full-map
hashes have passed validation.  A mismatch is a hard failure rather than a
reason to rerun or tune anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ov_probe.io import InputValidationError, sha256_file  # noqa: E402
from ov_probe.pixel_ovss import IGNORE_INDEX, assemble_semantic_map, method_predictions, method_score_matrices  # noqa: E402
from ov_probe.remoteclip_potsdam_baseline import CLASSES, COLORS, directory_sha256, pixel_confusion_fast  # noqa: E402
from ov_probe.vaihingen_blind import TEST_AREAS  # noqa: E402


RUN_METHODS = ("text_only", "C2", "CTP")
PARTIAL_COUNTS = (2, 3, 4)
METRIC_COLUMNS = ("OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU")
BOOTSTRAP_COLUMNS = ("OA", "macro_f1", "mIoU", "U_IoU", "H_IoU")


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv_exclusive(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _expected_subsets() -> OrderedDict[str, dict[str, Any]]:
    """Return the frozen all-bitmask registration in source-run order."""
    subsets: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for subset_index in range(1 << len(CLASSES)):
        supported = [name for index, name in enumerate(CLASSES) if (subset_index >> index) & 1]
        if len(supported) not in PARTIAL_COUNTS:
            continue
        subsets[f"subset_{subset_index}"] = {
            "subset_index": subset_index,
            "k": len(supported),
            "supported": supported,
            "unsupported": [name for name in CLASSES if name not in supported],
        }
    return subsets


def _aggregate(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(CLASSES), len(CLASSES)):
        raise InputValidationError("Confusion matrix shape differs from the frozen five-class protocol.")
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


def _subset_metrics(matrix: np.ndarray, subset: dict[str, Any]) -> dict[str, Any]:
    result = _aggregate(matrix)
    supported, unsupported = subset["supported"], subset["unsupported"]
    s_f1 = float(np.mean([result["per_class_f1"][name] for name in supported]))
    u_f1 = float(np.mean([result["per_class_f1"][name] for name in unsupported]))
    s_iou = float(np.mean([result["per_class_iou"][name] for name in supported]))
    u_iou = float(np.mean([result["per_class_iou"][name] for name in unsupported]))
    result.update({
        "S_F1": s_f1,
        "U_F1": u_f1,
        "H_F1": 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 else 0.0,
        "S_IoU": s_iou,
        "U_IoU": u_iou,
        "H_IoU": 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou else 0.0,
    })
    return result


def _pixel_accounting(prediction: np.ndarray, gt: np.ndarray, covered: np.ndarray) -> dict[str, int]:
    if prediction.shape != gt.shape or prediction.shape != covered.shape:
        raise InputValidationError("Prediction, GT, and candidate coverage shapes must agree.")
    ignored = prediction == IGNORE_INDEX
    return {
        "valid_pixels": int(((prediction != IGNORE_INDEX) & (gt != IGNORE_INDEX)).sum()),
        "assigned_pixels": int((prediction != IGNORE_INDEX).sum()),
        "conflict_ignore_pixels": int((ignored & covered).sum()),
        "uncovered_pixels": int((ignored & ~covered).sum()),
        "gt_ignore_pixels": int((gt == IGNORE_INDEX).sum()),
        "total_pixels": int(prediction.size),
    }


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    required = ("manifest.json", "features.npz", "records.jsonl", "metrics.json", "partial_metrics.json", "partial_metrics.csv")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise InputValidationError(f"Completed RemoteCLIP run is missing required artifacts: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("scientific_evidence") is not True or manifest.get("gt_read") is not True:
        raise InputValidationError("Source run is not a completed, evaluated RemoteCLIP run.")
    if manifest.get("methods") != list(RUN_METHODS):
        raise InputValidationError("Source run method matrix differs from frozen Text/C2/CTP.")
    if manifest.get("partial_support") != {"counts": [2, 3, 4], "policy": "all_deterministic_bitmasks"}:
        raise InputValidationError("Source run partial-support registration differs from frozen k=2/3/4 bitmasks.")
    for field, filename in (("features_sha256", "features.npz"), ("records_sha256", "records.jsonl"), ("metrics_sha256", "metrics.json"), ("partial_metrics_sha256", "partial_metrics.json")):
        if sha256_file(run_dir / filename) != manifest.get(field):
            raise InputValidationError(f"Source run {filename} hash differs from its manifest.")
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(records) != int(manifest.get("record_count", -1)) or not records:
        raise InputValidationError("Source run record count differs from manifest.")
    with np.load(run_dir / "features.npz", allow_pickle=False) as archive:
        required_keys = {"features", "text_scores", "visual_scores", "text_prototypes", "visual_prototypes", "text_pred"}
        if set(archive.files) != required_keys:
            raise InputValidationError("Frozen feature cache schema differs from expected RemoteCLIP cache.")
        cache = {key: archive[key].astype(np.int64 if key == "text_pred" else np.float32) for key in archive.files}
    n = len(records)
    if cache["features"].shape != (n, 512) or cache["text_scores"].shape != (n, len(CLASSES)) or cache["visual_scores"].shape != (n, len(CLASSES)):
        raise InputValidationError("Frozen score cache cardinality or feature dimensions are invalid.")
    if cache["text_prototypes"].shape != (len(CLASSES), 512) or cache["visual_prototypes"].shape != (len(CLASSES), 512):
        raise InputValidationError("Frozen prototype cache dimensions are invalid.")
    if cache["text_pred"].shape != (n,) or np.any(cache["text_pred"] < 0) or np.any(cache["text_pred"] >= len(CLASSES)):
        raise InputValidationError("Frozen stored text top-1 vector is invalid.")
    if not all(np.isfinite(value).all() for key, value in cache.items() if key != "text_pred"):
        raise InputValidationError("Frozen score cache contains non-finite values.")
    for position, record in enumerate(records):
        if record.get("split") != "test" or int(record.get("area", -1)) not in TEST_AREAS:
            raise InputValidationError("Records contain a non-test area or incorrect split.")
        if int(record.get("candidate_index", -1)) < 0 or not str(record.get("image_id", "")).startswith("vaih_area"):
            raise InputValidationError("Record schema differs from the completed RemoteCLIP run.")
        if position and int(record["row_index"]) != int(records[position - 1]["row_index"]) + 1:
            raise InputValidationError("Source records do not preserve contiguous frozen score order.")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    partial = json.loads((run_dir / "partial_metrics.json").read_text(encoding="utf-8"))
    _validate_source_partial(partial, run_dir / "partial_metrics.csv")
    return manifest, records, cache, metrics, partial


def _validate_source_partial(partial: dict[str, Any], csv_path: Path) -> None:
    expected = _expected_subsets()
    # The source JSON is deliberately written with ``sort_keys=True``; therefore
    # its lexical key order (subset_10 before subset_3) is not protocol order.
    # Membership, not serialized ordering, is the frozen registration invariant.
    if set(partial) != set(expected):
        raise InputValidationError("Source partial JSON subset keys differ from registered all-bitmask protocol.")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if len(rows) != len(expected) * len(RUN_METHODS):
        raise InputValidationError("Source partial CSV does not contain exactly 25 subsets x 3 methods.")
    seen: set[tuple[str, str]] = set()
    for key, info in expected.items():
        source = partial[key]
        if source.get("k") != info["k"] or source.get("supported") != info["supported"] or source.get("unsupported") != info["unsupported"]:
            raise InputValidationError(f"Source partial subset registration differs for {key}.")
        if set(source.get("methods", {})) != set(RUN_METHODS):
            raise InputValidationError(f"Source partial method matrix differs for {key}.")
    for row in rows:
        key = f"subset_{row.get('subset_index')}"
        method = row.get("method", "")
        if key not in expected or method not in RUN_METHODS or (key, method) in seen:
            raise InputValidationError("Source partial CSV has invalid, duplicate, or unregistered rows.")
        seen.add((key, method))
    if len(seen) != len(expected) * len(RUN_METHODS):
        raise InputValidationError("Source partial CSV omits a registered subset/method row.")


def _validate_static_bindings(manifest: dict[str, Any], candidates_dir: Path, ctp_config: Path, protocol: Path) -> dict[str, Any]:
    sources = manifest.get("sources", {})
    if _sha256_bytes(ctp_config) != sources.get("ctp_frozen_file", {}).get("sha256"):
        raise InputValidationError("CTP frozen configuration hash differs from the source RemoteCLIP run.")
    if json.loads(ctp_config.read_text(encoding="utf-8")).get("status") != "frozen":
        raise InputValidationError("CTP configuration is not marked frozen.")
    if _sha256_bytes(protocol) != sources.get("remoteclip_protocol_file", {}).get("sha256"):
        raise InputValidationError("RemoteCLIP protocol hash differs from the source RemoteCLIP run.")
    candidate_hash, candidate_count = directory_sha256(candidates_dir, "*.npz")
    expected = sources.get("candidates", {})
    if candidate_hash != expected.get("sha256") or candidate_count != expected.get("count"):
        raise InputValidationError("Candidate cache hash/count differs from the completed RemoteCLIP run.")
    return {
        "candidate_sha256": candidate_hash,
        "candidate_count": candidate_count,
        "ctp_v1_frozen_sha256": _sha256_bytes(ctp_config),
        "remoteclip_protocol_sha256": _sha256_bytes(protocol),
    }


def _load_bound_candidates(records: list[dict[str, Any]], candidates_dir: Path, python_root: Path) -> OrderedDict[str, tuple[tuple[int, int], list[dict[str, Any]], np.ndarray]]:
    if not python_root.is_dir():
        raise InputValidationError(f"SAM3 candidate reader root is missing: {python_root}")
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    try:
        from sam3_remote_wsss.candidate_cache import load_candidate_cache
    except Exception as exc:  # pragma: no cover - environment-only guard
        raise InputValidationError(f"Cannot import read-only candidate cache reader: {type(exc).__name__}: {exc}") from exc
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        grouped.setdefault(str(record["image_id"]), []).append(record)
    expected_ids = [f"vaih_area{area}" for area in TEST_AREAS]
    if list(grouped) != expected_ids:
        raise InputValidationError("Record image ordering differs from frozen Vaihingen test-area registration.")
    bound: OrderedDict[str, tuple[tuple[int, int], list[dict[str, Any]], np.ndarray]] = OrderedDict()
    for image_id, image_records in grouped.items():
        metadata, raw_regions = load_candidate_cache(candidates_dir, image_id)
        shape = tuple(int(value) for value in metadata["image_shape"])
        if len(raw_regions) != len(image_records):
            raise InputValidationError(f"Candidate count differs for {image_id}.")
        regions: list[dict[str, Any]] = []
        covered = np.zeros(shape, dtype=bool)
        for index, (record, raw) in enumerate(zip(image_records, raw_regions)):
            x0, y0 = int(raw.x0), int(raw.y0)
            mask = np.asarray(raw.mask, dtype=bool)
            if int(record.get("candidate_index", -1)) != index or int(record.get("x0", -1)) != x0 or int(record.get("y0", -1)) != y0 or record.get("sam3_source_label") != str(raw.class_name):
                raise InputValidationError(f"Candidate metadata/order differs for {image_id} index {index}.")
            if x0 < 0 or y0 < 0 or y0 + mask.shape[0] > shape[0] or x0 + mask.shape[1] > shape[1]:
                raise InputValidationError(f"Candidate bounds exceed cached image shape for {image_id}.")
            covered[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] |= mask
            regions.append({"mask": mask, "x0": x0, "y0": y0})
        bound[image_id] = (shape, regions, covered)
    return bound


def _score_sets(cache: dict[str, np.ndarray], subsets: OrderedDict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    fused = 0.5 * cache["text_prototypes"] + 0.5 * cache["visual_prototypes"]
    normalizers = np.linalg.norm(fused, axis=1)
    if np.any(normalizers <= 1e-8):
        raise InputValidationError("Frozen C2 prototypes have a zero norm.")
    anchored = (0.5 * cache["text_scores"] + 0.5 * cache["visual_scores"]) / normalizers[None, :]
    score_sets, prediction_sets = {}, {}
    for key, subset in subsets.items():
        support = np.asarray([name in subset["supported"] for name in CLASSES], dtype=bool)
        score = method_score_matrices(cache["text_scores"], cache["visual_scores"], anchored, support, cache["text_pred"])
        prediction = method_predictions(score, cache["text_pred"], support)
        score_sets[key] = {method: score[method] for method in RUN_METHODS}
        prediction_sets[key] = {method: prediction[method] for method in RUN_METHODS}
    return score_sets, prediction_sets


def _validate_full_maps(run_dir: Path, records: list[dict[str, Any]], bound: OrderedDict[str, tuple[tuple[int, int], list[dict[str, Any]], np.ndarray]], cache: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    """Prove f16 cache reconstruction retains every saved full-support map before GT."""
    all_supported = OrderedDict({"full": {"supported": list(CLASSES)}})
    score_sets, prediction_sets = _score_sets(cache, all_supported)
    position = 0
    grouped_positions: dict[str, np.ndarray] = {}
    for image_id in bound:
        length = sum(1 for record in records if record["image_id"] == image_id)
        grouped_positions[image_id] = np.arange(position, position + length, dtype=np.int64)
        position += length
    if position != len(records):
        raise InputValidationError("Record/image grouping does not cover frozen cache rows.")
    artifacts = manifest.get("artifacts", {})
    for image_id, (shape, regions, _) in bound.items():
        rows = grouped_positions[image_id]
        for method in RUN_METHODS:
            prediction = prediction_sets["full"][method][rows]
            scores = score_sets["full"][method][rows, prediction]
            rebuilt, _ = assemble_semantic_map(shape, regions, prediction, scores, CLASSES)
            target = run_dir / f"{method}_{image_id}_semantic.npz"
            if not target.is_file() or sha256_file(target) != artifacts.get(target.name):
                raise InputValidationError(f"Source semantic-map hash differs from manifest: {target.name}")
            with np.load(target, allow_pickle=False) as archive:
                if set(archive.files) != {"label_map"} or not np.array_equal(rebuilt, archive["label_map"]):
                    raise InputValidationError(f"Frozen cache cannot reproduce source semantic map: {target.name}")


def _gt_from_label(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    rgb = image[:, :, :3] if image.ndim == 3 else image
    gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for index, name in enumerate(CLASSES):
        gt[np.all(rgb == np.asarray(COLORS[name], dtype=np.uint8), axis=-1)] = index
    return gt


def _metric_delta(ctp: dict[str, Any], c2: dict[str, Any]) -> dict[str, float]:
    return {name: float(ctp[name] - c2[name]) for name in BOOTSTRAP_COLUMNS}


def _bootstrap(area_matrices: dict[int, dict[str, np.ndarray]], subset: dict[str, Any], seed: int, repeats: int) -> dict[str, Any]:
    areas = list(TEST_AREAS)
    if sorted(area_matrices) != areas:
        raise InputValidationError("Bootstrap cluster areas differ from frozen Vaihingen test areas.")
    point = _metric_delta(
        _subset_metrics(np.sum([area_matrices[area]["CTP"] for area in areas], axis=0), subset),
        _subset_metrics(np.sum([area_matrices[area]["C2"] for area in areas], axis=0), subset),
    )
    rng = np.random.default_rng(seed)
    values = {name: np.empty(repeats, dtype=np.float64) for name in BOOTSTRAP_COLUMNS}
    for repeat in range(repeats):
        drawn = rng.choice(areas, size=len(areas), replace=True)
        ctp = _subset_metrics(np.sum([area_matrices[int(area)]["CTP"] for area in drawn], axis=0), subset)
        c2 = _subset_metrics(np.sum([area_matrices[int(area)]["C2"] for area in drawn], axis=0), subset)
        delta = _metric_delta(ctp, c2)
        for name, value in delta.items():
            values[name][repeat] = value
    directions = {str(area): _metric_delta(_subset_metrics(area_matrices[area]["CTP"], subset), _subset_metrics(area_matrices[area]["C2"], subset)) for area in areas}
    return {
        "cluster_unit": "Vaihingen test area",
        "areas": areas,
        "seed": seed,
        "repeats": repeats,
        "point_estimate": point,
        "bootstrap": {name: {"mean": float(value.mean()), "ci95_low": float(np.quantile(value, 0.025)), "ci95_high": float(np.quantile(value, 0.975))} for name, value in values.items()},
        "direction_by_area": directions,
    }


def _source_partial_check(reconstructed: dict[str, dict[str, dict[str, Any]]], source: dict[str, Any]) -> None:
    for key, methods in reconstructed.items():
        for method, values in methods.items():
            expected = source[key]["methods"][method]
            for metric in METRIC_COLUMNS + ("valid_pixels",):
                if not np.isclose(float(values[metric]), float(expected[metric]), rtol=0.0, atol=0.0):
                    raise InputValidationError(f"Reconstructed cache metric differs from source partial metric: {key}/{method}/{metric}.")


def _summary_rows(results: dict[str, dict[str, dict[str, Any]]], subsets: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in PARTIAL_COUNTS:
        for method in RUN_METHODS:
            selected = [results[key][method] for key, subset in subsets.items() if subset["k"] == k]
            row: dict[str, Any] = {"k": k, "method": method, "subset_count": len(selected)}
            for metric in METRIC_COLUMNS:
                values = np.asarray([record[metric] for record in selected], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            rows.append(row)
    return rows


def _render_report(output: Path, source_run: Path, bindings: dict[str, Any], full_metrics: dict[str, Any], summaries: list[dict[str, Any]], bootstrap: dict[str, Any]) -> None:
    lines = [
        "# RemoteCLIP Vaihingen partial-support cache audit",
        "",
        "This is a cache-only reconstruction from the completed RemoteCLIP run. No RemoteCLIP model, feature extraction, SAM3 execution, prompt, alpha, threshold, prototype, CTP-v1 formula, candidate cache, or support subset was changed. The source run was never written.",
        "",
        "## Source binding",
        "",
        f"- Source run: `{source_run}`",
        f"- Candidate cache SHA-256: `{bindings['candidate_sha256']}` ({bindings['candidate_count']} files)",
        f"- CTP-v1 SHA-256: `{bindings['ctp_v1_frozen_sha256']}`",
        f"- RemoteCLIP protocol SHA-256: `{bindings['remoteclip_protocol_sha256']}`",
        "- Cache reconstruction was required to match all 15 saved full-support semantic maps before GT was opened.",
        "",
        "## Full-support source metrics",
        "",
        "| Method | OA | Macro F1 | mIoU |",
        "|---|---:|---:|---:|",
    ]
    for method in RUN_METHODS:
        row = full_metrics[method]
        lines.append(f"| {method} | {row['OA']:.6f} | {row['macro_f1']:.6f} | {row['mIoU']:.6f} |")
    lines.extend(["", "## Partial-support k-level mean ± population std", "", "| k | Method | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in summaries:
        lines.append(f"| {row['k']} | {row['method']} | {row['OA_mean']:.6f} ± {row['OA_std']:.6f} | {row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | {row['mIoU_mean']:.6f} ± {row['mIoU_std']:.6f} | {row['S_IoU_mean']:.6f} ± {row['S_IoU_std']:.6f} | {row['U_IoU_mean']:.6f} ± {row['U_IoU_std']:.6f} | {row['H_IoU_mean']:.6f} ± {row['H_IoU_std']:.6f} |")
    lines.extend(["", "## CTP − C2 area-cluster bootstrap", "", "All 25 registered support subsets were evaluated with test area as the cluster, seed 42, and 5,000 resamples. Per-subset point estimates, CIs, and per-area directions are in `bootstrap.json`; the complete rows and pixel accounting are in the CSV files.", ""])
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _run(args: argparse.Namespace) -> Path:
    if args.output_dir.exists():
        raise InputValidationError(f"Output directory already exists; refusing overwrite: {args.output_dir}")
    manifest, records, cache, full_metrics, source_partial = _load_run(args.run_dir)
    bindings = _validate_static_bindings(manifest, args.candidates_dir, args.ctp_config, args.remoteclip_protocol)
    bound = _load_bound_candidates(records, args.candidates_dir, args.sam3_python_root)
    _validate_full_maps(args.run_dir, records, bound, cache, manifest)
    subsets = _expected_subsets()
    score_sets, prediction_sets = _score_sets(cache, subsets)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pre_gt = {
        "format_version": 1,
        "status": "pre_gt_validated",
        "gt_read": False,
        "source_run": str(args.run_dir.resolve()),
        "source_run_manifest_sha256": sha256_file(args.run_dir / "manifest.json"),
        "source_features_sha256": sha256_file(args.run_dir / "features.npz"),
        "source_records_sha256": sha256_file(args.run_dir / "records.jsonl"),
        "source_metrics_sha256": sha256_file(args.run_dir / "metrics.json"),
        "source_partial_metrics_sha256": sha256_file(args.run_dir / "partial_metrics.json"),
        "source_artifact_sha256": manifest["artifacts"],
        "source_run_commit": manifest.get("code_commit"),
        "source_config_sha256": manifest.get("config_sha256"),
        "bindings": bindings,
        "test_areas": list(TEST_AREAS),
        "registered_subset_count": len(subsets),
        "methods": list(RUN_METHODS),
        "semantic_maps_saved": False,
        "model_or_feature_inference": False,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "code_commit": _git_commit(),
    }
    _write_json_exclusive(args.output_dir / "manifest.json", pre_gt)
    try:
        label_hash, label_count = directory_sha256(args.labels_dir, "*_label.tif")
        labels = manifest.get("labels", {})
        if label_hash != labels.get("sha256") or label_count != labels.get("count"):
            raise InputValidationError("Label directory hash/count differs from completed source run; GT remains unread.")
        matrices = {key: {method: {area: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64) for area in TEST_AREAS} for method in RUN_METHODS} for key in subsets}
        pixel_rows: list[dict[str, Any]] = []
        offset = 0
        for image_id, (shape, regions, covered) in bound.items():
            area = int(image_id.removeprefix("vaih_area"))
            length = len(regions)
            rows = np.arange(offset, offset + length, dtype=np.int64)
            offset += length
            gt = _gt_from_label(args.labels_dir / f"{image_id}_label.tif")
            if tuple(gt.shape) != tuple(shape):
                raise InputValidationError(f"GT/candidate image shape differs for {image_id}.")
            for key, subset in subsets.items():
                for method in RUN_METHODS:
                    prediction = prediction_sets[key][method][rows]
                    scores = score_sets[key][method][rows, prediction]
                    semantic, _ = assemble_semantic_map(shape, regions, prediction, scores, CLASSES)
                    matrix = np.asarray(pixel_confusion_fast(semantic, gt, CLASSES)["confusion_matrix"], dtype=np.int64)
                    matrices[key][method][area] = matrix
                    accounting = _pixel_accounting(semantic, gt, covered)
                    metrics = _subset_metrics(matrix, subset)
                    pixel_rows.append({"subset": key, "subset_index": subset["subset_index"], "k": subset["k"], "supported": "|".join(subset["supported"]), "unsupported": "|".join(subset["unsupported"]), "area": area, "method": method, **{name: metrics[name] for name in METRIC_COLUMNS}, **accounting})
        if offset != len(records):
            raise InputValidationError("Candidate groups do not cover all frozen cache records.")
        results: dict[str, dict[str, dict[str, Any]]] = {}
        subset_rows: list[dict[str, Any]] = []
        bootstraps: dict[str, Any] = {}
        for key, subset in subsets.items():
            results[key] = {}
            for method in RUN_METHODS:
                total = np.sum([matrices[key][method][area] for area in TEST_AREAS], axis=0)
                values = _subset_metrics(total, subset)
                results[key][method] = values
                accounting = {name: int(sum(row[name] for row in pixel_rows if row["subset"] == key and row["method"] == method)) for name in ("assigned_pixels", "conflict_ignore_pixels", "uncovered_pixels", "gt_ignore_pixels", "total_pixels")}
                subset_rows.append({"subset": key, "subset_index": subset["subset_index"], "k": subset["k"], "supported": "|".join(subset["supported"]), "unsupported": "|".join(subset["unsupported"]), "method": method, **{name: values[name] for name in METRIC_COLUMNS}, **accounting})
            bootstraps[key] = _bootstrap({area: {method: matrices[key][method][area] for method in RUN_METHODS} for area in TEST_AREAS}, subset, args.bootstrap_seed, args.bootstrap_repeats)
        _source_partial_check(results, source_partial)
        summaries = _summary_rows(results, subsets)
        _write_csv_exclusive(args.output_dir / "partial_metrics_per_subset.csv", list(subset_rows[0]), subset_rows)
        _write_csv_exclusive(args.output_dir / "partial_metrics_by_area.csv", list(pixel_rows[0]), pixel_rows)
        _write_csv_exclusive(args.output_dir / "partial_summary_by_k.csv", list(summaries[0]), summaries)
        _write_json_exclusive(args.output_dir / "bootstrap.json", {"format_version": 1, "comparison": "CTP_minus_C2", "seed": args.bootstrap_seed, "repeats": args.bootstrap_repeats, "cluster_unit": "Vaihingen test area", "subsets": bootstraps})
        _write_json_exclusive(args.output_dir / "metrics.json", {"full_support_source": full_metrics, "partial_support": results, "subsets": subsets})
        _render_report(args.output_dir / "report.md", args.run_dir, bindings, full_metrics, summaries, bootstraps)
        final = dict(pre_gt)
        final.update({"status": "completed", "gt_read": True, "labels": {"sha256": label_hash, "count": label_count}, "metrics_sha256": sha256_file(args.output_dir / "metrics.json"), "bootstrap_sha256": sha256_file(args.output_dir / "bootstrap.json"), "partial_metrics_per_subset_sha256": sha256_file(args.output_dir / "partial_metrics_per_subset.csv"), "partial_metrics_by_area_sha256": sha256_file(args.output_dir / "partial_metrics_by_area.csv"), "partial_summary_by_k_sha256": sha256_file(args.output_dir / "partial_summary_by_k.csv"), "report_sha256": sha256_file(args.output_dir / "report.md")})
        (args.output_dir / "manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return args.output_dir
    except Exception:
        failure = {"status": "failed", "gt_read": True, "error": traceback.format_exc()}
        if not (args.output_dir / "failure.json").exists():
            _write_json_exclusive(args.output_dir / "failure.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache-only RemoteCLIP Vaihingen partial-support evaluator.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidates-dir", required=True, type=Path)
    parser.add_argument("--sam3-python-root", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--ctp-config", required=True, type=Path)
    parser.add_argument("--remoteclip-protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_seed != 42 or args.bootstrap_repeats != 5000:
        raise InputValidationError("Frozen bootstrap requires seed=42 and repeats=5000.")
    try:
        output = _run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
