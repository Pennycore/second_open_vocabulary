"""One-shot, non-overwriting RemoteCLIP baseline for frozen Vaihingen protocol.

This runner is intentionally conservative.  It checks the exact 3090 runtime,
read-only images/labels/SAM3 candidate cache and RemoteCLIP checkpoint *before*
it creates a run directory.  Missing inputs print a machine-readable ``blocked``
status and do not download data, run SAM3, or create output artifacts.

With all inputs present it evaluates the registered full-support Text-only/C2/
CTP matrix (SCC may be opted in) followed by every deterministic partial-support
bitmask with k=2/3/4.  It reuses the frozen CTP/SCC/FusionCanvas primitives and
never changes prompts, alpha, prototype construction, thresholds or candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ov_probe.io import InputValidationError, sha256_file  # noqa: E402
from ov_probe.pixel_ovss import assemble_semantic_map, method_predictions, method_score_matrices  # noqa: E402
from ov_probe.remoteclip_potsdam_baseline import (  # noqa: E402
    CLASSES,
    COLORS,
    _aggregate,
    _load_model,
    _normalize,
    crop_views,
    directory_sha256,
    pixel_confusion_fast,
    text_prototypes,
)
from ov_probe.vaihingen_blind import TEST_AREAS, TRAIN_AREAS, _area_from_id  # noqa: E402


ALLOWED_METHODS = ("text_only", "C2", "SCC", "CTP")
MINIMUM_METHODS = ("text_only", "C2", "CTP")


@dataclass(frozen=True)
class Preflight:
    status: str
    errors: list[str]
    config: dict[str, Any]
    sources: dict[str, Any]
    environment: dict[str, Any]


def _project_root(config_path: Path) -> Path:
    """Resolve the project root for a config stored in ``<root>/configs``."""
    return config_path.resolve().parents[1]


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    config_path = config_path.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise InputValidationError("Configuration must be a YAML mapping.")
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Config must freeze experiment.overwrite=false.")
    root = _project_root(config_path)
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise InputValidationError("Configuration requires a paths mapping.")
    required = {
        "image_dir", "label_dir", "candidates_dir", "sam3_python_root",
        "remoteclip_checkpoint", "remoteclip_protocol_file", "ctp_frozen_file", "output_root",
    }
    missing = sorted(key for key in required if not paths.get(key))
    if missing:
        raise InputValidationError(f"Configuration missing paths: {missing}")
    for key, value in list(paths.items()):
        path = Path(os.path.expandvars(str(value)))
        if path.is_absolute():
            raise InputValidationError(f"Path must be project-relative, not absolute: paths.{key}")
        paths[key] = str((root / path).resolve())
    methods = cfg.get("matrix", {}).get("full_support_methods", [])
    partial_methods = cfg.get("matrix", {}).get("partial_support_methods", [])
    for name, values in (("full_support_methods", methods), ("partial_support_methods", partial_methods)):
        if not isinstance(values, list) or not set(MINIMUM_METHODS).issubset(values):
            raise InputValidationError(f"matrix.{name} must include frozen Text-only/C2/CTP.")
        if any(value not in ALLOWED_METHODS for value in values):
            raise InputValidationError(f"matrix.{name} contains a non-frozen method.")
    counts = cfg.get("matrix", {}).get("partial_support_counts")
    if counts != [2, 3, 4] or cfg.get("matrix", {}).get("partial_subset_policy") != "all_deterministic_bitmasks":
        raise InputValidationError("Partial-support protocol must be all deterministic k=2/3/4 bitmasks.")
    if list(cfg.get("split", {}).get("train_areas", [])) != TRAIN_AREAS or list(cfg.get("split", {}).get("test_areas", [])) != TEST_AREAS:
        raise InputValidationError("Vaihingen train/test split differs from the frozen protocol.")
    return cfg, root


def _environment_check(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    environment: dict[str, Any] = {"python": platform.python_version()}
    errors: list[str] = []
    if sys.version_info < (3, 9):
        errors.append(f"Python >=3.9 required; found {platform.python_version()}.")
    try:
        import torch
        environment["torch"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable to torch.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            environment["gpu"] = gpu_name
            required_gpu = str(cfg.get("runtime", {}).get("required_gpu_substring", ""))
            if required_gpu and required_gpu.lower() not in gpu_name.lower():
                errors.append(f"Required GPU '{required_gpu}' not found; detected '{gpu_name}'.")
    except Exception as exc:  # import diagnostics are part of preflight output
        errors.append(f"torch unavailable: {type(exc).__name__}: {exc}")
    try:
        import open_clip
        environment["open_clip"] = getattr(open_clip, "__version__", None)
        required = str(cfg["integrity"]["required_open_clip_version"])
        if environment["open_clip"] != required:
            errors.append(f"open_clip must be {required}; found {environment['open_clip']}.")
    except Exception as exc:
        errors.append(f"open_clip unavailable: {type(exc).__name__}: {exc}")
    return environment, errors


def _expected_ids(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    train = [f"vaih_area{area}" for area in cfg["split"]["train_areas"]]
    test = [f"vaih_area{area}" for area in cfg["split"]["test_areas"]]
    return train, test, train + test


def _preflight(config_path: Path) -> Preflight:
    cfg, root = _load_config(config_path)
    errors: list[str] = []
    environment, environment_errors = _environment_check(cfg)
    errors.extend(environment_errors)
    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    sources: dict[str, Any] = {
        "project_root": str(root), "config_sha256": _sha256_bytes(config_path.resolve()),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    for key in ("image_dir", "label_dir", "candidates_dir", "sam3_python_root"):
        if not paths[key].is_dir():
            errors.append(f"Required directory missing: {key}={paths[key]}")
    for key in ("remoteclip_checkpoint", "remoteclip_protocol_file", "ctp_frozen_file"):
        if not paths[key].is_file():
            errors.append(f"Required file missing: {key}={paths[key]}")
    if paths["remoteclip_checkpoint"].is_file():
        actual = sha256_file(paths["remoteclip_checkpoint"])
        sources["checkpoint"] = {"sha256": actual, "path": str(paths["remoteclip_checkpoint"])}
        expected = str(cfg["integrity"]["remoteclip_checkpoint_sha256"])
        if actual != expected:
            errors.append("RemoteCLIP checkpoint SHA-256 differs from the frozen deployment contract.")
    for name in ("remoteclip_protocol_file", "ctp_frozen_file"):
        if paths[name].is_file():
            sources[name] = {"sha256": _sha256_bytes(paths[name]), "path": str(paths[name])}
    if paths["remoteclip_protocol_file"].is_file():
        try:
            protocol = json.loads(paths["remoteclip_protocol_file"].read_text(encoding="utf-8"))
            model = protocol.get("model", {})
            if list(protocol.get("classes", [])) != CLASSES or model.get("checkpoint_sha256") != cfg["integrity"]["remoteclip_checkpoint_sha256"]:
                errors.append("RemoteCLIP protocol classes or checkpoint binding differs from the frozen contract.")
            if model.get("feature_dimension") != cfg["integrity"]["required_feature_dimension"]:
                errors.append("RemoteCLIP protocol feature dimension differs from the frozen contract.")
        except Exception as exc:
            errors.append(f"RemoteCLIP protocol is unreadable: {type(exc).__name__}: {exc}")
    if paths["ctp_frozen_file"].is_file():
        try:
            if json.loads(paths["ctp_frozen_file"].read_text(encoding="utf-8")).get("status") != "frozen":
                errors.append("CTP configuration is not marked frozen.")
        except Exception as exc:
            errors.append(f"CTP frozen file is unreadable: {type(exc).__name__}: {exc}")
    loader = paths["sam3_python_root"] / "sam3_remote_wsss" / "candidate_cache.py"
    if paths["sam3_python_root"].is_dir() and not loader.is_file():
        errors.append(f"SAM3 candidate-cache reader missing: {loader}")
    train_ids, test_ids, all_ids = _expected_ids(cfg)
    if paths["candidates_dir"].is_dir():
        candidate_ids = {path.stem for path in paths["candidates_dir"].glob("*.npz")}
        required_ids = set(all_ids)
        absent = sorted(required_ids - candidate_ids)
        if absent:
            errors.append(f"SAM3 candidate cache lacks required Vaihingen areas: {absent}")
        if candidate_ids:
            try:
                digest, count = directory_sha256(paths["candidates_dir"], "*.npz")
                sources["candidates"] = {"sha256": digest, "count": count}
            except Exception as exc:
                errors.append(f"Cannot hash candidate cache: {type(exc).__name__}: {exc}")
    for key, suffix, identifiers in (("image_dir", "_RGB.tif", all_ids), ("label_dir", "_label.tif", test_ids)):
        directory = paths[key]
        if directory.is_dir():
            absent = [image_id for image_id in identifiers if not (directory / f"{image_id}{suffix}").is_file()]
            if absent:
                errors.append(f"{key} lacks required files: {absent}")
    if paths["image_dir"].is_dir():
        try:
            digest, count = directory_sha256(paths["image_dir"], "*_RGB.tif")
            sources["images"] = {"sha256": digest, "count": count}
        except Exception as exc:
            errors.append(f"Cannot hash Vaihingen images: {type(exc).__name__}: {exc}")
    # Never read labels in preflight: only their paths are checked, retaining GT isolation.
    sources["label_files_checked"] = [f"{image_id}_label.tif" for image_id in test_ids]
    return Preflight("ready" if not errors else "blocked", errors, cfg, sources, environment)


def _read_rgb(path: Path) -> np.ndarray:
    import tifffile
    value = tifffile.imread(path)
    if value.ndim != 3 or value.shape[2] < 3:
        raise InputValidationError(f"Expected HxWxC image: {path}")
    rgb = value[:, :, :3]
    if rgb.dtype != np.uint8:
        low, high = float(np.percentile(rgb, 1)), float(np.percentile(rgb, 99))
        rgb = np.clip((rgb.astype(np.float32) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _load_candidates(paths: dict[str, Path], image_id: str) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    source_root = str(paths["sam3_python_root"])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from sam3_remote_wsss.candidate_cache import load_candidate_cache

    metadata, candidates = load_candidate_cache(paths["candidates_dir"], image_id)
    shape = tuple(int(value) for value in metadata["image_shape"])
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append({
            "index": index, "mask": np.asarray(candidate.mask, dtype=bool),
            "x0": int(candidate.x0), "y0": int(candidate.y0),
            "class_name": str(candidate.class_name), "score": float(candidate.score),
        })
    if not rows:
        raise InputValidationError(f"Candidate cache has no regions: {image_id}")
    return shape, rows


def _encode_regions(image: np.ndarray, regions: list[dict[str, Any]], model: Any, preprocess: Any, device: str, batch_size: int) -> np.ndarray:
    from PIL import Image
    import torch

    chunks: list[np.ndarray] = []
    for start in range(0, len(regions), batch_size):
        batch = []
        for region in regions[start:start + batch_size]:
            context, masked, _ = crop_views(image, region["mask"], region["x0"], region["y0"])
            batch.extend((preprocess(Image.fromarray(context)), preprocess(Image.fromarray(masked))))
        with torch.inference_mode():
            features = model.encode_image(torch.stack(batch).to(device)).float()
            features = features / features.norm(dim=1, keepdim=True)
            features = features.reshape(-1, 2, features.shape[-1]).mean(dim=1)
            features = features / features.norm(dim=1, keepdim=True)
        chunks.append(features.cpu().numpy())
    return _normalize(np.concatenate(chunks, axis=0))


def _metrics_for_subset(matrix: np.ndarray, supported: list[str], unsupported: list[str]) -> dict[str, Any]:
    result = _aggregate([{"confusion_matrix": matrix.tolist()}])
    s_f1 = float(np.mean([result["per_class_f1"][name] for name in supported]))
    u_f1 = float(np.mean([result["per_class_f1"][name] for name in unsupported]))
    s_iou = float(np.mean([result["per_class_iou"][name] for name in supported]))
    u_iou = float(np.mean([result["per_class_iou"][name] for name in unsupported]))
    result.update({
        "S_F1": s_f1, "U_F1": u_f1, "H_F1": 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 else 0.0,
        "S_IoU": s_iou, "U_IoU": u_iou, "H_IoU": 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou else 0.0,
    })
    return result


def _gt_map(path: Path) -> np.ndarray:
    rgb = _read_rgb(path)
    gt = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for index, name in enumerate(CLASSES):
        gt[np.all(rgb == np.asarray(COLORS[name], dtype=np.uint8), axis=-1)] = index
    return gt


def _new_run_dir(root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"{prefix}{stamp}_{uuid.uuid4().hex[:8]}"
    if candidate.exists():  # UUID collision is exceptionally unlikely; never reuse a directory.
        raise InputValidationError(f"Refusing to reuse existing run directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv_exclusive(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _run(preflight: Preflight, config_path: Path) -> Path:
    cfg, paths = preflight.config, {key: Path(value) for key, value in preflight.config["paths"].items()}
    run_dir = _new_run_dir(paths["output_root"], str(cfg["experiment"].get("run_prefix", "run_")))
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    log = log_path.open("x", encoding="utf-8", newline="\n")
    try:
        def note(message: str) -> None:
            log.write(message + "\n")
            log.flush()
            print(message, flush=True)

        protocol = json.loads(paths["remoteclip_protocol_file"].read_text(encoding="utf-8"))
        import torch
        device = "cuda"
        model, preprocess, tokenizer, torch = _load_model(paths["remoteclip_checkpoint"], protocol, device)
        text_proto, token_hash = text_prototypes(model, tokenizer, protocol, device, torch)
        if text_proto.shape != (len(CLASSES), int(cfg["integrity"]["required_feature_dimension"])):
            raise InputValidationError("RemoteCLIP text prototype feature dimension is invalid.")
        train_ids, test_ids, all_ids = _expected_ids(cfg)
        all_features: list[np.ndarray] = []
        train_rows: list[dict[str, Any]] = []
        test_by_image: dict[str, tuple[tuple[int, int], list[dict[str, Any]], list[dict[str, Any]]]] = {}
        records: list[dict[str, Any]] = []
        for image_id in all_ids:
            shape, regions = _load_candidates(paths, image_id)
            image = _read_rgb(paths["image_dir"] / f"{image_id}_RGB.tif")
            if tuple(image.shape[:2]) != shape:
                raise InputValidationError(f"Image/candidate shape mismatch for {image_id}: image={image.shape[:2]}, cache={shape}")
            features = _encode_regions(image, regions, model, preprocess, device, int(cfg["runtime"]["image_batch"]))
            split = "train" if image_id in train_ids else "test"
            local_rows = []
            for candidate_index, region in enumerate(regions):
                row = {
                    "row_index": len(records), "image_id": image_id, "candidate_index": candidate_index,
                    "sam3_source_label": region["class_name"], "sam3_score": region["score"],
                    "x0": region["x0"], "y0": region["y0"], "split": split, "area": _area_from_id(image_id),
                }
                records.append(row)
                local_rows.append(row)
                if split == "train":
                    train_rows.append(row)
            all_features.append(features)
            if split == "test":
                test_by_image[image_id] = (shape, regions, local_rows)
            note(f"encoded {image_id}: {len(regions)} candidates")
        features = _normalize(np.concatenate(all_features, axis=0))
        visual = np.empty_like(text_proto)
        prototype_counts: dict[str, int] = {}
        for class_index, class_name in enumerate(CLASSES):
            positions = [row["row_index"] for row in train_rows if row["sam3_source_label"] == class_name]
            if not positions:
                raise InputValidationError(f"No train-area SAM3 candidates for visual prototype '{class_name}'.")
            visual[class_index] = _normalize(features[positions].mean(axis=0, keepdims=True))[0]
            prototype_counts[class_name] = len(positions)
        test_rows = [row for row in records if row["split"] == "test"]
        test_position_by_row_index = {int(row["row_index"]): index for index, row in enumerate(test_rows)}
        test_positions = np.asarray([row["row_index"] for row in test_rows], dtype=np.int64)
        test_features = features[test_positions]
        text_scores = test_features @ text_proto.T
        visual_scores = test_features @ visual.T
        fused = 0.5 * text_proto + 0.5 * visual
        norms = np.linalg.norm(fused, axis=1)
        if np.any(norms <= 1e-8):
            raise InputValidationError("RemoteCLIP C2 prototype has zero norm.")
        anchored = (0.5 * text_scores + 0.5 * visual_scores) / norms[None, :]
        text_pred = np.argmax(text_scores, axis=1).astype(np.int64)
        with (run_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in test_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        with (run_dir / "features.npz").open("xb") as handle:
            np.savez_compressed(handle, features=test_features.astype(np.float16), text_scores=text_scores.astype(np.float16),
                                visual_scores=visual_scores.astype(np.float16), text_prototypes=text_proto.astype(np.float16),
                                visual_prototypes=visual.astype(np.float16), text_pred=text_pred)
        full_mask = np.ones(len(CLASSES), dtype=bool)
        full_scores = method_score_matrices(text_scores, visual_scores, anchored, full_mask, text_pred)
        full_preds = method_predictions(full_scores, text_pred, full_mask)
        full_methods = list(cfg["matrix"]["full_support_methods"])
        full_artifacts: dict[str, str] = {}
        full_confusions: dict[str, list[dict[str, Any]]] = {method: [] for method in full_methods}
        gt_by_image: dict[str, np.ndarray] = {}
        # Persist all prediction artifacts before the first GT label is opened.
        for image_id, (shape, regions, local_rows) in test_by_image.items():
            positions = [test_position_by_row_index[int(row["row_index"])] for row in local_rows]
            for method in full_methods:
                prediction = full_preds[method][positions]
                scores = full_scores[method][positions, prediction]
                label_map, _ = assemble_semantic_map(shape, regions, prediction, scores, CLASSES)
                output = run_dir / f"{method}_{image_id}_semantic.npz"
                with output.open("xb") as handle:
                    np.savez_compressed(handle, label_map=label_map)
                full_artifacts[output.name] = sha256_file(output)
        predict_manifest = {
            "format_version": 1, "status": "predict_completed", "scientific_evidence": False,
            "dataset": "ISPRS Vaihingen 2D Semantic Labeling", "methods": full_methods,
            "partial_support": {"counts": [2, 3, 4], "policy": "all_deterministic_bitmasks"},
            "gt_read": False, "config_sha256": preflight.sources["config_sha256"],
            "code_commit": _commit_hash(), "runner_sha256": sha256_file(Path(__file__)),
            "sources": preflight.sources, "environment": preflight.environment,
            "text_token_sha256": token_hash, "visual_prototype_counts": prototype_counts,
            "record_count": len(test_rows), "train_record_count": len(train_rows),
            "records_sha256": sha256_file(run_dir / "records.jsonl"), "features_sha256": sha256_file(run_dir / "features.npz"),
            "artifacts": full_artifacts, "labels": {"read": False},
        }
        _write_json_exclusive(run_dir / "manifest.json", predict_manifest)
        note("predict phase completed; opening GT only for frozen evaluation")
        for image_id in test_ids:
            gt_by_image[image_id] = _gt_map(paths["label_dir"] / f"{image_id}_label.tif")
        labels_hash, labels_count = directory_sha256(paths["label_dir"], "*_label.tif")
        for image_id, (_, _, local_rows) in test_by_image.items():
            positions = [test_position_by_row_index[int(row["row_index"])] for row in local_rows]
            for method in full_methods:
                with np.load(run_dir / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                    prediction = archive["label_map"].astype(np.int64)
                full_confusions[method].append(pixel_confusion_fast(prediction, gt_by_image[image_id], CLASSES))
        full_metrics = {method: _aggregate(full_confusions[method]) for method in full_methods}
        full_rows = [{"method": method, "OA": full_metrics[method]["OA"], "macro_f1": full_metrics[method]["macro_f1"],
                      "mIoU": full_metrics[method]["mIoU"], "valid_pixels": full_metrics[method]["valid_pixels"]} for method in full_methods]
        _write_csv_exclusive(run_dir / "metrics.csv", ["method", "OA", "macro_f1", "mIoU", "valid_pixels"], full_rows)
        _write_json_exclusive(run_dir / "metrics.json", full_metrics)
        partial_rows: list[dict[str, Any]] = []
        partial_details: dict[str, Any] = {}
        partial_methods = list(cfg["matrix"]["partial_support_methods"])
        for subset_index in range(1 << len(CLASSES)):
            mask = np.asarray([(subset_index >> index) & 1 for index in range(len(CLASSES))], dtype=bool)
            k = int(mask.sum())
            if k not in (2, 3, 4):
                continue
            supported = [name for name, enabled in zip(CLASSES, mask) if enabled]
            unsupported = [name for name, enabled in zip(CLASSES, mask) if not enabled]
            score_set = method_score_matrices(text_scores, visual_scores, anchored, mask, text_pred)
            prediction_set = method_predictions(score_set, text_pred, mask)
            accumulators = {method: [] for method in partial_methods}
            for image_id, (shape, regions, local_rows) in test_by_image.items():
                positions = [test_position_by_row_index[int(row["row_index"])] for row in local_rows]
                for method in partial_methods:
                    prediction = prediction_set[method][positions]
                    score = score_set[method][positions, prediction]
                    label_map, _ = assemble_semantic_map(shape, regions, prediction, score, CLASSES)
                    accumulators[method].append(pixel_confusion_fast(label_map, gt_by_image[image_id], CLASSES))
            subset_key = f"subset_{subset_index}"
            partial_details[subset_key] = {"k": k, "supported": supported, "unsupported": unsupported, "methods": {}}
            for method in partial_methods:
                result = _metrics_for_subset(np.asarray(_aggregate(accumulators[method])["confusion_matrix"], dtype=np.int64), supported, unsupported)
                partial_details[subset_key]["methods"][method] = result
                partial_rows.append({"subset_index": subset_index, "k": k, "supported": "|".join(supported), "unsupported": "|".join(unsupported), "method": method,
                                     **{field: result[field] for field in ("OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU", "valid_pixels")}})
            note(f"partial subset {subset_index} (k={k}) completed")
        fields = ["subset_index", "k", "supported", "unsupported", "method", "OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU", "valid_pixels"]
        _write_csv_exclusive(run_dir / "partial_metrics.csv", fields, partial_rows)
        _write_json_exclusive(run_dir / "partial_metrics.json", partial_details)
        report = ["# RemoteCLIP Vaihingen baseline", "", "Frozen RemoteCLIP replacement with unchanged prompts, visual prototypes, CTP-v1, FusionCanvas and support protocol.", "", "## Full support", "", "| Method | OA | Macro F1 | mIoU |", "|---|---:|---:|---:|"]
        report.extend(f"| {row['method']} | {row['OA']:.6f} | {row['macro_f1']:.6f} | {row['mIoU']:.6f} |" for row in full_rows)
        report.extend(["", "## Partial support", "", "All deterministic support bitmasks with k=2/3/4 are in `partial_metrics.csv`; no partial semantic maps are saved.", ""])
        with (run_dir / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(report))
        final = dict(predict_manifest)
        final.update({"status": "completed", "scientific_evidence": True, "gt_read": True,
                      "labels": {"read": True, "sha256": labels_hash, "count": labels_count},
                      "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                      "partial_metrics_sha256": sha256_file(run_dir / "partial_metrics.json"),
                      "report_sha256": sha256_file(run_dir / "report.md")})
        (run_dir / "manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        note("completed")
        return run_dir
    except Exception:
        failure = {"status": "failed", "error": traceback.format_exc(), "scientific_evidence": False}
        failed = run_dir / "failure.json"
        if not failed.exists():
            _write_json_exclusive(failed, failure)
        raise
    finally:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen RemoteCLIP Vaihingen full + partial baseline once.")
    parser.add_argument("--config", type=Path, default=Path("configs/remoteclip_vaihingen_all_v0.yaml"))
    parser.add_argument("--preflight", action="store_true", help="Validate only; never create outputs.")
    args = parser.parse_args()
    try:
        preflight = _preflight(args.config)
    except Exception as exc:
        _json_line({"status": "blocked", "reason": f"configuration error: {type(exc).__name__}: {exc}", "outputs_created": False})
        return 3
    if preflight.status != "ready":
        _json_line({"status": "blocked", "reasons": preflight.errors, "outputs_created": False,
                    "environment": preflight.environment, "sources": preflight.sources})
        return 3
    if args.preflight:
        _json_line({"status": "ready", "outputs_created": False, "environment": preflight.environment, "sources": preflight.sources})
        return 0
    try:
        run_dir = _run(preflight, args.config.resolve())
    except Exception as exc:
        _json_line({"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "outputs_created": True})
        return 2
    _json_line({"status": "completed", "run_dir": str(run_dir), "outputs_created": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
