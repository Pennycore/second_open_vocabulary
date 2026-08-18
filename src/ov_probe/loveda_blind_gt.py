"""Blind LoveDA Train GT evaluation for frozen OpenAI-CLIP region recognition.

Two strictly separated phases:

- predict: builds the frozen text and SAM3 visual prototypes from the
  development partition only, scores the heldout regions, and persists every
  prediction before any GT pixel is opened.
- evaluate: verifies the persisted predictions, then opens the LoveDA Train
  label PNGs, derives each heldout region's GT label by majority vote inside
  the registered candidate mask, and computes region-level accuracy, macro F1,
  per-class F1 and confusion matrices.

The protocol fixes the split, feature cache, checkpoint, prompts, weights and
GT color mapping; nothing is tuned here.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .io import InputValidationError, create_run_dir, load_config, sha256_file, write_json
from .openai_clip_visual_anchor import (
    _normalize,
    _read_jsonl,
    build_visual_prototypes,
    join_split_to_partitions,
)

_PROTOCOL_NAME = "loveda_blind_gt_protocol_v0.json"
_CLASSES = ["building", "road", "water", "barren", "forest", "agriculture"]
_BACKGROUND = (0, 0, 0)
_IGNORE = (255, 0, 255)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_ordinary_path(path: Path, name: str) -> None:
    for component in (path.absolute(), *path.absolute().parents):
        if component.exists() and (component.is_symlink() or bool(getattr(os, "isjunction", lambda _: False)(component))):
            raise InputValidationError(f"Blind-GT {name} may not traverse a symlink or junction.")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    # The LoveDA label directory is a read-only external dataset root; it is allowed
    # outside the project but must be an ordinary path (no symlink/junction chain).
    if name == "loveda_label_dir":
        _assert_ordinary_path(resolved, name)
        return str(resolved)
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"Blind-GT path escapes project root: {name}")
    _assert_ordinary_path(resolved, name)
    return str(resolved)


def load_loveda_blind_gt_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read blind-GT configuration.") from exc
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Blind-GT config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    required = {
        "pixel_pack", "split_manifest", "development_records", "heldout_records",
        "feature_cache_manifest", "features", "row_partitions",
        "openai_clip_checkpoint", "loveda_label_dir", "protocol_file", "output_root",
    }
    if not isinstance(paths, dict) or set(paths) != required:
        raise InputValidationError("Blind-GT config paths do not match the frozen schema.")
    for key, value in list(paths.items()):
        paths[key] = _resolve_project_path(value, root, key)
    output_root = Path(str(paths["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("Blind-GT output_root must be directly under outputs/.")
    expected = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(paths["protocol_file"])).resolve() != expected:
        raise InputValidationError("Blind-GT protocol must be the committed canonical protocol.")
    try:
        raw = expected.read_bytes()
        protocol = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read canonical blind-GT protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("Blind-GT protocol must be a JSON object.")
    protocol["path"] = str(expected)
    protocol["sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg, protocol


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_pre_result" or protocol.get("scientific_evidence") is not True:
        raise InputValidationError("Blind-GT protocol must be frozen and confirmatory.")
    if protocol.get("blind", {}).get("gt_read_before_predictions") is not False:
        raise InputValidationError("Blind-GT protocol must forbid GT reads before predictions.")
    if protocol.get("classes") != _CLASSES:
        raise InputValidationError("Blind-GT classes differ from the registered six-class vocabulary.")
    model = protocol.get("model", {})
    if model.get("architecture") != "ViT-B-32-quickgelu" or int(model.get("feature_dimension", -1)) != 512 or model.get("open_clip_version") != "3.3.0":
        raise InputValidationError("Blind-GT model registration is invalid.")
    strategy = protocol.get("strategy", {})
    if strategy.get("fixed_text_weight") != 0.5 or strategy.get("fixed_visual_weight") != 0.5:
        raise InputValidationError("Blind-GT fusion weights differ from the frozen design.")
    if any(protocol.get("constraints", {}).get(key) is not False for key in ("training", "adapter_or_new_network", "alpha_tuning", "prompt_tuning", "threshold_tuning", "grid_search", "model_selection_or_decision", "gt_before_predictions", "overwrite")):
        raise InputValidationError("Blind-GT constraints differ from the frozen protocol.")


def _validate_hashed_inputs(cfg: dict[str, Any], protocol: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    paths = cfg["paths"]
    registered = {
        "feature_cache_manifest": protocol["feature_cache"]["manifest_sha256"],
        "features": protocol["feature_cache"]["features_sha256"],
        "row_partitions": protocol["feature_cache"]["row_partitions_sha256"],
        "split_manifest": protocol["split_inputs"]["manifest_sha256"],
        "development_records": protocol["split_inputs"]["development_records_sha256"],
        "heldout_records": protocol["split_inputs"]["heldout_records_sha256"],
    }
    hashes: dict[str, str] = {}
    for name, expected in registered.items():
        candidate = Path(str(paths[name]))
        if not candidate.is_file() or (hashes.setdefault(name, sha256_file(candidate)) != expected):
            raise InputValidationError(f"Blind-GT immutable input differs from protocol: {name}")
    try:
        cache_manifest = json.loads(Path(str(paths["feature_cache_manifest"])).read_text(encoding="utf-8"))
        split_manifest = json.loads(Path(str(paths["split_manifest"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Blind-GT input manifest is invalid JSON.") from exc
    if cache_manifest.get("status") != "completed":
        raise InputValidationError("Feature cache is not the registered completed cache.")
    cache_outputs = cache_manifest.get("outputs", {})
    if cache_outputs.get("features_openai_clip", {}).get("sha256") != hashes["features"] or cache_outputs.get("row_partitions", {}).get("sha256") != hashes["row_partitions"]:
        raise InputValidationError("Feature-cache manifest does not bind its configured cache files.")
    if split_manifest.get("status") != "completed" or split_manifest.get("development", {}).get("records_sha256") != hashes["development_records"] or split_manifest.get("heldout", {}).get("records_sha256") != hashes["heldout_records"]:
        raise InputValidationError("Split manifest does not bind its configured split files.")
    cached = np.load(Path(str(paths["features"])), allow_pickle=False)
    if cached.dtype != np.dtype(protocol["feature_cache"]["feature_dtype"]):
        raise InputValidationError("Cached OpenAI-CLIP feature dtype differs from the frozen cache.")
    features = cached.astype(np.float32)
    expected_shape = tuple(protocol["feature_cache"]["feature_shape"])
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise InputValidationError("Cached OpenAI-CLIP features have invalid shape or values.")
    partitions = _read_jsonl(Path(str(paths["row_partitions"])), "cache row partitions")
    development = _read_jsonl(Path(str(paths["development_records"])), "development records")
    heldout = _read_jsonl(Path(str(paths["heldout_records"])), "heldout records")
    dev, hold = join_split_to_partitions(partitions, development, heldout, len(features))
    return features, dev, hold, hashes


def _text_prototypes(protocol: dict[str, Any], checkpoint: Path, device: str) -> tuple[np.ndarray, str]:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise InputValidationError("Blind-GT text construction requires torch and open_clip.") from exc
    if getattr(open_clip, "__version__", None) != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the frozen protocol.")
    model, _, _ = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.eval().to(device)
    templates = protocol["prompts"]["group_a_templates"]
    classes = protocol["classes"]
    texts = [template.format(**{"class": name}) for name in classes for template in templates]
    tokenizer = open_clip.get_tokenizer(protocol["model"]["architecture"])
    tokens = tokenizer(texts)
    token_hash = hashlib.sha256(tokens.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    with torch.inference_mode():
        encoded = model.encode_text(tokens.to(device)).float().cpu().numpy()
    encoded = _normalize(encoded).reshape(len(classes), len(templates), -1)
    return _normalize(encoded.mean(axis=1)), token_hash


def _load_record_masks(pixel_pack: str | Path) -> dict[int, dict[str, Any]]:
    """Load crop masks from shards keyed by row_index, plus crop boxes/shapes."""
    root = Path(pixel_pack)
    records = _read_jsonl(root / "records.jsonl", "pixel-pack records")
    shards: dict[str, dict[str, np.ndarray]] = {}
    by_row: dict[int, dict[str, Any]] = {}
    for row in records:
        shard_name = str(row["shard"])
        if shard_name not in shards:
            with np.load(root / shard_name, allow_pickle=False) as archive:
                shards[shard_name] = {name: archive[name] for name in archive.files}
        shard = shards[shard_name]
        order = {int(value): index for index, value in enumerate(shard["row_indices"].tolist())}
        position = order[int(row["row_index"])]
        crop_shape = tuple(int(value) for value in shard["crop_shapes"][position])
        mask_offset = int(shard["crop_mask_offsets"][position])
        mask_end = int(shard["crop_mask_offsets"][position + 1])
        bits = shard["crop_mask_bits"][mask_offset:mask_end]
        mask = np.unpackbits(bits, bitorder="little")[: crop_shape[0] * crop_shape[1]].reshape(crop_shape)
        by_row[int(row["row_index"])] = {
            "image_id": str(row["image_id"]),
            "crop_box": tuple(int(value) for value in row["crop_box"]),
            "crop_shape": crop_shape,
            "mask": mask.astype(bool),
        }
    return by_row


def _region_gt_from_label(mask_info: dict[str, Any], label_png: np.ndarray, color_map: dict[str, tuple[int, int, int]]) -> tuple[str | None, int, int]:
    x1, y1, x2, y2 = mask_info["crop_box"]
    crop = label_png[y1:y2, x1:x2]
    mask = mask_info["mask"]
    if crop.shape[:2] != mask.shape:
        raise InputValidationError("GT crop geometry differs from the registered candidate mask.")
    pixels = crop[mask]
    votes: dict[str, int] = {}
    for name, color in color_map.items():
        matches = (pixels == np.asarray(color, dtype=np.uint8)).all(axis=1)
        votes[name] = int(matches.sum())
    total = sum(votes.values())
    if total == 0:
        return None, 0, int(mask.sum())
    winner = max(votes, key=lambda name: votes[name])
    return winner, total, int(mask.sum())


def _classification_metrics(predictions: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, Any]:
    names = np.asarray([classes[int(index)] for index in predictions])
    truth = np.asarray([classes[int(index)] for index in labels])
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for predicted, actual in zip(names, truth):
        matrix[classes.index(actual), classes.index(predicted)] += 1
    per_class_f1: dict[str, float] = {}
    per_class_iou: dict[str, float] = {}
    for index, name in enumerate(classes):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        per_class_f1[name] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class_iou[name] = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    accuracy = float((names == truth).mean())
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    macro_iou = float(np.mean(list(per_class_iou.values())))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "per_class_f1": per_class_f1,
        "per_class_iou": per_class_iou,
        "confusion_matrix": matrix.tolist(),
        "count": int(len(names)),
    }


def _prepare_output_dir(output_root: str | Path) -> Path:
    destination = Path(output_root).resolve()
    _assert_ordinary_path(destination, "output")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise InputValidationError("Blind-GT output directory must be absent or empty.")
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)
    return destination


def _checkpoint_sha(cfg: dict[str, Any], protocol: dict[str, Any]) -> str:
    checkpoint_value = cfg["paths"].get("openai_clip_checkpoint")
    if not checkpoint_value:
        raise InputValidationError("Blind-GT config has no checkpoint; use the deployment config.")
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_file() or sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("OpenAI-CLIP checkpoint differs from the registered artifact.")
    return checkpoint


def _subsample_support(
    development: list[dict[str, Any]], fraction: float, seed: int, classes: list[str]
) -> list[dict[str, Any]]:
    """Per-class deterministic subsample of development records for stability probes."""
    if fraction >= 1.0:
        return development
    rng = random.Random(int(seed))
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in classes}
    for row in development:
        by_class.setdefault(str(row.get("sam3_source_label", "")), []).append(row)
    selected: list[dict[str, Any]] = []
    for name in classes:
        rows = by_class[name]
        if not rows:
            raise InputValidationError(f"No development rows for class {name} in support subsample.")
        count = max(1, int(round(len(rows) * float(fraction))))
        chosen = rng.sample(rows, count)
        selected.extend(chosen)
    return selected


def run_loveda_blind_gt_predict(
    cfg: dict[str, Any], protocol: dict[str, Any], *, support_fraction: float = 1.0, support_seed: int = 42
) -> dict[str, Any]:
    """Phase 1: score heldout regions and persist predictions without reading GT.

    support_fraction < 1.0 randomly subsamples the development records per class
    (fixed seed) to test prototype stability; all other settings stay frozen.
    """
    _validate_protocol(protocol)
    if not 0 < support_fraction <= 1.0:
        raise InputValidationError("support_fraction must be in (0, 1].")
    if cfg["paths"].get("loveda_label_dir"):
        raise InputValidationError("Predict phase must not configure a GT label directory.")
    destination = _prepare_output_dir(cfg["paths"]["output_root"])
    features, development, heldout, input_hashes = _validate_hashed_inputs(cfg, protocol)
    checkpoint = _checkpoint_sha(cfg, protocol)
    try:
        import torch
    except ImportError as exc:
        raise InputValidationError("Blind-GT runtime requires torch.") from exc
    requested = str(cfg.get("runtime", {}).get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    text, token_hash = _text_prototypes(protocol, checkpoint, device)
    support_rows = _subsample_support(development, support_fraction, support_seed, protocol["classes"])
    visual, counts = build_visual_prototypes(features, support_rows, list(protocol["classes"]))
    regions = _normalize(features)
    text_scores = regions @ _normalize(text).T
    visual_scores = regions @ _normalize(visual).T
    fused_scores = 0.5 * text_scores + 0.5 * visual_scores
    predictions = {
        "text_only": np.argmax(text_scores, axis=1).astype(np.int16),
        "visual_only": np.argmax(visual_scores, axis=1).astype(np.int16),
        "fused_text_visual": np.argmax(fused_scores, axis=1).astype(np.int16),
    }
    hold_indices = np.asarray([int(row["row_index"]) for row in heldout], dtype=np.int64)
    array_path = destination / "predictions.npz"
    with array_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            text_scores=text_scores.astype(np.float16),
            visual_scores=visual_scores.astype(np.float16),
            fused_scores=fused_scores.astype(np.float16),
            text_only=predictions["text_only"],
            visual_only=predictions["visual_only"],
            fused_text_visual=predictions["fused_text_visual"],
            heldout_row_indices=hold_indices,
            text_prototypes=text.astype(np.float16),
            visual_prototypes=visual.astype(np.float16),
        )
    heldout_keys = [
        {
            "row_index": int(row["row_index"]),
            "image_id": str(row["image_id"]),
            "candidate_index": int(row["candidate_index"]),
            "sam3_source_label": str(row.get("sam3_source_label", "")),
            "cam_label": str(row.get("cam_label", "")),
        }
        for row in heldout
    ]
    with (destination / "heldout_keys.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in heldout_keys:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "format_version": 1,
        "phase": "predict",
        "status": "completed",
        "scientific_evidence": True,
        "blind": protocol["blind"],
        "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "inputs": input_hashes,
        "model": protocol["model"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "text_token_sha256": token_hash,
        "visual_prototype_counts": counts,
        "support": {"fraction": support_fraction, "seed": support_seed, "rows": len(support_rows)},
        "strategy": protocol["strategy"],
        "constraints": protocol["constraints"],
        "device": device,
        "heldout": {
            "record_count": len(heldout),
            "image_count": len({str(row["image_id"]) for row in heldout}),
            "keys_sha256": sha256_file(destination / "heldout_keys.jsonl"),
        },
        "outputs": {
            "predictions": {"path": array_path.name, "sha256": sha256_file(array_path)},
            "heldout_keys": {"path": "heldout_keys.jsonl", "sha256": sha256_file(destination / "heldout_keys.jsonl")},
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def run_loveda_blind_gt_evaluate(cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Phase 2: after predictions exist, open GT and compute region metrics."""
    _validate_protocol(protocol)
    label_dir = cfg["paths"].get("loveda_label_dir")
    if not label_dir or not Path(label_dir).is_dir():
        raise InputValidationError("Evaluate phase requires the registered LoveDA label directory.")
    source = Path(output_dir).resolve()
    predict_manifest_path = source / "manifest.json"
    if not predict_manifest_path.is_file():
        raise InputValidationError("Predict-phase manifest is missing; run predict first.")
    predict_manifest = json.loads(predict_manifest_path.read_text(encoding="utf-8"))
    if predict_manifest.get("phase") != "predict" or predict_manifest.get("status") != "completed":
        raise InputValidationError("Predict-phase manifest is not a completed predict run.")
    arrays_path = source / "predictions.npz"
    if predict_manifest["outputs"]["predictions"]["sha256"] != sha256_file(arrays_path):
        raise InputValidationError("Prediction artifact changed since the predict phase.")
    with np.load(arrays_path, allow_pickle=False) as archive:
        required = {"text_only", "visual_only", "fused_text_visual", "heldout_row_indices"}
        if not required.issubset(set(archive.files)):
            raise InputValidationError("Prediction artifact is missing required arrays.")
        pred_arrays = {
            "text_only": archive["text_only"].astype(np.int64),
            "visual_only": archive["visual_only"].astype(np.int64),
            "fused_text_visual": archive["fused_text_visual"].astype(np.int64),
        }
        has_scores = {"text_scores", "visual_scores", "fused_scores"}.issubset(set(archive.files))
        if has_scores:
            text_scores = archive["text_scores"].astype(np.float32)
            visual_scores = archive["visual_scores"].astype(np.float32)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    records = _read_jsonl(source / "heldout_keys.jsonl", "heldout keys")
    if len(records) != len(hold_indices) or any(int(row["row_index"]) != int(index) for row, index in zip(records, hold_indices)):
        raise InputValidationError("Heldout key records do not match prediction row indices.")
    if predict_manifest["heldout"]["keys_sha256"] != sha256_file(source / "heldout_keys.jsonl"):
        raise InputValidationError("Heldout keys changed since the predict phase.")
    color_map = {name: tuple(int(value) for value in color) for name, color in protocol["ground_truth"]["color_map"].items()}
    if set(color_map) != set(_CLASSES):
        raise InputValidationError("GT color map differs from the frozen classes.")
    mask_info = _load_record_masks(cfg["paths"]["pixel_pack"])
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    unlabeled = 0
    labeled_count = 0
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            label_path = Path(label_dir) / f"{image_id}_label.png"
            if not label_path.is_file():
                raise InputValidationError(f"Missing LoveDA label PNG: {label_path}")
            try:
                from PIL import Image
                with Image.open(label_path) as image:
                    label_cache[image_id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                raise InputValidationError(f"Cannot read LoveDA label PNG {label_path}.") from exc
        info = mask_info.get(int(row["row_index"]))
        if info is None:
            raise InputValidationError(f"No pixel-pack mask for heldout row {row['row_index']}.")
        label, _, _ = _region_gt_from_label(info, label_cache[image_id], color_map)
        gt_labels.append(label)
        if label is None:
            unlabeled += 1
        else:
            labeled_count += 1
    label_index = {name: index for index, name in enumerate(_CLASSES)}
    labeled_mask = np.asarray([label is not None for label in gt_labels], dtype=bool)
    gt_index = np.asarray([label_index[label] for label in gt_labels if label is not None], dtype=np.int64)
    # predictions arrays are indexed by frozen cache row_index (6000 rows); hold_indices
    # carries the heldout row_index values in key-record order, aligned with gt_labels.
    if len(hold_indices) != len(labeled_mask):
        raise InputValidationError("Heldout row index count does not match GT label count.")
    metric_results: dict[str, dict[str, Any]] = {}
    for method, array in pred_arrays.items():
        selected = array[hold_indices[labeled_mask]]
        metric_results[method] = _classification_metrics(selected, gt_index, _CLASSES)
    # Leave-one-class-out variants reuse the frozen scores: the unsupported class's
    # visual prototype is dropped, so its fused score degenerates to the text score
    # while the other five classes keep the fixed 0.5/0.5 fusion.
    leave_one_out: dict[str, dict[str, Any]] = {}
    if has_scores:
        for unsupported in _CLASSES:
            unsupported_index = label_index[unsupported]
            loo_scores = 0.5 * text_scores + 0.5 * visual_scores
            loo_scores[:, unsupported_index] = text_scores[:, unsupported_index]
            loo_pred = np.argmax(loo_scores, axis=1).astype(np.int64)
            selected = loo_pred[hold_indices[labeled_mask]]
            result = _classification_metrics(selected, gt_index, _CLASSES)
            leave_one_out[unsupported] = result
            metric_results[f"fused_leave_out_{unsupported}"] = result
    metrics_csv = _write_metrics_csv(source, metric_results)
    confusion_csvs = _write_confusion_csvs(source, metric_results)
    loo_csv = None
    if leave_one_out:
        loo_csv = _write_loo_csv(source, leave_one_out, protocol["classes"], metric_results)
    summary = {
        "format_version": 1,
        "phase": "evaluate",
        "status": "completed",
        "scientific_evidence": True,
        "predict_manifest_sha256": sha256_file(predict_manifest_path),
        "predictions_sha256": predict_manifest["outputs"]["predictions"]["sha256"],
        "heldout": {
            "record_count": len(records),
            "labeled_count": labeled_count,
            "unlabeled_count": unlabeled,
            "unlabeled_rule": protocol["ground_truth"]["unlabeled_rule"],
        },
        "metrics": metric_results,
        "leave_one_class_out": {
            "rule": "unsupported class fused score degenerates to its text score; other five classes keep 0.5/0.5 fusion; full 6-class vocabulary prediction",
            "variants": {name: {"macro_f1": result["macro_f1"], "macro_iou": result["macro_iou"], "unsupported_class_f1": result["per_class_f1"][name], "unsupported_class_iou": result["per_class_iou"][name], "supported_macro_f1": float(np.mean([result["per_class_f1"][other] for other in protocol["classes"] if other != name])), "supported_macro_iou": float(np.mean([result["per_class_iou"][other] for other in protocol["classes"] if other != name]))} for name, result in leave_one_out.items()},
        },
        "outputs": {
            "metrics_csv": {"path": metrics_csv.name, "sha256": sha256_file(metrics_csv)},
            "confusion_matrices": {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in confusion_csvs.items()},
        },
    }
    if loo_csv is not None:
        summary["outputs"]["leave_one_out_csv"] = {"path": loo_csv.name, "sha256": sha256_file(loo_csv)}
    write_json(source / "evaluate_manifest.json", summary)
    return summary


def _write_loo_csv(destination: Path, leave_one_out: dict[str, dict[str, Any]], classes: list[str], metric_results: dict[str, dict[str, Any]]) -> Path:
    rows = [["unsupported_class", "supported_macro_f1", "supported_macro_iou", "unsupported_class_f1", "unsupported_class_iou", "all_macro_f1", "all_macro_iou", "text_only_all_macro_f1", "full_anchor_all_macro_f1"]]
    text_f1 = metric_results["text_only"]["macro_f1"]
    full_f1 = metric_results["fused_text_visual"]["macro_f1"]
    for name in classes:
        result = leave_one_out[name]
        supported_f1 = float(np.mean([result["per_class_f1"][other] for other in classes if other != name]))
        supported_iou = float(np.mean([result["per_class_iou"][other] for other in classes if other != name]))
        rows.append([
            name,
            f"{supported_f1:.6f}",
            f"{supported_iou:.6f}",
            f"{result['per_class_f1'][name]:.6f}",
            f"{result['per_class_iou'][name]:.6f}",
            f"{result['macro_f1']:.6f}",
            f"{result['macro_iou']:.6f}",
            f"{text_f1:.6f}",
            f"{full_f1:.6f}",
        ])
    path = destination / "leave_one_out_summary.csv"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(",".join(row) + "\n")
    return path


def _write_metrics_csv(destination: Path, metric_results: dict[str, dict[str, Any]]) -> Path:
    rows = [["method", "accuracy", "macro_f1", "macro_iou", "count"]]
    for method, result in metric_results.items():
        rows.append([method, f"{result['accuracy']:.6f}", f"{result['macro_f1']:.6f}", f"{result['macro_iou']:.6f}", str(result["count"])])
    path = destination / "metrics.csv"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(",".join(row) + "\n")
    per_class_rows = [["method", "class", "f1", "iou"]]
    for method, result in metric_results.items():
        for name in _CLASSES:
            per_class_rows.append([method, name, f"{result['per_class_f1'][name]:.6f}", f"{result['per_class_iou'][name]:.6f}"])
    per_class_path = destination / "per_class_f1.csv"
    with per_class_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in per_class_rows:
            handle.write(",".join(row) + "\n")
    return path


def _write_confusion_csvs(destination: Path, metric_results: dict[str, dict[str, Any]]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for method, result in metric_results.items():
        rows = [["actual\\predicted", *_CLASSES]]
        for actual_index, actual_name in enumerate(_CLASSES):
            rows.append([actual_name, *[str(int(value)) for value in result["confusion_matrix"][actual_index]]])
        path = destination / f"confusion_matrix_{method}.csv"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(",".join(row) + "\n")
        outputs[method] = path
    return outputs


__all__ = [
    "load_loveda_blind_gt_config",
    "run_loveda_blind_gt_predict",
    "run_loveda_blind_gt_evaluate",
]
