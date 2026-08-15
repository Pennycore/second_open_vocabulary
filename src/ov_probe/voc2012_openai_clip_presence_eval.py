"""Fail-closed, post-inference VOC2012 image-presence evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import InputValidationError, sha256_file


_PROTOCOL_NAME = "voc2012_openai_clip_presence_eval_protocol_v1.json"
_VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor",
)
_IMAGE_ID_RE = re.compile(r"^[0-9]{4}_[0-9]{6}$")
_SCORE_MANIFEST_SHA256 = "67f4e371999e856bfb5b3169d7343576d958c6e3c27a2abf49db3204530301f5"
_SCORES_SHA256 = "1b88e70f2456d6b89a7f552fc6cd6ecf0b03c38f56cf0cfb12abe65a406ef24d"
_IMAGE_IDS_SHA256 = "1439193bc81ad369518d66d4f87003877aa0b535e201dac65a42362637e04af7"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(os.path, "isjunction", lambda _: False)(path))


def _assert_ordinary(path: Path, label: str) -> None:
    for component in (path.absolute(), *path.absolute().parents):
        if component.exists() and _link_like(component):
            raise InputValidationError(f"VOC presence evaluation {label} may not traverse a symlink or junction.")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        raise InputValidationError(f"VOC presence evaluation path must be project-contained and relative: {name}")
    untrusted = root / candidate
    _assert_ordinary(untrusted, name)
    resolved = untrusted.resolve()
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"VOC presence evaluation path escapes project root: {name}")
    return str(resolved)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_pre_result" or protocol.get("scientific_evidence") is not False or protocol.get("evaluation_evidence") is not True:
        raise InputValidationError("VOC presence evaluation protocol must disclose its frozen evaluation-only scope.")
    if protocol.get("role") != "PASCAL VOC 2012 Segmentation val image-presence evaluation of one immutable OpenAI-CLIP zero-shot score cache":
        raise InputValidationError("VOC presence evaluation role differs from the frozen scope.")
    if protocol.get("classes") != list(_VOC_CLASSES):
        raise InputValidationError("VOC presence evaluation class order differs from the frozen score cache.")
    expected_dataset = {
        "name": "PASCAL VOC 2012", "split": "Segmentation val", "mask_root_relative": "VOCdevkit/VOC2012/SegmentationClass",
        "image_count": 1449, "class_id_range": [1, 20], "background_id": 0, "ignore_id": 255,
    }
    if protocol.get("dataset") != expected_dataset:
        raise InputValidationError("VOC presence evaluation dataset registration differs from the frozen values.")
    expected_cache = {
        "manifest_sha256": _SCORE_MANIFEST_SHA256, "scores_sha256": _SCORES_SHA256, "image_ids_sha256": _IMAGE_IDS_SHA256,
        "scores_shape": [1449, 20], "scores_dtype": "float32", "image_id_count": 1449,
    }
    if protocol.get("score_cache") != expected_cache:
        raise InputValidationError("VOC presence evaluation score-cache registration differs from the frozen values.")
    metric = protocol.get("metric", {})
    if metric.get("name") != "uninterpolated integral average precision" or metric.get("zero_support") != "fail_closed":
        raise InputValidationError("VOC presence evaluation metric differs from the frozen endpoint.")
    required_false = (
        "model_execution", "network_download", "score_mutation", "threshold_selection", "prediction_export", "prompt_selection",
        "prototype_selection", "class_selection", "segmentation_miou", "sam3_rerun", "training", "overwrite",
    )
    if any(protocol.get("constraints", {}).get(key) is not False for key in required_false):
        raise InputValidationError("VOC presence evaluation constraints differ from the frozen scope.")


def load_voc2012_openai_clip_presence_eval_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read VOC presence evaluation configuration.") from exc
    required = {"score_cache_manifest", "raw_mask_root", "protocol_file", "output_root"}
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False or not isinstance(cfg.get("paths"), dict) or set(cfg["paths"]) != required:
        raise InputValidationError("VOC presence evaluation config paths do not match the frozen schema.")
    for key, value in list(cfg["paths"].items()):
        cfg["paths"][key] = _resolve_project_path(value, root, key)
    output_root = Path(str(cfg["paths"]["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("VOC presence evaluation output_root must be directly under outputs/.")
    canonical = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(cfg["paths"]["protocol_file"])).resolve() != canonical:
        raise InputValidationError("VOC presence evaluation protocol must be the committed canonical protocol.")
    try:
        raw = canonical.read_bytes()
        protocol = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read canonical VOC presence evaluation protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("VOC presence evaluation protocol must be a JSON object.")
    _validate_protocol(protocol)
    protocol["path"] = str(canonical)
    protocol["sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg, protocol


def verify_voc2012_openai_clip_presence_eval_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("VOC presence evaluation repository anchor could not read git state.") from exc
    actual = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual != expected_protocol_sha256:
        raise InputValidationError("VOC presence evaluation requires the approved clean commit and protocol SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def integral_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Uninterpolated AP with equal scores treated as one threshold."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if values.size == 0 or values.size != truth.size or not np.isfinite(values).all():
        raise InputValidationError("Average precision requires equal-length finite score and label arrays.")
    positives = int(truth.sum())
    if positives == 0:
        raise InputValidationError("Average precision is undefined for a zero-support class.")
    order = np.argsort(-values, kind="mergesort")
    sorted_scores, sorted_truth = values[order], truth[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    cumulative_positive = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    cumulative_count = ends + 1
    recall = cumulative_positive.astype(np.float64) / positives
    precision = cumulative_positive.astype(np.float64) / cumulative_count
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def presence_from_mask(mask: np.ndarray, class_count: int = 20) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise InputValidationError("VOC SegmentationClass mask must decode to a non-empty two-dimensional integer ID array.")
    allowed = (values == 0) | (values == 255) | ((values >= 1) & (values <= class_count))
    if not bool(np.all(allowed)):
        raise InputValidationError("VOC SegmentationClass mask contains an invalid class ID.")
    return np.asarray([(values == class_id).any() for class_id in range(1, class_count + 1)], dtype=bool)


def _mask_path(mask_root: Path, image_id: str) -> Path:
    if not _IMAGE_ID_RE.fullmatch(image_id):
        raise InputValidationError("VOC score cache contains an invalid image ID.")
    candidate = (mask_root / f"{image_id}.png").resolve()
    if not _is_relative_to(candidate, mask_root) or candidate.parent != mask_root or candidate.name != f"{image_id}.png":
        raise InputValidationError("VOC mask path escapes the configured SegmentationClass root.")
    _assert_ordinary(candidate, "mask")
    if not candidate.is_file():
        raise InputValidationError(f"VOC mask is missing for immutable image ID {image_id}.")
    return candidate


def _read_presence_labels(mask_root: Path, image_ids: list[str]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise InputValidationError("VOC presence evaluation requires Pillow to decode PNG masks.") from exc
    labels = np.empty((len(image_ids), len(_VOC_CLASSES)), dtype=bool)
    for index, image_id in enumerate(image_ids):
        mask_path = _mask_path(mask_root, image_id)
        with Image.open(mask_path) as image:
            labels[index] = presence_from_mask(np.asarray(image))
    return labels


def _load_fixed_score_cache(manifest_path: Path, protocol: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if sha256_file(manifest_path) != protocol["score_cache"]["manifest_sha256"]:
        raise InputValidationError("VOC presence evaluation score manifest SHA-256 differs from the frozen input.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("VOC presence evaluation score manifest is not valid JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "completed" or manifest.get("scientific_evidence") is not False:
        raise InputValidationError("VOC presence evaluation needs a completed immutable non-evaluative score manifest.")
    if manifest.get("image_count") != 1449 or manifest.get("role") != "PASCAL VOC 2012 validation-image OpenAI-CLIP zero-shot score cache; input-only and no evaluation":
        raise InputValidationError("VOC presence evaluation score manifest identity differs from the frozen input.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"scores", "image_ids", "image_feature_stats"}:
        raise InputValidationError("VOC presence evaluation score manifest output schema differs from the frozen cache.")
    scores_entry, ids_entry = outputs["scores"], outputs["image_ids"]
    if not isinstance(scores_entry, dict) or not isinstance(ids_entry, dict):
        raise InputValidationError("VOC presence evaluation score manifest output entries are invalid.")
    frozen = protocol["score_cache"]
    if scores_entry != {"path": "scores_openai_clip.npy", "sha256": frozen["scores_sha256"], "dtype": "float32", "shape": [1449, 20]}:
        raise InputValidationError("VOC presence evaluation score artifact binding differs from the frozen cache.")
    if ids_entry != {"path": "image_ids.json", "sha256": frozen["image_ids_sha256"], "count": 1449}:
        raise InputValidationError("VOC presence evaluation image-ID artifact binding differs from the frozen cache.")
    scores_path, ids_path = manifest_path.parent / scores_entry["path"], manifest_path.parent / ids_entry["path"]
    if scores_path.parent != manifest_path.parent or ids_path.parent != manifest_path.parent:
        raise InputValidationError("VOC presence evaluation score artifacts must be direct siblings of their manifest.")
    _assert_ordinary(scores_path, "score cache")
    _assert_ordinary(ids_path, "score cache")
    if not scores_path.is_file() or not ids_path.is_file() or sha256_file(scores_path) != frozen["scores_sha256"] or sha256_file(ids_path) != frozen["image_ids_sha256"]:
        raise InputValidationError("VOC presence evaluation score artifacts differ from the frozen input.")
    scores = np.load(scores_path, allow_pickle=False)
    if scores.shape != (1449, 20) or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise InputValidationError("VOC presence evaluation scores have an invalid shape, dtype, or value.")
    try:
        image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("VOC presence evaluation image IDs are not valid JSON.") from exc
    if not isinstance(image_ids, list) or len(image_ids) != 1449 or any(not isinstance(item, str) for item in image_ids) or len(set(image_ids)) != len(image_ids) or any(not _IMAGE_ID_RE.fullmatch(item) for item in image_ids):
        raise InputValidationError("VOC presence evaluation image IDs do not exactly bind the score rows.")
    return scores, image_ids, manifest


def run_voc2012_openai_clip_presence_eval(cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    _validate_protocol(protocol)
    score_manifest_value, mask_root_value = cfg["paths"].get("score_cache_manifest"), cfg["paths"].get("raw_mask_root")
    if not score_manifest_value or not mask_root_value:
        raise InputValidationError("Tracked VOC presence evaluation config is non-runnable; deployment paths are required.")
    score_manifest, mask_root = Path(str(score_manifest_value)), Path(str(mask_root_value))
    if not score_manifest.is_file() or not mask_root.is_dir() or mask_root.name != "SegmentationClass":
        raise InputValidationError("VOC presence evaluation deployment inputs are unavailable or not a direct SegmentationClass root.")
    _assert_ordinary(score_manifest, "score manifest")
    _assert_ordinary(mask_root, "mask root")
    scores, image_ids, score_manifest_payload = _load_fixed_score_cache(score_manifest, protocol)
    labels = _read_presence_labels(mask_root, image_ids)
    if labels.shape != scores.shape:
        raise InputValidationError("VOC presence labels do not exactly align with immutable score rows and class columns.")
    supports = labels.sum(axis=0, dtype=np.int64)
    if np.any(supports == 0):
        raise InputValidationError("VOC presence evaluation has a zero-support class and must fail closed.")
    ap = np.asarray([integral_average_precision(scores[:, index], labels[:, index]) for index in range(len(_VOC_CLASSES))], dtype=np.float64)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise InputValidationError("VOC presence evaluation output directory must be empty.")
    destination.mkdir(parents=True, exist_ok=True)
    anchor = cfg.get("repository_anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"code_commit", "protocol_sha256"} or anchor["protocol_sha256"] != protocol.get("sha256"):
        raise InputValidationError("VOC presence evaluation runner must supply the exact repository anchor.")
    metrics = {
        "format_version": 1, "metric": protocol["metric"], "image_count": len(image_ids), "class_order": list(_VOC_CLASSES),
        "per_class": [{"class_id": index + 1, "class_name": label, "support": int(supports[index]), "average_precision": float(ap[index])} for index, label in enumerate(_VOC_CLASSES)],
        "macro_map": float(ap.mean()),
    }
    metrics_path = destination / protocol["output"]["metrics_file"]
    with metrics_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "format_version": 1, "status": "completed", "scientific_evidence": False, "evaluation_evidence": True, "role": protocol["role"],
        "scope_disclosure": protocol["scope_disclosure"], "repository_anchor": anchor,
        "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "score_cache": {"manifest_sha256": sha256_file(score_manifest), "scores_sha256": protocol["score_cache"]["scores_sha256"], "image_ids_sha256": protocol["score_cache"]["image_ids_sha256"], "source_role": score_manifest_payload["role"]},
        "dataset": {"name": protocol["dataset"]["name"], "split": protocol["dataset"]["split"], "mask_root_relative": protocol["dataset"]["mask_root_relative"], "image_count": len(image_ids)},
        "target": protocol["target"], "constraints": protocol["constraints"],
        "outputs": {"metrics": {"path": metrics_path.name, "sha256": sha256_file(metrics_path)}},
    }
    with (destination / protocol["output"]["manifest_file"]).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
