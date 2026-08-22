"""Fail-closed core for CTP-v1.1 development-only hyperparameter selection.

This module deliberately contains no model invocation and no GT-aware feature or
prototype construction.  A future deployment runner may consume a manifest-bound
development cache, but can never discover or score registered test areas.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .io import InputValidationError, create_run_dir, sha256_file, write_json
from .loveda_partial_support import ctp_predictions, scc_scores
from .openai_clip_visual_anchor import _normalize
from .pixel_ovss import FusionCanvas, IGNORE_INDEX


REGISTERED_TEST_AREAS = (11, 15, 28, 30, 34)
DEVELOPMENT_AREAS = (1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37)
ALPHA_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
TAU_GRID = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
BASELINE = (0.5, 0.03)
_BANNED_PATH_MARKERS = ("test", "potsdam", "remoteclip", "segearth", "common_support", "baseline")


def canonical_grid() -> tuple[tuple[float, float], ...]:
    return tuple((alpha, tau) for alpha in ALPHA_GRID for tau in TAU_GRID)


def support_subsets(classes: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    if len(classes) != 5 or len(set(classes)) != len(classes):
        raise InputValidationError("CTP-v1.1 requires exactly five unique Vaihingen classes.")
    result = tuple(
        subset
        for k in (2, 3, 4)
        for subset in itertools.combinations(tuple(classes), k)
    )
    if len(result) != 25:
        raise InputValidationError("The registered partial-support enumeration must contain 25 subsets.")
    return result


def _as_int_areas(values: Iterable[Any], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{label} must be integer area IDs.") from exc
    if len(result) != len(set(result)):
        raise InputValidationError(f"{label} contains duplicate area IDs.")
    return result


def assert_exact_development_areas(values: Iterable[Any], label: str = "development areas") -> tuple[int, ...]:
    areas = _as_int_areas(values, label)
    overlap = sorted(set(areas) & set(REGISTERED_TEST_AREAS))
    if overlap:
        raise InputValidationError(f"{label} intersects registered test IDs: {overlap}")
    if areas != DEVELOPMENT_AREAS:
        raise InputValidationError(
            f"{label} must exactly equal the registered development IDs {list(DEVELOPMENT_AREAS)}."
        )
    return areas


def assert_no_registered_test_ids(values: Iterable[Any], label: str) -> None:
    areas = _as_int_areas(values, label)
    overlap = sorted(set(areas) & set(REGISTERED_TEST_AREAS))
    if overlap:
        raise InputValidationError(f"{label} intersects registered test IDs: {overlap}")


def assert_safe_source_path(value: str | Path | None, label: str, *, allow_null: bool = False) -> Path | None:
    if value is None:
        if allow_null:
            return None
        raise InputValidationError(f"{label} is required.")
    path = Path(value)
    text = path.as_posix().lower()
    if any(marker in text for marker in _BANNED_PATH_MARKERS):
        raise InputValidationError(f"{label} has a forbidden test/external source marker.")
    for area in REGISTERED_TEST_AREAS:
        if f"area{area}" in text or f"area_{area}" in text:
            raise InputValidationError(f"{label} names a registered test area.")
    return path


def load_protocol(path: str | Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        protocol = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read CTP-v1.1 tuning protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("CTP-v1.1 tuning protocol must be an object.")
    if tuple(protocol.get("registered_test_area_ids", ())) != REGISTERED_TEST_AREAS:
        raise InputValidationError("Protocol registered test IDs differ from the immutable registry.")
    assert_exact_development_areas(protocol.get("development_area_ids", ()), "Protocol development areas")
    allowed = protocol.get("allowed_parameters", {})
    if tuple(float(v) for v in allowed.get("alpha", ())) != ALPHA_GRID:
        raise InputValidationError("Protocol alpha grid is not the registered seven-value grid.")
    if tuple(float(v) for v in allowed.get("tau_conflict", ())) != TAU_GRID:
        raise InputValidationError("Protocol tau grid is not the registered seven-value grid.")
    if tuple(protocol.get("partial_support", {}).get("subset_sizes", ())) != (2, 3, 4):
        raise InputValidationError("Protocol partial-support subset sizes are not registered.")
    if protocol.get("partial_support", {}).get("expected_subset_count") != 25:
        raise InputValidationError("Protocol partial-support count is not 25.")
    protocol = dict(protocol)
    protocol["sha256"] = hashlib.sha256(raw).hexdigest()
    return protocol


def load_deployment_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read CTP-v1.1 deployment config.") from exc
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("CTP-v1.1 config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    required = {
        "protocol_file", "development_manifest", "feature_cache", "candidate_dir", "image_dir",
        "openai_clip_checkpoint", "label_dir", "output_root",
    }
    if not isinstance(paths, dict) or set(paths) != required:
        raise InputValidationError("CTP-v1.1 config paths do not match the fixed schema.")
    expected_protocol = (root / "configs" / "ctp_v1_1_tuning_protocol.json").resolve()
    protocol_value = Path(str(paths["protocol_file"]))
    protocol_path = (root / protocol_value).resolve() if not protocol_value.is_absolute() else protocol_value.resolve()
    if protocol_path != expected_protocol:
        raise InputValidationError("CTP-v1.1 must use the committed canonical protocol.")
    protocol = load_protocol(protocol_path)
    assert_exact_development_areas(cfg.get("development", {}).get("area_ids", ()), "Config development areas")
    output_value = Path(str(paths["output_root"]))
    output_path = (root / output_value).resolve() if not output_value.is_absolute() else output_value.resolve()
    allowed_output = (root / "outputs" / "ctp_tuning_v1_1").resolve()
    if output_path != allowed_output:
        raise InputValidationError("CTP-v1.1 output_root must be exactly outputs/ctp_tuning_v1_1.")
    for name, value in paths.items():
        if name in {"protocol_file", "output_root"}:
            continue
        if value is None:
            assert_safe_source_path(value, name, allow_null=True)
            continue
        raw = Path(str(value))
        resolved = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        assert_safe_source_path(resolved, name)
        paths[name] = str(resolved)
    paths["protocol_file"] = str(protocol_path)
    paths["output_root"] = str(output_path)
    return cfg, protocol


def validate_development_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise InputValidationError("Development manifest must be a mapping.")
    areas = assert_exact_development_areas(manifest.get("development_area_ids", ()), "Manifest development areas")
    if manifest.get("registered_test_area_ids") not in (None, list(REGISTERED_TEST_AREAS), tuple(REGISTERED_TEST_AREAS)):
        raise InputValidationError("Manifest test registry differs from the immutable registry.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise InputValidationError("Development manifest needs non-empty records.")
    seen: set[tuple[int, str, int]] = set()
    actual_areas: set[int] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise InputValidationError("Development manifest record is not an object.")
        try:
            area = int(row["area_id"])
            image_id = str(row["image_id"])
            candidate_index = int(row["candidate_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InputValidationError("Development manifest record has invalid key fields.") from exc
        if area not in DEVELOPMENT_AREAS:
            raise InputValidationError("Development manifest contains a non-development area.")
        if f"area{area}" not in image_id.lower().replace("_", ""):
            raise InputValidationError("Development manifest image_id does not agree with area_id.")
        key = (area, image_id, candidate_index)
        if candidate_index < 0 or key in seen:
            raise InputValidationError("Development manifest has duplicate/invalid candidate keys.")
        seen.add(key)
        actual_areas.add(area)
    if actual_areas != set(areas):
        raise InputValidationError("Development manifest does not cover every registered development area.")
    return {"development_area_ids": list(areas), "record_count": len(records)}


def load_development_manifest(path: str | Path) -> dict[str, Any]:
    safe = assert_safe_source_path(path, "development_manifest")
    assert safe is not None
    try:
        manifest = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read development manifest.") from exc
    validate_development_manifest(manifest)
    return manifest


def build_visual_prototypes(features: np.ndarray, sam3_source_labels: Sequence[str], classes: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Build weakly supervised visual prototypes; this API intentionally has no GT input."""
    vectors = _normalize(np.asarray(features, dtype=np.float32))
    names = tuple(str(name) for name in sam3_source_labels)
    if len(vectors) != len(names):
        raise InputValidationError("Feature rows and SAM3 source labels differ in length.")
    prototypes: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for class_name in classes:
        index = np.asarray([i for i, label in enumerate(names) if label == class_name], dtype=np.int64)
        if len(index) == 0:
            raise InputValidationError(f"No frozen SAM3 weak labels exist for class {class_name}.")
        prototypes.append(_normalize(vectors[index].mean(axis=0, keepdims=True))[0])
        counts[str(class_name)] = int(len(index))
    return np.asarray(prototypes, dtype=np.float32), counts


def anchored_prototypes(text_prototypes: np.ndarray, visual_prototypes: np.ndarray, alpha: float) -> np.ndarray:
    """The sole permitted alpha use: global scalar normalized text/visual anchoring."""
    alpha = float(alpha)
    if alpha not in ALPHA_GRID:
        raise InputValidationError("alpha is outside the registered grid.")
    text = _normalize(np.asarray(text_prototypes, dtype=np.float32))
    visual = _normalize(np.asarray(visual_prototypes, dtype=np.float32))
    if text.shape != visual.shape:
        raise InputValidationError("Text and visual prototype shapes differ.")
    return _normalize((1.0 - alpha) * text + alpha * visual)


def score_region_features(features: np.ndarray, text_prototypes: np.ndarray, visual_prototypes: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT-free score construction for one global alpha."""
    regions = _normalize(np.asarray(features, dtype=np.float32))
    text = _normalize(np.asarray(text_prototypes, dtype=np.float32))
    anchored = anchored_prototypes(text, visual_prototypes, alpha)
    if regions.shape[1] != text.shape[1]:
        raise InputValidationError("Feature dimension and prototype dimension differ.")
    text_scores = regions @ text.T
    anchored_scores = regions @ anchored.T
    text_top1 = np.argmax(text_scores, axis=1).astype(np.int64)
    return text_scores.astype(np.float32), anchored_scores.astype(np.float32), text_top1


def ctp_predictions_for_subset(text_scores: np.ndarray, anchored_scores: np.ndarray, text_top1: np.ndarray, supported_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scc = scc_scores(text_scores, anchored_scores, supported_mask)
    return ctp_predictions(text_top1, text_scores, scc, supported_mask), scc


def validate_tau(tau_conflict: float) -> float:
    tau = float(tau_conflict)
    if tau not in TAU_GRID:
        raise InputValidationError("tau_conflict is outside the registered grid.")
    return tau


def evaluate_class_predictions(pred: np.ndarray, gt: np.ndarray, classes: Sequence[str], supported_mask: np.ndarray) -> dict[str, Any]:
    """Development evaluation only; GT enters nowhere else in this module."""
    predicted = np.asarray(pred, dtype=np.int64).reshape(-1)
    truth = np.asarray(gt, dtype=np.int64).reshape(-1)
    if predicted.shape != truth.shape:
        raise InputValidationError("Prediction and development GT shapes differ.")
    n_classes = len(classes)
    if supported_mask.shape != (n_classes,):
        raise InputValidationError("Support mask does not match classes.")
    if np.any((truth < 0) | (truth >= n_classes)) or np.any((predicted < 0) | (predicted >= n_classes)):
        raise InputValidationError("Development evaluation expects class indices only.")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    per_iou: dict[str, float] = {}
    per_f1: dict[str, float] = {}
    for index, name in enumerate(classes):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        per_iou[str(name)] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_f1[str(name)] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    supported = [str(name) for name, keep in zip(classes, supported_mask) if keep]
    unsupported = [str(name) for name, keep in zip(classes, supported_mask) if not keep]
    s_iou = float(np.mean([per_iou[name] for name in supported]))
    u_iou = float(np.mean([per_iou[name] for name in unsupported]))
    s_f1 = float(np.mean([per_f1[name] for name in supported]))
    u_f1 = float(np.mean([per_f1[name] for name in unsupported]))
    return {
        "OA": float(np.mean(predicted == truth)),
        "MacroF1": float(np.mean(list(per_f1.values()))),
        "mIoU": float(np.mean(list(per_iou.values()))),
        "per_class_iou": per_iou,
        "S_F1": s_f1,
        "U_F1": u_f1,
        "H_F1": 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 else 0.0,
        "S_IoU": s_iou,
        "U_IoU": u_iou,
        "H_IoU": 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def annotate_feasibility(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    baseline_rows = [row for row in result if (float(row["alpha"]), float(row["tau_conflict"])) == BASELINE]
    if len(baseline_rows) != 1:
        raise InputValidationError("Exactly one untouched CTP-v1 baseline row is required.")
    baseline_h = float(baseline_rows[0]["partial_mean_H_IoU"])
    for row in result:
        alpha, tau = float(row["alpha"]), float(row["tau_conflict"])
        if (alpha, tau) not in canonical_grid():
            raise InputValidationError("A selection row is outside the registered 49-cell grid.")
        collapse = int(row["collapse_subset_count"])
        h_value = float(row["partial_mean_H_IoU"])
        row["feasible"] = bool(collapse == 0 and h_value >= baseline_h)
        row["selected"] = False
    if len({(float(row["alpha"]), float(row["tau_conflict"])) for row in result}) != len(canonical_grid()):
        raise InputValidationError("Selection requires each of the 49 registered configurations exactly once.")
    return result


def select_configuration(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = annotate_feasibility(rows)
    feasible = [row for row in annotated if row["feasible"]]
    if not feasible:
        raise InputValidationError("No CTP-v1.1 configuration satisfies the partial-support safety constraint.")
    best_miou = max(float(row["full_mIoU"]) for row in feasible)
    contenders = [row for row in feasible if best_miou - float(row["full_mIoU"]) < 0.001]
    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
        alpha, tau = float(row["alpha"]), float(row["tau_conflict"])
        distance = math.hypot(alpha - BASELINE[0], tau - BASELINE[1])
        return (-float(row["partial_mean_H_IoU"]), -float(row["partial_mean_U_IoU"]), float(row["abstention_ratio"]), distance, alpha, tau)
    selected = min(contenders, key=key)
    for row in annotated:
        row["selected"] = (float(row["alpha"]), float(row["tau_conflict"])) == (float(selected["alpha"]), float(selected["tau_conflict"]))
    return {"selected": dict(selected), "grid": annotated, "baseline": dict(next(row for row in annotated if (float(row["alpha"]), float(row["tau_conflict"])) == BASELINE))}


def write_grid_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    columns = [
        "alpha", "tau_conflict", "full_OA", "full_MacroF1", "full_mIoU", "abstention_ratio",
        "partial_mean_S_IoU", "partial_mean_U_IoU", "partial_mean_H_IoU", "partial_min_U_IoU",
        "collapse_subset_count", "feasible", "selected",
    ]
    destination = Path(path)
    if destination.exists():
        raise InputValidationError("CTP-v1.1 grid CSV already exists; overwrite is forbidden.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def preflight_status(cfg: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Read metadata only; never invoke model/GPU or read GT."""
    paths = cfg["paths"]
    expected_checkpoint = str(protocol["openai_clip_binding"]["checkpoint_sha256"])
    status: dict[str, Any] = {
        "phase": "preflight",
        "model_or_gpu_invoked": False,
        "gt_read": False,
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "protocol_sha256": protocol["sha256"],
        "items": {},
    }
    manifest_path = assert_safe_source_path(paths["development_manifest"], "development_manifest", allow_null=True)
    if manifest_path and manifest_path.is_file():
        try:
            status["items"]["development_manifest"] = {"ready": True, **validate_development_manifest(json.loads(manifest_path.read_text(encoding="utf-8"))), "sha256": sha256_file(manifest_path)}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, InputValidationError) as exc:
            status["items"]["development_manifest"] = {"ready": False, "reason": str(exc)}
    else:
        status["items"]["development_manifest"] = {"ready": False, "reason": "missing"}
    for key in ("feature_cache", "candidate_dir", "image_dir"):
        path = assert_safe_source_path(paths[key], key, allow_null=True)
        status["items"][key] = {"ready": bool(path and path.exists()), "path": str(path) if path else None}
    checkpoint = assert_safe_source_path(paths["openai_clip_checkpoint"], "openai_clip_checkpoint", allow_null=True)
    checkpoint_ready = bool(checkpoint and checkpoint.is_file())
    checkpoint_status: dict[str, Any] = {"ready": checkpoint_ready, "path": str(checkpoint) if checkpoint else None, "expected_sha256": expected_checkpoint}
    if checkpoint_ready:
        actual = sha256_file(checkpoint)
        checkpoint_status.update({"actual_sha256": actual, "matches_expected": actual == expected_checkpoint, "ready": actual == expected_checkpoint})
    status["items"]["openai_clip_checkpoint"] = checkpoint_status
    label = paths["label_dir"]
    status["items"]["label_dir"] = {"configured": label is not None, "allowed": "evaluate_only"}
    status["ready_for_cache"] = all(status["items"][key]["ready"] for key in ("candidate_dir", "image_dir", "openai_clip_checkpoint"))
    status["ready_for_grid"] = status["ready_for_cache"] and status["items"]["development_manifest"]["ready"] and status["items"]["feature_cache"]["ready"]
    return status


def _candidate_paths(candidate_dir: Path, image_id: str) -> tuple[Path, Path]:
    return candidate_dir / f"{image_id}.npz", candidate_dir / f"{image_id}.json"


def load_frozen_candidates(candidate_dir: str | Path, image_id: str) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Load the v1 SAM3 cache format read-only without importing the first project.

    The loader is deliberately format-specific, validates both JSON/NPZ members,
    and never generates candidates.  It has no GT argument or side effect.
    """
    root = assert_safe_source_path(candidate_dir, "candidate_dir")
    assert root is not None
    try:
        area = int(image_id.rsplit("area", 1)[1])
    except (IndexError, ValueError) as exc:
        raise InputValidationError("Candidate image_id has no valid Vaihingen area suffix.") from exc
    assert_no_registered_test_ids([area], "Candidate image ID")
    if area not in DEVELOPMENT_AREAS:
        raise InputValidationError("Candidate image ID is outside the fixed development pool.")
    data_path, metadata_path = _candidate_paths(root, image_id)
    if not data_path.is_file() or not metadata_path.is_file():
        raise InputValidationError(f"Missing frozen candidate pair for {image_id}.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(data_path, allow_pickle=False) as archive:
            if int(archive["format_version"][0]) != 1 or int(metadata.get("format_version", -1)) != 1:
                raise InputValidationError("Unsupported frozen candidate cache format.")
            image_shape = tuple(int(v) for v in archive["image_shape"])
            if len(image_shape) != 2 or min(image_shape) <= 0:
                raise InputValidationError("Frozen candidate image shape is invalid.")
            packed = archive["packed_masks"]
            offsets = archive["offsets"]
            shapes = archive["shapes"]
            origins = archive["origins"]
            scores = archive["scores"]
            class_ids = archive["class_ids"]
            prompt_ids = archive["prompt_ids"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Cannot load frozen candidate cache for {image_id}.") from exc
    if str(metadata.get("image_id")) != image_id or int(metadata.get("candidate_count", -1)) != len(scores):
        raise InputValidationError("Frozen candidate JSON/NPZ identity mismatch.")
    prompts = {int(item["id"]): item for item in metadata.get("prompts", []) if isinstance(item, Mapping)}
    candidates: list[dict[str, Any]] = []
    height_limit, width_limit = image_shape
    for index in range(len(scores)):
        try:
            height, width = (int(v) for v in shapes[index])
            x0, y0 = (int(v) for v in origins[index])
            start, end = int(offsets[index]), int(offsets[index + 1])
            prompt = prompts[int(prompt_ids[index])]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise InputValidationError("Frozen candidate arrays are inconsistent.") from exc
        if height <= 0 or width <= 0 or x0 < 0 or y0 < 0 or x0 + width > width_limit or y0 + height > height_limit:
            raise InputValidationError("Frozen candidate bounds exceed its image geometry.")
        mask = np.unpackbits(packed[start:end], bitorder="little", count=height * width).reshape(height, width).astype(bool, copy=False)
        candidates.append({
            "candidate_index": index,
            "class_id": int(class_ids[index]),
            "sam3_source_label": str(prompt["class_name"]),
            "score": float(scores[index]),
            "mask": mask,
            "x0": x0,
            "y0": y0,
        })
    return (height_limit, width_limit), candidates


def _read_rgb_image(path: Path) -> np.ndarray:
    import tifffile
    array = tifffile.imread(path)
    if array.ndim != 3 or array.shape[2] < 3:
        raise InputValidationError(f"Expected RGB image at {path}.")
    rgb = array[:, :, :3]
    if rgb.dtype != np.uint8:
        lo, hi = float(np.percentile(rgb, 1)), float(np.percentile(rgb, 99))
        rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _crop_view(image_rgb: np.ndarray, candidate: Mapping[str, Any], crop: Mapping[str, Any]) -> np.ndarray:
    """Frozen masked crop implementation; only RGB plus frozen candidate geometry."""
    mask = np.asarray(candidate["mask"], dtype=bool)
    x0, y0 = int(candidate["x0"]), int(candidate["y0"])
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise InputValidationError("Frozen candidate contains an empty mask.")
    left, top = x0 + int(xs.min()), y0 + int(ys.min())
    right, bottom = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
    target = max(right - left, bottom - top)
    size = max(int(crop["min_crop_size"]), int(np.ceil(target * (1 + 2 * float(crop["context_ratio"])))) )
    image_height, image_width = image_rgb.shape[:2]
    size = min(size, image_height, image_width)
    crop_left = max(0, min(int(np.floor((left + right - size) / 2)), image_width - size))
    crop_top = max(0, min(int(np.floor((top + bottom - size) / 2)), image_height - size))
    crop_rgb = np.ascontiguousarray(image_rgb[crop_top:crop_top + size, crop_left:crop_left + size])
    crop_mask = np.zeros(crop_rgb.shape[:2], dtype=bool)
    il, it = max(crop_left, x0), max(crop_top, y0)
    ir, ib = min(crop_left + size, x0 + mask.shape[1]), min(crop_top + size, y0 + mask.shape[0])
    if il < ir and it < ib:
        crop_mask[it - crop_top:ib - crop_top, il - crop_left:ir - crop_left] = mask[it - y0:ib - y0, il - x0:ir - x0]
    masked = crop_rgb.astype(np.float32)
    masked[~crop_mask] *= float(crop["background_retain"])
    return np.rint(masked).clip(0, 255).astype(np.uint8)


def _load_openai_clip(protocol: Mapping[str, Any], checkpoint: Path, device: str):
    """Load only the protocol-bound OpenAI CLIP checkpoint."""
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise InputValidationError("OpenCLIP and torch are required for development feature caching.") from exc
    binding = protocol["openai_clip_binding"]
    if getattr(open_clip, "__version__", None) != binding["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the frozen CTP-v1.1 binding.")
    if sha256_file(checkpoint) != binding["checkpoint_sha256"]:
        raise InputValidationError("OpenAI CLIP checkpoint hash differs from the frozen binding.")
    model, _, preprocess = open_clip.create_model_and_transforms(binding["architecture"], pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.eval().to(device)
    return model, preprocess, open_clip, torch


def build_development_feature_cache(
    image_dir: str | Path,
    candidate_dir: str | Path,
    checkpoint: str | Path,
    protocol: Mapping[str, Any],
    output_root: str | Path,
    *,
    batch_size: int = 32,
) -> Path:
    """Create one unique GT-free, float32 development cache from frozen RGB/candidates."""
    image_root = assert_safe_source_path(image_dir, "image_dir")
    candidate_root = assert_safe_source_path(candidate_dir, "candidate_dir")
    checkpoint_path = assert_safe_source_path(checkpoint, "openai_clip_checkpoint")
    assert image_root is not None and candidate_root is not None and checkpoint_path is not None
    if batch_size <= 0:
        raise InputValidationError("Feature cache batch size must be positive.")
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, open_clip, torch = _load_openai_clip(protocol, checkpoint_path, device)
    run_dir = create_run_dir(output_root)
    rows: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    row_index = 0
    source_images: dict[str, str] = {}
    source_candidates: dict[str, dict[str, str]] = {}
    for area in DEVELOPMENT_AREAS:
        image_id = f"vaih_area{area}"
        image_path = image_root / f"{image_id}_RGB.tif"
        if not image_path.is_file():
            raise InputValidationError(f"Missing development RGB image: {image_path}")
        shape, candidates = load_frozen_candidates(candidate_root, image_id)
        image = _read_rgb_image(image_path)
        if tuple(image.shape[:2]) != shape:
            raise InputValidationError("Frozen candidate image geometry differs from RGB geometry.")
        features = np.empty((len(candidates), int(protocol["openai_clip_binding"]["feature_dimension"])), dtype=np.float32)
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            tensors = torch.stack([preprocess(Image.fromarray(_crop_view(image, candidate, protocol["openai_clip_binding"]["crop"]))).to(device) for candidate in chunk])
            with torch.inference_mode():
                encoded = model.encode_image(tensors).float().cpu().numpy()
            features[start:start + len(chunk)] = _normalize(encoded)
        feature_blocks.append(features)
        source_images[image_id] = sha256_file(image_path)
        data_path, metadata_path = _candidate_paths(candidate_root, image_id)
        source_candidates[image_id] = {"npz_sha256": sha256_file(data_path), "json_sha256": sha256_file(metadata_path)}
        for candidate in candidates:
            rows.append({
                "row_index": row_index,
                "area_id": area,
                "image_id": image_id,
                "candidate_index": int(candidate["candidate_index"]),
                "sam3_source_label": str(candidate["sam3_source_label"]),
            })
            row_index += 1
    features_all = np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
    classes = tuple(protocol["classes"])
    templates = tuple(protocol["openai_clip_binding"]["group_a_templates"])
    texts = [template.format(**{"class": class_name}) for class_name in classes for template in templates]
    tokenizer = open_clip.get_tokenizer(protocol["openai_clip_binding"]["architecture"])
    tokens = tokenizer(texts)
    token_hash = hashlib.sha256(tokens.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    with torch.inference_mode():
        encoded_text = model.encode_text(tokens.to(device)).float().cpu().numpy()
    text_prototypes = _normalize(_normalize(encoded_text).reshape(len(classes), len(templates), -1).mean(axis=1))
    cache_path = run_dir / "features_float32.npz"
    with cache_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            features=features_all,
            text_prototypes=text_prototypes.astype(np.float32),
            row_index=np.asarray([row["row_index"] for row in rows], dtype=np.int64),
            area_id=np.asarray([row["area_id"] for row in rows], dtype=np.int16),
            image_id=np.asarray([row["image_id"] for row in rows], dtype="U32"),
            candidate_index=np.asarray([row["candidate_index"] for row in rows], dtype=np.int32),
            sam3_source_label=np.asarray([row["sam3_source_label"] for row in rows], dtype="U64"),
        )
    manifest = {
        "format_version": 1,
        "phase": "cache",
        "status": "completed",
        "gt_read": False,
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "records": rows,
        "feature_cache": {"path": cache_path.name, "sha256": sha256_file(cache_path), "dtype": "float32", "shape": list(features_all.shape)},
        "text_token_sha256": token_hash,
        "model": dict(protocol["openai_clip_binding"]),
        "sources": {"images": source_images, "candidates": source_candidates},
        "protocol_sha256": protocol["sha256"],
    }
    write_json(run_dir / "cache_manifest.json", manifest)
    return run_dir


def load_development_feature_cache(cache_path: str | Path, cache_manifest_path: str | Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    cache = assert_safe_source_path(cache_path, "feature_cache")
    manifest_path = assert_safe_source_path(cache_manifest_path, "development_manifest")
    assert cache is not None and manifest_path is not None
    manifest = load_development_manifest(manifest_path)
    if manifest.get("phase") != "cache" or manifest.get("status") != "completed" or manifest.get("gt_read") is not False:
        raise InputValidationError("Feature cache manifest is not a completed GT-free cache run.")
    if manifest.get("protocol_sha256") != protocol["sha256"]:
        raise InputValidationError("Feature cache protocol hash differs from CTP-v1.1 protocol.")
    entry = manifest.get("feature_cache", {})
    if entry.get("sha256") != sha256_file(cache):
        raise InputValidationError("Feature cache hash differs from its manifest.")
    try:
        with np.load(cache, allow_pickle=False) as archive:
            features = archive["features"]
            text = archive["text_prototypes"]
            rows = {name: archive[name] for name in ("row_index", "area_id", "image_id", "candidate_index", "sam3_source_label")}
    except (OSError, KeyError, ValueError) as exc:
        raise InputValidationError("Cannot load development feature cache.") from exc
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != protocol["openai_clip_binding"]["feature_dimension"]:
        raise InputValidationError("Development feature cache must be float32 [N,512].")
    if not np.isfinite(features).all() or text.shape != (len(protocol["classes"]), features.shape[1]):
        raise InputValidationError("Development feature cache contains invalid feature/text data.")
    if len(features) != len(manifest["records"]):
        raise InputValidationError("Development feature cache row count differs from manifest records.")
    loaded_records = []
    for index, original in enumerate(manifest["records"]):
        if int(rows["row_index"][index]) != int(original["row_index"]) or int(rows["area_id"][index]) != int(original["area_id"]):
            raise InputValidationError("Feature cache row mapping differs from manifest.")
        loaded_records.append(dict(original))
    return {"features": features, "text_prototypes": text.astype(np.float32), "records": loaded_records, "manifest": manifest}


def _ordered_dev_candidates(candidate_dir: str | Path, records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, tuple[int, int]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_image: dict[str, list[dict[str, Any]]] = {f"vaih_area{area}": [] for area in DEVELOPMENT_AREAS}
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in by_image:
            raise InputValidationError("Feature records contain an unregistered image ID.")
        by_image[image_id].append(dict(row))
    shapes: dict[str, tuple[int, int]] = {}
    candidates_by_image: dict[str, list[dict[str, Any]]] = {}
    ordered_rows: list[dict[str, Any]] = []
    for area in DEVELOPMENT_AREAS:
        image_id = f"vaih_area{area}"
        rows = sorted(by_image[image_id], key=lambda row: int(row["candidate_index"]))
        shape, candidates = load_frozen_candidates(candidate_dir, image_id)
        if len(rows) != len(candidates) or [int(row["candidate_index"]) for row in rows] != list(range(len(candidates))):
            raise InputValidationError("Feature cache/candidate cache row alignment is incomplete.")
        for row, candidate in zip(rows, candidates):
            if str(row.get("sam3_source_label")) != str(candidate["sam3_source_label"]):
                raise InputValidationError("Feature cache SAM3 source label differs from frozen candidate cache.")
        shapes[image_id], candidates_by_image[image_id] = shape, candidates
        ordered_rows.extend(rows)
    if [int(row["row_index"]) for row in ordered_rows] != list(range(len(ordered_rows))):
        raise InputValidationError("Feature cache rows are not in a stable complete order.")
    return shapes, candidates_by_image, ordered_rows


def _support_mask(classes: Sequence[str], subset: Sequence[str] | None) -> np.ndarray:
    if subset is None:
        return np.ones(len(classes), dtype=bool)
    selected = set(subset)
    return np.asarray([name in selected for name in classes], dtype=bool)


def _subset_key(index: int | None) -> str:
    return "full" if index is None else f"subset_{index:02d}"


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def source_code_identity() -> dict[str, str]:
    """Bind a run to this standalone dev-only module and its CLI."""
    project_root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("Cannot determine CTP-v1.1 source repository identity.") from exc
    module = Path(__file__).resolve()
    runner = project_root / "scripts" / "run_ctp_v11_dev_tuning.py"
    if not runner.is_file():
        raise InputValidationError("CTP-v1.1 runner source file is missing.")
    return {"repo_head": head, "module_sha256": sha256_file(module), "runner_sha256": sha256_file(runner)}


def full_support_per_class_rows(rows: Iterable[Mapping[str, Any]], classes: Sequence[str]) -> list[dict[str, Any]]:
    """Explode the 49 full-support per-class IoU mappings into a complete CSV table."""
    result: list[dict[str, Any]] = []
    grid = list(rows)
    if len(grid) != len(canonical_grid()):
        raise InputValidationError("Per-class IoU export requires all 49 grid configurations.")
    for row in grid:
        per_class = row.get("per_class_iou")
        if not isinstance(per_class, Mapping) or set(per_class) != set(classes):
            raise InputValidationError("Full-support row lacks the exact registered per-class IoU mapping.")
        for class_name in classes:
            result.append({"alpha": float(row["alpha"]), "tau_conflict": float(row["tau_conflict"]), "class": str(class_name), "IoU": float(per_class[class_name])})
    if len(result) != len(canonical_grid()) * len(classes):
        raise InputValidationError("Per-class IoU row count is incomplete.")
    return result


def required_output_hashes(run_dir: str | Path, names: Sequence[str]) -> dict[str, str]:
    root = Path(run_dir)
    result: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise InputValidationError(f"Required CTP-v1.1 output is missing: {name}")
        result[str(name)] = sha256_file(path)
    return result


def write_prediction_archives(
    features: np.ndarray,
    text_prototypes: np.ndarray,
    visual_prototypes: np.ndarray,
    classes: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist all GT-free region predictions before any evaluation can begin."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    subsets = support_subsets(classes)
    archives: dict[str, dict[str, Any]] = {}
    for alpha in ALPHA_GRID:
        text_scores, anchored_scores, text_top1 = score_region_features(features, text_prototypes, visual_prototypes, alpha)
        payload: dict[str, np.ndarray] = {"text_scores": text_scores, "text_top1": text_top1}
        for subset_index, subset in [(None, None), *list(enumerate(subsets))]:
            mask = _support_mask(classes, subset)
            prediction, scc = ctp_predictions_for_subset(text_scores, anchored_scores, text_top1, mask)
            key = _subset_key(subset_index)
            payload[f"{key}_prediction"] = prediction.astype(np.int16)
            payload[f"{key}_score"] = scc[np.arange(len(prediction)), prediction].astype(np.float32)
        path = root / f"alpha_{alpha:.3f}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        archives[f"{alpha:.3f}"] = {"path": str(path.relative_to(root.parent)), "sha256": sha256_file(path)}
    return {"directory": root.name, "archives": archives, "subset_count": len(subsets) + 1}


def _assemble_map(
    shape: tuple[int, int],
    candidates: Sequence[Mapping[str, Any]],
    prediction: np.ndarray,
    score: np.ndarray,
    tau_conflict: float,
) -> tuple[np.ndarray, dict[str, int]]:
    tau = validate_tau(tau_conflict)
    if len(candidates) != len(prediction) or len(prediction) != len(score):
        raise InputValidationError("Region prediction arrays do not align with frozen candidates.")
    canvas = FusionCanvas(height=int(shape[0]), width=int(shape[1]), conflict_margin=tau)
    covered = np.zeros(shape, dtype=bool)
    for candidate, class_id, value in zip(candidates, prediction, score):
        mask = np.asarray(candidate["mask"], dtype=bool)
        x0, y0 = int(candidate["x0"]), int(candidate["y0"])
        covered[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] |= mask
        canvas.add_mask(mask, int(class_id), float(value), x0, y0)
    label_map = canvas.result()
    ignored = label_map == IGNORE_INDEX
    return label_map, {
        "pixels_total": int(label_map.size),
        "pixels_assigned": int((~ignored).sum()),
        "pixels_conflict_ignored": int((ignored & covered).sum()),
        "pixels_uncovered": int((ignored & ~covered).sum()),
    }


def _read_development_gt(label_dir: str | Path, classes: Sequence[str]) -> dict[str, np.ndarray]:
    """The sole GT reader, called only after a prediction manifest is sealed."""
    root = assert_safe_source_path(label_dir, "label_dir")
    assert root is not None
    colors = {
        "impervious_surface": (255, 255, 255), "building": (0, 0, 255),
        "low_vegetation": (0, 255, 255), "tree": (0, 255, 0), "car": (255, 255, 0),
    }
    if tuple(colors) != tuple(classes):
        raise InputValidationError("Development GT mapping differs from the registered five-class protocol.")
    result: dict[str, np.ndarray] = {}
    for area in DEVELOPMENT_AREAS:
        image_id = f"vaih_area{area}"
        path = root / f"{image_id}_label.tif"
        if not path.is_file():
            raise InputValidationError(f"Missing development GT label: {path}")
        rgb = _read_rgb_image(path)
        labels = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for index, name in enumerate(classes):
            labels[np.all(rgb == np.asarray(colors[name], dtype=np.uint8), axis=-1)] = index
        result[image_id] = labels
    return result


def _evaluate_pixel_maps(predicted: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray], classes: Sequence[str], supported: np.ndarray) -> dict[str, Any]:
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid_pixels = 0
    ignored_predictions = 0
    for image_id in sorted(predicted):
        pred = predicted[image_id]
        gt = truth[image_id]
        if pred.shape != gt.shape:
            raise InputValidationError("Development prediction/GT geometry differs.")
        valid = (gt != IGNORE_INDEX) & (pred != IGNORE_INDEX)
        valid_pixels += int(valid.sum())
        ignored_predictions += int(((gt != IGNORE_INDEX) & (pred == IGNORE_INDEX)).sum())
        np.add.at(matrix, (gt[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)
    per_iou: dict[str, float] = {}
    per_f1: dict[str, float] = {}
    for index, name in enumerate(classes):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    names_s = [name for name, keep in zip(classes, supported) if keep]
    names_u = [name for name, keep in zip(classes, supported) if not keep]
    s_iou, u_iou = float(np.mean([per_iou[name] for name in names_s])), float(np.mean([per_iou[name] for name in names_u]))
    s_f1, u_f1 = float(np.mean([per_f1[name] for name in names_s])), float(np.mean([per_f1[name] for name in names_u]))
    return {
        "OA": float(np.trace(matrix) / valid_pixels) if valid_pixels else 0.0,
        "MacroF1": float(np.mean(list(per_f1.values()))), "mIoU": float(np.mean(list(per_iou.values()))),
        "per_class_iou": per_iou, "S_F1": s_f1, "U_F1": u_f1,
        "H_F1": 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 else 0.0,
        "S_IoU": s_iou, "U_IoU": u_iou,
        "H_IoU": 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou else 0.0,
        "valid_pixels": valid_pixels, "abstained_valid_pixels": ignored_predictions, "confusion_matrix": matrix.tolist(),
    }


def _load_prediction_payload(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise InputValidationError("GT-free prediction archive changed before development evaluation.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise InputValidationError("Cannot load sealed GT-free prediction archive.") from exc


def _heatmap(rows: Sequence[Mapping[str, Any]], field: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    values = np.asarray([[float(next(row[field] for row in rows if float(row["alpha"]) == alpha and float(row["tau_conflict"]) == tau)) for tau in TAU_GRID] for alpha in ALPHA_GRID])
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(values, aspect="auto", origin="lower")
    axis.set_xticks(range(len(TAU_GRID)), [f"{tau:.3f}" for tau in TAU_GRID])
    axis.set_yticks(range(len(ALPHA_GRID)), [f"{alpha:.1f}" for alpha in ALPHA_GRID])
    axis.set_xlabel("tau_conflict")
    axis.set_ylabel("alpha")
    axis.set_title(field)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _selection_stability(selected: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alpha, tau = float(selected["alpha"]), float(selected["tau_conflict"])
    neighbors = [
        dict(row) for row in rows
        if (abs(float(row["alpha"]) - alpha) in (0.0, 0.1) and abs(float(row["tau_conflict"]) - tau) in (0.0, 0.005))
        and not (float(row["alpha"]) == alpha and float(row["tau_conflict"]) == tau)
    ]
    if not neighbors:
        return {"classification": "sharp_optimum", "neighbors": []}
    selected_value = float(selected["full_mIoU"])
    stable = all(abs(selected_value - float(row["full_mIoU"])) < 0.001 for row in neighbors)
    return {"classification": "stable_optimum" if stable else "sharp_optimum", "neighbors": neighbors}


def run_development_grid(
    cache_path: str | Path,
    cache_manifest_path: str | Path,
    candidate_dir: str | Path,
    label_dir: str | Path,
    protocol: Mapping[str, Any],
    output_root: str | Path,
) -> Path:
    """Execute the sealed dev-only 49x25 grid; never accepts test inputs.

    It first creates and hashes region-level prediction archives for every alpha
    and every required support subset.  Only after the prediction manifest is
    atomically written does it read development GT and render/evaluate the fixed
    tau grid.
    """
    bundle = load_development_feature_cache(cache_path, cache_manifest_path, protocol)
    classes = tuple(protocol["classes"])
    shapes, candidates_by_image, ordered_rows = _ordered_dev_candidates(candidate_dir, bundle["records"])
    if ordered_rows != bundle["records"]:
        raise InputValidationError("Feature cache record order differs from frozen candidate order.")
    visual_prototypes, prototype_counts = build_visual_prototypes(
        bundle["features"], [str(row["sam3_source_label"]) for row in bundle["records"]], classes
    )
    run_dir = create_run_dir(output_root)
    source_identity = source_code_identity()
    cache_binding = {
        "feature_cache_sha256": sha256_file(Path(cache_path)),
        "feature_cache_manifest_sha256": sha256_file(Path(cache_manifest_path)),
        "text_prototypes_sha256": _sha256_array(bundle["text_prototypes"]),
        "visual_prototypes_sha256": _sha256_array(visual_prototypes),
        "visual_prototype_counts": prototype_counts,
    }
    prediction_info = write_prediction_archives(
        bundle["features"], bundle["text_prototypes"], visual_prototypes, classes, run_dir / "region_predictions"
    )
    prediction_manifest = {
        "format_version": 1,
        "phase": "predict",
        "status": "completed",
        "gt_read": False,
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "feature_cache": {"path": str(cache_path), "sha256": cache_binding["feature_cache_sha256"]},
        "feature_cache_manifest": {"path": str(cache_manifest_path), "sha256": cache_binding["feature_cache_manifest_sha256"]},
        "candidate_dir": str(candidate_dir),
        "candidate_row_count": len(ordered_rows),
        "cache_prototype_text_binding": cache_binding,
        "source_code": source_identity,
        "prediction_archives": prediction_info,
        "configuration_plan": [{"alpha": alpha, "tau_conflict": tau} for alpha, tau in canonical_grid()],
        "partial_support_subsets": [list(subset) for subset in support_subsets(classes)],
        "protocol_sha256": protocol["sha256"],
    }
    prediction_manifest_path = run_dir / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)

    # From this point on evaluation can read only development GT.  All predictions
    # and their hashes are already bound above.
    gt_by_image = _read_development_gt(label_dir, classes)
    subsets = support_subsets(classes)
    full_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    archive_root = run_dir / prediction_info["directory"]
    for alpha in ALPHA_GRID:
        archive_entry = prediction_info["archives"][f"{alpha:.3f}"]
        archive_path = run_dir / archive_entry["path"]
        payload = _load_prediction_payload(archive_path, archive_entry["sha256"])
        for tau in TAU_GRID:
            full_prediction = payload["full_prediction"].astype(np.int64)
            full_score = payload["full_score"].astype(np.float32)
            full_maps: dict[str, np.ndarray] = {}
            account = {"pixels_total": 0, "pixels_assigned": 0, "pixels_conflict_ignored": 0, "pixels_uncovered": 0}
            offset = 0
            for area in DEVELOPMENT_AREAS:
                image_id = f"vaih_area{area}"
                count = len(candidates_by_image[image_id])
                label_map, stats = _assemble_map(shapes[image_id], candidates_by_image[image_id], full_prediction[offset:offset + count], full_score[offset:offset + count], tau)
                full_maps[image_id] = label_map
                for key in account:
                    account[key] += stats[key]
                offset += count
            full_metric = _evaluate_pixel_maps(full_maps, gt_by_image, classes, np.ones(len(classes), dtype=bool))
            full_rows.append({
                "alpha": alpha, "tau_conflict": tau, "full_OA": full_metric["OA"], "full_MacroF1": full_metric["MacroF1"],
                "full_mIoU": full_metric["mIoU"], "abstention_ratio": account["pixels_conflict_ignored"] / account["pixels_total"],
                "partial_mean_S_IoU": 0.0, "partial_mean_U_IoU": 0.0, "partial_mean_H_IoU": 0.0,
                "partial_min_U_IoU": 0.0, "collapse_subset_count": 0, "per_class_iou": full_metric["per_class_iou"],
            })
            accounting_rows.append({"alpha": alpha, "tau_conflict": tau, **account, "full_valid_pixels": full_metric["valid_pixels"], "full_abstained_valid_pixels": full_metric["abstained_valid_pixels"]})
            per_subset_metrics: list[dict[str, Any]] = []
            for subset_index, subset in enumerate(subsets):
                key = _subset_key(subset_index)
                prediction = payload[f"{key}_prediction"].astype(np.int64)
                score = payload[f"{key}_score"].astype(np.float32)
                maps: dict[str, np.ndarray] = {}
                offset = 0
                for area in DEVELOPMENT_AREAS:
                    image_id = f"vaih_area{area}"
                    count = len(candidates_by_image[image_id])
                    maps[image_id], _ = _assemble_map(shapes[image_id], candidates_by_image[image_id], prediction[offset:offset + count], score[offset:offset + count], tau)
                    offset += count
                mask = _support_mask(classes, subset)
                metric = _evaluate_pixel_maps(maps, gt_by_image, classes, mask)
                per_subset_metrics.append(metric)
                partial_rows.append({
                    "alpha": alpha, "tau_conflict": tau, "subset_index": subset_index, "k": len(subset),
                    "supported": "|".join(subset), "unsupported": "|".join(name for name in classes if name not in subset),
                    **{key: metric[key] for key in ("OA", "MacroF1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU", "valid_pixels", "abstained_valid_pixels")},
                })
            row = full_rows[-1]
            row.update({
                "partial_mean_S_IoU": float(np.mean([metric["S_IoU"] for metric in per_subset_metrics])),
                "partial_mean_U_IoU": float(np.mean([metric["U_IoU"] for metric in per_subset_metrics])),
                "partial_mean_H_IoU": float(np.mean([metric["H_IoU"] for metric in per_subset_metrics])),
                "partial_min_U_IoU": float(min(metric["U_IoU"] for metric in per_subset_metrics)),
                "collapse_subset_count": int(sum(metric["U_IoU"] == 0.0 for metric in per_subset_metrics)),
            })
    selected = select_configuration(full_rows)
    write_grid_csv(run_dir / "grid_search_full.csv", selected["grid"])
    per_class_rows = full_support_per_class_rows(selected["grid"], classes)
    partial_columns = ["alpha", "tau_conflict", "subset_index", "k", "supported", "unsupported", "OA", "MacroF1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU", "valid_pixels", "abstained_valid_pixels"]
    accounting_columns = ["alpha", "tau_conflict", "pixels_total", "pixels_assigned", "pixels_conflict_ignored", "pixels_uncovered", "full_valid_pixels", "full_abstained_valid_pixels"]
    per_class_columns = ["alpha", "tau_conflict", "class", "IoU"]
    for name, rows, columns in (("partial_all_subsets.csv", partial_rows, partial_columns), ("full_support_accounting.csv", accounting_rows, accounting_columns), ("full_support_per_class.csv", per_class_rows, per_class_columns)):
        with (run_dir / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    for field, filename in (("full_mIoU", "heatmap_full_miou.png"), ("partial_mean_H_IoU", "heatmap_mean_h_iou.png"), ("partial_mean_U_IoU", "heatmap_mean_u_iou.png"), ("abstention_ratio", "heatmap_abstention_ratio.png")):
        _heatmap(selected["grid"], field, run_dir / filename)
    stability = _selection_stability(selected["selected"], selected["grid"])
    output_hashes = required_output_hashes(run_dir, ("grid_search_full.csv", "partial_all_subsets.csv", "full_support_accounting.csv", "full_support_per_class.csv"))
    candidate_config = {
        "name": "CTP-v1.1-tuned-candidate", "status": "development_selected_pending_final_test",
        "alpha": selected["selected"]["alpha"], "tau_conflict": selected["selected"]["tau_conflict"],
        "selection_rule": protocol["selection"], "development_area_ids": list(DEVELOPMENT_AREAS),
        "protocol_sha256": protocol["sha256"], "artifacts": output_hashes, "cache_prototype_text_binding": cache_binding,
        "source_code": source_identity, "stability": stability,
    }
    write_json(run_dir / "ctp_v1_1_tuned_candidate.json", candidate_config)
    final_manifest = {
        "format_version": 1, "phase": "evaluate", "status": "completed", "development_only": True,
        "registered_test_evaluation": False, "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "selected": candidate_config, "cache_prototype_text_binding": cache_binding, "source_code": source_identity,
        "outputs": required_output_hashes(run_dir, (*output_hashes.keys(), "ctp_v1_1_tuned_candidate.json")),
    }
    write_json(run_dir / "run_manifest.json", final_manifest)
    return run_dir


__all__ = [
    "ALPHA_GRID", "BASELINE", "DEVELOPMENT_AREAS", "REGISTERED_TEST_AREAS", "TAU_GRID",
    "annotate_feasibility", "anchored_prototypes", "assert_exact_development_areas",
    "assert_no_registered_test_ids", "assert_safe_source_path", "build_visual_prototypes", "canonical_grid",
    "ctp_predictions_for_subset", "evaluate_class_predictions", "load_deployment_config", "load_protocol",
    "preflight_status", "score_region_features", "select_configuration", "support_subsets", "validate_development_manifest",
    "validate_tau", "write_grid_csv", "load_frozen_candidates", "build_development_feature_cache",
    "load_development_feature_cache", "write_prediction_archives", "run_development_grid",
    "source_code_identity", "full_support_per_class_rows", "required_output_hashes",
]
