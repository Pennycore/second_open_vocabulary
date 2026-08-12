from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import FeatureBundle, InputValidationError, sha256_file


_FORBIDDEN_PATH_TOKEN = re.compile(r"(?:^|[_-])(val|validation|oracle)(?:[_-]|$)")
_LOVEDA_TRAIN_IMAGE_ID = re.compile(r"^loveda_train_(?:rural|urban)_.+$")


@dataclass(frozen=True)
class NativeCandidate:
    class_id: int
    mask: np.ndarray
    x0: int
    y0: int


@dataclass(frozen=True)
class NativeRegionRecord:
    feature: np.ndarray
    cam_label: str
    sam3_source_label: str
    image_id: str
    candidate_index: int


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"JSON metadata must be an object: {path}")
    return value


def _assert_read_only_source_path(path: Path, label: str) -> None:
    if not path.exists():
        raise InputValidationError(f"Missing {label}: {path}")
    for part in path.resolve().parts:
        lowered = part.lower()
        if _FORBIDDEN_PATH_TOKEN.search(lowered):
            raise InputValidationError(
                f"{label} path contains a forbidden Val/oracle token: {path}"
            )


def _walk_mapping(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower(), item
            yield from _walk_mapping(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mapping(item)


def _reject_forbidden_declarations(metadata: dict[str, Any], source: Path) -> None:
    false_required = {
        "pixel_gt_used",
        "uses_pixel_gt",
        "use_pixel_gt",
        "allow_pixel_gt",
        "uses_oracle",
        "use_oracle",
        "allow_oracle",
        "love_da_val_used",
        "validation_used",
    }
    for key, value in _walk_mapping(metadata):
        if key in false_required and value is not False:
            raise InputValidationError(
                f"{source} declares forbidden or ambiguous {key}={value!r}."
            )
        if key in {"split", "data_split"} and str(value).lower() != "train":
            raise InputValidationError(f"{source} is not declared as Train-only.")
        if isinstance(value, str) and "oracle" in value.lower():
            raise InputValidationError(f"{source} contains an oracle provenance reference.")


def candidate_cache_fingerprint(candidate_dir: str | Path, image_id: str) -> str:
    root = Path(candidate_dir)
    digest = hashlib.sha256()
    for path in (root / f"{image_id}.npz", root / f"{image_id}.json"):
        if not path.is_file():
            raise InputValidationError(f"Missing candidate cache companion: {path}")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _require_arrays(archive: Any, names: set[str], source: Path) -> None:
    missing = sorted(names - set(archive.files))
    if missing:
        raise InputValidationError(f"{source} lacks required arrays: {missing}")


def _require_dtype(array: np.ndarray, dtype: np.dtype[Any], field: str, source: Path) -> None:
    if array.dtype != np.dtype(dtype):
        raise InputValidationError(
            f"{source} field {field!r} must have dtype {np.dtype(dtype)}, got {array.dtype}."
        )


def _configured_id_to_name(cfg: dict[str, Any]) -> dict[int, str]:
    return {int(item["id"]): str(item["name"]) for item in cfg["data"]["classes"]}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_registered_scope(
    manifest: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    scope = manifest.get("registered_scope")
    if not isinstance(scope, dict):
        raise InputValidationError("Region protocol lacks a registered_scope object.")
    class_names = [str(item["name"]) for item in cfg["data"]["classes"]]
    source_counts = scope.get("expected_source_counts")
    if not isinstance(source_counts, dict) or set(source_counts) != set(class_names):
        raise InputValidationError(
            "registered_scope.expected_source_counts must name every configured class exactly."
        )
    normalized_counts = {name: int(source_counts[name]) for name in class_names}
    if any(value <= 0 for value in normalized_counts.values()):
        raise InputValidationError("Registered source counts must all be positive.")
    expected_images = int(scope.get("expected_image_count", 0))
    expected_candidates = int(scope.get("expected_candidate_count", 0))
    registered_cap = int(scope.get("max_regions_per_class", 0))
    if expected_images <= 0 or expected_candidates <= 0 or registered_cap <= 0:
        raise InputValidationError("Registered scope counts and sampling cap must be positive.")
    if sum(normalized_counts.values()) != expected_candidates:
        raise InputValidationError(
            "Registered per-class source counts do not sum to expected_candidate_count."
        )
    if scope.get("limit_images", "missing") is not None:
        raise InputValidationError("Registered scope must declare limit_images=null.")
    expected_literals = {
        "require_all_pairs": True,
        "require_all_classes": True,
        "cam_method": "mean",
        "sampling_reference": "sam3_source_class",
        "normalize_features": True,
    }
    for key, expected in expected_literals.items():
        if scope.get(key) != expected:
            raise InputValidationError(
                f"Registered scope must declare {key}={expected!r}."
            )
    for key in ("prompt_config_sha256", "evaluation_config_sha256"):
        value = str(scope.get(key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise InputValidationError(f"Registered scope has an invalid {key}.")
    return {
        **scope,
        "expected_image_count": expected_images,
        "expected_candidate_count": expected_candidates,
        "expected_source_counts": normalized_counts,
        "seed": int(scope.get("seed", -1)),
        "max_regions_per_class": registered_cap,
    }


def is_registered_region_scope(
    cfg: dict[str, Any], manifest: dict[str, Any] | None = None
) -> bool:
    if manifest is None:
        value = cfg["paths"].get("region_provenance_file")
        if not value:
            raise InputValidationError("paths.region_provenance_file is required.")
        manifest = _read_json_object(Path(value))
    scope = _validated_registered_scope(manifest, cfg)
    options = cfg.get("region_input", {})
    return bool(
        options.get("limit_images") is None
        and options.get("require_all_pairs") is True
        and options.get("require_all_classes") is True
        and options.get("cam_method") == scope["cam_method"]
        and options.get("sampling_reference") == scope["sampling_reference"]
        and int(options.get("max_regions_per_class", -1))
        == scope["max_regions_per_class"]
        and int(cfg["experiment"]["seed"]) == scope["seed"]
        and cfg["model"].get("normalize_features") is scope["normalize_features"]
        and _canonical_sha256(cfg.get("prompts")) == scope["prompt_config_sha256"]
        and _canonical_sha256(cfg.get("evaluation")) == scope["evaluation_config_sha256"]
    )


def load_region_protocol_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg["paths"].get("region_provenance_file")
    if not value:
        raise InputValidationError("paths.region_provenance_file is required for a formal region probe.")
    path = Path(value)
    _assert_read_only_source_path(path, "region protocol manifest")
    if not path.is_file():
        raise InputValidationError(f"Region protocol manifest must be a file: {path}")
    manifest = _read_json_object(path)
    if int(manifest.get("format_version", -1)) != 1:
        raise InputValidationError("Unsupported region protocol manifest format.")
    if str(manifest.get("dataset", "")).lower() != str(cfg["data"]["dataset_name"]).lower():
        raise InputValidationError("Region protocol dataset differs from the configured dataset.")
    if str(manifest.get("split", "")).lower() != "train":
        raise InputValidationError("Region protocol must declare split=train.")
    required_false = (
        "direct_pixel_gt_used",
        "love_da_val_used",
        "oracle_used",
        "e2_used",
        "selection_used_gt",
        "selection_used_text_prediction",
    )
    for key in required_false:
        if manifest.get(key) is not False:
            raise InputValidationError(f"Region protocol must explicitly declare {key}=false.")
    if manifest.get("candidate_subset") != "all_train_candidates":
        raise InputValidationError(
            "Only all_train_candidates may be loaded before registered source-class sampling."
        )
    weak_origin = str(manifest.get("image_level_weak_tag_origin", ""))
    if "train" not in weak_origin.lower() or "pixel" not in weak_origin.lower():
        raise InputValidationError(
            "Region protocol must disclose the first-paper Train pixel-mask-derived image tags."
        )
    label_derivations = manifest.get("label_derivations")
    expected_derivations = {
        "sam3_source": "sam3_candidate_source_class",
        "cam": "cam_mask_mean_top1",
    }
    if label_derivations != expected_derivations:
        raise InputValidationError(
            "Region labels must be independently derived from SAM3 candidate source and CAM mask mean top1."
        )
    registered_scope = _validated_registered_scope(manifest, cfg)
    feature_source = manifest.get("feature_source")
    if not isinstance(feature_source, dict):
        raise InputValidationError("Region protocol lacks feature_source metadata.")
    if feature_source.get("array") != "region_features":
        raise InputValidationError("Only the native region_features array is permitted.")
    if feature_source.get("model_name") != cfg["model"]["model_name"]:
        raise InputValidationError("Region protocol model differs from the configured model.")
    if int(feature_source.get("feature_dimension", -1)) != int(cfg["model"]["feature_dim"]):
        raise InputValidationError("Region protocol feature dimension is invalid.")
    if feature_source.get("view_fusion") != "normalized mean of context and mask-emphasized features":
        raise InputValidationError("Region protocol view-fusion identity is invalid.")
    checkpoint = cfg["paths"].get("remoteclip_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise InputValidationError("A readable RemoteCLIP checkpoint is required for identity binding.")
    checkpoint_sha = sha256_file(Path(checkpoint))
    if str(feature_source.get("checkpoint_sha256", "")).lower() != checkpoint_sha.lower():
        raise InputValidationError("Region protocol checkpoint SHA-256 differs from the configured checkpoint.")
    return {
        **manifest,
        "registered_scope": registered_scope,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "validated_checkpoint_sha256": checkpoint_sha,
    }


def load_native_candidate_cache(
    candidate_dir: str | Path,
    image_id: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[NativeCandidate], str]:
    root = Path(candidate_dir)
    data_path = root / f"{image_id}.npz"
    metadata_path = root / f"{image_id}.json"
    metadata = _read_json_object(metadata_path)
    _reject_forbidden_declarations(metadata, metadata_path)
    if int(metadata.get("format_version", -1)) != 1:
        raise InputValidationError(f"Unsupported candidate metadata format: {metadata_path}")
    if str(metadata.get("image_id")) != image_id:
        raise InputValidationError(f"Candidate image_id mismatch for {image_id}.")
    if metadata.get("foreground_only") is not True:
        raise InputValidationError(f"Candidate cache is not declared foreground_only: {metadata_path}")
    if metadata.get("mask_encoding") != "flattened-packbits-little":
        raise InputValidationError(f"Unsupported candidate mask encoding: {metadata_path}")
    if metadata.get("data_file") != data_path.name:
        raise InputValidationError(f"Candidate data_file does not identify its sibling NPZ: {metadata_path}")

    required = {
        "format_version",
        "image_shape",
        "packed_masks",
        "offsets",
        "shapes",
        "origins",
        "boxes",
        "areas",
        "scores",
        "class_ids",
        "prompt_ids",
    }
    with np.load(data_path, allow_pickle=False) as archive:
        _require_arrays(archive, required, data_path)
        arrays = {name: archive[name] for name in required}

    if arrays["format_version"].shape != (1,) or int(arrays["format_version"][0]) != 1:
        raise InputValidationError(f"Unsupported candidate NPZ format: {data_path}")
    _require_dtype(arrays["format_version"], np.int16, "format_version", data_path)
    _require_dtype(arrays["image_shape"], np.int32, "image_shape", data_path)
    _require_dtype(arrays["packed_masks"], np.uint8, "packed_masks", data_path)
    _require_dtype(arrays["offsets"], np.int64, "offsets", data_path)
    _require_dtype(arrays["shapes"], np.int32, "shapes", data_path)
    _require_dtype(arrays["origins"], np.int32, "origins", data_path)
    _require_dtype(arrays["boxes"], np.int32, "boxes", data_path)
    _require_dtype(arrays["areas"], np.int64, "areas", data_path)
    _require_dtype(arrays["scores"], np.float32, "scores", data_path)
    _require_dtype(arrays["class_ids"], np.int16, "class_ids", data_path)
    _require_dtype(arrays["prompt_ids"], np.int16, "prompt_ids", data_path)

    count = int(metadata.get("candidate_count", -1))
    image_shape = np.asarray(arrays["image_shape"])
    if image_shape.shape != (2,) or np.any(image_shape <= 0):
        raise InputValidationError(f"Invalid candidate image_shape: {data_path}")
    if list(map(int, image_shape)) != [int(x) for x in metadata.get("image_shape", [])]:
        raise InputValidationError(f"Candidate JSON/NPZ image_shape mismatch: {image_id}")
    expected_shapes = {
        "shapes": (count, 2),
        "origins": (count, 2),
        "boxes": (count, 4),
        "areas": (count,),
        "scores": (count,),
        "class_ids": (count,),
        "prompt_ids": (count,),
        "offsets": (count + 1,),
    }
    for field, expected in expected_shapes.items():
        if arrays[field].shape != expected:
            raise InputValidationError(
                f"Candidate field {field!r} shape mismatch for {image_id}: "
                f"expected {expected}, got {arrays[field].shape}."
            )
    offsets = arrays["offsets"]
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(arrays["packed_masks"]):
        raise InputValidationError(f"Candidate packed-mask offsets are invalid: {image_id}")
    if not np.isfinite(arrays["scores"]).all():
        raise InputValidationError(f"Candidate scores contain NaN/Inf: {image_id}")

    id_to_name = _configured_id_to_name(cfg)
    prompt_records = metadata.get("prompts")
    if not isinstance(prompt_records, list):
        raise InputValidationError(f"Candidate prompt table is missing: {metadata_path}")
    prompt_table: dict[int, dict[str, Any]] = {}
    for record in prompt_records:
        if not isinstance(record, dict) or "id" not in record:
            raise InputValidationError(f"Malformed candidate prompt record: {metadata_path}")
        prompt_id = int(record["id"])
        if prompt_id in prompt_table:
            raise InputValidationError(f"Duplicate candidate prompt id {prompt_id}: {metadata_path}")
        prompt_table[prompt_id] = record
    if sorted(prompt_table) != list(range(len(prompt_table))):
        raise InputValidationError(f"Candidate prompt IDs must be contiguous from zero: {metadata_path}")

    candidates: list[NativeCandidate] = []
    computed_class_counts: dict[str, int] = {}
    image_height, image_width = (int(value) for value in image_shape)
    for index in range(count):
        class_id = int(arrays["class_ids"][index])
        if class_id not in id_to_name:
            raise InputValidationError(f"Unmapped SAM3 source class id {class_id}: {image_id}")
        prompt_id = int(arrays["prompt_ids"][index])
        if prompt_id not in prompt_table:
            raise InputValidationError(f"Unknown prompt id {prompt_id}: {image_id}")
        prompt_record = prompt_table[prompt_id]
        if int(prompt_record.get("class_id", -1)) != class_id:
            raise InputValidationError(f"Prompt/source class mismatch for {image_id}[{index}].")
        if str(prompt_record.get("class_name")) != id_to_name[class_id]:
            raise InputValidationError(f"Prompt class name mismatch for {image_id}[{index}].")

        height, width = (int(value) for value in arrays["shapes"][index])
        x0, y0 = (int(value) for value in arrays["origins"][index])
        if height <= 0 or width <= 0 or x0 < 0 or y0 < 0:
            raise InputValidationError(f"Invalid candidate geometry for {image_id}[{index}].")
        if x0 + width > image_width or y0 + height > image_height:
            raise InputValidationError(f"Candidate bounds exceed image for {image_id}[{index}].")
        start, end = int(offsets[index]), int(offsets[index + 1])
        expected_bytes = math.ceil(height * width / 8)
        if end - start != expected_bytes:
            raise InputValidationError(f"Packed mask length mismatch for {image_id}[{index}].")
        flat = np.unpackbits(
            arrays["packed_masks"][start:end],
            bitorder="little",
            count=height * width,
        )
        mask = flat.reshape(height, width).astype(bool, copy=False)
        area = int(mask.sum())
        if area <= 0 or area != int(arrays["areas"][index]):
            raise InputValidationError(f"Candidate mask area mismatch for {image_id}[{index}].")
        ys, xs = np.nonzero(mask)
        expected_box = np.asarray(
            [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1],
            dtype=np.int32,
        )
        if not np.array_equal(expected_box, arrays["boxes"][index]):
            raise InputValidationError(f"Candidate box/mask mismatch for {image_id}[{index}].")
        candidates.append(NativeCandidate(class_id=class_id, mask=mask, x0=x0, y0=y0))
        class_name = id_to_name[class_id]
        computed_class_counts[class_name] = computed_class_counts.get(class_name, 0) + 1

    declared_counts = metadata.get("class_candidate_counts")
    if declared_counts != dict(sorted(computed_class_counts.items())):
        raise InputValidationError(f"Candidate class_candidate_counts mismatch: {image_id}")

    fingerprint = candidate_cache_fingerprint(root, image_id)
    return metadata, candidates, fingerprint


def load_native_region_score(
    region_dir: str | Path,
    image_id: str,
    candidate_fingerprint: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(region_dir)
    data_path = root / f"{image_id}.npz"
    metadata_path = root / f"{image_id}.json"
    metadata = _read_json_object(metadata_path)
    _reject_forbidden_declarations(metadata, metadata_path)
    if int(metadata.get("format_version", -1)) != 1:
        raise InputValidationError(f"Unsupported region metadata format: {metadata_path}")
    if str(metadata.get("image_id")) != image_id:
        raise InputValidationError(f"Region image_id mismatch for {image_id}.")
    if metadata.get("pixel_gt_used") is not False:
        raise InputValidationError(f"Region cache must explicitly declare pixel_gt_used=false: {image_id}")
    if metadata.get("region_features_saved") is not True:
        raise InputValidationError(f"Region cache lacks saved visual features: {image_id}")
    if metadata.get("region_feature_dtype") != "float16":
        raise InputValidationError(f"Unexpected region feature dtype declaration: {image_id}")
    if str(metadata.get("model_name")) != str(cfg["model"]["model_name"]):
        raise InputValidationError(f"Region cache model mismatch: {image_id}")
    if str(metadata.get("candidate_cache_sha256", "")).lower() != candidate_fingerprint.lower():
        raise InputValidationError(f"Candidate cache fingerprint mismatch for {image_id}.")
    checkpoint = cfg["paths"].get("remoteclip_checkpoint")
    weights_source = metadata.get("weights_source")
    if not weights_source:
        raise InputValidationError(f"Region cache lacks weights_source: {image_id}")
    if checkpoint and Path(str(weights_source)).name != Path(str(checkpoint)).name:
        raise InputValidationError(f"Region/cache checkpoint identity mismatch: {image_id}")
    if metadata.get("view_fusion") != "normalized mean of context and mask-emphasized features":
        raise InputValidationError(f"Unexpected region view-fusion rule: {image_id}")

    required = {
        "format_version",
        "candidate_indices",
        "scores",
        "class_ids",
        "active_class_ids",
        "predicted_class_ids",
        "margins",
        "crop_boxes",
        "mask_fractions",
        "region_features",
    }
    with np.load(data_path, allow_pickle=False) as archive:
        _require_arrays(archive, required, data_path)
        arrays = {name: archive[name] for name in required}
    if arrays["format_version"].shape != (1,) or int(arrays["format_version"][0]) != 1:
        raise InputValidationError(f"Unsupported region NPZ format: {data_path}")
    _require_dtype(arrays["format_version"], np.int16, "format_version", data_path)
    _require_dtype(arrays["candidate_indices"], np.int32, "candidate_indices", data_path)
    _require_dtype(arrays["scores"], np.float32, "scores", data_path)
    _require_dtype(arrays["class_ids"], np.int16, "class_ids", data_path)
    _require_dtype(arrays["active_class_ids"], np.int16, "active_class_ids", data_path)
    _require_dtype(arrays["predicted_class_ids"], np.int16, "predicted_class_ids", data_path)
    _require_dtype(arrays["margins"], np.float32, "margins", data_path)
    _require_dtype(arrays["crop_boxes"], np.int32, "crop_boxes", data_path)
    _require_dtype(arrays["mask_fractions"], np.float32, "mask_fractions", data_path)
    _require_dtype(arrays["region_features"], np.float16, "region_features", data_path)

    count = int(metadata.get("candidate_count", -1))
    dimension = int(cfg["model"]["feature_dim"])
    class_count = len(arrays["class_ids"])
    expected_shapes = {
        "candidate_indices": (count,),
        "scores": (count, class_count),
        "predicted_class_ids": (count,),
        "margins": (count,),
        "crop_boxes": (count, 4),
        "mask_fractions": (count,),
        "region_features": (count, dimension),
    }
    for field, expected in expected_shapes.items():
        if arrays[field].shape != expected:
            raise InputValidationError(
                f"Region field {field!r} shape mismatch for {image_id}: "
                f"expected {expected}, got {arrays[field].shape}."
            )
    if not np.array_equal(arrays["candidate_indices"], np.arange(count, dtype=np.int32)):
        raise InputValidationError(f"Region candidate ordering is invalid for {image_id}.")
    id_to_name = _configured_id_to_name(cfg)
    class_ids = [int(value) for value in arrays["class_ids"]]
    if class_ids != list(id_to_name):
        raise InputValidationError(f"Region score class mapping mismatch for {image_id}.")
    active_ids = [int(value) for value in arrays["active_class_ids"]]
    if active_ids != sorted(set(active_ids)) or not set(active_ids).issubset(id_to_name):
        raise InputValidationError(f"Region active class mapping is invalid for {image_id}.")
    if count and not active_ids:
        raise InputValidationError(f"Non-empty region cache has no active classes: {image_id}")
    old_predictions = {int(value) for value in arrays["predicted_class_ids"]}
    if not old_predictions.issubset(set(active_ids)):
        raise InputValidationError(f"Stored RemoteCLIP predictions are outside active classes: {image_id}")
    if not np.isfinite(arrays["scores"]).all():
        raise InputValidationError(f"Region scores contain NaN/Inf: {image_id}")
    class_columns = {class_id: index for index, class_id in enumerate(class_ids)}
    active_columns = np.asarray([class_columns[class_id] for class_id in active_ids], dtype=np.int64)
    if count:
        active_scores = arrays["scores"][:, active_columns]
        expected_predictions = np.asarray(active_ids, dtype=np.int16)[np.argmax(active_scores, axis=1)]
        if not np.array_equal(expected_predictions, arrays["predicted_class_ids"]):
            raise InputValidationError(f"Stored region predictions fail score-integrity validation: {image_id}")
        if len(active_ids) == 1:
            if not np.isnan(arrays["margins"]).all():
                raise InputValidationError(f"Single-active-class region margins must be NaN: {image_id}")
        else:
            expected_margins = np.diff(np.sort(active_scores, axis=1)[:, -2:], axis=1)[:, 0]
            if not np.isfinite(arrays["margins"]).all() or not np.allclose(
                arrays["margins"], expected_margins, atol=1e-6
            ):
                raise InputValidationError(f"Stored region margins fail score-integrity validation: {image_id}")
    features = arrays["region_features"].astype(np.float32)
    if not np.isfinite(features).all():
        raise InputValidationError(f"Region features contain NaN/Inf: {image_id}")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        raise InputValidationError(f"Region features contain zero vectors: {image_id}")
    fractions = arrays["mask_fractions"]
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise InputValidationError(f"Region mask fractions are invalid: {image_id}")
    json_class_ids = [int(value) for value in metadata.get("class_ids", [])]
    json_active_ids = [int(value) for value in metadata.get("active_class_ids", [])]
    if json_class_ids != class_ids or json_active_ids != active_ids:
        raise InputValidationError(f"Region JSON/NPZ class mapping mismatch: {image_id}")
    arrays["region_features"] = features
    return metadata, arrays


def load_cam_cache(
    cam_dir: str | Path,
    image_id: str,
    expected_image_shape: tuple[int, int],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cam_dir) / f"{image_id}.npz"
    with np.load(path, allow_pickle=False) as archive:
        _require_arrays(archive, {"cams", "class_ids"}, path)
        raw_cams = np.asarray(archive["cams"])
        raw_class_ids = np.asarray(archive["class_ids"])
    if not np.issubdtype(raw_cams.dtype, np.floating):
        raise InputValidationError(f"CAM data must be floating point: {image_id}")
    if not np.issubdtype(raw_class_ids.dtype, np.integer):
        raise InputValidationError(f"CAM class_ids must be integral: {image_id}")
    cams = raw_cams.astype(np.float32, copy=False)
    class_ids = raw_class_ids.reshape(-1)
    if cams.ndim != 3 or cams.shape[0] != len(class_ids):
        raise InputValidationError(f"CAM array/class_ids shape mismatch: {image_id}")
    if cams.shape[1:] != expected_image_shape:
        raise InputValidationError(
            f"CAM/candidate image shape mismatch for {image_id}: "
            f"{cams.shape[1:]} != {expected_image_shape}."
        )
    if not np.isfinite(cams).all():
        raise InputValidationError(f"CAM contains NaN/Inf: {image_id}")
    if np.any(cams < -1e-4) or np.any(cams > 1.0001):
        raise InputValidationError(f"CAM values fall outside the normalized [0,1] range: {image_id}")
    configured_ids = set(_configured_id_to_name(cfg))
    values = [int(value) for value in class_ids]
    if len(values) != len(set(values)) or set(values) != configured_ids:
        raise InputValidationError(f"CAM class mapping mismatch: {image_id}")
    return cams, class_ids.astype(np.int64, copy=False)


def mean_cam_prediction(
    candidate: NativeCandidate,
    cams: np.ndarray,
    cam_class_ids: np.ndarray,
    active_class_ids: list[int],
) -> int:
    if candidate.class_id not in active_class_ids:
        raise InputValidationError(
            f"SAM3 source class {candidate.class_id} is absent from the image active classes."
        )
    class_to_channel = {int(class_id): index for index, class_id in enumerate(cam_class_ids)}
    missing = sorted(set(active_class_ids) - set(class_to_channel))
    if missing:
        raise InputValidationError(f"Active classes missing from CAM channels: {missing}")
    height, width = candidate.mask.shape
    region = cams[
        :,
        candidate.y0 : candidate.y0 + height,
        candidate.x0 : candidate.x0 + width,
    ]
    if region.shape[1:] != candidate.mask.shape or not candidate.mask.any():
        raise InputValidationError("Candidate mask cannot be aligned with the CAM array.")
    channels = np.asarray([class_to_channel[class_id] for class_id in active_class_ids])
    mean_scores = region[channels][:, candidate.mask].mean(axis=1)
    return int(active_class_ids[int(np.argmax(mean_scores))])


def discover_native_region_ids(region_dir: str | Path, require_all_pairs: bool = True) -> list[str]:
    root = Path(region_dir)
    # The first-paper run directory also contains summary.json and the
    # visual_prototypes JSON/NPZ pair.  Only registered LoveDA Train image
    # identifiers are native region records; treating every stem as an image
    # either creates a false orphan or evaluates a calibration artifact.
    json_ids = {
        path.stem
        for path in root.glob("loveda_train_*.json")
        if path.is_file() and _LOVEDA_TRAIN_IMAGE_ID.fullmatch(path.stem)
    }
    npz_ids = {
        path.stem
        for path in root.glob("loveda_train_*.npz")
        if path.is_file() and _LOVEDA_TRAIN_IMAGE_ID.fullmatch(path.stem)
    }
    if require_all_pairs and json_ids != npz_ids:
        missing_npz = sorted(json_ids - npz_ids)[:5]
        missing_json = sorted(npz_ids - json_ids)[:5]
        raise InputValidationError(
            "Region directory contains orphan companions: "
            f"missing_npz={missing_npz}, missing_json={missing_json}."
        )
    image_ids = sorted(json_ids & npz_ids)
    if not image_ids:
        raise InputValidationError(f"No paired native region caches found under {root}")
    return image_ids


def discover_native_input_ids(
    region_dir: str | Path,
    candidate_dir: str | Path,
    cam_dir: str | Path,
    require_all_pairs: bool = True,
) -> list[str]:
    region_ids = set(discover_native_region_ids(region_dir, require_all_pairs))
    candidate_root = Path(candidate_dir)
    cam_root = Path(cam_dir)
    candidate_json_ids = {
        path.stem
        for path in candidate_root.glob("loveda_train_*.json")
        if path.is_file() and _LOVEDA_TRAIN_IMAGE_ID.fullmatch(path.stem)
    }
    candidate_npz_ids = {
        path.stem
        for path in candidate_root.glob("loveda_train_*.npz")
        if path.is_file() and _LOVEDA_TRAIN_IMAGE_ID.fullmatch(path.stem)
    }
    if candidate_json_ids != candidate_npz_ids:
        raise InputValidationError("Candidate directory contains orphan LoveDA Train companions.")
    candidate_ids = candidate_json_ids
    cam_ids = {
        path.stem
        for path in cam_root.glob("loveda_train_*.npz")
        if path.is_file() and _LOVEDA_TRAIN_IMAGE_ID.fullmatch(path.stem)
    }
    if region_ids != candidate_ids or region_ids != cam_ids:
        raise InputValidationError(
            "Region, candidate, and CAM LoveDA Train image-ID sets are not identical."
        )
    return sorted(region_ids)


def _native_source_stat_snapshot(
    region_dir: Path,
    candidate_dir: Path,
    cam_dir: Path,
    image_ids: list[str],
) -> dict[str, Any]:
    rows: list[str] = []
    total_bytes = 0
    for image_id in image_ids:
        for label, path in (
            ("region_json", region_dir / f"{image_id}.json"),
            ("region_npz", region_dir / f"{image_id}.npz"),
            ("candidate_json", candidate_dir / f"{image_id}.json"),
            ("candidate_npz", candidate_dir / f"{image_id}.npz"),
            ("cam_npz", cam_dir / f"{image_id}.npz"),
        ):
            stat = path.stat()
            total_bytes += int(stat.st_size)
            rows.append(f"{label}/{image_id}\0{stat.st_size}\0{stat.st_mtime_ns}")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "stat_inventory_sha256": hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(),
    }


def inspect_native_region_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = ("region_feature_cache", "candidate_cache_dir", "cam_cache_dir")
    result: dict[str, Any] = {"ready": True, "items": {}}
    for key in keys:
        value = cfg["paths"].get(key)
        path = Path(value) if value else None
        ready = bool(path and path.is_dir())
        result["items"][key] = {
            "path": str(path) if path else None,
            "status": "ready" if ready else "missing_required",
        }
        if not ready:
            result["ready"] = False
    if result["ready"]:
        try:
            ids = discover_native_input_ids(
                cfg["paths"]["region_feature_cache"],
                cfg["paths"]["candidate_cache_dir"],
                cfg["paths"]["cam_cache_dir"],
                bool(cfg.get("region_input", {}).get("require_all_pairs", True)),
            )
            result["paired_region_images"] = len(ids)
        except InputValidationError as exc:
            result["ready"] = False
            result["error"] = str(exc)
    return result


def load_native_region_directory(cfg: dict[str, Any]) -> FeatureBundle:
    region_dir = Path(cfg["paths"]["region_feature_cache"])
    candidate_dir = Path(cfg["paths"]["candidate_cache_dir"])
    cam_dir = Path(cfg["paths"]["cam_cache_dir"])
    for path, label in (
        (region_dir, "region feature cache"),
        (candidate_dir, "candidate cache"),
        (cam_dir, "CAM cache"),
    ):
        _assert_read_only_source_path(path, label)
        if not path.is_dir():
            raise InputValidationError(f"{label} must be a directory: {path}")

    protocol = load_region_protocol_manifest(cfg)

    options = cfg.get("region_input", {})
    if str(options.get("cam_method", "mean")) != "mean":
        raise InputValidationError("region_probe_v0 is registered to the first-paper CAM mean rule.")
    cap_value = options.get("max_regions_per_class")
    cap = None if cap_value is None else int(cap_value)
    if cap is not None and cap <= 0:
        raise InputValidationError("max_regions_per_class must be null or positive.")
    limit_value = options.get("limit_images")
    limit = None if limit_value is None else int(limit_value)
    if limit is not None and limit <= 0:
        raise InputValidationError("limit_images must be null or positive.")
    require_all = bool(options.get("require_all_pairs", True))
    require_all_classes = bool(options.get("require_all_classes", True))

    all_image_ids = discover_native_input_ids(region_dir, candidate_dir, cam_dir, require_all)
    registered_scope = is_registered_region_scope(cfg, protocol)
    scope = protocol["registered_scope"]
    if registered_scope and len(all_image_ids) != scope["expected_image_count"]:
        raise InputValidationError(
            "Registered image-count gate failed: "
            f"{len(all_image_ids)} != {scope['expected_image_count']}."
        )
    image_ids = all_image_ids
    if limit is not None:
        image_ids = image_ids[:limit]
    source_snapshot_before = _native_source_stat_snapshot(
        region_dir, candidate_dir, cam_dir, image_ids
    )
    id_to_name = _configured_id_to_name(cfg)
    class_order = [str(item["name"]) for item in cfg["data"]["classes"]]
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    reservoirs: dict[str, list[NativeRegionRecord]] = {name: [] for name in class_order}
    seen_counts: dict[str, int] = {name: 0 for name in class_order}
    total_candidates = 0
    candidate_fingerprints: list[str] = []

    for image_id in image_ids:
        candidate_npz = candidate_dir / f"{image_id}.npz"
        candidate_json = candidate_dir / f"{image_id}.json"
        cam_path = cam_dir / f"{image_id}.npz"
        missing = [str(path) for path in (candidate_npz, candidate_json, cam_path) if not path.is_file()]
        if missing:
            raise InputValidationError(f"Missing native companions for {image_id}: {missing}")
        candidate_meta, candidates, fingerprint = load_native_candidate_cache(
            candidate_dir, image_id, cfg
        )
        candidate_fingerprints.append(f"{image_id}:{fingerprint}")
        region_meta, region = load_native_region_score(region_dir, image_id, fingerprint, cfg)
        if len(candidates) != len(region["region_features"]):
            raise InputValidationError(f"Candidate/region feature count mismatch for {image_id}.")
        image_shape = tuple(int(value) for value in candidate_meta["image_shape"])
        if [int(value) for value in region_meta.get("candidate_image_shape", [])] != list(image_shape):
            raise InputValidationError(f"Region/candidate image shape metadata mismatch for {image_id}.")
        cams, cam_class_ids = load_cam_cache(cam_dir, image_id, image_shape, cfg)
        active_ids = [int(value) for value in region["active_class_ids"]]
        total_candidates += len(candidates)

        for row, candidate_index in enumerate(region["candidate_indices"]):
            index = int(candidate_index)
            candidate = candidates[index]
            sam3_name = id_to_name[candidate.class_id]
            cam_id = mean_cam_prediction(candidate, cams, cam_class_ids, active_ids)
            cam_name = id_to_name[cam_id]
            record = NativeRegionRecord(
                feature=np.asarray(region["region_features"][row], dtype=np.float32).copy(),
                cam_label=cam_name,
                sam3_source_label=sam3_name,
                image_id=image_id,
                candidate_index=index,
            )
            seen_counts[sam3_name] += 1
            reservoir = reservoirs[sam3_name]
            if cap is None or len(reservoir) < cap:
                reservoir.append(record)
            else:
                replacement = int(rng.integers(0, seen_counts[sam3_name]))
                if replacement < cap:
                    reservoir[replacement] = record

    missing_classes = [name for name in class_order if not reservoirs[name]]
    if require_all_classes and missing_classes:
        raise InputValidationError(
            f"Native region input has no selected SAM3-source records for classes: {missing_classes}"
        )
    records = [record for name in class_order for record in reservoirs[name]]
    if not records:
        raise InputValidationError("Native region join produced no records.")
    features = np.stack([record.feature for record in records]).astype(np.float32)
    cam_labels = [record.cam_label for record in records]
    sam3_labels = [record.sam3_source_label for record in records]
    selected_records = [
        {"image_id": record.image_id, "candidate_index": record.candidate_index}
        for record in records
    ]
    selected_counts = {name: len(reservoirs[name]) for name in class_order}
    source_snapshot_after = _native_source_stat_snapshot(
        region_dir, candidate_dir, cam_dir, image_ids
    )
    if source_snapshot_before != source_snapshot_after:
        raise InputValidationError("Native source files changed while the read-only probe was running.")
    if registered_scope:
        expected_seen = scope["expected_source_counts"]
        expected_selected = {
            name: scope["max_regions_per_class"] for name in class_order
        }
        if total_candidates != scope["expected_candidate_count"]:
            raise InputValidationError(
                "Registered candidate-count gate failed: "
                f"{total_candidates} != {scope['expected_candidate_count']}."
            )
        if seen_counts != expected_seen:
            raise InputValidationError(
                f"Registered per-class source-count gate failed: {seen_counts} != {expected_seen}."
            )
        if selected_counts != expected_selected:
            raise InputValidationError(
                f"Registered selected-count gate failed: {selected_counts} != {expected_selected}."
            )
    record_keys = [f"{record.image_id}:{record.candidate_index}" for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise InputValidationError("Joined native region records contain duplicate row keys.")
    ordered_key_sha256 = hashlib.sha256("\n".join(record_keys).encode("utf-8")).hexdigest()
    metadata = {
        "shape": list(features.shape),
        "provenance": {
            "dataset": cfg["data"]["dataset_name"],
            "split": "train",
            "direct_pixel_gt_used": False,
            "uses_oracle": False,
            "construction": (
                "Read-only join of first-paper region_features, SAM3 candidate source class, "
                "and candidate-mask CAM mean top1."
            ),
            "image_level_weak_tag_origin": options.get(
                "image_level_weak_tag_origin", protocol["image_level_weak_tag_origin"]
            ),
            "protocol_manifest_path": protocol["path"],
            "protocol_manifest_sha256": protocol["sha256"],
            "checkpoint_sha256": protocol["validated_checkpoint_sha256"],
        },
        "source_paths": {
            "region_feature_cache": str(region_dir.resolve()),
            "candidate_cache_dir": str(candidate_dir.resolve()),
            "cam_cache_dir": str(cam_dir.resolve()),
        },
        "image_count": len(image_ids),
        "registered_formal_scope": registered_scope,
        "registered_scope_expectations": scope,
        "candidate_count_scanned": total_candidates,
        "candidate_fingerprint_inventory_sha256": hashlib.sha256(
            "\n".join(candidate_fingerprints).encode("utf-8")
        ).hexdigest(),
        "source_stat_inventory_before": source_snapshot_before,
        "source_stat_inventory_after": source_snapshot_after,
        "sam3_source_seen_counts": seen_counts,
        "selected_counts": selected_counts,
        "selected_records": selected_records,
        "ordered_record_key_sha256": ordered_key_sha256,
        "sampling": {
            "method": "per-SAM3-source reservoir sampling" if cap is not None else "all records",
            "seed": int(cfg["experiment"]["seed"]),
            "max_regions_per_class": cap,
        },
        "cam_method": "mean",
        "stored_remoteclip_predicted_class_ids_used_as_labels": False,
        "stored_remoteclip_scores_used_as_targets": False,
        "input_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
        "input_norm_std": float(np.linalg.norm(features, axis=1).std()),
    }
    return FeatureBundle(
        features=features,
        class_names=["unknown"] * len(features),
        metadata=metadata,
        cam_labels=cam_labels,
        sam3_source_labels=sam3_labels,
    )
