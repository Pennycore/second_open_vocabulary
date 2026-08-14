"""Frozen OpenAI-CLIP feature-cache infrastructure with no semantic evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .encoder_compare import _encode_model, _load_records
from .io import InputValidationError, sha256_file
from .pixel_pack import validate_region_pixel_pack


_PROTOCOL_NAME = "openai_clip_feature_cache_protocol_v1.json"
_PIXEL_PACK_PROTOCOL_NAME = "encoder_compare_protocol_v0.json"
_PARTITION_FIELDS = ("row_index", "image_id", "candidate_index", "partition")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction is not None and isjunction(path))


def _assert_no_link_components(path: Path) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and _is_link_or_junction(component):
            raise InputValidationError(
                f"Feature-cache output may not traverse a symlink or junction: {component}"
            )


def _prepare_empty_destination(output_dir: str | Path) -> tuple[Path, bool]:
    """Accept only an absent or empty ordinary directory for exclusive outputs."""
    requested = Path(output_dir)
    _assert_no_link_components(requested)
    if _is_link_or_junction(requested):
        raise InputValidationError("Feature-cache output may not be a symlink or junction.")
    destination = requested.resolve()
    if not destination.exists():
        return destination, True
    if _is_link_or_junction(destination) or not destination.is_dir():
        raise InputValidationError("Feature-cache output must be an ordinary directory.")
    try:
        next(destination.iterdir())
    except StopIteration:
        return destination, False
    except OSError as exc:
        raise InputValidationError("Cannot verify that feature-cache output is empty.") from exc
    raise InputValidationError("Feature-cache output must be empty when it already exists.")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"Feature-cache path escapes project root: {name}")
    return str(resolved)


def load_openai_clip_feature_cache_config(
    path: str | Path, project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either the tracked base config or an ignored deployment override."""
    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read feature-cache configuration.") from exc
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Feature-cache config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise InputValidationError("Feature-cache config requires paths.")
    required = {
        "pixel_pack",
        "split_manifest",
        "development_records",
        "heldout_records",
        "openai_clip_checkpoint",
        "protocol_file",
        "output_root",
    }
    if set(paths) != required:
        raise InputValidationError("Feature-cache config paths do not match the frozen schema.")
    for key, value in list(paths.items()):
        paths[key] = _resolve_project_path(value, root, key)
    output_root = Path(str(paths["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("Feature-cache output_root must be directly under outputs/.")
    expected_protocol = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(paths["protocol_file"])).resolve() != expected_protocol:
        raise InputValidationError("Feature-cache protocol must be the committed canonical protocol.")
    try:
        protocol_bytes = expected_protocol.read_bytes()
        protocol = json.loads(protocol_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read the canonical feature-cache protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("Feature-cache protocol must be a JSON object.")
    protocol["path"] = str(expected_protocol)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_openai_clip_feature_cache_anchor(
    project_root: str | Path, expected_commit: str, expected_protocol_sha256: str
) -> dict[str, str]:
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("Feature-cache repository anchor could not read git state.") from exc
    actual_protocol_sha256 = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual_protocol_sha256 != expected_protocol_sha256:
        raise InputValidationError(
            "Feature-cache creation requires the approved clean commit and protocol SHA-256."
        )
    return {"code_commit": commit, "protocol_sha256": actual_protocol_sha256}


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputValidationError(f"Malformed {label} JSON at line {line_number}.") from exc
                if not isinstance(row, dict):
                    raise InputValidationError(f"{label} rows must be JSON objects.")
                rows.append(row)
    except OSError as exc:
        raise InputValidationError(f"Cannot read {label}.") from exc
    if not rows:
        raise InputValidationError(f"{label} may not be empty.")
    return rows


def _row_key(row: dict[str, Any], label: str) -> tuple[int, str, int]:
    try:
        row_index = int(row["row_index"])
        image_id = str(row["image_id"])
        candidate_index = int(row["candidate_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(
            f"{label} rows require row_index, image_id, and candidate_index."
        ) from exc
    if row_index < 0 or not image_id or candidate_index < 0:
        raise InputValidationError(f"{label} has an invalid region key.")
    return row_index, image_id, candidate_index


def validate_split_partition_mapping(
    pixel_records: Iterable[dict[str, Any]],
    development_records: Iterable[dict[str, Any]],
    heldout_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source-row-ordered, label-free partition records after exact joining."""
    source_rows = list(pixel_records)
    development_rows = list(development_records)
    heldout_rows = list(heldout_records)
    if not source_rows or not development_rows or not heldout_rows:
        raise InputValidationError("Pixel and both split partitions must be non-empty.")
    source_by_key: dict[tuple[str, int], int] = {}
    source_by_index: dict[int, tuple[str, int]] = {}
    for row in source_rows:
        row_index, image_id, candidate_index = _row_key(row, "Pixel-pack")
        key = (image_id, candidate_index)
        if key in source_by_key or row_index in source_by_index:
            raise InputValidationError("Pixel-pack region keys and row indices must be unique.")
        source_by_key[key] = row_index
        source_by_index[row_index] = key
    if sorted(source_by_index) != list(range(len(source_rows))):
        raise InputValidationError("Pixel-pack source rows must have contiguous ordered row indices.")

    partition_by_key: dict[tuple[str, int], str] = {}
    image_partition: dict[str, str] = {}
    for partition, rows in (("development", development_rows), ("heldout", heldout_rows)):
        for row in rows:
            row_index, image_id, candidate_index = _row_key(row, f"{partition} split")
            key = (image_id, candidate_index)
            if key not in source_by_key:
                raise InputValidationError("Split record does not occur in the frozen pixel pack.")
            if source_by_key[key] != row_index:
                raise InputValidationError("Split row_index does not match the frozen pixel pack.")
            if key in partition_by_key:
                raise InputValidationError("Split region keys must cover the pixel pack exactly once.")
            partition_by_key[key] = partition
            previous = image_partition.setdefault(image_id, partition)
            if previous != partition:
                raise InputValidationError("Development and heldout partitions must be image-disjoint.")
    if set(partition_by_key) != set(source_by_key):
        raise InputValidationError("Split partitions do not cover every frozen pixel-pack row.")
    return [
        {
            "row_index": row_index,
            "image_id": source_by_index[row_index][0],
            "candidate_index": source_by_index[row_index][1],
            "partition": partition_by_key[source_by_index[row_index]],
        }
        for row_index in range(len(source_rows))
    ]


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_pre_result":
        raise InputValidationError("Feature-cache protocol must be frozen_pre_result.")
    if protocol.get("role") != "infrastructure cache; no strategy evaluation":
        raise InputValidationError("Feature-cache protocol role is not the registered infrastructure scope.")
    if protocol.get("scientific_evidence") is not False:
        raise InputValidationError("Feature-cache protocol must declare scientific_evidence=false.")
    if protocol.get("dataset") != "LoveDA" or protocol.get("split") != "train":
        raise InputValidationError("Feature-cache protocol must be scoped to LoveDA Train.")
    model = protocol.get("model", {})
    if (
        model.get("architecture") != "ViT-B-32-quickgelu"
        or int(model.get("feature_dimension", -1)) != 512
        or model.get("open_clip_version") != "3.3.0"
    ):
        raise InputValidationError("Feature-cache model registration is invalid.")
    constraints = protocol.get("constraints", {})
    required_false = (
        "sam3_rerun",
        "training",
        "pixel_gt",
        "validation_split",
        "remoteclip_feature_or_text_reuse",
        "weak_labels_used_for_cache",
        "predictions",
        "similarity_scores",
        "metrics",
        "model_selection_or_decision",
        "overwrite",
    )
    if any(constraints.get(field) is not False for field in required_false):
        raise InputValidationError("Feature-cache constraints differ from the frozen protocol.")


def _validate_split_inputs(
    cfg: dict[str, Any], protocol: dict[str, Any], pixel_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    split = protocol["split_inputs"]
    paths = cfg["paths"]
    sources = {
        "manifest": (Path(str(paths["split_manifest"])), str(split["manifest_sha256"])),
        "development_records": (
            Path(str(paths["development_records"])),
            str(split["development_records_sha256"]),
        ),
        "heldout_records": (
            Path(str(paths["heldout_records"])),
            str(split["heldout_records_sha256"]),
        ),
    }
    actual_hashes: dict[str, str] = {}
    for name, (path, expected_hash) in sources.items():
        if not path.is_file():
            raise InputValidationError(f"Feature-cache split input is missing: {name}")
        actual_hashes[name] = sha256_file(path)
        if actual_hashes[name] != expected_hash:
            raise InputValidationError(f"Feature-cache split input hash differs from protocol: {name}")
    try:
        split_manifest = json.loads(sources["manifest"][0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Feature-cache split manifest is not valid JSON.") from exc
    if not isinstance(split_manifest, dict) or split_manifest.get("status") != "completed":
        raise InputValidationError("Feature-cache split manifest is not a completed split.")
    if split_manifest.get("development", {}).get("records_sha256") != actual_hashes["development_records"]:
        raise InputValidationError("Split manifest does not bind the development records.")
    if split_manifest.get("heldout", {}).get("records_sha256") != actual_hashes["heldout_records"]:
        raise InputValidationError("Split manifest does not bind the heldout records.")
    development = _read_jsonl(sources["development_records"][0], "development split")
    heldout = _read_jsonl(sources["heldout_records"][0], "heldout split")
    partitions = validate_split_partition_mapping(pixel_rows, development, heldout)
    return partitions, actual_hashes


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
    except Exception:
        raise


def create_openai_clip_feature_cache(
    cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Create one OpenAI-CLIP image-feature cache without any semantic operation."""
    _validate_protocol(protocol)
    paths = cfg.get("paths", {})
    checkpoint_value = paths.get("openai_clip_checkpoint")
    if not checkpoint_value:
        raise InputValidationError(
            "Feature-cache checkpoint is null in the tracked base config; use an ignored deployment config."
        )
    package = Path(str(paths["pixel_pack"])).resolve()
    validation = validate_region_pixel_pack(package, package / _PIXEL_PACK_PROTOCOL_NAME)
    for field, expected in protocol["pixel_pack"].items():
        if validation.get(field) != expected:
            raise InputValidationError(f"Pixel package differs from frozen feature-cache protocol: {field}")
    pixel_rows = _load_records(package)
    partitions, split_hashes = _validate_split_inputs(cfg, protocol, pixel_rows)
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_file():
        raise InputValidationError("OpenAI CLIP checkpoint is unavailable.")
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("OpenAI CLIP checkpoint differs from the registered artifact.")
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise InputValidationError("Feature cache requires torch and open_clip.") from exc
    if getattr(open_clip, "__version__", None) != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the registered protocol.")
    requested = str(cfg.get("runtime", {}).get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    batch_regions = int(cfg.get("runtime", {}).get("batch_regions", 0))
    if batch_regions != 64:
        raise InputValidationError("Feature-cache batch_regions must remain the frozen value 64.")
    destination, must_create_destination = _prepare_empty_destination(output_dir)
    if must_create_destination:
        destination.mkdir(parents=True, exist_ok=False)

    features, model, _tokenizer, preprocess = _encode_model(
        package, checkpoint, protocol["model"]["architecture"], batch_regions, device
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    features = np.asarray(features, dtype=np.float32)
    expected_shape = tuple(int(value) for value in protocol["output"]["feature_shape"])
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise InputValidationError("Encoded feature cache has an invalid shape or non-finite values.")
    if np.any(np.linalg.norm(features, axis=1) <= 0):
        raise InputValidationError("Encoded feature cache contains a zero vector.")
    feature_path = destination / str(protocol["output"]["feature_file"])
    partition_path = destination / str(protocol["output"]["row_partition_file"])
    _write_npy_exclusive(feature_path, features.astype(np.float16))
    _write_jsonl_exclusive(partition_path, partitions)
    anchor = cfg.get("repository_anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"code_commit", "protocol_sha256"}:
        raise InputValidationError("Feature-cache runner must supply the exact repository anchor.")
    if anchor["protocol_sha256"] != protocol.get("sha256"):
        raise InputValidationError("Feature-cache repository anchor does not bind the loaded protocol.")
    manifest = {
        "format_version": 1,
        "status": "completed",
        "scientific_evidence": False,
        "role": protocol["role"],
        "repository_anchor": anchor,
        "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "pixel_pack": {
            field: validation[field]
            for field in ("bundle_id", "record_count", "image_count", "ordered_record_key_sha256")
        },
        "split_inputs": {
            "manifest_sha256": split_hashes["manifest"],
            "development_records_sha256": split_hashes["development_records"],
            "heldout_records_sha256": split_hashes["heldout_records"],
            "development_record_count": sum(row["partition"] == "development" for row in partitions),
            "heldout_record_count": sum(row["partition"] == "heldout" for row in partitions),
            "image_disjoint": True,
            "all_pixel_rows_covered_exactly_once": True,
        },
        "model": protocol["model"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "preprocess": str(preprocess),
        "view_fusion": protocol["software"]["view_fusion"],
        "outputs": {
            "features_openai_clip": {
                "path": feature_path.name,
                "sha256": sha256_file(feature_path),
                "dtype": "float16",
                "shape": list(expected_shape),
            },
            "row_partitions": {
                "path": partition_path.name,
                "sha256": sha256_file(partition_path),
                "fields": list(_PARTITION_FIELDS),
            },
        },
        "constraints": protocol["constraints"],
    }
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
