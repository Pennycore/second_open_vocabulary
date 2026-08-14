"""Frozen, post-hoc exploratory visual anchors over an immutable OpenAI-CLIP cache."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .io import InputValidationError, sha256_file


_PROTOCOL_NAME = "openai_clip_visual_anchor_exploratory_protocol_v1.json"
_PARTITION_FIELDS = ("row_index", "image_id", "candidate_index", "partition")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(os.path, "isjunction", lambda _: False)(path))


def _assert_ordinary_path(path: Path, name: str) -> None:
    for component in (path.absolute(), *path.absolute().parents):
        if component.exists() and _link_like(component):
            raise InputValidationError(f"Visual-anchor {name} may not traverse a symlink or junction.")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"Visual-anchor path escapes project root: {name}")
    _assert_ordinary_path(resolved, name)
    return str(resolved)


def load_openai_clip_visual_anchor_config(
    path: str | Path, project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the committed visual-anchor protocol and project-contained inputs."""
    root = Path(project_root).resolve()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read visual-anchor configuration.") from exc
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Visual-anchor config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    required = {
        "feature_cache_manifest", "features", "row_partitions", "split_manifest",
        "development_records", "heldout_records", "openai_clip_checkpoint", "protocol_file", "output_root",
    }
    if not isinstance(paths, dict) or set(paths) != required:
        raise InputValidationError("Visual-anchor config paths do not match the frozen schema.")
    for key, value in list(paths.items()):
        paths[key] = _resolve_project_path(value, root, key)
    output_root = Path(str(paths["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("Visual-anchor output_root must be directly under outputs/.")
    expected = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(paths["protocol_file"])).resolve() != expected:
        raise InputValidationError("Visual-anchor protocol must be the committed canonical protocol.")
    try:
        raw = expected.read_bytes()
        protocol = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read canonical visual-anchor protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("Visual-anchor protocol must be a JSON object.")
    protocol["path"] = str(expected)
    protocol["sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg, protocol


def verify_openai_clip_visual_anchor_anchor(
    project_root: str | Path, expected_commit: str, expected_protocol_sha256: str
) -> dict[str, str]:
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("Visual-anchor repository anchor could not read git state.") from exc
    actual = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual != expected_protocol_sha256:
        raise InputValidationError("Visual-anchor run requires the approved clean commit and protocol SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise InputValidationError("Visual-anchor vectors must be finite two-dimensional arrays.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise InputValidationError("Visual-anchor vectors may not contain zero rows.")
    return array / norms


def build_visual_prototypes(
    features: np.ndarray, development_records: Iterable[dict[str, Any]], classes: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Build exactly one class prototype exclusively from development weak labels."""
    vectors = _normalize(features)
    rows = list(development_records)
    prototypes: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for name in classes:
        indices = []
        for row in rows:
            if str(row.get("sam3_source_label", "")) == name:
                try:
                    index = int(row["row_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise InputValidationError("Development record lacks a valid row_index.") from exc
                if index < 0 or index >= len(vectors):
                    raise InputValidationError("Development row_index is outside the frozen feature cache.")
                indices.append(index)
        if not indices:
            raise InputValidationError(f"Development partition has no SAM3 rows for {name}.")
        if len(set(indices)) != len(indices):
            raise InputValidationError("Development records contain duplicate row_index values.")
        counts[name] = len(indices)
        prototypes.append(_normalize(vectors[np.asarray(indices, dtype=np.int64)].mean(axis=0, keepdims=True))[0])
    return np.asarray(prototypes, dtype=np.float32), counts


def fuse_scores(region_features: np.ndarray, text_prototypes: np.ndarray, visual_prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    regions = _normalize(region_features)
    text = _normalize(text_prototypes)
    visual = _normalize(visual_prototypes)
    if text.shape != visual.shape or regions.shape[1] != text.shape[1]:
        raise InputValidationError("Visual-anchor feature dimensions do not match.")
    text_scores = regions @ text.T
    return text_scores, (0.5 * text_scores + 0.5 * (regions @ visual.T)).astype(np.float32)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise InputValidationError(f"{label} row {number} is not an object.")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Cannot read {label} JSONL.") from exc
    if not rows:
        raise InputValidationError(f"{label} may not be empty.")
    return rows


def _key(row: dict[str, Any], label: str) -> tuple[int, str, int]:
    try:
        result = int(row["row_index"]), str(row["image_id"]), int(row["candidate_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(f"{label} needs row_index, image_id, and candidate_index.") from exc
    if result[0] < 0 or result[2] < 0 or not result[1]:
        raise InputValidationError(f"{label} has an invalid region key.")
    return result


def join_split_to_partitions(
    partition_rows: Iterable[dict[str, Any]], development: Iterable[dict[str, Any]], heldout: Iterable[dict[str, Any]], feature_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail-closed keyed join proving complete, image-disjoint split/cache alignment."""
    partitions = list(partition_rows)
    if len(partitions) != feature_count:
        raise InputValidationError("Cache partition count does not equal feature-row count.")
    by_key: dict[tuple[int, str, int], str] = {}
    image_partition: dict[str, str] = {}
    for row in partitions:
        if set(row) != set(_PARTITION_FIELDS):
            raise InputValidationError("Cache row partitions have an unexpected schema.")
        key = _key(row, "Cache partition")
        part = row["partition"]
        if part not in {"development", "heldout"} or key in by_key:
            raise InputValidationError("Cache row partitions have duplicate or invalid keys.")
        by_key[key] = str(part)
        previous = image_partition.setdefault(key[1], str(part))
        if previous != part:
            raise InputValidationError("Cache partitions are not image-disjoint.")
    if {key[0] for key in by_key} != set(range(feature_count)):
        raise InputValidationError("Cache row_index values must exactly cover feature rows.")

    def validate(rows: Iterable[dict[str, Any]], expected: str) -> list[dict[str, Any]]:
        result = list(rows)
        keys = [_key(row, f"{expected} split") for row in result]
        if len(keys) != len(set(keys)):
            raise InputValidationError(f"{expected} split contains duplicate keys.")
        if any(by_key.get(key) != expected for key in keys):
            raise InputValidationError(f"{expected} split does not exactly match cache partitions.")
        return result

    dev, hold = validate(development, "development"), validate(heldout, "heldout")
    if set(_key(row, "development split") for row in dev) | set(_key(row, "heldout split") for row in hold) != set(by_key):
        raise InputValidationError("Split records do not cover all cached rows exactly once.")
    return dev, hold


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_pre_result" or protocol.get("scientific_evidence") is not False:
        raise InputValidationError("Visual-anchor protocol must be frozen and non-scientific.")
    if protocol.get("post_hoc_exploratory") is not True or "not blind" not in str(protocol.get("holdout_interpretation")):
        raise InputValidationError("Visual-anchor protocol must disclose its post-hoc, non-blind scope.")
    if protocol.get("classes") != ["building", "road", "water", "barren", "forest", "agriculture"]:
        raise InputValidationError("Visual-anchor classes differ from the registered six-class vocabulary.")
    model = protocol.get("model", {})
    if model.get("architecture") != "ViT-B-32-quickgelu" or int(model.get("feature_dimension", -1)) != 512 or model.get("open_clip_version") != "3.3.0":
        raise InputValidationError("Visual-anchor model registration is invalid.")
    expected_templates = ["{class}", "a photo of {class}", "an aerial image of {class}", "a satellite image of {class}", "a remote sensing image of {class}", "{class} in an aerial image", "{class} in a satellite image", "a region of {class} viewed from above"]
    strategy = protocol.get("strategy", {})
    if protocol.get("prompts", {}).get("group_a_templates") != expected_templates or strategy.get("fixed_text_weight") != 0.5 or strategy.get("fixed_visual_weight") != 0.5:
        raise InputValidationError("Visual-anchor prompts or fusion weights differ from the frozen design.")
    if any(protocol.get("constraints", {}).get(key) is not False for key in ("sam3_rerun", "training", "pixel_gt", "region_reencoding", "prompt_tuning", "alpha_tuning", "grid_search", "model_selection_or_decision", "overwrite")):
        raise InputValidationError("Visual-anchor constraints differ from the frozen protocol.")


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
            raise InputValidationError(f"Visual-anchor immutable input differs from protocol: {name}")
    try:
        cache_manifest = json.loads(Path(str(paths["feature_cache_manifest"])).read_text(encoding="utf-8"))
        split_manifest = json.loads(Path(str(paths["split_manifest"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Visual-anchor input manifest is invalid JSON.") from exc
    if cache_manifest.get("status") != "completed" or cache_manifest.get("scientific_evidence") is not False:
        raise InputValidationError("Feature cache is not the registered non-semantic completed cache.")
    if not isinstance(cache_manifest.get("preprocess"), str) or not cache_manifest["preprocess"]:
        raise InputValidationError("Feature-cache manifest lacks its frozen preprocessing representation.")
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
        raise InputValidationError("Visual-anchor text construction requires torch and open_clip.") from exc
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


def _metrics(predictions: np.ndarray, records: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    names = np.asarray([classes[int(index)] for index in predictions])
    sam3 = np.asarray([str(row.get("sam3_source_label", "")) for row in records])
    cam = np.asarray([str(row.get("cam_label", "")) for row in records])
    if any(name not in classes for name in sam3) or any(name not in classes for name in cam):
        raise InputValidationError("Split records require registered SAM3 and CAM labels.")
    per_class = {name: float(np.mean(names[sam3 == name] == name)) for name in classes}
    return {"sam3_agreement": float(np.mean(names == sam3)), "macro_sam3_agreement": float(np.mean(list(per_class.values()))), "cam_agreement": float(np.mean(names == cam)), "three_way_agreement": float(np.mean((names == sam3) & (names == cam))), "per_class_sam3_agreement": per_class, "prediction_counts": {name: int(np.sum(names == name)) for name in classes}}


def run_openai_clip_visual_anchor(cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run the one frozen no-gradient visual-anchor diagnostic over cached features."""
    _validate_protocol(protocol)
    anchor = cfg.get("repository_anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"code_commit", "protocol_sha256"} or anchor["protocol_sha256"] != protocol.get("sha256"):
        raise InputValidationError("Visual-anchor runner must supply the exact repository anchor.")
    checkpoint_value = cfg["paths"].get("openai_clip_checkpoint")
    if not checkpoint_value:
        raise InputValidationError("Tracked visual-anchor config has no checkpoint; use an ignored deployment config.")
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_file() or sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("OpenAI-CLIP checkpoint differs from the registered artifact.")
    destination = Path(output_dir)
    _assert_ordinary_path(destination, "output")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise InputValidationError("Visual-anchor output directory must be absent or empty.")
    features, development, heldout, input_hashes = _validate_hashed_inputs(cfg, protocol)
    try:
        import torch
    except ImportError as exc:
        raise InputValidationError("Visual-anchor runtime requires torch.") from exc
    requested = str(cfg.get("runtime", {}).get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    text, token_hash = _text_prototypes(protocol, checkpoint, device)
    visual, counts = build_visual_prototypes(features, development, list(protocol["classes"]))
    text_scores, fused_scores = fuse_scores(features, text, visual)
    baseline = np.argmax(text_scores, axis=1).astype(np.int16)
    fused = np.argmax(fused_scores, axis=1).astype(np.int16)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=False)
    array_path = destination / "arrays.npz"
    with array_path.open("xb") as handle:
        np.savez_compressed(handle, text_prototypes=text, visual_prototypes=visual, baseline_predictions=baseline, fused_predictions=fused)
    index_dev = np.asarray([int(row["row_index"]) for row in development], dtype=np.int64)
    index_hold = np.asarray([int(row["row_index"]) for row in heldout], dtype=np.int64)
    manifest = {
        "format_version": 1, "status": "completed", "scientific_evidence": False,
        "post_hoc_exploratory": True, "holdout_interpretation": protocol["holdout_interpretation"],
        "role": protocol["role"], "repository_anchor": anchor,
        "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "inputs": input_hashes, "model": protocol["model"], "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess": cache_manifest_preprocess(cfg),
        "device": device, "text_token_sha256": token_hash, "visual_prototype_counts": counts,
        "strategy": protocol["strategy"], "constraints": protocol["constraints"],
        "metrics": {"development": {"baseline_text": _metrics(baseline[index_dev], development, protocol["classes"]), "fused_visual_anchor": _metrics(fused[index_dev], development, protocol["classes"])}, "heldout": {"baseline_text": _metrics(baseline[index_hold], heldout, protocol["classes"]), "fused_visual_anchor": _metrics(fused[index_hold], heldout, protocol["classes"])}},
        "outputs": {"arrays": {"path": array_path.name, "sha256": sha256_file(array_path), "fields": ["text_prototypes", "visual_prototypes", "baseline_predictions", "fused_predictions"]}},
    }
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def cache_manifest_preprocess(cfg: dict[str, Any]) -> dict[str, str]:
    """Record the exact cache preprocessing representation without re-encoding a region."""
    try:
        value = json.loads(Path(str(cfg["paths"]["feature_cache_manifest"])).read_text(encoding="utf-8"))["preprocess"]
    except (OSError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Feature-cache manifest lacks its frozen preprocessing representation.") from exc
    text = str(value)
    return {"representation": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
