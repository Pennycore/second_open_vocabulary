"""Fail-closed VOC2012 image-only OpenAI-CLIP zero-shot score cache."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import InputValidationError, sha256_file
from .voc2012_external_data import validate_voc_val_image_path


_PROTOCOL_NAME = "voc2012_openai_clip_zeroshot_protocol_v1.json"
_VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor",
)
_GROUP_A = (
    "{class}", "a photo of {class}", "an aerial image of {class}", "a satellite image of {class}",
    "a remote sensing image of {class}", "{class} in an aerial image", "{class} in a satellite image", "a region of {class} viewed from above",
)


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
            raise InputValidationError(f"VOC zero-shot {label} may not traverse a symlink or junction.")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        raise InputValidationError(f"VOC zero-shot path must be project-contained and relative: {name}")
    resolved = (root / candidate).resolve()
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"VOC zero-shot path escapes project root: {name}")
    _assert_ordinary(resolved, name)
    return str(resolved)


def _normalize(rows: np.ndarray) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise InputValidationError("VOC zero-shot vectors must be finite two-dimensional arrays.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise InputValidationError("VOC zero-shot vectors may not contain zero rows.")
    return array / norms


def build_text_prototypes(prompt_features: np.ndarray, class_count: int = len(_VOC_CLASSES)) -> np.ndarray:
    """L2 -> class mean -> L2 for the frozen class/template order."""
    values = _normalize(prompt_features)
    if values.shape[0] != class_count * len(_GROUP_A):
        raise InputValidationError("Prompt feature count differs from the frozen classes/templates.")
    means = values.reshape(class_count, len(_GROUP_A), values.shape[1]).mean(axis=1)
    return _normalize(means)


def cosine_scores(image_features: np.ndarray, text_prototypes: np.ndarray) -> np.ndarray:
    images, texts = _normalize(image_features), _normalize(text_prototypes)
    if images.shape[1] != texts.shape[1]:
        raise InputValidationError("VOC zero-shot image/text feature dimensions differ.")
    return (images @ texts.T).astype(np.float32)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    dataset = protocol.get("dataset", {})
    model = protocol.get("model", {})
    if protocol.get("status") != "frozen_pre_result" or protocol.get("scientific_evidence") is not False:
        raise InputValidationError("VOC zero-shot protocol must be frozen and non-scientific.")
    if protocol.get("role") != "PASCAL VOC 2012 validation-image OpenAI-CLIP zero-shot score cache; input-only and no evaluation":
        raise InputValidationError("VOC zero-shot role differs from the frozen scope.")
    if dataset != {
        "name": "PASCAL VOC 2012", "split": "Segmentation val",
        "manifest_sha256": "8c2ec1b7a115f2d0b4892b8ef3dfa80d8a863488b449769c02e642295f965a6c",
        "archive_md5": "6cd6e144f989b92b3379bac3b3de84fd", "val_image_count": 1449,
    }:
        raise InputValidationError("VOC zero-shot dataset registration differs from the frozen values.")
    if protocol.get("classes") != list(_VOC_CLASSES) or protocol.get("prompts", {}).get("group_a_templates") != list(_GROUP_A):
        raise InputValidationError("VOC zero-shot classes or Group A templates differ from the frozen values.")
    if model.get("architecture") != "ViT-B-32-quickgelu" or model.get("checkpoint_sha256") != "9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e" or int(model.get("feature_dimension", -1)) != 512 or model.get("open_clip_version") != "3.3.0" or model.get("load_policy") != "weights_only=True, strict=True, eval mode":
        raise InputValidationError("VOC zero-shot model registration differs from the frozen values.")
    if protocol.get("scoring", {}).get("dtype") != "float32" or protocol.get("scoring", {}).get("shape") != [1449, 20]:
        raise InputValidationError("VOC zero-shot output scoring registration differs from the frozen values.")
    required_false = ("network_download", "semantic_label_read_or_decode", "detection_label_read_or_decode", "class_specific_file_read", "ground_truth_retrieval", "thresholds", "predictions", "metrics", "class_selection", "prompt_selection", "prototype_selection", "sam3_rerun", "training", "overwrite")
    if any(protocol.get("constraints", {}).get(name) is not False for name in required_false):
        raise InputValidationError("VOC zero-shot constraints differ from the frozen scope.")


def load_voc2012_openai_clip_zeroshot_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read VOC zero-shot configuration.") from exc
    required = {"dataset_manifest", "raw_data_root", "openai_clip_checkpoint", "protocol_file", "output_root"}
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False or not isinstance(cfg.get("paths"), dict) or set(cfg["paths"]) != required:
        raise InputValidationError("VOC zero-shot config paths do not match the frozen schema.")
    for key, value in list(cfg["paths"].items()):
        cfg["paths"][key] = _resolve_project_path(value, root, key)
    output_root = Path(str(cfg["paths"]["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("VOC zero-shot output_root must be directly under outputs/.")
    canonical = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(cfg["paths"]["protocol_file"])).resolve() != canonical:
        raise InputValidationError("VOC zero-shot protocol must be the committed canonical protocol.")
    try:
        raw = canonical.read_bytes()
        protocol = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read canonical VOC zero-shot protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("VOC zero-shot protocol must be a JSON object.")
    _validate_protocol(protocol)
    protocol["path"] = str(canonical)
    protocol["sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg, protocol


def verify_voc2012_openai_clip_zeroshot_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("VOC zero-shot repository anchor could not read git state.") from exc
    actual = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual != expected_protocol_sha256:
        raise InputValidationError("VOC zero-shot run requires the approved clean commit and protocol SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def _read_image_manifest(manifest_path: Path, raw_root: Path, protocol: dict[str, Any]) -> tuple[list[str], list[Path], str]:
    if sha256_file(manifest_path) != protocol["dataset"]["manifest_sha256"]:
        raise InputValidationError("VOC zero-shot dataset manifest hash differs from the frozen protocol.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("VOC zero-shot dataset manifest is not valid JSON.") from exc
    allowed_fields = {
        "format_version", "status", "scientific_evidence", "role", "repository_anchor", "protocol",
        "dataset", "archive", "raw_data_root", "val_split", "archive_jpeg_inventory",
        "model_input_files", "model_input_image_sha256_aggregate", "quarantined_annotation_roots", "constraints",
    }
    if set(manifest) != allowed_fields:
        raise InputValidationError("VOC zero-shot manifest has an unexpected or label-bearing schema field.")
    if manifest.get("status") != "completed" or manifest.get("scientific_evidence") is not False:
        raise InputValidationError("VOC zero-shot requires a completed non-scientific image manifest.")
    if manifest.get("archive", {}).get("md5") != protocol["dataset"]["archive_md5"]:
        raise InputValidationError("VOC zero-shot manifest archive MD5 differs from the frozen protocol.")
    external_constraints = manifest.get("constraints", {})
    label_barriers = ("semantic_label_read_or_decode", "detection_label_read_or_decode", "class_specific_file_read", "ground_truth_retrieval")
    if not isinstance(external_constraints, dict) or any(external_constraints.get(name) is not False for name in label_barriers):
        raise InputValidationError("VOC zero-shot manifest does not preserve the label-access quarantine.")
    files = manifest.get("model_input_files")
    if not isinstance(files, list) or len(files) != protocol["dataset"]["val_image_count"]:
        raise InputValidationError("VOC zero-shot manifest image count differs from the frozen protocol.")
    ids: list[str] = []
    paths: list[Path] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"image_id", "path", "sha256"}:
            raise InputValidationError("VOC zero-shot manifest image entries must be label-free image records.")
        image_id, rel, expected = str(entry["image_id"]), str(entry["path"]), str(entry["sha256"])
        if rel != f"JPEGImages/{image_id}.jpg" or len(expected) != 64:
            raise InputValidationError("VOC zero-shot manifest has an invalid JPEG record.")
        candidate = validate_voc_val_image_path(raw_root, raw_root / rel, image_id)
        if sha256_file(candidate) != expected:
            raise InputValidationError("VOC zero-shot JPEG differs from the immutable manifest inventory.")
        ids.append(image_id)
        paths.append(candidate)
    if len(ids) != len(set(ids)):
        raise InputValidationError("VOC zero-shot manifest contains duplicate image IDs.")
    return ids, paths, str(manifest.get("model_input_image_sha256_aggregate", ""))


def _load_model(checkpoint: Path, protocol: dict[str, Any], device: str) -> tuple[Any, Any, Any, str]:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise InputValidationError("VOC zero-shot requires torch and open_clip.") from exc
    if getattr(open_clip, "__version__", None) != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the frozen VOC zero-shot protocol.")
    model, _, preprocess = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if not isinstance(state, dict):
        raise InputValidationError("VOC zero-shot checkpoint must contain a state dictionary.")
    model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.eval().to(device)
    return model, open_clip.get_tokenizer(protocol["model"]["architecture"]), preprocess, repr(preprocess)


def _encode_text(model: Any, tokenizer: Any, protocol: dict[str, Any], device: str) -> tuple[np.ndarray, str]:
    import torch

    prompts = [template.format(**{"class": label}) for label in _VOC_CLASSES for template in _GROUP_A]
    tokens = tokenizer(prompts)
    token_hash = hashlib.sha256(tokens.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    with torch.inference_mode():
        values = model.encode_text(tokens.to(device)).float().cpu().numpy()
    return build_text_prototypes(values), token_hash


def _encode_images(model: Any, preprocess: Any, paths: list[Path], batch_size: int, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    encoded = np.empty((len(paths), 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            values = model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
            encoded[start:start + len(tensors)] = _normalize(values)
    return encoded


def _exclusive_npy(path: Path, values: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        np.save(handle, values, allow_pickle=False)


def run_voc2012_openai_clip_zeroshot(cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    _validate_protocol(protocol)
    paths = cfg["paths"]
    if not paths.get("dataset_manifest") or not paths.get("raw_data_root") or not paths.get("openai_clip_checkpoint"):
        raise InputValidationError("Tracked VOC zero-shot config is non-runnable; deployment paths are required.")
    manifest_path, raw_root, checkpoint = Path(str(paths["dataset_manifest"])), Path(str(paths["raw_data_root"])), Path(str(paths["openai_clip_checkpoint"]))
    if not manifest_path.is_file() or not raw_root.is_dir() or not checkpoint.is_file():
        raise InputValidationError("VOC zero-shot deployment inputs are unavailable.")
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("VOC zero-shot checkpoint differs from the frozen artifact.")
    image_ids, image_paths, aggregate = _read_image_manifest(manifest_path, raw_root, protocol)
    import torch
    requested = str(cfg.get("runtime", {}).get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    batch_size = int(cfg.get("runtime", {}).get("batch_images", 0))
    if batch_size != 32:
        raise InputValidationError("VOC zero-shot batch_images must remain frozen at 32.")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise InputValidationError("VOC zero-shot output directory must be empty.")
    destination.mkdir(parents=True, exist_ok=True)
    anchor = cfg.get("repository_anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"code_commit", "protocol_sha256"} or anchor["protocol_sha256"] != protocol.get("sha256"):
        raise InputValidationError("VOC zero-shot runner must supply the exact repository anchor.")
    model, tokenizer, preprocess, preprocess_repr = _load_model(checkpoint, protocol, device)
    text, token_hash = _encode_text(model, tokenizer, protocol, device)
    images = _encode_images(model, preprocess, image_paths, batch_size, device)
    scores = cosine_scores(images, text)
    if scores.shape != (1449, 20) or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise InputValidationError("VOC zero-shot scores have an invalid shape, dtype, or value.")
    score_path, ids_path, stats_path = (destination / protocol["output"][key] for key in ("score_file", "image_id_file", "image_feature_stats_file"))
    _exclusive_npy(score_path, scores)
    with ids_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(image_ids, handle, ensure_ascii=False)
        handle.write("\n")
    stats = {"dtype": "float32", "shape": list(images.shape), "norm_min": float(np.linalg.norm(images, axis=1).min()), "norm_max": float(np.linalg.norm(images, axis=1).max()), "norm_mean": float(np.linalg.norm(images, axis=1).mean())}
    with stats_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "format_version": 1, "status": "completed", "scientific_evidence": False, "role": protocol["role"],
        "repository_anchor": anchor, "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "dataset_manifest_sha256": sha256_file(manifest_path), "archive_md5": protocol["dataset"]["archive_md5"], "image_count": len(image_ids), "image_inventory_aggregate_sha256": aggregate,
        "model": protocol["model"], "checkpoint_sha256": sha256_file(checkpoint), "device": device, "preprocess": preprocess_repr, "prompt_token_sha256": token_hash,
        "outputs": {"scores": {"path": score_path.name, "sha256": sha256_file(score_path), "dtype": "float32", "shape": [1449, 20]}, "image_ids": {"path": ids_path.name, "sha256": sha256_file(ids_path), "count": 1449}, "image_feature_stats": {"path": stats_path.name, "sha256": sha256_file(stats_path)}},
        "constraints": protocol["constraints"],
    }
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
