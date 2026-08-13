from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from .io import InputValidationError, sha256_file
from .pixel_pack import validate_region_pixel_pack


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise InputValidationError("Bridge features contain invalid or zero rows.")
    return values / norms


def load_bridge_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise InputValidationError("Bridge config must be a mapping.")
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Bridge outputs must never overwrite existing runs.")
    for key, value in cfg["paths"].items():
        candidate = Path(str(value))
        cfg["paths"][key] = str((root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    output_root = Path(cfg["paths"]["output_root"])
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("Bridge output must be a named directory directly under project outputs/.")
    for key in ("pixel_pack", "checkpoint", "protocol_file"):
        source = Path(cfg["paths"][key])
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise InputValidationError(f"Bridge {key} must remain inside the project root.") from exc
    protocol_path = Path(cfg["paths"]["protocol_file"])
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol["path"] = str(protocol_path)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_bridge_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual_commit != expected_commit:
        raise InputValidationError("Bridge code commit differs from the externally approved commit.")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if status.strip():
        raise InputValidationError("Tracked bridge worktree must be clean.")
    protocol_path = root / "configs" / "remoteclip_bridge_protocol_v0.json"
    actual_protocol = sha256_file(protocol_path)
    if actual_protocol != expected_protocol_sha256:
        raise InputValidationError("Bridge protocol differs from the externally approved SHA-256.")
    return {"code_commit": actual_commit, "protocol_sha256": actual_protocol}


def _iter_pixel_views(package: Path) -> Iterator[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    shard_names = sorted(name for name in manifest["artifacts"] if name.startswith("shards/"))
    for shard_name in shard_names:
        with np.load(package / shard_name, allow_pickle=False) as archive:
            rows = np.asarray(archive["row_indices"], dtype=np.int32)
            shapes = np.asarray(archive["crop_shapes"], dtype=np.int32)
            rgb_offsets = np.asarray(archive["crop_rgb_offsets"], dtype=np.int64)
            rgb_flat = np.asarray(archive["crop_rgb_flat"], dtype=np.uint8)
            mask_offsets = np.asarray(archive["crop_mask_offsets"], dtype=np.int64)
            mask_bits = np.asarray(archive["crop_mask_bits"], dtype=np.uint8)
        contexts: list[np.ndarray] = []
        masked_views: list[np.ndarray] = []
        for index, (height, width) in enumerate(shapes):
            start, end = int(rgb_offsets[index]), int(rgb_offsets[index + 1])
            context = rgb_flat[start:end].reshape(int(height), int(width), 3).copy()
            mask_start, mask_end = int(mask_offsets[index]), int(mask_offsets[index + 1])
            mask = np.unpackbits(
                mask_bits[mask_start:mask_end], bitorder="little", count=int(height * width)
            ).reshape(int(height), int(width)).astype(bool)
            masked = context.astype(np.float32)
            masked[~mask] *= 0.25
            masked = np.rint(masked).clip(0, 255).astype(np.uint8)
            contexts.append(context)
            masked_views.append(masked)
        yield rows, contexts, masked_views


def run_remoteclip_bridge(
    cfg: dict[str, Any], protocol: dict[str, Any], repository_anchor: dict[str, str]
) -> tuple[dict[str, Any], np.ndarray]:
    import open_clip
    import torch
    from PIL import Image

    package = Path(cfg["paths"]["pixel_pack"])
    checkpoint = Path(cfg["paths"]["checkpoint"])
    registered_pack = protocol["pixel_pack"]
    package_validation = validate_region_pixel_pack(
        package, package / "encoder_compare_protocol_v0.json"
    )
    for key in ("bundle_id", "record_count", "image_count", "ordered_record_key_sha256"):
        if package_validation[key] != registered_pack[key]:
            raise InputValidationError(f"Pixel pack differs from bridge registration: {key}")
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("RemoteCLIP checkpoint hash differs from bridge registration.")
    if open_clip.__version__ != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from bridge registration.")
    requested = str(cfg["model"].get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    architecture = protocol["model"]["architecture"]
    model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    reference = _normalize(np.load(package / "reference_region_features.npy", allow_pickle=False))
    reproduced = np.empty_like(reference, dtype=np.float32)
    batch_regions = int(cfg["model"]["batch_regions"])
    with torch.inference_mode():
        for rows, contexts, masked_views in _iter_pixel_views(package):
            for start in range(0, len(rows), batch_regions):
                stop = min(start + batch_regions, len(rows))
                images = []
                for context, masked in zip(contexts[start:stop], masked_views[start:stop]):
                    images.extend([preprocess(Image.fromarray(context)), preprocess(Image.fromarray(masked))])
                batch = torch.stack(images).to(device)
                encoded = model.encode_image(batch).float()
                encoded = encoded / encoded.norm(dim=1, keepdim=True)
                encoded = encoded.reshape(-1, 2, encoded.shape[-1]).mean(dim=1)
                encoded = encoded / encoded.norm(dim=1, keepdim=True)
                reproduced[rows[start:stop]] = encoded.cpu().numpy()
    reproduced = _normalize(reproduced)
    cosine = np.sum(reference * reproduced, axis=1)
    norms = np.linalg.norm(reproduced, axis=1)
    metrics = {
        "record_count": int(len(cosine)),
        "mean_cosine": float(cosine.mean()),
        "p01_cosine": float(np.quantile(cosine, 0.01)),
        "minimum_cosine": float(cosine.min()),
        "fraction_cosine_ge_0_999": float(np.mean(cosine >= 0.999)),
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "mean_l2_error": float(np.linalg.norm(reference - reproduced, axis=1).mean()),
    }
    thresholds = protocol["thresholds"]
    checks = {
        "mean_cosine": metrics["mean_cosine"] >= thresholds["mean_cosine_min"],
        "p01_cosine": metrics["p01_cosine"] >= thresholds["p01_cosine_min"],
        "minimum_cosine": metrics["minimum_cosine"] >= thresholds["minimum_cosine_min"],
        "fraction_cosine_ge_0_999": metrics["fraction_cosine_ge_0_999"] >= thresholds["fraction_cosine_ge_0_999_min"],
        "maximum_norm_error": metrics["maximum_norm_error"] <= thresholds["maximum_norm_error"],
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "scientific_evidence": False,
        "role": "RemoteCLIP environment bridge gate; no model comparison",
        "repository_anchor": repository_anchor,
        "pixel_pack_validation": package_validation,
        "checkpoint_sha256": sha256_file(checkpoint),
        "architecture": architecture,
        "device": device,
        "preprocess": repr(preprocess),
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "constraints": protocol["constraints"],
    }
    return summary, cosine.astype(np.float32)

