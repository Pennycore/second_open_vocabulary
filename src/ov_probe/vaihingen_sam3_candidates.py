"""Fail-closed SAM3 candidate-cache generation for the frozen Vaihingen protocol.

This module owns only proposal generation.  It never imports CTP, reads GT, or
derives support from labels.  At runtime it uses a read-only copy of the
first-paper SAM3 backend and its ``candidate_cache.save_candidate_cache`` writer
so the resulting ``.npz + .json`` pairs remain wire-compatible with the existing
pixel/RemoteCLIP evaluators.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io import InputValidationError, sha256_file


CHECKPOINT_PLACEHOLDER = "UNSET_REPLACE_WITH_VERIFIED_SHA256_BEFORE_RUN"
DTYPE_GUARD = "first_paper_fp32_input_hooks"
REQUIRED_CACHE_KEYS = {
    "format_version", "image_shape", "packed_masks", "offsets", "shapes",
    "origins", "boxes", "areas", "scores", "class_ids", "prompt_ids",
}


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Preflight:
    status: str
    config: dict[str, Any]
    protocol: dict[str, Any]
    errors: list[str]
    manifest: dict[str, Any]


def enumerate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Use the first-paper shift-at-edge tiling convention without importing SAM3."""
    if width <= 0 or height <= 0:
        raise InputValidationError("Image width and height must be positive.")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise InputValidationError("tile_overlap must be in [0, tile_size).")
    stride = tile_size - overlap

    def positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        result = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if result[-1] != last:
            result.append(last)
        return result

    return [
        Tile(x0=x0, y0=y0, x1=min(x0 + tile_size, width), y1=min(y0 + tile_size, height))
        for y0 in positions(height) for x0 in positions(width)
    ]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def _resolve_relative(root: Path, value: Any, key: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise InputValidationError(f"paths.{key} must be project-relative, not absolute.")
    return (root / path).resolve()


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import yaml

    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise InputValidationError("Candidate config must be a YAML mapping.")
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("experiment.overwrite must remain false.")
    root = _project_root(config_path)
    paths = cfg.get("paths")
    required = {
        "protocol_file", "image_dir", "label_dir", "sam3_python_root", "sam3_repo",
        "sam3_checkpoint", "output_root",
    }
    if not isinstance(paths, dict):
        raise InputValidationError("Candidate config requires a paths mapping.")
    missing = sorted(required - set(paths))
    if missing:
        raise InputValidationError(f"Candidate config lacks paths: {missing}")
    resolved: dict[str, str] = {}
    for key in required:
        resolved[key] = str(_resolve_relative(root, paths[key], key))
    output_root = Path(resolved["output_root"])
    allowed = (root / "outputs").resolve()
    if output_root == allowed or allowed not in output_root.parents:
        raise InputValidationError("paths.output_root must be a named child of project outputs/.")
    for key, value in resolved.items():
        if key == "output_root":
            continue
        source = Path(value)
        if source == output_root or output_root in source.parents or source in output_root.parents:
            raise InputValidationError(f"Output/input path overlap is forbidden for paths.{key}.")
    cfg["paths"] = resolved
    protocol_path = Path(resolved["protocol_file"])
    if not protocol_path.is_file():
        raise InputValidationError(f"Protocol file is missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol_binding(cfg, protocol)
    return cfg, protocol, root


def _validate_protocol_binding(cfg: dict[str, Any], protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_pre_run":
        raise InputValidationError("Candidate protocol must be frozen_pre_run.")
    proposal = protocol.get("proposal", {})
    expected = {
        "tile_size": 512, "tile_overlap": 128, "score_threshold": 0.55,
        "mask_threshold": 0.5, "min_mask_area": 32, "max_mask_area_ratio": 0.95,
        "conflict_margin": 0.03, "rgb_band_indices": [0, 1, 2],
    }
    if any(proposal.get(key) != value for key, value in expected.items()):
        raise InputValidationError("Proposal parameters differ from the frozen Vaihingen SAM3 protocol.")
    prompting = protocol.get("prompting", {})
    if prompting != {"style": "remoteclip_b2c", "include_manual_prompts": True, "max_prompts_per_class": 4}:
        raise InputValidationError("Prompting protocol differs from the frozen first-paper settings.")
    classes = protocol.get("classes", [])
    names = [item.get("name") for item in classes if isinstance(item, dict)]
    if names != ["impervious_surface", "building", "low_vegetation", "tree", "car"]:
        raise InputValidationError("Candidate protocol must retain the five registered Vaihingen classes.")
    if any(len(item.get("prompts", [])) != 4 for item in classes if isinstance(item, dict)):
        raise InputValidationError("Each class must retain exactly four frozen manual prompts.")
    configured = str(cfg.get("integrity", {}).get("sam3_checkpoint_sha256", ""))
    if configured != proposal.get("checkpoint_sha256"):
        raise InputValidationError("Config/protocol SAM3 checkpoint hash binding differs.")


def _input_entry(path: Path, hash_file: bool) -> dict[str, Any]:
    entry = {"path": str(path), "exists": path.exists(), "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing"}
    if hash_file and path.is_file():
        entry["sha256"] = sha256_file(path)
    return entry


def _preflight(config_path: str | Path, require_checkpoint_hash: bool = True) -> Preflight:
    cfg, protocol, root = _load_config(config_path)
    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    errors: list[str] = []
    for key in ("image_dir", "label_dir", "sam3_python_root", "sam3_repo"):
        if not paths[key].is_dir():
            errors.append(f"Required directory missing: {key}={paths[key]}")
    for key in ("sam3_checkpoint",):
        if not paths[key].is_file():
            errors.append(f"Required file missing: {key}={paths[key]}")
    source_files = [
        paths["sam3_python_root"] / "sam3_remote_wsss" / "candidate_cache.py",
        paths["sam3_python_root"] / "sam3_remote_wsss" / "sam3_backend.py",
        paths["sam3_python_root"] / "sam3_remote_wsss" / "fusion.py",
        paths["sam3_python_root"] / "sam3_remote_wsss" / "prompts.py",
    ]
    for source in source_files:
        if paths["sam3_python_root"].is_dir() and not source.is_file():
            errors.append(f"Read-only first-paper SAM3 source is incomplete: {source}")
    ids = list(protocol["input_contract"]["image_ids"])
    image_pattern = str(protocol["input_contract"]["image_filename_pattern"])
    label_pattern = str(protocol["input_contract"]["label_filename_pattern"])
    image_paths = [paths["image_dir"] / image_pattern.format(image_id=image_id) for image_id in ids]
    label_paths = [paths["label_dir"] / label_pattern.format(image_id=image_id) for image_id in ids]
    missing_images = [path.name for path in image_paths if not path.is_file()]
    missing_labels = [path.name for path in label_paths if not path.is_file()]
    if missing_images:
        errors.append(f"Required region-level image TIFFs are missing: {missing_images}")
    if missing_labels:
        errors.append(f"Required region-level label TIFFs are missing: {missing_labels}")
    expected_hash = str(cfg["integrity"]["sam3_checkpoint_sha256"])
    checkpoint_entry = _input_entry(paths["sam3_checkpoint"], hash_file=paths["sam3_checkpoint"].is_file() and expected_hash != CHECKPOINT_PLACEHOLDER)
    if require_checkpoint_hash:
        if expected_hash == CHECKPOINT_PLACEHOLDER:
            errors.append("SAM3 checkpoint SHA-256 is unresolved; replace the frozen placeholder before execution.")
        elif paths["sam3_checkpoint"].is_file() and checkpoint_entry.get("sha256") != expected_hash:
            errors.append("SAM3 checkpoint SHA-256 differs from the frozen deployment contract.")
    environment = {"python": platform.python_version(), "cuda_checked": False}
    manifest = {
        "format_version": 1,
        "status": "ready" if not errors else "blocked",
        "scientific_evidence": False,
        "project_root": str(root),
        "config_sha256": sha256_file(Path(config_path).resolve()),
        "protocol_sha256": sha256_file(paths["protocol_file"]),
        "protocol_content_sha256": _sha256_json(protocol),
        "inputs": {
            "images": {"required_count": len(ids), "paths": [str(path) for path in image_paths], "missing": missing_images},
            # Existence-only binding is deliberate: never hash or decode GT labels here.
            "labels": {"required_count": len(ids), "paths": [str(path) for path in label_paths], "missing": missing_labels, "read": False, "hashed": False},
            "sam3_checkpoint": checkpoint_entry,
            "sam3_python_root": _input_entry(paths["sam3_python_root"], hash_file=False),
            "sam3_repo": _input_entry(paths["sam3_repo"], hash_file=False),
        },
        "environment": environment,
        "errors": errors,
    }
    return Preflight("ready" if not errors else "blocked", cfg, protocol, errors, manifest)


def _load_runtime(preflight: Preflight) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in preflight.config["paths"].items()}
    root = str(paths["sam3_python_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from sam3_remote_wsss.candidate_cache import CandidateMask, save_candidate_cache
        from sam3_remote_wsss.config import ClassSpec, PromptingConfig
        from sam3_remote_wsss.fusion import filter_masks
        from sam3_remote_wsss.prompts import prompts_for_class
        from sam3_remote_wsss.sam3_backend import SAM3ImageBackend, _install_fp32_dtype_hooks
    except Exception as exc:
        raise InputValidationError(f"Cannot load read-only SAM3 backend: {type(exc).__name__}: {exc}") from exc
    return {
        "CandidateMask": CandidateMask, "save_candidate_cache": save_candidate_cache,
        "ClassSpec": ClassSpec, "PromptingConfig": PromptingConfig,
        "filter_masks": filter_masks, "prompts_for_class": prompts_for_class,
        "SAM3ImageBackend": SAM3ImageBackend,
        "install_fp32_dtype_hooks": _install_fp32_dtype_hooks,
    }


def _build_backend(runtime: dict[str, Any], paths: dict[str, Path], proposal: dict[str, Any]) -> Any:
    """Construct the read-only backend and apply its existing FP32 input guard."""
    backend = runtime["SAM3ImageBackend"](
        paths["sam3_repo"], paths["sam3_checkpoint"],
        device=str(proposal["device"]), confidence_threshold=float(proposal["score_threshold"]),
    )
    # This is the first-paper helper, applied only at the second-paper runtime
    # boundary to prevent BF16 SAM3 activations reaching FP32 ViTDet weights.
    runtime["install_fp32_dtype_hooks"](backend.model)
    return backend


def _read_rgb(path: Path, bands: list[int]) -> np.ndarray:
    import tifffile

    array = tifffile.imread(path)
    if array.ndim != 3 or array.shape[2] <= max(bands):
        raise InputValidationError(f"Expected HxWxC region-level TIFF with requested RGB bands: {path}")
    rgb = np.asarray(array[:, :, bands])
    if rgb.dtype != np.uint8:
        low, high = float(np.percentile(rgb, 1)), float(np.percentile(rgb, 99))
        rgb = np.clip((rgb.astype(np.float32) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _validate_candidate_cache_schema(cache_dir: Path, image_id: str, image_shape: tuple[int, int]) -> dict[str, Any]:
    data_path, metadata_path = cache_dir / f"{image_id}.npz", cache_dir / f"{image_id}.json"
    if not data_path.is_file() or not metadata_path.is_file():
        raise InputValidationError(f"Candidate writer did not create a complete .npz + .json pair for {image_id}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("image_id") != image_id or metadata.get("image_shape") != list(image_shape):
        raise InputValidationError(f"Candidate cache metadata schema mismatch for {image_id}.")
    with np.load(data_path, allow_pickle=False) as archive:
        if not REQUIRED_CACHE_KEYS.issubset(set(archive.files)):
            raise InputValidationError(f"Candidate cache array schema mismatch for {image_id}.")
        if int(archive["format_version"][0]) != 1 or archive["image_shape"].tolist() != list(image_shape):
            raise InputValidationError(f"Candidate cache data binding mismatch for {image_id}.")
        count = int(np.asarray(archive["scores"]).size)
        offsets = np.asarray(archive["offsets"])
        if offsets.size != count + 1 or int(metadata.get("candidate_count", -1)) != count:
            raise InputValidationError(f"Candidate cache count mismatch for {image_id}.")
    return {"npz_sha256": sha256_file(data_path), "json_sha256": sha256_file(metadata_path), "candidate_count": count}


def _new_run_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    path = output_root / name
    path.mkdir(mode=0o755, exist_ok=False)
    return path


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise InputValidationError(f"Refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)  # create-only publication; fails rather than replacing an existing target
    finally:
        if temporary.exists():
            temporary.unlink()


def _commit_hash() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _run(preflight: Preflight) -> Path:
    if preflight.status != "ready":
        raise InputValidationError("Candidate generation is blocked by preflight; no SAM3 import was attempted.")
    paths = {key: Path(value) for key, value in preflight.config["paths"].items()}
    protocol, proposal = preflight.protocol, preflight.protocol["proposal"]
    runtime = _load_runtime(preflight)
    try:
        import torch
        if not torch.cuda.is_available():
            raise InputValidationError("CUDA is unavailable; refusing to run SAM3 on an unintended device.")
        required_gpu = str(preflight.config.get("runtime", {}).get("required_gpu_substring", ""))
        gpu = torch.cuda.get_device_name(0)
        if required_gpu and required_gpu.lower() not in gpu.lower():
            raise InputValidationError(f"Required GPU '{required_gpu}' not found; detected '{gpu}'.")
    except ImportError as exc:
        raise InputValidationError("torch is required for SAM3 candidate generation.") from exc
    run_dir = _new_run_dir(paths["output_root"], str(preflight.config["experiment"].get("run_prefix", "run_")))
    stage = run_dir / ".staging" / "candidates"
    stage.mkdir(parents=True, exist_ok=False)
    started = dict(preflight.manifest)
    started.update({
        "status": "running", "run_dir": str(run_dir), "code_commit": _commit_hash(),
        "runtime_compatibility": {"dtype_guard": DTYPE_GUARD},
        "environment": {**preflight.manifest["environment"], "cuda_checked": True, "gpu": gpu},
    })
    _write_json_exclusive(run_dir / "manifest.json", started)
    try:
        specs = [runtime["ClassSpec"](id=int(item["id"]), name=str(item["name"]), label_color=tuple(item["label_color"]), prompts=tuple(item["prompts"])) for item in protocol["classes"]]
        prompting = runtime["PromptingConfig"](**protocol["prompting"])
        prompts = {spec.name: runtime["prompts_for_class"](spec, prompting) for spec in specs}
        expected_prompts = {item["name"]: item["prompts"] for item in protocol["classes"]}
        if prompts != expected_prompts:
            raise InputValidationError("Read-only prompt backend produced prompts different from the frozen protocol.")
        backend = _build_backend(runtime, paths, proposal)
        candidate_files: dict[str, Any] = {}
        for image_id in protocol["input_contract"]["image_ids"]:
            image_path = paths["image_dir"] / protocol["input_contract"]["image_filename_pattern"].format(image_id=image_id)
            image = _read_rgb(image_path, list(proposal["rgb_band_indices"]))
            height, width = image.shape[:2]
            candidates = []
            for tile in enumerate_tiles(width, height, int(proposal["tile_size"]), int(proposal["tile_overlap"])):
                tile_image = image[tile.y0:tile.y1, tile.x0:tile.x1]
                jobs = [(spec, prompt) for spec in specs for prompt in prompts[spec.name]]
                outputs = backend.predict_texts(tile_image, [prompt for _, prompt in jobs])
                if len(outputs) != len(jobs):
                    raise InputValidationError(f"SAM3 returned {len(outputs)} prompt outputs for {len(jobs)} jobs on {image_id}.")
                for (spec, prompt), output in zip(jobs, outputs):
                    kept = runtime["filter_masks"](output["masks"], output["scores"], float(proposal["score_threshold"]), int(proposal["min_mask_area"]), float(proposal["max_mask_area_ratio"]))
                    candidates.extend(runtime["CandidateMask"](class_id=spec.id, class_name=spec.name, prompt=prompt, score=score, mask=mask, x0=tile.x0, y0=tile.y0) for mask, score in kept)
            runtime["save_candidate_cache"](stage, image_id, (height, width), candidates, provenance={
                "protocol_sha256": preflight.manifest["protocol_sha256"], "image_sha256": sha256_file(image_path),
                "sam3_checkpoint_sha256": preflight.config["integrity"]["sam3_checkpoint_sha256"],
                "tile_size": proposal["tile_size"], "tile_overlap": proposal["tile_overlap"],
                "score_threshold": proposal["score_threshold"], "min_mask_area": proposal["min_mask_area"],
                "max_mask_area_ratio": proposal["max_mask_area_ratio"], "labels_read": False,
                "dtype_guard": DTYPE_GUARD,
            })
            candidate_files[image_id] = _validate_candidate_cache_schema(stage, image_id, (height, width))
        final_candidates = run_dir / "candidates"
        if final_candidates.exists():
            raise InputValidationError(f"Refusing to overwrite existing candidate directory: {final_candidates}")
        os.rename(stage, final_candidates)  # atomic directory publication inside the unique run directory
        stage.parent.rmdir()
        complete = dict(started)
        complete.update({"status": "completed", "scientific_evidence": True, "labels_read": False, "candidate_directory": str(final_candidates), "candidates": candidate_files})
        _write_json_exclusive(run_dir / "completed_manifest.json", complete)
        return run_dir
    except Exception:
        failure = {"status": "failed", "scientific_evidence": False, "labels_read": False, "error": traceback.format_exc()}
        _write_json_exclusive(run_dir / "failure.json", failure)
        raise


def dry_run_manifest(config_path: str | Path) -> dict[str, Any]:
    """Return a GT-free, SAM3-free preflight manifest for CLI/tests."""
    return _preflight(config_path, require_checkpoint_hash=True).manifest


__all__ = [
    "CHECKPOINT_PLACEHOLDER", "DTYPE_GUARD", "Preflight", "Tile", "_build_backend", "_load_config", "_preflight",
    "_run", "_validate_candidate_cache_schema", "dry_run_manifest", "enumerate_tiles",
]
