from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .io import InputValidationError, load_config, sha256_file, write_json
from .native_region import (
    NativeCandidate,
    candidate_cache_fingerprint,
    load_native_candidate_cache,
    load_native_region_score,
)


PIXEL_PACK_FORMAT_VERSION = 1
_EXPECTED_PROTOCOL_NAME = "encoder_compare_protocol_v0.json"
_EMBEDDED_PROTOCOL_NAME = "encoder_compare_protocol_v0.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"loveda_train_(?:rural|urban)_\d+")
_REQUIRED_SHARD_ARRAYS = {
    "format_version",
    "row_indices",
    "crop_shapes",
    "crop_boxes",
    "crop_rgb_offsets",
    "crop_rgb_flat",
    "crop_mask_offsets",
    "crop_mask_bits",
}
_REQUIRED_RECORD_FIELDS = {
    "row_index",
    "image_id",
    "candidate_index",
    "sam3_source_label",
    "cam_label",
    "image_shape",
    "crop_shape",
    "crop_box",
    "mask_area",
    "mask_fraction",
    "image_sha256",
    "candidate_cache_sha256",
    "region_cache_sha256",
    "context_sha256",
    "crop_mask_sha256",
    "masked_view_sha256",
    "shard",
}


@dataclass(frozen=True)
class SelectedRecord:
    row_index: int
    image_id: str
    candidate_index: int
    sam3_source_label: str
    cam_label: str


@dataclass(frozen=True)
class PixelViews:
    context: np.ndarray
    crop_mask: np.ndarray
    masked: np.ndarray
    crop_box: tuple[int, int, int, int]
    mask_fraction: float


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_utf8_text_bytes(value: bytes, label: str) -> bytes:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"{label} must be UTF-8 text.") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _assert_no_link_components(path: Path) -> None:
    absolute = path.absolute()
    chain = [absolute, *absolute.parents]
    for component in reversed(chain):
        if not component.exists():
            continue
        info = os.lstat(component)
        is_reparse = bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        is_junction = bool(
            hasattr(component, "is_junction") and component.is_junction()
        )
        if stat.S_ISLNK(info.st_mode) or is_reparse or is_junction:
            raise InputValidationError(
                f"Symlink/junction/reparse paths are forbidden: {component}"
            )


def _stable_read_bytes(path: Path) -> bytes:
    _assert_no_link_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputValidationError(f"Cannot safely open source file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InputValidationError(f"Source must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(value) != before.st_size:
        raise InputValidationError(f"Source changed while it was read: {path}")
    return value


def _stable_sha256_file(path: Path) -> str:
    _assert_no_link_components(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputValidationError(f"Cannot safely open source file: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InputValidationError(f"Source must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise InputValidationError(f"Source changed while it was hashed: {path}")
    return digest.hexdigest()


def _safe_source_relative(root: Path, value: Any, label: str) -> Path:
    text = str(value)
    if not text or "\\" in text:
        raise InputValidationError(f"Registered {label} must be a POSIX relative path.")
    relative = Path(*text.split("/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise InputValidationError(f"Unsafe registered relative path for {label}: {text!r}")
    root_resolved = root.resolve()
    result = (root_resolved / relative).resolve()
    try:
        result.relative_to(root_resolved)
    except ValueError as exc:
        raise InputValidationError(f"Registered {label} escapes its source root.") from exc
    _assert_no_link_components(result)
    return result


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_key_sha256(records: Sequence[SelectedRecord]) -> str:
    keys = [f"{record.image_id}:{record.candidate_index}" for record in records]
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def read_selected_records(
    path: str | Path, protocol: dict[str, Any]
) -> list[SelectedRecord]:
    source = Path(path)
    expected = protocol["selection"]
    source_bytes = _stable_read_bytes(source)
    if _sha256_bytes(source_bytes) != expected["selected_records_sha256"]:
        raise InputValidationError("Selected-record file SHA-256 is not registered.")
    records: list[SelectedRecord] = []
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputValidationError("Selected-record file must be UTF-8.") from exc
    with io.StringIO(source_text) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    f"Malformed selected-record JSON at line {line_number}."
                ) from exc
            if not isinstance(value, dict):
                raise InputValidationError("Selected-record rows must be JSON objects.")
            try:
                records.append(
                    SelectedRecord(
                        row_index=int(value["row_index"]),
                        image_id=str(value["image_id"]),
                        candidate_index=int(value["candidate_index"]),
                        sam3_source_label=str(value["sam3_source_label"]),
                        cam_label=str(value["cam_label"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise InputValidationError(
                    f"Selected-record schema error at line {line_number}."
                ) from exc
    expected_count = int(expected["record_count"])
    if len(records) != expected_count:
        raise InputValidationError(
            f"Selected-record count mismatch: {len(records)} != {expected_count}."
        )
    if [record.row_index for record in records] != list(range(expected_count)):
        raise InputValidationError("Selected row_index must be contiguous and ordered.")
    keys = [(record.image_id, record.candidate_index) for record in records]
    if any(
        not _IMAGE_ID.fullmatch(record.image_id) or record.candidate_index < 0
        for record in records
    ):
        raise InputValidationError("Selected records contain an invalid image/region key.")
    if len(keys) != len(set(keys)):
        raise InputValidationError("Selected records contain duplicate region keys.")
    if ordered_key_sha256(records) != expected["ordered_record_key_sha256"]:
        raise InputValidationError("Selected ordered region-key SHA-256 is invalid.")
    if len({record.image_id for record in records}) != int(expected["image_count"]):
        raise InputValidationError("Selected unique image count is invalid.")
    counts = Counter(record.sam3_source_label for record in records)
    if dict(counts) != expected["class_counts"]:
        raise InputValidationError(
            f"Selected SAM3 source counts are invalid: {dict(counts)}."
        )
    return records


def _centered_bounds(center: float, size: int, limit: int) -> tuple[int, int]:
    bounded_size = min(int(size), int(limit))
    start = int(np.floor(center - bounded_size / 2))
    start = max(0, min(start, limit - bounded_size))
    return start, start + bounded_size


def make_pixel_views(
    image_rgb: np.ndarray,
    candidate: NativeCandidate,
    *,
    context_ratio: float = 0.25,
    min_crop_size: int = 48,
    background_retain: float = 0.25,
) -> PixelViews:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise InputValidationError("Pixel pack requires an HxWx3 uint8 RGB image.")
    if context_ratio < 0 or min_crop_size <= 0 or not 0 <= background_retain <= 1:
        raise InputValidationError("Invalid registered crop parameters.")
    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise InputValidationError("Candidate mask must be a non-empty 2D array.")
    height, width = mask.shape
    image_height, image_width = image_rgb.shape[:2]
    x0, y0 = int(candidate.x0), int(candidate.y0)
    if x0 < 0 or y0 < 0 or x0 + width > image_width or y0 + height > image_height:
        raise InputValidationError("Candidate bounds exceed the source image.")
    ys, xs = np.nonzero(mask)
    left, top = x0 + int(xs.min()), y0 + int(ys.min())
    right, bottom = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
    target_size = max(right - left, bottom - top)
    crop_size = max(
        int(min_crop_size), int(np.ceil(target_size * (1 + 2 * context_ratio)))
    )
    crop_left, crop_right = _centered_bounds(
        (left + right) / 2, crop_size, image_width
    )
    crop_top, crop_bottom = _centered_bounds(
        (top + bottom) / 2, crop_size, image_height
    )
    crop = np.ascontiguousarray(
        image_rgb[crop_top:crop_bottom, crop_left:crop_right]
    )
    crop_mask = np.zeros(crop.shape[:2], dtype=bool)
    intersect_left, intersect_top = max(crop_left, x0), max(crop_top, y0)
    intersect_right = min(crop_right, x0 + width)
    intersect_bottom = min(crop_bottom, y0 + height)
    if intersect_left < intersect_right and intersect_top < intersect_bottom:
        crop_mask[
            intersect_top - crop_top : intersect_bottom - crop_top,
            intersect_left - crop_left : intersect_right - crop_left,
        ] = mask[
            intersect_top - y0 : intersect_bottom - y0,
            intersect_left - x0 : intersect_right - x0,
        ]
    if not crop_mask.any():
        raise InputValidationError("Candidate mask does not intersect the computed crop.")
    masked = crop.astype(np.float32)
    masked[~crop_mask] *= float(background_retain)
    masked = np.rint(masked).clip(0, 255).astype(np.uint8)
    return PixelViews(
        context=crop,
        crop_mask=np.ascontiguousarray(crop_mask),
        masked=np.ascontiguousarray(masked),
        crop_box=(crop_left, crop_top, crop_right, crop_bottom),
        mask_fraction=float(crop_mask.mean()),
    )


def reconstruct_masked_view(
    context: np.ndarray, crop_mask: np.ndarray, background_retain: float
) -> np.ndarray:
    value = np.asarray(context, dtype=np.uint8).astype(np.float32)
    mask = np.asarray(crop_mask, dtype=bool)
    if value.shape[:2] != mask.shape or value.shape[-1] != 3:
        raise InputValidationError("Context/mask geometry mismatch.")
    value[~mask] *= float(background_retain)
    return np.rint(value).clip(0, 255).astype(np.uint8)


def _read_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(cfg["_meta"]["project_root"]).resolve()
    expected_path = (project_root / "configs" / _EXPECTED_PROTOCOL_NAME).resolve()
    configured = Path(cfg["paths"]["export_protocol_file"]).resolve()
    if configured != expected_path or not configured.is_file():
        raise InputValidationError(
            f"Export protocol must be the committed canonical file: {expected_path}"
        )
    protocol_bytes = _canonical_utf8_text_bytes(
        _stable_read_bytes(configured), "Export protocol"
    )
    try:
        protocol = json.loads(protocol_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read the export protocol.") from exc
    if int(protocol.get("format_version", -1)) != PIXEL_PACK_FORMAT_VERSION:
        raise InputValidationError("Unsupported export protocol format.")
    if protocol.get("dataset") != "LoveDA" or protocol.get("split") != "train":
        raise InputValidationError("Export protocol must be LoveDA Train.")
    for key in ("direct_pixel_gt_used", "love_da_val_used", "oracle_used", "e2_used"):
        if protocol.get(key) is not False:
            raise InputValidationError(f"Export protocol must declare {key}=false.")
    return {
        **protocol,
        "path": str(configured),
        "sha256": _sha256_bytes(protocol_bytes),
        "bytes": protocol_bytes,
    }


def load_export_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(path, project_root)
    protocol = _read_protocol(cfg)
    path_specs = protocol["source_paths"]
    roots = {
        "source_project_root": Path(cfg["paths"]["source_project_root"]),
        "dataset_root": Path(cfg["paths"]["dataset_root"]),
        "checkpoint_root": Path(cfg["paths"]["checkpoint_root"]),
    }
    registered_roots = {
        "source_project_root": path_specs["source_project_root_canonical"],
        "dataset_root": path_specs["dataset_root_canonical"],
        "checkpoint_root": path_specs["checkpoint_root_canonical"],
    }
    for name, root in roots.items():
        if not root.is_absolute():
            raise InputValidationError(f"Configured source root must be absolute: {name}")
        _assert_no_link_components(root)
        if root.resolve() != Path(str(registered_roots[name])).resolve():
            raise InputValidationError(f"Configured {name} is not the registered source root.")
    expected_paths = {
        "candidate_cache_dir": _safe_source_relative(
            roots["source_project_root"],
            path_specs["candidate_cache_dir_relative_to_source_project_root"],
            "candidate cache",
        ),
        "region_feature_cache": _safe_source_relative(
            roots["source_project_root"],
            path_specs["region_feature_cache_relative_to_source_project_root"],
            "region cache",
        ),
        "image_dir": _safe_source_relative(
            roots["dataset_root"],
            path_specs["image_dir_relative_to_dataset_root"],
            "image directory",
        ),
        "remoteclip_checkpoint": _safe_source_relative(
            roots["checkpoint_root"],
            path_specs["remoteclip_checkpoint_relative_to_checkpoint_root"],
            "RemoteCLIP checkpoint",
        ),
    }
    for key, expected in expected_paths.items():
        actual = Path(cfg["paths"][key]).resolve()
        if actual != expected.resolve():
            raise InputValidationError(f"Configured {key} is outside its registered source path.")
        if not actual.exists():
            raise InputValidationError(f"Missing registered source {key}: {actual}")
    checkpoint_sha = _stable_sha256_file(
        Path(cfg["paths"]["remoteclip_checkpoint"])
    )
    if checkpoint_sha != protocol["reference_features"]["checkpoint_sha256"]:
        raise InputValidationError("RemoteCLIP checkpoint SHA-256 is not registered.")
    if int(cfg["export"]["shard_size"]) != int(protocol["export"]["shard_size"]):
        raise InputValidationError("Configured shard size differs from the protocol.")
    implementation_paths = path_specs.get(
        "implementation_files_relative_to_source_project_root"
    )
    implementation_hashes = protocol.get("source_implementation_sha256")
    if (
        not isinstance(implementation_paths, dict)
        or not isinstance(implementation_hashes, dict)
        or set(implementation_paths) != set(implementation_hashes)
    ):
        raise InputValidationError("Source implementation path/hash registration is invalid.")
    verified_implementation_paths: dict[str, str] = {}
    source_root = roots["source_project_root"].resolve()
    for name, relative in implementation_paths.items():
        source_path = _safe_source_relative(
            source_root, relative, f"implementation {name}"
        )
        if not source_path.is_file():
            raise InputValidationError(f"Missing registered implementation file: {source_path}")
        if _sha256_bytes(_stable_read_bytes(source_path)) != str(
            implementation_hashes[name]
        ).lower():
            raise InputValidationError(
                f"Registered implementation SHA-256 mismatch: {name}"
            )
        verified_implementation_paths[name] = str(source_path)
    source_run_hashes = protocol["selection"].get("source_run_artifacts_sha256")
    run_bindings = {
        "selection_input_manifest": "input_manifest.json",
        "selection_summary": "summary_metrics.json",
        "selection_validation": "validated_region_input.json",
    }
    if not isinstance(source_run_hashes, dict) or set(source_run_hashes) != set(
        run_bindings.values()
    ):
        raise InputValidationError("Source run artifact registration is invalid.")
    run_parents: set[Path] = set()
    for config_key, filename in run_bindings.items():
        artifact_path = Path(cfg["paths"].get(config_key) or "")
        if not artifact_path.is_absolute() or artifact_path.name != filename:
            raise InputValidationError(f"Invalid source run artifact path: {config_key}")
        _assert_no_link_components(artifact_path)
        if _sha256_bytes(_stable_read_bytes(artifact_path)) != source_run_hashes[filename]:
            raise InputValidationError(f"Source run artifact SHA-256 mismatch: {filename}")
        run_parents.add(artifact_path.resolve().parent)
    selection_path = Path(cfg["paths"].get("selection_file") or "")
    if (
        len(run_parents) != 1
        or selection_path.resolve().parent not in run_parents
        or selection_path.name != "selected_region_records.jsonl"
        or selection_path.resolve().parent.name
        != str(protocol["selection"]["source_run_name"])
    ):
        raise InputValidationError("Selection artifacts are not one registered source run.")
    input_manifest = json.loads(
        _stable_read_bytes(Path(cfg["paths"]["selection_input_manifest"]))
    )
    summary = json.loads(_stable_read_bytes(Path(cfg["paths"]["selection_summary"])))
    validated = json.loads(_stable_read_bytes(Path(cfg["paths"]["selection_validation"])))
    expected_historical_paths = {
        "source_project_root": roots["source_project_root"].resolve(),
        "remoteclip_checkpoint": expected_paths["remoteclip_checkpoint"].resolve(),
        "region_feature_cache": expected_paths["region_feature_cache"].resolve(),
        "candidate_cache_dir": expected_paths["candidate_cache_dir"].resolve(),
    }
    for name, expected_path in expected_historical_paths.items():
        record = input_manifest.get(name)
        if not isinstance(record, dict) or Path(str(record.get("path"))).resolve() != expected_path:
            raise InputValidationError(
                f"Configured source differs from the registered run manifest: {name}"
            )
    historical_source_paths = validated.get("source_paths")
    if not isinstance(historical_source_paths, dict):
        raise InputValidationError("Registered source run lacks validated source paths.")
    for name in ("candidate_cache_dir", "region_feature_cache"):
        if Path(str(historical_source_paths.get(name))).resolve() != expected_paths[
            name
        ].resolve():
            raise InputValidationError(
                f"Configured source differs from validated source path: {name}"
            )
    if (
        summary.get("status") != "completed"
        or summary.get("scientific_evidence") is not True
        or summary.get("registered_formal_scope") is not True
        or validated.get("registered_formal_scope") is not True
        or validated.get("candidate_count_scanned") != 270641
        or validated.get("image_count") != 2522
        or validated.get("selected_counts") != protocol["selection"]["class_counts"]
        or validated.get("ordered_record_key_sha256")
        != protocol["selection"]["ordered_record_key_sha256"]
    ):
        raise InputValidationError("Source run formal evidence gate failed.")
    protocol["verified_implementation_paths"] = verified_implementation_paths
    protocol["verified_source_run_dir"] = str(next(iter(run_parents)))
    return cfg, protocol


def _stat_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[str] = []
    total_bytes = 0
    for path in sorted({item.resolve() for item in paths}, key=str):
        stat = path.stat()
        total_bytes += int(stat.st_size)
        rows.append(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "path_size_mtime_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf-8")
        ).hexdigest(),
    }


def _file_content_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[str] = []
    total_bytes = 0
    for path in sorted({item.resolve() for item in paths}, key=str):
        _assert_no_link_components(path)
        size = path.stat().st_size
        total_bytes += int(size)
        rows.append(f"{path}\0{size}\0{_stable_sha256_file(path)}")
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "path_size_content_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf-8")
        ).hexdigest(),
    }


def verify_repository_anchor(
    project_root: str | Path,
    expected_commit: str,
    expected_protocol_sha256: str,
) -> dict[str, str]:
    root = Path(project_root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise InputValidationError("Expected code commit must be a full lowercase SHA-1.")
    if not _SHA256.fullmatch(expected_protocol_sha256):
        raise InputValidationError("Expected protocol SHA-256 must be full lowercase hex.")

    def git(*args: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.STDOUT
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise InputValidationError("Cannot verify the committed exporter identity.") from exc

    head = git("rev-parse", "HEAD").decode("ascii").strip().lower()
    if head != expected_commit:
        raise InputValidationError(f"Exporter HEAD differs from approved commit: {head}")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise InputValidationError("Tracked exporter worktree must be clean.")
    relative = f"configs/{_EXPECTED_PROTOCOL_NAME}"
    git("ls-files", "--error-unmatch", relative)
    head_protocol = _canonical_utf8_text_bytes(
        git("show", f"HEAD:{relative}"), "Committed protocol"
    )
    worktree_protocol = _canonical_utf8_text_bytes(
        _stable_read_bytes(root / relative), "Worktree protocol"
    )
    if head_protocol != worktree_protocol:
        raise InputValidationError("Protocol bytes differ from the approved Git object.")
    actual_protocol_sha = _sha256_bytes(head_protocol)
    if actual_protocol_sha != expected_protocol_sha256:
        raise InputValidationError("Protocol differs from the externally approved SHA-256.")
    return {"code_commit": head, "protocol_sha256": actual_protocol_sha}


def _source_paths_for_records(
    cfg: dict[str, Any],
    protocol: dict[str, Any],
    records: Sequence[SelectedRecord],
) -> list[Path]:
    image_dir = Path(cfg["paths"]["image_dir"])
    candidate_dir = Path(cfg["paths"]["candidate_cache_dir"])
    region_dir = Path(cfg["paths"]["region_feature_cache"])
    result = [
        Path(cfg["paths"]["selection_file"]),
        Path(cfg["paths"]["selection_input_manifest"]),
        Path(cfg["paths"]["selection_summary"]),
        Path(cfg["paths"]["selection_validation"]),
        Path(cfg["paths"]["remoteclip_checkpoint"]),
        Path(protocol["path"]),
        *(Path(value) for value in protocol["verified_implementation_paths"].values()),
    ]
    for image_id in sorted({record.image_id for record in records}):
        result.extend(
            [
                image_dir / f"{image_id}_RGB.png",
                candidate_dir / f"{image_id}.json",
                candidate_dir / f"{image_id}.npz",
                region_dir / f"{image_id}.json",
                region_dir / f"{image_id}.npz",
            ]
        )
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise InputValidationError(f"Missing selected source files: {missing[:5]}")
    return result


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_npy_exclusive(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


def _append_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _pair_fingerprint(first: Path, second: Path) -> str:
    digest = hashlib.sha256()
    for path in (first, second):
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_package_relative(value: str) -> Path:
    if not value or "\\" in value:
        raise InputValidationError("Package paths must be non-empty POSIX relative paths.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InputValidationError(f"Unsafe package-relative path: {value!r}")
    relative = Path(*parts)
    if relative.is_absolute():
        raise InputValidationError(f"Absolute package path is forbidden: {value!r}")
    return relative


def _content_inventory(rows: dict[str, str]) -> dict[str, Any]:
    ordered = dict(sorted(rows.items()))
    return {"file_count": len(ordered), "sha256": canonical_json_sha256(ordered)}


def _manifest_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "bundle_id"}


def sync_pixel_pack_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def export_region_pixel_pack(
    cfg: dict[str, Any],
    protocol: dict[str, Any],
    package_dir: str | Path,
    repository_anchor: dict[str, str],
) -> dict[str, Any]:
    from PIL import Image

    package_root = Path(package_dir)
    package_root.mkdir(parents=False, exist_ok=False)
    shards_dir = package_root / "shards"
    shards_dir.mkdir()
    records = read_selected_records(cfg["paths"]["selection_file"], protocol)
    source_paths = _source_paths_for_records(cfg, protocol, records)
    before = _stat_inventory(source_paths)
    content_before = _file_content_inventory(source_paths)
    crop_params = protocol["crop_views"]
    shard_size = int(protocol["export"]["shard_size"])
    class_names = {int(item["id"]): str(item["name"]) for item in cfg["data"]["classes"]}
    native_cfg = {
        "paths": {"remoteclip_checkpoint": cfg["paths"]["remoteclip_checkpoint"]},
        "model": cfg["model"],
        "data": cfg["data"],
    }
    reference = np.empty((len(records), int(cfg["model"]["feature_dim"])), dtype=np.float16)
    record_rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    image_sha_cache: dict[str, str] = {}
    candidate_fingerprint_cache: dict[str, str] = {}
    region_fingerprint_cache: dict[str, str] = {}
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        selected = records[start : start + shard_size]
        grouped: dict[str, list[SelectedRecord]] = defaultdict(list)
        for record in selected:
            grouped[record.image_id].append(record)
        produced: dict[
            int,
            tuple[
                np.ndarray,
                np.ndarray,
                tuple[int, int, int, int],
                float,
                str,
                str,
                str,
                str,
            ],
        ] = {}
        for image_id, image_records in grouped.items():
            image_path = Path(cfg["paths"]["image_dir"]) / f"{image_id}_RGB.png"
            image_bytes = _stable_read_bytes(image_path)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if image_rgb.shape != (1024, 1024, 3):
                raise InputValidationError(f"Unexpected LoveDA image shape for {image_id}.")
            candidate_meta, candidates, fingerprint = load_native_candidate_cache(
                cfg["paths"]["candidate_cache_dir"], image_id, native_cfg
            )
            region_meta, region = load_native_region_score(
                cfg["paths"]["region_feature_cache"], image_id, fingerprint, native_cfg
            )
            del candidate_meta, region_meta
            if image_id not in image_sha_cache:
                image_sha_cache[image_id] = _sha256_bytes(image_bytes)
            image_sha = image_sha_cache[image_id]
            candidate_fingerprint_cache[image_id] = fingerprint
            if image_id not in region_fingerprint_cache:
                region_root = Path(cfg["paths"]["region_feature_cache"])
                region_fingerprint_cache[image_id] = _pair_fingerprint(
                    region_root / f"{image_id}.npz",
                    region_root / f"{image_id}.json",
                )
            region_fingerprint = region_fingerprint_cache[image_id]
            for record in image_records:
                index = record.candidate_index
                if index < 0 or index >= len(candidates):
                    raise InputValidationError(f"Candidate index out of range: {image_id}[{index}]")
                candidate = candidates[index]
                source_name = class_names[candidate.class_id]
                if source_name != record.sam3_source_label:
                    raise InputValidationError(
                        f"SAM3 source label mismatch for {image_id}[{index}]."
                    )
                views = make_pixel_views(
                    image_rgb,
                    candidate,
                    context_ratio=float(crop_params["context_ratio"]),
                    min_crop_size=int(crop_params["min_crop_size"]),
                    background_retain=float(crop_params["background_retain"]),
                )
                cached_box = tuple(int(value) for value in region["crop_boxes"][index])
                if views.crop_box != cached_box:
                    raise InputValidationError(f"Cached crop box mismatch for {image_id}[{index}].")
                cached_fraction = float(region["mask_fractions"][index])
                if abs(views.mask_fraction - cached_fraction) > 1e-7:
                    raise InputValidationError(
                        f"Cached mask fraction mismatch for {image_id}[{index}]."
                    )
                reference[record.row_index] = region["region_features"][index]
                produced[record.row_index] = (
                    views.context,
                    views.crop_mask,
                    views.crop_box,
                    views.mask_fraction,
                    image_sha,
                    fingerprint,
                    region_fingerprint,
                    hashlib.sha256(views.masked.tobytes()).hexdigest(),
                )
        row_indices = np.asarray([record.row_index for record in selected], dtype=np.int32)
        contexts = [produced[int(row)][0] for row in row_indices]
        masks = [produced[int(row)][1] for row in row_indices]
        rgb_lengths = [int(value.size) for value in contexts]
        mask_bytes = [
            np.packbits(mask.reshape(-1), bitorder="little").astype(np.uint8)
            for mask in masks
        ]
        rgb_offsets = np.asarray([0, *np.cumsum(rgb_lengths)], dtype=np.int64)
        mask_offsets = np.asarray(
            [0, *np.cumsum([len(value) for value in mask_bytes])], dtype=np.int64
        )
        shard_path = shards_dir / f"part-{shard_index:04d}.npz"
        _write_npz_exclusive(
            shard_path,
            {
                "format_version": np.asarray([PIXEL_PACK_FORMAT_VERSION], dtype=np.int16),
                "row_indices": row_indices,
                "crop_shapes": np.asarray([value.shape[:2] for value in contexts], dtype=np.int32),
                "crop_boxes": np.asarray(
                    [produced[int(row)][2] for row in row_indices], dtype=np.int32
                ),
                "crop_rgb_offsets": rgb_offsets,
                "crop_rgb_flat": np.concatenate(
                    [value.reshape(-1) for value in contexts]
                ).astype(np.uint8, copy=False),
                "crop_mask_offsets": mask_offsets,
                "crop_mask_bits": np.concatenate(mask_bytes).astype(np.uint8, copy=False),
            },
        )
        relative_shard = f"shards/{shard_path.name}"
        artifacts[relative_shard] = _artifact_entry(shard_path)
        for record in selected:
            (
                context,
                mask,
                crop_box,
                fraction,
                image_sha,
                fingerprint,
                region_fingerprint,
                masked_sha,
            ) = produced[
                record.row_index
            ]
            packed_mask = np.packbits(mask.reshape(-1), bitorder="little")
            record_rows.append(
                {
                    "row_index": record.row_index,
                    "image_id": record.image_id,
                    "candidate_index": record.candidate_index,
                    "sam3_source_label": record.sam3_source_label,
                    "cam_label": record.cam_label,
                    "image_shape": [1024, 1024],
                    "crop_shape": list(context.shape[:2]),
                    "crop_box": list(crop_box),
                    "mask_area": int(mask.sum()),
                    "mask_fraction": fraction,
                    "image_sha256": image_sha,
                    "candidate_cache_sha256": fingerprint,
                    "region_cache_sha256": region_fingerprint,
                    "context_sha256": hashlib.sha256(context.tobytes()).hexdigest(),
                    "crop_mask_sha256": hashlib.sha256(packed_mask.tobytes()).hexdigest(),
                    "masked_view_sha256": masked_sha,
                    "shard": relative_shard,
                }
            )
    records_path = package_root / "records.jsonl"
    reference_path = package_root / "reference_region_features.npy"
    _append_jsonl_exclusive(records_path, record_rows)
    _write_npy_exclusive(reference_path, reference)
    artifacts[records_path.name] = _artifact_entry(records_path)
    artifacts[reference_path.name] = _artifact_entry(reference_path)
    embedded_protocol_path = package_root / _EMBEDDED_PROTOCOL_NAME
    with embedded_protocol_path.open("xb") as destination:
        destination.write(protocol["bytes"])
    artifacts[embedded_protocol_path.name] = _artifact_entry(embedded_protocol_path)
    after = _stat_inventory(source_paths)
    content_after = _file_content_inventory(source_paths)
    if before != after:
        raise InputValidationError("Source files changed during pixel-pack export.")
    if content_before != content_after:
        raise InputValidationError("Source content changed during pixel-pack export.")
    if _sha256_bytes(_stable_read_bytes(Path(cfg["paths"]["selection_file"]))) != protocol["selection"][
        "selected_records_sha256"
    ]:
        raise InputValidationError("Selected-record file changed during export.")
    if _stable_sha256_file(Path(cfg["paths"]["remoteclip_checkpoint"])) != protocol[
        "reference_features"
    ]["checkpoint_sha256"]:
        raise InputValidationError("RemoteCLIP checkpoint changed during export.")
    if _sha256_bytes(_stable_read_bytes(Path(protocol["path"]))) != protocol["sha256"]:
        raise InputValidationError("Export protocol changed during export.")
    for name, source_path in protocol["verified_implementation_paths"].items():
        if _sha256_bytes(_stable_read_bytes(Path(source_path))) != protocol["source_implementation_sha256"][name]:
            raise InputValidationError(f"Source implementation changed during export: {name}")
    manifest = {
        "format_version": PIXEL_PACK_FORMAT_VERSION,
        "dataset": "LoveDA",
        "split": "train",
        "direct_pixel_gt_used": False,
        "love_da_val_used": False,
        "oracle_used": False,
        "e2_used": False,
        "record_count": len(records),
        "image_count": len({record.image_id for record in records}),
        "class_counts": protocol["selection"]["class_counts"],
        "ordered_record_key_sha256": ordered_key_sha256(records),
        "selected_records_sha256": protocol["selection"]["selected_records_sha256"],
        "protocol_sha256": protocol["sha256"],
        "crop_views": protocol["crop_views"],
        "reference_features": protocol["reference_features"],
        "source_implementation_sha256": protocol["source_implementation_sha256"],
        "source_inventory_before": before,
        "source_inventory_after": after,
        "source_content_inventory_before": content_before,
        "source_content_inventory_after": content_after,
        "source_run": {
            "code_commit": protocol["selection"]["source_run_code_commit"],
            "name": protocol["selection"]["source_run_name"],
            "artifact_sha256": protocol["selection"]["source_run_artifacts_sha256"],
        },
        "exporter_repository_anchor": repository_anchor,
        "image_content_inventory": _content_inventory(image_sha_cache),
        "candidate_content_inventory": _content_inventory(candidate_fingerprint_cache),
        "region_content_inventory": _content_inventory(region_fingerprint_cache),
        "artifacts": dict(sorted(artifacts.items())),
    }
    manifest["bundle_id"] = canonical_json_sha256(_manifest_identity_payload(manifest))
    manifest_path = package_root / "manifest.json"
    write_json(manifest_path, manifest)
    checksum_rows = [
        f"{sha256_file(path)}  {path.relative_to(package_root).as_posix()}"
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
    ]
    with (package_root / "checksums.sha256").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(checksum_rows) + "\n")
    with (package_root / "COMPLETE").open("xb") as handle:
        handle.write(b"pixel-pack-v1\n")
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    f"Malformed package record at line {line_number}."
                ) from exc
            if not isinstance(value, dict):
                raise InputValidationError("Package records must be JSON objects.")
            rows.append(value)
    return rows


def validate_region_pixel_pack(
    package_dir: str | Path,
    expected_protocol_file: str | Path,
) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    expected_protocol_path = Path(expected_protocol_file).resolve()
    _assert_no_link_components(root)
    _assert_no_link_components(expected_protocol_path)
    if not expected_protocol_path.is_file():
        raise InputValidationError("Expected pixel-pack protocol is missing.")
    expected_protocol_bytes = _stable_read_bytes(expected_protocol_path)
    expected_protocol_sha = _sha256_bytes(expected_protocol_bytes)
    manifest_path = root / "manifest.json"
    checksums_path = root / "checksums.sha256"
    complete_path = root / "COMPLETE"
    if (
        not manifest_path.is_file()
        or not checksums_path.is_file()
        or _stable_read_bytes(complete_path) != b"pixel-pack-v1\n"
    ):
        raise InputValidationError("Pixel package lacks manifest/checksums.")
    manifest = json.loads(_stable_read_bytes(manifest_path))
    if int(manifest.get("format_version", -1)) != PIXEL_PACK_FORMAT_VERSION:
        raise InputValidationError("Unsupported pixel-package format.")
    for key in ("direct_pixel_gt_used", "love_da_val_used", "oracle_used", "e2_used"):
        if manifest.get(key) is not False:
            raise InputValidationError(f"Pixel package does not declare {key}=false.")
    if manifest.get("protocol_sha256") != expected_protocol_sha:
        raise InputValidationError("Pixel package protocol SHA-256 is not registered.")
    embedded_protocol_path = root / _EMBEDDED_PROTOCOL_NAME
    if (
        not embedded_protocol_path.is_file()
        or sha256_file(embedded_protocol_path) != expected_protocol_sha
        or _stable_read_bytes(embedded_protocol_path) != expected_protocol_bytes
    ):
        raise InputValidationError("Embedded pixel-pack protocol is not byte-identical.")
    expected_protocol = json.loads(expected_protocol_bytes)
    expected_manifest_values = {
        "dataset": expected_protocol["dataset"],
        "split": expected_protocol["split"],
        "record_count": expected_protocol["selection"]["record_count"],
        "image_count": expected_protocol["selection"]["image_count"],
        "class_counts": expected_protocol["selection"]["class_counts"],
        "ordered_record_key_sha256": expected_protocol["selection"][
            "ordered_record_key_sha256"
        ],
        "selected_records_sha256": expected_protocol["selection"][
            "selected_records_sha256"
        ],
        "crop_views": expected_protocol["crop_views"],
        "reference_features": expected_protocol["reference_features"],
        "source_implementation_sha256": expected_protocol[
            "source_implementation_sha256"
        ],
    }
    for key, expected_value in expected_manifest_values.items():
        if manifest.get(key) != expected_value:
            raise InputValidationError(f"Manifest field is not protocol-registered: {key}")
    for path in root.rglob("*"):
        if path.is_symlink() or bool(
            hasattr(path, "is_junction") and path.is_junction()
        ):
            raise InputValidationError("Symlinks are forbidden inside a pixel package.")
    checksum_entries: dict[str, str] = {}
    for line in _stable_read_bytes(checksums_path).decode("utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or not _SHA256.fullmatch(digest)
            or relative in checksum_entries
        ):
            raise InputValidationError("Malformed or duplicate checksum entry.")
        _safe_package_relative(relative)
        checksum_entries[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path not in {checksums_path, complete_path}
    }
    expected_shard_count = math.ceil(
        int(expected_protocol["selection"]["record_count"])
        / int(expected_protocol["export"]["shard_size"])
    )
    expected_artifact_files = {
        "records.jsonl",
        "reference_region_features.npy",
        _EMBEDDED_PROTOCOL_NAME,
        *{f"shards/part-{index:04d}.npz" for index in range(expected_shard_count)},
    }
    expected_package_files = {"manifest.json", *expected_artifact_files}
    if actual_files != expected_package_files:
        raise InputValidationError("Pixel package has missing or unregistered files.")
    if set(checksum_entries) != actual_files:
        raise InputValidationError("Package checksum file set is incomplete or has extras.")
    for relative, digest in checksum_entries.items():
        path = root / _safe_package_relative(relative)
        if sha256_file(path) != digest:
            raise InputValidationError(f"Package checksum mismatch: {relative}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise InputValidationError("Pixel package lacks artifact metadata.")
    if set(artifacts) != expected_artifact_files:
        raise InputValidationError("Manifest artifact file set is incomplete or has extras.")
    for relative, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise InputValidationError(f"Malformed manifest artifact entry: {relative}")
        path = root / _safe_package_relative(relative)
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise InputValidationError(f"Manifest artifact mismatch: {relative}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise InputValidationError(f"Manifest artifact size mismatch: {relative}")
    rows = _load_jsonl(root / "records.jsonl")
    expected_count = int(manifest["record_count"])
    if len(rows) != expected_count:
        raise InputValidationError("Pixel-package record count mismatch.")
    if [int(row.get("row_index", -1)) for row in rows] != list(range(expected_count)):
        raise InputValidationError("Pixel-package row order is invalid.")
    allowed_labels = set(manifest["class_counts"])
    for row in rows:
        if set(row) != _REQUIRED_RECORD_FIELDS:
            raise InputValidationError("Pixel-package record schema is invalid.")
        if not re.fullmatch(
            r"loveda_train_(?:rural|urban)_\d+", str(row["image_id"])
        ):
            raise InputValidationError("Pixel-package image_id is invalid.")
        if int(row["candidate_index"]) < 0:
            raise InputValidationError("Pixel-package candidate_index is invalid.")
        if (
            str(row["sam3_source_label"]) not in allowed_labels
            or str(row["cam_label"]) not in allowed_labels
        ):
            raise InputValidationError("Pixel-package weak label is out of vocabulary.")
        for hash_field in (
            "image_sha256",
            "candidate_cache_sha256",
            "region_cache_sha256",
            "context_sha256",
            "crop_mask_sha256",
            "masked_view_sha256",
        ):
            if not _SHA256.fullmatch(str(row[hash_field])):
                raise InputValidationError(
                    f"Pixel-package record hash is invalid: {hash_field}"
                )
    keys = [(str(row["image_id"]), int(row["candidate_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise InputValidationError("Pixel-package row keys are not unique.")
    key_digest = hashlib.sha256(
        "\n".join(f"{image_id}:{index}" for image_id, index in keys).encode("utf-8")
    ).hexdigest()
    if key_digest != manifest["ordered_record_key_sha256"]:
        raise InputValidationError("Pixel-package ordered key SHA-256 mismatch.")
    if len({key[0] for key in keys}) != int(manifest["image_count"]):
        raise InputValidationError("Pixel-package image count mismatch.")
    if dict(Counter(str(row["sam3_source_label"]) for row in rows)) != manifest[
        "class_counts"
    ]:
        raise InputValidationError("Pixel-package class counts mismatch.")
    row_by_index = {int(row["row_index"]): row for row in rows}
    visited: list[int] = []
    background_retain = float(manifest["crop_views"]["background_retain"])
    shard_paths = sorted((root / "shards").glob("part-*.npz"))
    if not shard_paths:
        raise InputValidationError("Pixel package contains no shards.")
    for shard_path in shard_paths:
        relative_shard = shard_path.relative_to(root).as_posix()
        with np.load(shard_path, allow_pickle=False) as archive:
            missing = _REQUIRED_SHARD_ARRAYS - set(archive.files)
            extras = set(archive.files) - _REQUIRED_SHARD_ARRAYS
            if missing or extras:
                raise InputValidationError(
                    f"Shard schema mismatch {shard_path.name}: missing={missing}, extras={extras}."
                )
            arrays = {name: archive[name] for name in _REQUIRED_SHARD_ARRAYS}
        if arrays["format_version"].dtype != np.int16 or arrays["format_version"].tolist() != [1]:
            raise InputValidationError("Shard format_version is invalid.")
        row_indices = arrays["row_indices"]
        count = len(row_indices)
        expected_specs = {
            "row_indices": (np.int32, (count,)),
            "crop_shapes": (np.int32, (count, 2)),
            "crop_boxes": (np.int32, (count, 4)),
            "crop_rgb_offsets": (np.int64, (count + 1,)),
            "crop_rgb_flat": (np.uint8, (len(arrays["crop_rgb_flat"]),)),
            "crop_mask_offsets": (np.int64, (count + 1,)),
            "crop_mask_bits": (np.uint8, (len(arrays["crop_mask_bits"]),)),
        }
        for name, (dtype, shape) in expected_specs.items():
            if arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != shape:
                raise InputValidationError(f"Shard field {name} has invalid dtype/shape.")
        rgb_offsets = arrays["crop_rgb_offsets"]
        mask_offsets = arrays["crop_mask_offsets"]
        if (
            rgb_offsets[0] != 0
            or mask_offsets[0] != 0
            or np.any(np.diff(rgb_offsets) < 0)
            or np.any(np.diff(mask_offsets) < 0)
            or rgb_offsets[-1] != len(arrays["crop_rgb_flat"])
            or mask_offsets[-1] != len(arrays["crop_mask_bits"])
        ):
            raise InputValidationError("Shard offsets are invalid.")
        for position, row_value in enumerate(row_indices):
            row_index = int(row_value)
            if row_index not in row_by_index:
                raise InputValidationError("Shard references an unknown row index.")
            visited.append(row_index)
            row = row_by_index[row_index]
            height, width = (int(value) for value in arrays["crop_shapes"][position])
            if height <= 0 or width <= 0:
                raise InputValidationError("Shard contains an invalid crop shape.")
            rgb_start, rgb_end = int(rgb_offsets[position]), int(rgb_offsets[position + 1])
            if rgb_end - rgb_start != height * width * 3:
                raise InputValidationError("Shard crop RGB length is invalid.")
            context = arrays["crop_rgb_flat"][rgb_start:rgb_end].reshape(height, width, 3)
            mask_start, mask_end = int(mask_offsets[position]), int(mask_offsets[position + 1])
            if mask_end - mask_start != math.ceil(height * width / 8):
                raise InputValidationError("Shard packed mask length is invalid.")
            crop_mask = np.unpackbits(
                arrays["crop_mask_bits"][mask_start:mask_end],
                bitorder="little",
                count=height * width,
            ).reshape(height, width).astype(bool)
            crop_box = list(map(int, arrays["crop_boxes"][position]))
            left, top, right, bottom = crop_box
            if (
                not crop_mask.any()
                or crop_box != row["crop_box"]
                or row["crop_shape"] != [height, width]
                or row["image_shape"] != [1024, 1024]
                or row["shard"] != relative_shard
                or left < 0
                or top < 0
                or right > 1024
                or bottom > 1024
                or right - left != width
                or bottom - top != height
            ):
                raise InputValidationError("Shard mask/crop metadata is invalid.")
            if hashlib.sha256(context.tobytes()).hexdigest() != row["context_sha256"]:
                raise InputValidationError("Shard context SHA-256 mismatch.")
            packed = np.packbits(crop_mask.reshape(-1), bitorder="little")
            if hashlib.sha256(packed.tobytes()).hexdigest() != row["crop_mask_sha256"]:
                raise InputValidationError("Shard crop-mask SHA-256 mismatch.")
            masked = reconstruct_masked_view(context, crop_mask, background_retain)
            if hashlib.sha256(masked.tobytes()).hexdigest() != row["masked_view_sha256"]:
                raise InputValidationError("Shard masked-view SHA-256 mismatch.")
            if int(crop_mask.sum()) != int(row["mask_area"]):
                raise InputValidationError("Shard mask area mismatch.")
            if abs(float(crop_mask.mean()) - float(row["mask_fraction"])) > 1e-12:
                raise InputValidationError("Shard mask fraction mismatch.")
    if visited != list(range(expected_count)):
        raise InputValidationError("Shard row coverage/order is incomplete or duplicated.")
    reference = np.load(root / "reference_region_features.npy", allow_pickle=False)
    expected_shape = tuple(int(value) for value in manifest["reference_features"]["shape"])
    if reference.dtype != np.float16 or reference.shape != expected_shape:
        raise InputValidationError("Reference region feature dtype/shape mismatch.")
    if not np.isfinite(reference).all() or np.any(np.linalg.norm(reference.astype(np.float32), axis=1) == 0):
        raise InputValidationError("Reference region features contain invalid vectors.")
    image_inventory = {
        str(row["image_id"]): str(row["image_sha256"]) for row in rows
    }
    candidate_inventory = {
        str(row["image_id"]): str(row["candidate_cache_sha256"]) for row in rows
    }
    region_inventory = {
        str(row["image_id"]): str(row["region_cache_sha256"]) for row in rows
    }
    if (
        manifest.get("image_content_inventory") != _content_inventory(image_inventory)
        or manifest.get("candidate_content_inventory")
        != _content_inventory(candidate_inventory)
        or manifest.get("region_content_inventory")
        != _content_inventory(region_inventory)
    ):
        raise InputValidationError("Pixel package source content inventory is invalid.")
    if manifest.get("source_inventory_before") != manifest.get(
        "source_inventory_after"
    ):
        raise InputValidationError("Pixel package source stat inventories differ.")
    if manifest.get("source_content_inventory_before") != manifest.get(
        "source_content_inventory_after"
    ):
        raise InputValidationError("Pixel package source content inventories differ.")
    expected_source_run = {
        "code_commit": expected_protocol["selection"]["source_run_code_commit"],
        "name": expected_protocol["selection"]["source_run_name"],
        "artifact_sha256": expected_protocol["selection"][
            "source_run_artifacts_sha256"
        ],
    }
    if manifest.get("source_run") != expected_source_run:
        raise InputValidationError("Pixel package source run provenance is invalid.")
    exporter_anchor = manifest.get("exporter_repository_anchor")
    if (
        not isinstance(exporter_anchor, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(exporter_anchor.get("code_commit", "")))
        or exporter_anchor.get("protocol_sha256") != expected_protocol_sha
    ):
        raise InputValidationError("Pixel package exporter repository anchor is invalid.")
    expected_bundle_id = canonical_json_sha256(_manifest_identity_payload(manifest))
    if manifest.get("bundle_id") != expected_bundle_id:
        raise InputValidationError("Pixel package bundle_id is invalid.")
    return {
        "status": "valid",
        "bundle_id": manifest["bundle_id"],
        "record_count": expected_count,
        "image_count": manifest["image_count"],
        "ordered_record_key_sha256": key_digest,
        "artifact_count": len(artifacts),
    }
