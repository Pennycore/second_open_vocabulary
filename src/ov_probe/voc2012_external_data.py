"""Fail-closed PASCAL VOC 2012 validation-image acquisition manifest helpers.

This module deliberately treats VOC annotations as quarantined.  It reads only
the segmentation validation *ID list* and hashes only the corresponding JPEG
files; it never opens a label, annotation, or class-specific file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

from .io import InputValidationError, sha256_file


_PROTOCOL_NAME = "voc2012_external_data_protocol_v1.json"
_VAL_LIST = Path("ImageSets") / "Segmentation" / "val.txt"
_JPEG_ROOT = Path("JPEGImages")
_QUARANTINED_ROOTS = ("SegmentationClass", "SegmentationObject", "Annotations")
_VOC_ARCHIVE_ROOT = PurePosixPath("VOCdevkit/VOC2012")
_VOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


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


def _assert_no_link_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and _is_link_or_junction(component):
            raise InputValidationError(f"{label} may not traverse a symlink or junction: {component}")


def _resolve_project_path(value: str | None, root: Path, name: str) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        raise InputValidationError(f"VOC config path must be project-contained and relative: {name}")
    _assert_no_link_components(root / candidate, f"VOC config path {name}")
    resolved = (root / candidate).resolve()
    if not _is_relative_to(resolved, root):
        raise InputValidationError(f"VOC config path escapes project root: {name}")
    return str(resolved)


def _sha256_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_safe_tar_members(archive_path: str | Path) -> list[tarfile.TarInfo]:
    """Return only regular-file/directory members after rejecting unsafe tar entries."""
    archive = Path(archive_path)
    try:
        with tarfile.open(archive, "r:") as handle:
            members = handle.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise InputValidationError("VOC archive cannot be read as an uncompressed tar file.") from exc
    if not members:
        raise InputValidationError("VOC archive may not be empty.")
    for member in members:
        name = member.name.replace("\\", "/")
        pure = PurePosixPath(name)
        if not name or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise InputValidationError(f"VOC archive has an unsafe member path: {member.name!r}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isreg()):
            raise InputValidationError(f"VOC archive has a forbidden non-regular member: {member.name!r}")
    return members


def safe_extract_voc_trainval_archive(archive_path: str | Path, destination_root: str | Path) -> Path:
    """Safely extract a validated VOC archive to an absent, ordinary destination.

    This helper is intentionally not invoked by the acquisition CLI: that CLI
    accepts an already-prepared local data root only.  It exists so a separately
    approved offline preparation command can use the same fail-closed policy.
    """
    archive = Path(archive_path).resolve()
    destination = Path(destination_root)
    _assert_no_link_components(destination, "VOC extraction destination")
    if destination.exists():
        raise InputValidationError("VOC extraction destination must be absent; extraction never overwrites.")
    members = validate_safe_tar_members(archive)
    try:
        with tarfile.open(archive, "r:") as handle:
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                target = destination.joinpath(*PurePosixPath(member.name.replace("\\", "/")).parts)
                resolved_target = target.resolve()
                if not _is_relative_to(resolved_target, destination.resolve()):
                    raise InputValidationError("Validated VOC archive member escaped extraction destination.")
                handle.extract(member, path=destination, set_attrs=False, numeric_owner=False)
    except Exception:
        # Do not clean up: an incomplete destination is evidence and must not be reused.
        raise
    return destination / Path(*_VOC_ARCHIVE_ROOT.parts)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    dataset = protocol.get("dataset", {})
    if protocol.get("status") != "frozen_pre_result" or protocol.get("scientific_evidence") is not False:
        raise InputValidationError("VOC protocol must be frozen_pre_result with scientific_evidence=false.")
    if protocol.get("role") != "external PASCAL VOC 2012 validation-image acquisition and manifest only; no model evaluation":
        raise InputValidationError("VOC protocol role differs from the frozen scope.")
    required_dataset = {
        "name": "PASCAL VOC 2012", "year": "2012", "image_domain": "natural imagery",
        "purpose": "external independent evaluation", "archive_filename": "VOCtrainval_11-May-2012.tar",
        "archive_md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "archive_url": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        # The VOC page's 11,530 statistic is retained for disclosure, but the
        # downloaded trainval archive contains 17,125 JPEGs under JPEGImages.
        # Only the latter is an inventory invariant for this archive.
        "official_trainval_image_count": 11530, "expected_archive_jpeg_count": 17125,
        "expected_val_id_count": 1449,
    }
    if any(dataset.get(key) != value for key, value in required_dataset.items()):
        raise InputValidationError("VOC protocol dataset registration differs from the frozen values.")
    scope = protocol.get("input_scope", {})
    if scope.get("allowed_split_list") != _VAL_LIST.as_posix() or scope.get("allowed_image_root") != "JPEGImages":
        raise InputValidationError("VOC protocol allows an unexpected input scope.")
    if tuple(scope.get("quarantined_annotation_roots", [])) != _QUARANTINED_ROOTS:
        raise InputValidationError("VOC protocol quarantine roots differ from the frozen values.")
    if scope.get("quarantined_class_specific_files") is not True:
        raise InputValidationError("VOC protocol must quarantine class-specific files.")
    required_false = (
        "network_download", "semantic_label_read_or_decode", "detection_label_read_or_decode",
        "class_specific_file_read", "downstream_evaluation", "model_execution", "prompt_selection",
        "prototype_selection", "ground_truth_retrieval", "sam3_rerun", "training", "overwrite",
    )
    if any(protocol.get("constraints", {}).get(field) is not False for field in required_false):
        raise InputValidationError("VOC protocol constraints differ from the frozen scope.")


def load_voc2012_external_data_config(
    path: str | Path, project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("Cannot read VOC external-data configuration.") from exc
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("VOC external-data config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"archive_file", "raw_data_root", "protocol_file", "output_root"}:
        raise InputValidationError("VOC external-data config paths do not match the frozen schema.")
    for key, value in list(paths.items()):
        paths[key] = _resolve_project_path(value, root, key)
    output_root = Path(str(paths["output_root"])).resolve()
    if output_root.parent != (root / "outputs").resolve():
        raise InputValidationError("VOC external-data output_root must be directly under outputs/.")
    expected_protocol = (root / "configs" / _PROTOCOL_NAME).resolve()
    if Path(str(paths["protocol_file"])).resolve() != expected_protocol:
        raise InputValidationError("VOC external-data protocol must be the committed canonical protocol.")
    try:
        protocol_bytes = expected_protocol.read_bytes()
        protocol = json.loads(protocol_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("Cannot read the canonical VOC external-data protocol.") from exc
    if not isinstance(protocol, dict):
        raise InputValidationError("VOC external-data protocol must be a JSON object.")
    _validate_protocol(protocol)
    protocol["path"] = str(expected_protocol)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_voc2012_external_data_anchor(
    project_root: str | Path, expected_commit: str, expected_protocol_sha256: str
) -> dict[str, str]:
    root = Path(project_root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("VOC repository anchor could not read git state.") from exc
    actual_protocol_sha256 = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual_protocol_sha256 != expected_protocol_sha256:
        raise InputValidationError("VOC manifest creation requires the approved clean commit and protocol SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual_protocol_sha256}


def _read_val_ids(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputValidationError("VOC validation ID list cannot be read as ASCII text.") from exc
    ids = [line.strip() for line in lines if line.strip()]
    if not ids or any(not _VOC_ID_RE.fullmatch(image_id) for image_id in ids):
        raise InputValidationError("VOC validation ID list contains an invalid image ID.")
    if len(ids) != len(set(ids)):
        raise InputValidationError("VOC validation ID list contains duplicate image IDs.")
    return ids


def validate_voc_val_image_path(raw_data_root: str | Path, image_path: str | Path, image_id: str) -> Path:
    """Accept only the exact val JPEG path for one allowed ID, never an annotation path."""
    raw_root = Path(raw_data_root).resolve()
    candidate = Path(image_path)
    _assert_no_link_components(candidate, "VOC model input")
    resolved = candidate.resolve()
    jpeg_root = (raw_root / _JPEG_ROOT).resolve()
    if not _is_relative_to(resolved, jpeg_root) or resolved.parent != jpeg_root:
        raise InputValidationError("VOC model input must be a direct JPEGImages/<id>.jpg file, never an annotation.")
    if resolved.name != f"{image_id}.jpg" or not resolved.is_file() or _is_link_or_junction(resolved):
        raise InputValidationError(f"VOC validation JPEG is missing or unsafe: {image_id}")
    return resolved


def _list_trainval_jpegs(image_root: Path) -> list[Path]:
    images = sorted(image_root.glob("*.jpg"))
    if any(not image.is_file() or _is_link_or_junction(image) for image in images):
        raise InputValidationError("VOC JPEGImages contains a non-regular or linked entry.")
    return images


def _assert_untracked_data_root(root: Path, raw_data_root: Path) -> None:
    try:
        relative = raw_data_root.resolve().relative_to(root).as_posix()
        tracked = subprocess.check_output(["git", "ls-files", "--", relative], cwd=root, text=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise InputValidationError("VOC raw data root must be project-contained and absent from Git.") from exc
    if tracked.strip():
        raise InputValidationError("VOC raw data root must be absent from Git.")


def _prepare_empty_destination(output_dir: str | Path) -> tuple[Path, bool]:
    requested = Path(output_dir)
    _assert_no_link_components(requested, "VOC output")
    if _is_link_or_junction(requested):
        raise InputValidationError("VOC output may not be a symlink or junction.")
    destination = requested.resolve()
    if not destination.exists():
        return destination, True
    if _is_link_or_junction(destination) or not destination.is_dir():
        raise InputValidationError("VOC output must be an ordinary directory.")
    try:
        next(destination.iterdir())
    except StopIteration:
        return destination, False
    except OSError as exc:
        raise InputValidationError("Cannot verify that VOC output directory is empty.") from exc
    raise InputValidationError("VOC output directory must be empty when it already exists.")


def create_voc2012_external_data_manifest(
    cfg: dict[str, Any], protocol: dict[str, Any], output_dir: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Validate a local VOC root and write an image-only, label-free manifest."""
    _validate_protocol(protocol)
    archive_value, raw_value = cfg["paths"].get("archive_file"), cfg["paths"].get("raw_data_root")
    if not archive_value or not raw_value:
        raise InputValidationError("Tracked VOC config is non-runnable; archive_file and raw_data_root are required.")
    root = Path(project_root).resolve()
    archive, raw_root = Path(str(archive_value)).resolve(), Path(str(raw_value)).resolve()
    _assert_no_link_components(archive, "VOC archive")
    _assert_no_link_components(raw_root, "VOC raw data root")
    if not archive.is_file() or _is_link_or_junction(archive):
        raise InputValidationError("VOC archive must be an ordinary local file.")
    if archive.name != protocol["dataset"]["archive_filename"]:
        raise InputValidationError("VOC archive filename differs from the frozen protocol.")
    archive_md5 = md5_file(archive)
    if not _MD5_RE.fullmatch(archive_md5) or archive_md5 != protocol["dataset"]["archive_md5"]:
        raise InputValidationError("VOC archive MD5 differs from the frozen protocol.")
    validate_safe_tar_members(archive)
    if not raw_root.is_dir() or _is_link_or_junction(raw_root):
        raise InputValidationError("VOC raw_data_root must be an existing ordinary directory.")
    _assert_untracked_data_root(root, raw_root)
    val_list = raw_root / _VAL_LIST
    image_root = raw_root / _JPEG_ROOT
    if not val_list.is_file() or not image_root.is_dir():
        raise InputValidationError("VOC raw_data_root lacks the allowed val ID list or JPEGImages root.")
    val_ids = _read_val_ids(val_list)
    if len(val_ids) != int(protocol["dataset"]["expected_val_id_count"]):
        raise InputValidationError("VOC validation ID count differs from the frozen protocol.")
    images: list[dict[str, str]] = []
    for image_id in val_ids:
        image = validate_voc_val_image_path(raw_root, image_root / f"{image_id}.jpg", image_id)
        images.append({"image_id": image_id, "path": (_JPEG_ROOT / image.name).as_posix(), "sha256": sha256_file(image)})
    trainval_images = _list_trainval_jpegs(image_root)
    if len(trainval_images) != int(protocol["dataset"]["expected_archive_jpeg_count"]):
        raise InputValidationError("VOC JPEGImages count differs from the frozen protocol.")
    destination, must_create_destination = _prepare_empty_destination(output_dir)
    if must_create_destination:
        destination.mkdir(parents=True, exist_ok=False)
    anchor = cfg.get("repository_anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"code_commit", "protocol_sha256"}:
        raise InputValidationError("VOC runner must supply the exact repository anchor.")
    if anchor["protocol_sha256"] != protocol.get("sha256"):
        raise InputValidationError("VOC repository anchor does not bind the loaded protocol.")
    image_lines = [f"{entry['image_id']}\t{entry['sha256']}" for entry in images]
    manifest = {
        "format_version": 1,
        "status": "completed",
        "scientific_evidence": False,
        "role": protocol["role"],
        "repository_anchor": anchor,
        "protocol": {"sha256": protocol["sha256"], "status": protocol["status"]},
        "dataset": {key: protocol["dataset"][key] for key in ("name", "year", "image_domain", "purpose")},
        "archive": {"path": str(archive), "filename": archive.name, "size_bytes": archive.stat().st_size, "md5": archive_md5},
        "raw_data_root": str(raw_root),
        "val_split": {
            "id_list": _VAL_LIST.as_posix(), "id_list_sha256": sha256_file(val_list),
            "id_count": len(val_ids), "canonical_id_sha256": _sha256_lines(val_ids),
        },
        "archive_jpeg_inventory": {
            "jpeg_count": len(trainval_images),
            "expected_archive_jpeg_count": protocol["dataset"]["expected_archive_jpeg_count"],
            "official_trainval_image_count": protocol["dataset"]["official_trainval_image_count"],
        },
        "model_input_files": images,
        "model_input_image_sha256_aggregate": _sha256_lines(image_lines),
        "quarantined_annotation_roots": list(_QUARANTINED_ROOTS),
        "constraints": protocol["constraints"],
    }
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
