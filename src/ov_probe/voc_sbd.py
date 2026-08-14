from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import uuid
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


VOC_ARCHIVE = "VOCtrainval_11-May-2012.tar"
SBD_ARCHIVE = "benchmark.tgz"
TRAIN_NOVAL = "train_noval.txt"

EXPECTED_MD5 = {
    VOC_ARCHIVE: "6cd6e144f989b92b3379bac3b3de84fd",
    SBD_ARCHIVE: "82b4d87ceb2ed10f6038a1cba92111cb",
    TRAIN_NOVAL: "79bff800c5f0b1ec6b21080a3c066722",
}


class DatasetPreparationError(RuntimeError):
    pass


def load_voc_image_level_tags(
    voc_root: Path,
    class_names: Iterable[str],
    *,
    split: str = "train",
    difficult_policy: str = "positive_presence",
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Load VOC classification-task labels without reading segmentation masks."""
    if split != "train":
        raise DatasetPreparationError("Weak-label loading is restricted to the VOC train split.")
    if difficult_policy not in {"positive_presence", "exclude"}:
        raise DatasetPreparationError(f"Unsupported difficult policy: {difficult_policy}")
    names = tuple(class_names)
    if not names or len(names) != len(set(names)):
        raise DatasetPreparationError("VOC class names must be unique and non-empty.")

    labels_by_image: dict[str, set[str]] = {}
    ordered_ids: list[str] | None = None
    difficult_counts: dict[str, int] = {}
    main = voc_root / "ImageSets" / "Main"
    for class_name in names:
        path = main / f"{class_name}_{split}.txt"
        if not path.is_file() or path.is_symlink():
            raise DatasetPreparationError(f"Missing VOC classification label file: {path}")
        records: list[tuple[str, int]] = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parts = raw.split()
            if len(parts) != 2:
                raise DatasetPreparationError(f"Malformed VOC label at {path}:{line_number}")
            image_id, value_raw = parts
            if "/" in image_id or "\\" in image_id or image_id in {".", ".."}:
                raise DatasetPreparationError(f"Unsafe VOC image ID at {path}:{line_number}")
            try:
                value = int(value_raw)
            except ValueError as exc:
                raise DatasetPreparationError(f"Non-integer VOC label at {path}:{line_number}") from exc
            if value not in {-1, 0, 1}:
                raise DatasetPreparationError(f"Unexpected VOC label at {path}:{line_number}: {value}")
            records.append((image_id, value))
        ids = [image_id for image_id, _ in records]
        if len(ids) != len(set(ids)):
            raise DatasetPreparationError(f"Duplicate VOC image ID in {path}")
        if ordered_ids is None:
            ordered_ids = ids
            labels_by_image = {image_id: set() for image_id in ids}
        elif ids != ordered_ids:
            raise DatasetPreparationError(f"VOC class label rows are not identically ordered: {path}")
        difficult_counts[class_name] = sum(value == 0 for _, value in records)
        for image_id, value in records:
            if value == 1 or (value == 0 and difficult_policy == "positive_presence"):
                labels_by_image[image_id].add(class_name)

    assert ordered_ids is not None
    tags = {image_id: tuple(name for name in names if name in labels_by_image[image_id]) for image_id in ordered_ids}
    metadata = {
        "split": split,
        "image_count": len(tags),
        "class_count": len(names),
        "difficult_policy": difficult_policy,
        "difficult_counts": difficult_counts,
        "images_without_positive_tags": sum(not value for value in tags.values()),
        "segmentation_masks_read": False,
    }
    return tags, metadata


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> Path:
    if "\\" in name:
        raise DatasetPreparationError(f"Backslash in archive member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise DatasetPreparationError(f"Unsafe archive member path: {name!r}")
    if ":" in pure.parts[0]:
        raise DatasetPreparationError(f"Drive-qualified archive member path: {name!r}")
    return Path(*pure.parts)


def inspect_archive(path: Path) -> list[tuple[tarfile.TarInfo, Path]]:
    seen: set[Path] = set()
    members: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            relative = _safe_member_path(member.name)
            if relative in seen:
                raise DatasetPreparationError(f"Duplicate archive member: {member.name!r}")
            seen.add(relative)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise DatasetPreparationError(f"Unsupported archive member type: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise DatasetPreparationError(f"Unknown archive member type: {member.name!r}")
            members.append((member, relative))
    return members


def extract_archive_safely(path: Path, target: Path) -> int:
    inspected = inspect_archive(path)
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    count = 0
    with tarfile.open(path, "r:*") as archive:
        by_name = {member.name: member for member in archive.getmembers()}
        for inspected_member, relative in inspected:
            member = by_name[inspected_member.name]
            destination = target / relative
            resolved_parent = destination.parent.resolve()
            if resolved_parent != target_root and target_root not in resolved_parent.parents:
                raise DatasetPreparationError(f"Archive member escapes target: {member.name!r}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise DatasetPreparationError(f"Unable to read archive member: {member.name!r}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if destination.stat().st_size != member.size:
                raise DatasetPreparationError(f"Extracted size mismatch: {member.name!r}")
            count += 1
    return count


def read_split(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise DatasetPreparationError(f"Duplicate image IDs in split: {path}")
    if any("/" in value or "\\" in value or value in {".", ".."} for value in values):
        raise DatasetPreparationError(f"Unsafe image ID in split: {path}")
    return values


def _names(directory: Path, suffix: str) -> set[str]:
    return {path.stem for path in directory.glob(f"*{suffix}") if path.is_file()}


def audit_extracted_dataset(extracted: Path, train_noval_file: Path) -> dict[str, Any]:
    voc = extracted / "VOCdevkit" / "VOC2012"
    sbd = extracted / "benchmark_RELEASE" / "dataset"
    required = [
        voc / "JPEGImages",
        voc / "SegmentationClass",
        voc / "ImageSets" / "Segmentation" / "train.txt",
        voc / "ImageSets" / "Segmentation" / "val.txt",
        sbd / "img",
        sbd / "cls",
        sbd / "train.txt",
        sbd / "val.txt",
        train_noval_file,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise DatasetPreparationError(f"Missing extracted dataset paths: {missing}")

    voc_train = read_split(voc / "ImageSets" / "Segmentation" / "train.txt")
    voc_val = read_split(voc / "ImageSets" / "Segmentation" / "val.txt")
    sbd_train = read_split(sbd / "train.txt")
    sbd_val = read_split(sbd / "val.txt")
    train_noval = read_split(train_noval_file)
    sbd_images = _names(sbd / "img", ".jpg")
    sbd_masks = _names(sbd / "cls", ".mat")
    voc_images = _names(voc / "JPEGImages", ".jpg")
    voc_masks = _names(voc / "SegmentationClass", ".png")

    train_noval_set = set(train_noval)
    unknown = train_noval_set - sbd_images
    missing_masks = train_noval_set - sbd_masks
    val_overlap = train_noval_set & set(voc_val)
    if unknown or missing_masks or val_overlap:
        raise DatasetPreparationError(
            "Invalid SBD train_noval split: "
            f"unknown_images={len(unknown)}, missing_masks={len(missing_masks)}, "
            f"voc_val_overlap={len(val_overlap)}"
        )
    if set(voc_train) & set(voc_val):
        raise DatasetPreparationError("VOC train and val splits overlap.")
    if not set(voc_train).issubset(voc_images) or not set(voc_val).issubset(voc_images):
        raise DatasetPreparationError("VOC split references a missing image.")
    if not set(voc_train).issubset(voc_masks) or not set(voc_val).issubset(voc_masks):
        raise DatasetPreparationError("VOC segmentation split references a missing mask.")

    return {
        "voc": {
            "jpeg_images": len(voc_images),
            "segmentation_masks": len(voc_masks),
            "train_ids": len(voc_train),
            "val_ids": len(voc_val),
            "train_val_overlap": 0,
        },
        "sbd": {
            "jpeg_images": len(sbd_images),
            "class_mat_files": len(sbd_masks),
            "train_ids": len(sbd_train),
            "val_ids": len(sbd_val),
            "train_noval_ids": len(train_noval),
            "train_noval_voc_val_overlap": 0,
        },
        "pixel_annotation_values_read": False,
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_voc_sbd(
    dataset_root: Path,
    *,
    protocol_path: Path | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    raw = root / "raw"
    final = root / "extracted"
    if final.exists():
        raise DatasetPreparationError(f"Refusing to overwrite existing extraction: {final}")
    archives = [raw / VOC_ARCHIVE, raw / SBD_ARCHIVE]
    support = raw / TRAIN_NOVAL
    preparation_identity: dict[str, Any] = {"code_commit": code_commit}
    if protocol_path is not None:
        protocol = protocol_path.resolve()
        if not protocol.is_file() or protocol.is_symlink():
            raise DatasetPreparationError(f"Preparation protocol is not a regular file: {protocol}")
        protocol_bytes = protocol.read_bytes()
        payload = json.loads(protocol_bytes.decode("utf-8"))
        registered_md5 = {
            name: record["md5"] for name, record in payload.get("artifacts", {}).items()
        }
        if payload.get("dataset_id") != "voc2012_sbd" or registered_md5 != EXPECTED_MD5:
            raise DatasetPreparationError("Preparation protocol does not match registered artifacts.")
        if payload.get("status") != "frozen_before_extraction":
            raise DatasetPreparationError("Preparation protocol was not frozen before extraction.")
        preparation_identity.update(
            {
                "protocol_name": protocol.name,
                "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            }
        )

    for path in [*archives, support]:
        if not path.is_file() or path.is_symlink():
            raise DatasetPreparationError(f"Required regular input file is missing: {path}")

    inputs: dict[str, Any] = {}
    for path in [*archives, support]:
        actual_md5 = file_digest(path, "md5")
        if actual_md5 != EXPECTED_MD5[path.name]:
            raise DatasetPreparationError(f"MD5 mismatch for {path.name}: {actual_md5}")
        inputs[path.name] = {
            "bytes": path.stat().st_size,
            "md5": actual_md5,
            "sha256": file_digest(path, "sha256"),
        }

    staging = root / f".extracted.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    try:
        extracted_counts = {
            archive.name: extract_archive_safely(archive, staging) for archive in archives
        }
        installed_train_noval = staging / "benchmark_RELEASE" / "dataset" / TRAIN_NOVAL
        with support.open("rb") as source, installed_train_noval.open("xb") as output:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        if file_digest(installed_train_noval, "md5") != EXPECTED_MD5[TRAIN_NOVAL]:
            raise DatasetPreparationError("Installed train_noval split failed its MD5 check.")
        audit = audit_extracted_dataset(staging, installed_train_noval)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": "voc2012_sbd",
            "sources": {
                VOC_ARCHIVE: {
                    "official_url": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                    "download_url": "https://d2l-data.s3-accelerate.amazonaws.com/VOCtrainval_11-May-2012.tar",
                    "mirror_disclosed": True,
                },
                SBD_ARCHIVE: {
                    "official_url": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
                    "download_url": "same_as_official",
                },
                TRAIN_NOVAL: {
                    "official_url": "https://www.cs.cornell.edu/~bharathh/train_noval.txt",
                    "download_url": "same_as_official",
                },
            },
            "inputs": inputs,
            "preparation_identity": preparation_identity,
            "extracted_regular_files": extracted_counts,
            "audit": audit,
            "policies": {
                "voc_val_used_for_training": False,
                "pixel_annotation_values_read_during_preparation": False,
                "archives_extracted_without_links_or_special_files": True,
            },
        }
        _write_json_exclusive(staging / "dataset_manifest.json", manifest)
        os.replace(staging, final)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_voc_segmentation_train_tags(
    dataset_root: Path,
    class_names: Iterable[str],
    *,
    output_name: str = "voc_segmentation_train_tags_v0",
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Export mask-free image tags for the VOC segmentation train IDs."""
    root = dataset_root.resolve()
    extracted = root / "extracted"
    voc = extracted / "VOCdevkit" / "VOC2012"
    dataset_manifest_path = extracted / "dataset_manifest.json"
    if not dataset_manifest_path.is_file() or dataset_manifest_path.is_symlink():
        raise DatasetPreparationError("A completed dataset manifest is required before tag export.")
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    dataset_manifest = json.loads(dataset_manifest_bytes.decode("utf-8"))
    if dataset_manifest.get("status") != "complete":
        raise DatasetPreparationError("Dataset preparation is not complete.")
    if dataset_manifest.get("audit", {}).get("pixel_annotation_values_read") is not False:
        raise DatasetPreparationError("Dataset preparation did not preserve the no-pixel-read boundary.")

    names = tuple(class_names)
    all_tags, tag_metadata = load_voc_image_level_tags(voc, names)
    segmentation_split = voc / "ImageSets" / "Segmentation" / "train.txt"
    segmentation_ids = read_split(segmentation_split)
    missing = [image_id for image_id in segmentation_ids if image_id not in all_tags]
    if missing:
        raise DatasetPreparationError(f"VOC segmentation train IDs missing classification labels: {len(missing)}")
    empty = [image_id for image_id in segmentation_ids if not all_tags[image_id]]
    if empty:
        raise DatasetPreparationError(f"VOC segmentation train IDs without positive tags: {len(empty)}")

    derived = root / "derived"
    derived.mkdir(exist_ok=True)
    final = derived / output_name
    if final.exists():
        raise DatasetPreparationError(f"Refusing to overwrite existing tag export: {final}")
    staging = derived / f".{output_name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        records_path = staging / "records.jsonl"
        class_counts = {name: 0 for name in names}
        digest = hashlib.sha256()
        with records_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row_index, image_id in enumerate(segmentation_ids):
                labels = all_tags[image_id]
                for label in labels:
                    class_counts[label] += 1
                record = {"row_index": row_index, "image_id": image_id, "class_names": list(labels)}
                line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

        source_files = [
            segmentation_split,
            *[voc / "ImageSets" / "Main" / f"{name}_train.txt" for name in names],
        ]
        source_hashes = {
            str(path.relative_to(extracted)).replace("\\", "/"): file_digest(path, "sha256")
            for path in source_files
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "dataset_id": "voc2012_sbd",
            "split": "voc2012_segmentation_train",
            "record_count": len(segmentation_ids),
            "class_names": list(names),
            "class_positive_counts": class_counts,
            "difficult_policy": tag_metadata["difficult_policy"],
            "difficult_counts_in_full_voc_train": tag_metadata["difficult_counts"],
            "records_sha256": digest.hexdigest(),
            "dataset_manifest_sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
            "source_file_sha256": source_hashes,
            "code_commit": code_commit,
            "segmentation_masks_read": False,
            "voc_val_read": False,
        }
        _write_json_exclusive(staging / "manifest.json", manifest)
        os.replace(staging, final)
        directory_fd = os.open(derived, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
