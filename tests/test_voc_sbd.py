from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from ov_probe.voc_sbd import (
    DatasetPreparationError,
    audit_extracted_dataset,
    extract_archive_safely,
    inspect_archive,
    load_voc_image_level_tags,
)


def _archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    _archive(archive, {"../escape.txt": b"bad"})
    with pytest.raises(DatasetPreparationError, match="Unsafe archive member"):
        inspect_archive(archive)


def test_safe_extract_rejects_windows_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad-windows.tar"
    _archive(archive, {"..\\escape.txt": b"bad"})
    with pytest.raises(DatasetPreparationError, match="Backslash"):
        inspect_archive(archive)


def test_safe_extract_round_trip(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tar"
    _archive(archive, {"root/a.txt": b"alpha", "root/b.bin": b"beta"})
    target = tmp_path / "out"
    assert extract_archive_safely(archive, target) == 2
    assert (target / "root" / "a.txt").read_bytes() == b"alpha"
    assert (target / "root" / "b.bin").read_bytes() == b"beta"


def _touch_many(directory: Path, names: list[str], suffix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}{suffix}").write_bytes(b"x")


def test_split_audit_rejects_voc_val_leakage(tmp_path: Path) -> None:
    voc = tmp_path / "VOCdevkit" / "VOC2012"
    sbd = tmp_path / "benchmark_RELEASE" / "dataset"
    _touch_many(voc / "JPEGImages", ["train", "val"], ".jpg")
    _touch_many(voc / "SegmentationClass", ["train", "val"], ".png")
    split_dir = voc / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("train\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("val\n", encoding="utf-8")
    _touch_many(sbd / "img", ["train", "val"], ".jpg")
    _touch_many(sbd / "cls", ["train", "val"], ".mat")
    (sbd / "train.txt").write_text("train\n", encoding="utf-8")
    (sbd / "val.txt").write_text("val\n", encoding="utf-8")
    train_noval = tmp_path / "train_noval.txt"
    train_noval.write_text("train\nval\n", encoding="utf-8")
    with pytest.raises(DatasetPreparationError, match="voc_val_overlap=1"):
        audit_extracted_dataset(tmp_path, train_noval)


def test_split_audit_accepts_disjoint_train_noval(tmp_path: Path) -> None:
    voc = tmp_path / "VOCdevkit" / "VOC2012"
    sbd = tmp_path / "benchmark_RELEASE" / "dataset"
    _touch_many(voc / "JPEGImages", ["train", "val"], ".jpg")
    _touch_many(voc / "SegmentationClass", ["train", "val"], ".png")
    split_dir = voc / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("train\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("val\n", encoding="utf-8")
    _touch_many(sbd / "img", ["train", "extra"], ".jpg")
    _touch_many(sbd / "cls", ["train", "extra"], ".mat")
    (sbd / "train.txt").write_text("train\n", encoding="utf-8")
    (sbd / "val.txt").write_text("extra\n", encoding="utf-8")
    train_noval = tmp_path / "train_noval.txt"
    train_noval.write_text("train\nextra\n", encoding="utf-8")
    audit = audit_extracted_dataset(tmp_path, train_noval)
    assert audit["sbd"]["train_noval_ids"] == 2
    assert audit["sbd"]["train_noval_voc_val_overlap"] == 0
    assert audit["pixel_annotation_values_read"] is False


def test_voc_classification_tags_use_registered_difficult_policy(tmp_path: Path) -> None:
    main = tmp_path / "ImageSets" / "Main"
    main.mkdir(parents=True)
    (main / "cat_train.txt").write_text("a 1\nb 0\nc -1\n", encoding="utf-8")
    (main / "dog_train.txt").write_text("a -1\nb 1\nc 0\n", encoding="utf-8")
    tags, metadata = load_voc_image_level_tags(tmp_path, ["cat", "dog"])
    assert tags == {"a": ("cat",), "b": ("cat", "dog"), "c": ("dog",)}
    assert metadata["difficult_counts"] == {"cat": 1, "dog": 1}
    assert metadata["segmentation_masks_read"] is False


def test_voc_classification_tags_reject_val_and_row_reordering(tmp_path: Path) -> None:
    main = tmp_path / "ImageSets" / "Main"
    main.mkdir(parents=True)
    (main / "cat_train.txt").write_text("a 1\nb -1\n", encoding="utf-8")
    (main / "dog_train.txt").write_text("b 1\na -1\n", encoding="utf-8")
    with pytest.raises(DatasetPreparationError, match="not identically ordered"):
        load_voc_image_level_tags(tmp_path, ["cat", "dog"])
    with pytest.raises(DatasetPreparationError, match="restricted to the VOC train split"):
        load_voc_image_level_tags(tmp_path, ["cat"], split="val")
