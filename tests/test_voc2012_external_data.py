from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from ov_probe.io import InputValidationError
from ov_probe.voc2012_external_data import (
    _read_val_ids,
    create_voc2012_external_data_manifest,
    load_voc2012_external_data_config,
    validate_voc_val_image_path,
    validate_safe_tar_members,
)


def _tar_with_member(path: Path, name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> None:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = 1 if kind == tarfile.REGTYPE else 0
    with tarfile.open(path, "w") as archive:
        archive.addfile(info, io.BytesIO(b"x") if info.size else None)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("VOCdevkit/link", tarfile.SYMTYPE), ("VOCdevkit/hard", tarfile.LNKTYPE)],
)
def test_safe_tar_validation_rejects_traversal_and_links(tmp_path: Path, name: str, kind: bytes) -> None:
    archive = tmp_path / "unsafe.tar"
    _tar_with_member(archive, name, kind)
    with pytest.raises(InputValidationError, match="unsafe|forbidden"):
        validate_safe_tar_members(archive)


def _protocol() -> dict[str, object]:
    return {
        "status": "frozen_pre_result", "scientific_evidence": False,
        "role": "external PASCAL VOC 2012 validation-image acquisition and manifest only; no model evaluation",
        "dataset": {"name": "PASCAL VOC 2012", "year": "2012", "image_domain": "natural imagery", "purpose": "external independent evaluation", "archive_url": "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", "archive_filename": "VOCtrainval_11-May-2012.tar", "archive_md5": "6cd6e144f989b92b3379bac3b3de84fd", "expected_trainval_image_count": 11530, "expected_val_id_count": 1449},
        "input_scope": {"allowed_split_list": "ImageSets/Segmentation/val.txt", "allowed_image_root": "JPEGImages", "quarantined_annotation_roots": ["SegmentationClass", "SegmentationObject", "Annotations"], "quarantined_class_specific_files": True},
        "constraints": {key: False for key in ("network_download", "semantic_label_read_or_decode", "detection_label_read_or_decode", "class_specific_file_read", "downstream_evaluation", "model_execution", "prompt_selection", "prototype_selection", "ground_truth_retrieval", "sam3_rerun", "training", "overwrite")},
        "sha256": "a" * 64,
    }


def _fixture_voc_root(tmp_path: Path, *, duplicate: bool = False, missing: bool = False) -> tuple[Path, list[str]]:
    raw = tmp_path / "datasets" / "VOCdevkit" / "VOC2012"
    (raw / "ImageSets" / "Segmentation").mkdir(parents=True)
    images = raw / "JPEGImages"
    images.mkdir()
    ids = [f"2012_{index:06d}" for index in range(1449)]
    listed = ids + ([ids[0]] if duplicate else [])
    (raw / "ImageSets" / "Segmentation" / "val.txt").write_text("\n".join(listed) + "\n", encoding="ascii")
    for image_id in ids[:-1] if missing else ids:
        (images / f"{image_id}.jpg").write_bytes(b"not-decoded-jpeg")
    return raw, ids


def test_manifest_generation_uses_1449_id_fixture_and_only_jpegs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw, ids = _fixture_voc_root(tmp_path)
    archive = tmp_path / "VOCtrainval_11-May-2012.tar"
    _tar_with_member(archive, "VOCdevkit/VOC2012/JPEGImages/2012_000000.jpg")
    cfg = {"paths": {"archive_file": str(archive), "raw_data_root": str(raw)}, "repository_anchor": {"code_commit": "b" * 40, "protocol_sha256": "a" * 64}}
    protocol = _protocol()
    monkeypatch.setattr("ov_probe.voc2012_external_data.md5_file", lambda _: "6cd6e144f989b92b3379bac3b3de84fd")
    monkeypatch.setattr("ov_probe.voc2012_external_data._assert_untracked_data_root", lambda *_: None)
    monkeypatch.setattr("ov_probe.voc2012_external_data._list_trainval_jpegs", lambda _: [Path("fixture.jpg")] * 11530)
    manifest = create_voc2012_external_data_manifest(cfg, protocol, tmp_path / "out", tmp_path)
    assert manifest["val_split"]["id_count"] == 1449
    assert [entry["image_id"] for entry in manifest["model_input_files"]] == ids
    assert all(entry["path"].startswith("JPEGImages/") for entry in manifest["model_input_files"])
    assert manifest["quarantined_annotation_roots"] == ["SegmentationClass", "SegmentationObject", "Annotations"]
    annotation = raw / "Annotations" / f"{ids[0]}.xml"
    annotation.parent.mkdir()
    annotation.write_text("not read", encoding="utf-8")
    with pytest.raises(InputValidationError, match="never an annotation"):
        validate_voc_val_image_path(raw, annotation, ids[0])
    assert json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))["model_input_files"] == manifest["model_input_files"]


def test_manifest_rejects_missing_images_and_duplicate_val_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "VOCtrainval_11-May-2012.tar"
    _tar_with_member(archive, "VOCdevkit/VOC2012/JPEGImages/2012_000000.jpg")
    monkeypatch.setattr("ov_probe.voc2012_external_data.md5_file", lambda _: "6cd6e144f989b92b3379bac3b3de84fd")
    monkeypatch.setattr("ov_probe.voc2012_external_data._assert_untracked_data_root", lambda *_: None)
    for issue in ("missing", "duplicate"):
        raw, _ = _fixture_voc_root(tmp_path / issue, **{issue: True})
        cfg = {"paths": {"archive_file": str(archive), "raw_data_root": str(raw)}, "repository_anchor": {"code_commit": "b" * 40, "protocol_sha256": "a" * 64}}
        with pytest.raises(InputValidationError, match="missing|duplicate"):
            create_voc2012_external_data_manifest(cfg, _protocol(), tmp_path / f"out-{issue}", tmp_path)


def test_config_is_project_contained_and_protocol_scope_is_frozen(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    protocol = json.loads((source / "configs" / "voc2012_external_data_protocol_v1.json").read_text(encoding="utf-8"))
    (project / "configs" / "voc2012_external_data_protocol_v1.json").write_text(json.dumps(protocol), encoding="utf-8")
    config = {"experiment": {"overwrite": False}, "paths": {"archive_file": None, "raw_data_root": None, "protocol_file": "configs/voc2012_external_data_protocol_v1.json", "output_root": "outputs/voc2012_external_data_v1"}}
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, registered = load_voc2012_external_data_config(config_path, project)
    assert loaded["paths"]["raw_data_root"] is None
    assert registered["dataset"]["expected_val_id_count"] == 1449
    config["paths"]["archive_file"] = "../escape.tar"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(InputValidationError, match="escapes|relative"):
        load_voc2012_external_data_config(config_path, project)
    protocol["constraints"]["model_execution"] = True
    (project / "configs" / "voc2012_external_data_protocol_v1.json").write_text(json.dumps(protocol), encoding="utf-8")
    config["paths"]["archive_file"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(InputValidationError, match="constraints"):
        load_voc2012_external_data_config(config_path, project)


def test_val_id_reader_rejects_duplicates_without_label_access(tmp_path: Path) -> None:
    ids = tmp_path / "val.txt"
    ids.write_text("a\na\n", encoding="ascii")
    with pytest.raises(InputValidationError, match="duplicate"):
        _read_val_ids(ids)
