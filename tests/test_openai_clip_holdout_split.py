from __future__ import annotations

import hashlib
import json

import pytest

from ov_probe.io import InputValidationError
from ov_probe import openai_clip_holdout_split
from ov_probe.openai_clip_holdout_split import split_records
from scripts import create_openai_clip_holdout_split as holdout_runner


def _records(image_count: int = 12) -> list[dict[str, object]]:
    labels = ["building", "road", "water", "barren", "forest", "agriculture"]
    return [
        {"image_id": f"image-{index:03d}", "candidate_index": 0, "sam3_source_label": labels[index % len(labels)]}
        for index in range(image_count)
    ]


def test_split_is_deterministic_exact_and_image_disjoint() -> None:
    rows = _records()
    first = split_records(rows, seed=42, development_image_count=8)
    second = split_records(list(reversed(rows)), seed=42, development_image_count=8)
    first_ids = {row["image_id"] for row in first[0]}
    second_ids = {row["image_id"] for row in second[0]}
    assert first_ids == second_ids
    assert len(first_ids) == 8
    assert not first_ids & {row["image_id"] for row in first[1]}
    assert len(first[0]) + len(first[1]) == len(rows)


def test_split_rejects_duplicate_record_keys() -> None:
    rows = _records()
    rows.append(dict(rows[0]))
    with pytest.raises(InputValidationError, match="unique"):
        split_records(rows, seed=42, development_image_count=8)


def test_create_rejects_class_missing_partition(tmp_path, monkeypatch) -> None:
    rows = _records(6)
    package = tmp_path / "pixel_pack"
    package.mkdir()
    (package / "records.jsonl").write_text(
        "".join(f'{__import__("json").dumps(row)}\n' for row in rows), encoding="utf-8"
    )
    validation = {
        "bundle_id": "synthetic",
        "record_count": 6,
        "image_count": 6,
        "ordered_record_key_sha256": openai_clip_holdout_split._ordered_key_sha256(rows),
    }
    monkeypatch.setattr(openai_clip_holdout_split, "validate_region_pixel_pack", lambda *_: validation)
    protocol = {
        "pixel_pack": validation,
        "classes": ["building", "road", "water", "barren", "forest", "agriculture"],
        "partition": {"seed": 42, "development_image_count": 3, "heldout_image_count": 3},
    }
    with pytest.raises(InputValidationError, match="SAM3 label"):
        openai_clip_holdout_split.create_holdout_split(package, protocol, tmp_path / "output")


def _valid_create_inputs(tmp_path, monkeypatch):
    labels = ["building", "road", "water", "barren", "forest", "agriculture"]
    rows = [
        {
            "image_id": f"image-{image_index:03d}",
            "candidate_index": label_index,
            "sam3_source_label": label,
        }
        for image_index in range(24)
        for label_index, label in enumerate(labels)
    ]
    package = tmp_path / "pixel_pack"
    package.mkdir()
    (package / "records.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )
    validation = {
        "bundle_id": "synthetic",
        "record_count": len(rows),
        "image_count": 24,
        "ordered_record_key_sha256": openai_clip_holdout_split._ordered_key_sha256(rows),
    }
    monkeypatch.setattr(openai_clip_holdout_split, "validate_region_pixel_pack", lambda *_: validation)
    protocol = {
        "pixel_pack": validation,
        "classes": labels,
        "partition": {"seed": 42, "development_image_count": 12, "heldout_image_count": 12},
    }
    return package, protocol


def test_create_allows_precreated_empty_output_directory(tmp_path, monkeypatch) -> None:
    package, protocol = _valid_create_inputs(tmp_path, monkeypatch)
    output = tmp_path / "reserved_run"
    output.mkdir()

    manifest = openai_clip_holdout_split.create_holdout_split(package, protocol, output)

    assert manifest["status"] == "completed"
    assert (output / "manifest.json").is_file()
    assert (output / "development_records.jsonl").is_file()
    assert (output / "heldout_records.jsonl").is_file()


def test_create_rejects_precreated_nonempty_output_directory(tmp_path, monkeypatch) -> None:
    package, protocol = _valid_create_inputs(tmp_path, monkeypatch)
    output = tmp_path / "occupied_run"
    output.mkdir()
    (output / "existing.txt").write_text("protected", encoding="utf-8")

    with pytest.raises(InputValidationError, match="empty"):
        openai_clip_holdout_split.create_holdout_split(package, protocol, output)


def test_repository_anchor_rejects_mismatched_protocol_hash(tmp_path, monkeypatch) -> None:
    protocol = tmp_path / "openai_clip_holdout_split_protocol_v1.json"
    protocol.write_text('{"status":"frozen"}\n', encoding="utf-8")
    monkeypatch.setattr(holdout_runner, "CANONICAL_PROTOCOL_PATH", protocol)

    def fake_run_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        assert args == ("status", "--porcelain", "--untracked-files=no")
        return ""

    monkeypatch.setattr(holdout_runner, "_run_git", fake_run_git)
    actual_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert actual_hash != "b" * 64
    with pytest.raises(InputValidationError, match="protocol hash"):
        holdout_runner._verify_repository_anchor("a" * 40, "b" * 64)
