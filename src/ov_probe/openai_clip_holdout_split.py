"""Deterministically partition a validated pixel package without model computation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import InputValidationError, sha256_file
from .pixel_pack import validate_region_pixel_pack


_REQUIRED_ROW_FIELDS = {"image_id", "candidate_index", "sam3_source_label"}


def _ordered_key_sha256(records: Iterable[dict[str, Any]]) -> str:
    text = "\n".join(
        f"{str(row['image_id'])}:{int(row['candidate_index'])}" for row in records
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise InputValidationError("Split source records may not be empty.")
    keys: set[tuple[str, int]] = set()
    for row in records:
        if not isinstance(row, dict) or not _REQUIRED_ROW_FIELDS.issubset(row):
            raise InputValidationError("Each split record needs image_id, candidate_index, and sam3_source_label.")
        image_id = str(row["image_id"])
        if not image_id:
            raise InputValidationError("Split record image_id may not be empty.")
        try:
            candidate_index = int(row["candidate_index"])
        except (TypeError, ValueError) as exc:
            raise InputValidationError("Split record candidate_index must be an integer.") from exc
        if candidate_index < 0:
            raise InputValidationError("Split record candidate_index must be non-negative.")
        key = (image_id, candidate_index)
        if key in keys:
            raise InputValidationError("Split source (image_id, candidate_index) keys must be unique.")
        keys.add(key)
        if not str(row["sam3_source_label"]):
            raise InputValidationError("Split record sam3_source_label may not be empty.")


def split_records(
    records: list[dict[str, Any]], seed: int, development_image_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministic image-disjoint development and heldout record lists."""
    _validate_records(records)
    image_ids = {str(row["image_id"]) for row in records}
    if development_image_count <= 0 or development_image_count >= len(image_ids):
        raise InputValidationError("development_image_count must be between zero and the image count.")
    ranked = sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest(),
    )
    development_ids = set(ranked[:development_image_count])
    development = [row for row in records if str(row["image_id"]) in development_ids]
    heldout = [row for row in records if str(row["image_id"]) not in development_ids]
    if not development or not heldout:
        raise InputValidationError("Both split partitions must contain records.")
    if {str(row["image_id"]) for row in development} & {str(row["image_id"]) for row in heldout}:
        raise InputValidationError("Split partitions are not image-disjoint.")
    if len(development) + len(heldout) != len(records):
        raise InputValidationError("Split does not cover every source record exactly once.")
    return development, heldout


def _read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"Malformed source record at line {line_number}.") from exc
            if not isinstance(value, dict):
                raise InputValidationError("Split source records must be JSON objects.")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _partition_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "image_count": len({str(row["image_id"]) for row in rows}),
        "sam3_source_label_counts": dict(sorted(Counter(str(row["sam3_source_label"]) for row in rows).items())),
        "ordered_record_key_sha256": _ordered_key_sha256(rows),
    }


def create_holdout_split(
    pixel_pack: str | Path, protocol: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Validate the immutable package and create one new split directory."""
    package = Path(pixel_pack).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise InputValidationError("Split output directory must not already exist.")
    validation = validate_region_pixel_pack(package, package / "encoder_compare_protocol_v0.json")
    registered = protocol.get("pixel_pack", {})
    for field in ("bundle_id", "record_count", "image_count", "ordered_record_key_sha256"):
        if validation.get(field) != registered.get(field):
            raise InputValidationError(f"Pixel package differs from frozen holdout protocol: {field}")
    rows = _read_records(package / "records.jsonl")
    _validate_records(rows)
    if len(rows) != int(registered["record_count"]):
        raise InputValidationError("Source record count differs from frozen holdout protocol.")
    if _ordered_key_sha256(rows) != registered["ordered_record_key_sha256"]:
        raise InputValidationError("Source ordered record key hash differs from frozen holdout protocol.")
    partition = protocol.get("partition", {})
    development, heldout = split_records(
        rows, int(partition["seed"]), int(partition["development_image_count"])
    )
    if len({str(row["image_id"]) for row in heldout}) != int(partition["heldout_image_count"]):
        raise InputValidationError("Heldout image count differs from frozen holdout protocol.")
    expected_classes = set(protocol["classes"])
    for name, subset in (("development", development), ("heldout", heldout)):
        labels = {str(row["sam3_source_label"]) for row in subset}
        if labels != expected_classes:
            raise InputValidationError(f"{name} partition does not contain exactly the frozen SAM3 label set.")
    destination.mkdir(parents=True, exist_ok=False)
    development_path = destination / "development_records.jsonl"
    heldout_path = destination / "heldout_records.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(heldout_path, heldout)
    manifest = {
        "format_version": 1,
        "status": "completed",
        "scientific_evidence": False,
        "no_model_computation": True,
        "role": protocol.get("role"),
        "source_pixel_package_validation": validation,
        "frozen_pixel_pack": registered,
        "partition": partition,
        "coverage": {
            "source_record_count": len(rows),
            "covered_record_count": len(development) + len(heldout),
            "image_disjoint": True,
            "all_source_records_covered_exactly_once": True,
        },
        "development": {**_partition_metadata(development), "records_sha256": sha256_file(development_path)},
        "heldout": {**_partition_metadata(heldout), "records_sha256": sha256_file(heldout_path)},
    }
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
