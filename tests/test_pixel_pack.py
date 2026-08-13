import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ov_probe.io import InputValidationError, sha256_file
from ov_probe.native_region import NativeCandidate
from ov_probe.pixel_pack import (
    canonical_json_sha256,
    make_pixel_views,
    ordered_key_sha256,
    read_selected_records,
    validate_region_pixel_pack,
)


CLASS_NAMES = ["building", "road", "water", "barren", "forest", "agriculture"]


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    if checksum_path.exists():
        checksum_path.unlink()
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_minimal_pack(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "pack"
    shards = root / "shards"
    shards.mkdir(parents=True)
    image_id = "loveda_train_rural_1"
    keys = [f"{image_id}:0", f"{image_id}:1"]
    ordered_hash = _digest("\n".join(keys).encode("utf-8"))
    class_counts = {"building": 1, "road": 1}
    protocol = {
        "format_version": 1,
        "dataset": "LoveDA",
        "split": "train",
        "direct_pixel_gt_used": False,
        "love_da_val_used": False,
        "oracle_used": False,
        "e2_used": False,
        "selection": {
            "selected_records_sha256": "1" * 64,
            "ordered_record_key_sha256": ordered_hash,
            "record_count": 2,
            "image_count": 1,
            "sampling_reference": "sam3_source_class",
            "sampling_seed": 42,
            "uses_gt_derived_image_tags": True,
            "image_level_weak_tag_origin": (
                "LoveDA Train pixel-mask presence simulation with a >=16-pixel threshold"
            ),
            "class_counts": class_counts,
        },
        "source_paths": {},
        "crop_views": {
            "image_mode": "RGB",
            "context_ratio": 0.25,
            "min_crop_size": 48,
            "background_retain": 0.25,
            "background_rounding": "numpy.rint then clip [0,255] then uint8",
            "mask_encoding": "flattened-packbits-little",
            "view_order": ["context", "mask-emphasized"],
            "view_fusion": "L2 each encoded view, arithmetic mean, L2 fused result",
        },
        "reference_features": {
            "array": "region_features",
            "shape": [2, 512],
            "dtype": "float16",
            "model_name": "ViT-B-32",
            "checkpoint_sha256": "2" * 64,
        },
        "source_implementation_sha256": {
            "candidate_region_scores.py": "3" * 64,
            "remoteclip_backend.py": "4" * 64,
            "candidate_cache.py": "5" * 64,
            "score_candidate_regions.py": "6" * 64,
        },
        "export": {
            "shard_size": 2,
            "compression": "numpy savez_compressed",
            "include_reference_region_features": True,
            "forbid_lossy_images": True,
        },
    }
    protocol_path = tmp_path / "encoder_compare_protocol_v0.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (root / protocol_path.name).write_bytes(protocol_path.read_bytes())

    contexts = [
        np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        np.arange(18, 36, dtype=np.uint8).reshape(2, 3, 3),
    ]
    masks = [
        np.asarray([[1, 0, 1], [0, 1, 0]], dtype=bool),
        np.asarray([[0, 1, 0], [1, 1, 1]], dtype=bool),
    ]
    packed = [np.packbits(mask.reshape(-1), bitorder="little") for mask in masks]
    shard_path = shards / "part-0000.npz"
    np.savez_compressed(
        shard_path,
        format_version=np.asarray([1], dtype=np.int16),
        row_indices=np.asarray([0, 1], dtype=np.int32),
        crop_shapes=np.asarray([[2, 3], [2, 3]], dtype=np.int32),
        crop_boxes=np.asarray([[0, 0, 3, 2], [1, 1, 4, 3]], dtype=np.int32),
        crop_rgb_offsets=np.asarray([0, 18, 36], dtype=np.int64),
        crop_rgb_flat=np.concatenate([item.reshape(-1) for item in contexts]),
        crop_mask_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        crop_mask_bits=np.concatenate(packed),
    )
    rows = []
    labels = ["building", "road"]
    for index, (context, mask, label) in enumerate(zip(contexts, masks, labels)):
        masked = context.astype(np.float32)
        masked[~mask] *= 0.25
        masked = np.rint(masked).clip(0, 255).astype(np.uint8)
        rows.append(
            {
                "row_index": index,
                "image_id": image_id,
                "candidate_index": index,
                "sam3_source_label": label,
                "cam_label": label,
                "image_shape": [1024, 1024],
                "crop_shape": [2, 3],
                "crop_box": [index, index, index + 3, index + 2],
                "mask_area": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "image_sha256": "7" * 64,
                "candidate_cache_sha256": "8" * 64,
                "region_cache_sha256": "9" * 64,
                "context_sha256": _digest(context.tobytes()),
                "crop_mask_sha256": _digest(packed[index].tobytes()),
                "masked_view_sha256": _digest(masked.tobytes()),
                "shard": "shards/part-0000.npz",
            }
        )
    records_path = root / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    reference = np.zeros((2, 512), dtype=np.float16)
    reference[0, 0] = 1
    reference[1, 1] = 1
    np.save(root / "reference_region_features.npy", reference, allow_pickle=False)
    artifact_paths = [
        root / protocol_path.name,
        records_path,
        root / "reference_region_features.npy",
        shard_path,
    ]
    artifacts = {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    protocol_sha = sha256_file(protocol_path)
    manifest = {
        "format_version": 1,
        "dataset": "LoveDA",
        "split": "train",
        "direct_pixel_gt_used": False,
        "love_da_val_used": False,
        "oracle_used": False,
        "e2_used": False,
        "record_count": 2,
        "image_count": 1,
        "class_counts": class_counts,
        "ordered_record_key_sha256": ordered_hash,
        "selected_records_sha256": "1" * 64,
        "protocol_sha256": protocol_sha,
        "crop_views": protocol["crop_views"],
        "reference_features": protocol["reference_features"],
        "source_implementation_sha256": protocol["source_implementation_sha256"],
        "source_inventory_before": {"same": True},
        "source_inventory_after": {"same": True},
        "source_content_inventory_before": {"same": True},
        "source_content_inventory_after": {"same": True},
        "source_run": {
            "code_commit": None,
            "name": None,
            "artifact_sha256": None,
        },
        "exporter_repository_anchor": {
            "code_commit": "a" * 40,
            "protocol_sha256": protocol_sha,
        },
        "image_content_inventory": {
            "file_count": 1,
            "sha256": canonical_json_sha256({image_id: "7" * 64}),
        },
        "candidate_content_inventory": {
            "file_count": 1,
            "sha256": canonical_json_sha256({image_id: "8" * 64}),
        },
        "region_content_inventory": {
            "file_count": 1,
            "sha256": canonical_json_sha256({image_id: "9" * 64}),
        },
        "artifacts": artifacts,
    }
    protocol["selection"].update(
        {
            "source_run_code_commit": None,
            "source_run_name": None,
            "source_run_artifacts_sha256": None,
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (root / protocol_path.name).write_bytes(protocol_path.read_bytes())
    protocol_sha = sha256_file(protocol_path)
    manifest["protocol_sha256"] = protocol_sha
    manifest["exporter_repository_anchor"]["protocol_sha256"] = protocol_sha
    artifacts[protocol_path.name] = {
        "size_bytes": (root / protocol_path.name).stat().st_size,
        "sha256": sha256_file(root / protocol_path.name),
    }
    manifest["bundle_id"] = canonical_json_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_checksums(root)
    (root / "COMPLETE").write_bytes(b"pixel-pack-v1\n")
    return root, protocol_path


def test_make_pixel_views_matches_registered_rounding_at_boundary():
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    mask = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=bool)
    candidate = NativeCandidate(class_id=1, mask=mask, x0=0, y0=0)
    result = make_pixel_views(
        image,
        candidate,
        context_ratio=0.25,
        min_crop_size=1,
        background_retain=0.25,
    )
    assert result.crop_box == (0, 0, 3, 3)
    assert np.array_equal(result.context, image[:3, :3])
    expected_mask = np.zeros((3, 3), dtype=bool)
    expected_mask[:2, :3] = mask
    assert np.array_equal(result.crop_mask, expected_mask)
    expected_masked = image[:3, :3].astype(np.float32)
    expected_masked[~expected_mask] *= 0.25
    assert np.array_equal(
        result.masked, np.rint(expected_masked).clip(0, 255).astype(np.uint8)
    )


def test_selected_record_reader_rejects_registered_reordering(tmp_path):
    rows = [
        {
            "row_index": 0,
            "image_id": "loveda_train_rural_1",
            "candidate_index": 0,
            "sam3_source_label": "building",
            "cam_label": "road",
        },
        {
            "row_index": 1,
            "image_id": "loveda_train_rural_1",
            "candidate_index": 1,
            "sam3_source_label": "road",
            "cam_label": "road",
        },
    ]
    path = tmp_path / "selected.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    records_for_hash = [
        type("Record", (), row) for row in rows
    ]
    protocol = {
        "selection": {
            "selected_records_sha256": sha256_file(path),
            "ordered_record_key_sha256": ordered_key_sha256(records_for_hash),
            "record_count": 2,
            "image_count": 1,
            "class_counts": {"building": 1, "road": 1},
        }
    }
    assert len(read_selected_records(path, protocol)) == 2
    rows.reverse()
    for index, row in enumerate(rows):
        row["row_index"] = index
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol["selection"]["selected_records_sha256"] = sha256_file(path)
    with pytest.raises(InputValidationError, match="ordered region-key"):
        read_selected_records(path, protocol)


def test_pixel_pack_validator_accepts_complete_registered_pack(tmp_path):
    root, protocol = _write_minimal_pack(tmp_path)
    result = validate_region_pixel_pack(root, protocol)
    assert result["status"] == "valid"
    assert result["record_count"] == 2


def test_pixel_pack_validator_rejects_content_corruption(tmp_path):
    root, protocol = _write_minimal_pack(tmp_path)
    with (root / "shards" / "part-0000.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(InputValidationError, match="checksum mismatch"):
        validate_region_pixel_pack(root, protocol)


def test_pixel_pack_validator_rejects_self_declared_protocol(tmp_path):
    root, protocol = _write_minimal_pack(tmp_path)
    forged = tmp_path / "forged_protocol.json"
    value = json.loads(protocol.read_text(encoding="utf-8"))
    value["selection"]["record_count"] = 1
    forged.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(InputValidationError, match="protocol SHA-256"):
        validate_region_pixel_pack(root, forged)


def test_pixel_pack_validator_rejects_checksum_path_traversal(tmp_path):
    root, protocol = _write_minimal_pack(tmp_path)
    with (root / "checksums.sha256").open("a", encoding="utf-8") as handle:
        handle.write(f"{'0' * 64}  ../escape\n")
    with pytest.raises(InputValidationError, match="Unsafe package-relative path"):
        validate_region_pixel_pack(root, protocol)
