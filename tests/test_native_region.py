import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ov_probe.io import InputValidationError, load_config, load_region_bundle
from ov_probe.native_region import (
    candidate_cache_fingerprint,
    discover_native_region_ids,
    load_native_region_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.reshape(-1), bitorder="little")


def _write_protocol(path: Path, checkpoint: Path, *, text_selection: bool = False) -> None:
    payload = {
        "format_version": 1,
        "dataset": "LoveDA",
        "split": "train",
        "candidate_subset": "all_train_candidates",
        "direct_pixel_gt_used": False,
        "love_da_val_used": False,
        "oracle_used": False,
        "e2_used": False,
        "selection_used_gt": False,
        "selection_used_text_prediction": text_selection,
        "uses_gt_derived_image_tags": True,
        "image_level_weak_tag_origin": (
            "LoveDA Train pixel-mask presence simulation with a >=16-pixel threshold"
        ),
        "label_derivations": {
            "sam3_source": "sam3_candidate_source_class",
            "cam": "cam_mask_mean_top1",
        },
        "feature_source": {
            "array": "region_features",
            "model_name": "ViT-B-32",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "feature_dimension": 512,
            "stored_dtype": "float16",
            "view_fusion": "normalized mean of context and mask-emphasized features",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_native_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path, str]:
    candidate_dir = tmp_path / "candidates"
    region_dir = tmp_path / "regions"
    cam_dir = tmp_path / "cams_train"
    candidate_dir.mkdir()
    region_dir.mkdir()
    cam_dir.mkdir()
    image_id = "loveda_train_rural_0001"
    mask0 = np.asarray([[1, 1, 0], [0, 1, 0]], dtype=bool)
    mask1 = np.asarray([[1, 1], [1, 1]], dtype=bool)
    packed0, packed1 = _pack(mask0), _pack(mask1)
    np.savez_compressed(
        candidate_dir / f"{image_id}.npz",
        format_version=np.asarray([1], dtype=np.int16),
        image_shape=np.asarray([4, 5], dtype=np.int32),
        packed_masks=np.concatenate([packed0, packed1]).astype(np.uint8),
        offsets=np.asarray([0, len(packed0), len(packed0) + len(packed1)], dtype=np.int64),
        shapes=np.asarray([[2, 3], [2, 2]], dtype=np.int32),
        origins=np.asarray([[0, 0], [3, 2]], dtype=np.int32),
        boxes=np.asarray([[0, 0, 2, 2], [3, 2, 5, 4]], dtype=np.int32),
        areas=np.asarray([3, 4], dtype=np.int64),
        scores=np.asarray([0.8, 0.7], dtype=np.float32),
        class_ids=np.asarray([1, 2], dtype=np.int16),
        prompt_ids=np.asarray([0, 1], dtype=np.int16),
    )
    candidate_metadata = {
        "format_version": 1,
        "image_id": image_id,
        "image_shape": [4, 5],
        "candidate_count": 2,
        "foreground_only": True,
        "mask_encoding": "flattened-packbits-little",
        "data_file": f"{image_id}.npz",
        "prompts": [
            {"id": 0, "class_id": 1, "class_name": "building", "prompt": "building"},
            {"id": 1, "class_id": 2, "class_name": "road", "prompt": "road"},
        ],
        "class_candidate_counts": {"building": 1, "road": 1},
        "provenance": {"split": "train", "pixel_gt_used": False},
    }
    (candidate_dir / f"{image_id}.json").write_text(
        json.dumps(candidate_metadata), encoding="utf-8"
    )
    fingerprint = candidate_cache_fingerprint(candidate_dir, image_id)

    features = np.zeros((2, 512), dtype=np.float16)
    features[0, 0] = 1
    features[1, 1] = 1
    scores = np.zeros((2, 6), dtype=np.float32)
    scores[0, :2] = [0.1, 0.9]
    scores[1, :2] = [0.2, 0.8]
    np.savez_compressed(
        region_dir / f"{image_id}.npz",
        format_version=np.asarray([1], dtype=np.int16),
        candidate_indices=np.asarray([0, 1], dtype=np.int32),
        scores=scores,
        class_ids=np.arange(1, 7, dtype=np.int16),
        active_class_ids=np.asarray([1, 2], dtype=np.int16),
        predicted_class_ids=np.asarray([2, 2], dtype=np.int16),
        margins=np.asarray([0.8, 0.6], dtype=np.float32),
        crop_boxes=np.asarray([[0, 0, 3, 3], [2, 1, 5, 4]], dtype=np.int32),
        mask_fractions=np.asarray([0.3, 0.4], dtype=np.float32),
        region_features=features,
    )
    region_metadata = {
        "format_version": 1,
        "image_id": image_id,
        "candidate_count": 2,
        "candidate_cache_sha256": fingerprint,
        "region_features_saved": True,
        "region_feature_dtype": "float16",
        "candidate_image_shape": [4, 5],
        "model_name": "ViT-B-32",
        "weights_source": "/read-only/checkpoints/RemoteCLIP-ViT-B-32.pt",
        "view_fusion": "normalized mean of context and mask-emphasized features",
        "class_ids": [1, 2, 3, 4, 5, 6],
        "active_class_ids": [1, 2],
        "pixel_gt_used": False,
    }
    (region_dir / f"{image_id}.json").write_text(
        json.dumps(region_metadata), encoding="utf-8"
    )

    cams = np.zeros((6, 4, 5), dtype=np.float16)
    cams[0] = 0.9
    cams[1] = 0.1
    np.savez_compressed(
        cam_dir / f"{image_id}.npz",
        cams=cams,
        class_ids=np.arange(1, 7, dtype=np.int16),
    )
    checkpoint = tmp_path / "RemoteCLIP-ViT-B-32.pt"
    checkpoint.write_bytes(b"tiny-test-checkpoint")
    protocol = tmp_path / "region_protocol.json"
    _write_protocol(protocol, checkpoint)

    cfg = load_config(ROOT / "configs" / "region_probe_v0.yaml", ROOT)
    cfg["paths"].update({
        "remoteclip_checkpoint": str(checkpoint),
        "region_feature_cache": str(region_dir),
        "candidate_cache_dir": str(candidate_dir),
        "cam_cache_dir": str(cam_dir),
        "region_provenance_file": str(protocol),
    })
    cfg["region_input"].update({
        "max_regions_per_class": None,
        "require_all_classes": False,
        "limit_images": None,
    })
    return cfg, candidate_dir, region_dir, cam_dir, protocol, image_id


def test_native_region_join_recomputes_independent_labels(tmp_path):
    cfg, _, _, _, _, _ = _write_native_fixture(tmp_path)
    bundle = load_native_region_directory(cfg)
    assert bundle.features.shape == (2, 512)
    assert bundle.sam3_source_labels == ["building", "road"]
    assert bundle.cam_labels == ["building", "building"]
    assert bundle.metadata["stored_remoteclip_predicted_class_ids_used_as_labels"] is False
    assert bundle.metadata["selected_records"] == [
        {"image_id": "loveda_train_rural_0001", "candidate_index": 0},
        {"image_id": "loveda_train_rural_0001", "candidate_index": 1},
    ]


def test_discovery_ignores_summary_and_prototype_artifacts(tmp_path):
    cfg, _, region_dir, _, _, image_id = _write_native_fixture(tmp_path)
    del cfg
    (region_dir / "summary.json").write_text("{}", encoding="utf-8")
    (region_dir / "visual_prototypes.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(region_dir / "visual_prototypes.npz", prototypes=np.zeros((6, 512)))
    assert discover_native_region_ids(region_dir, require_all_pairs=True) == [image_id]


def test_discovery_rejects_orphan_loveda_train_record(tmp_path):
    cfg, _, region_dir, _, _, _ = _write_native_fixture(tmp_path)
    del cfg
    (region_dir / "loveda_train_urban_orphan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InputValidationError, match="orphan companions"):
        discover_native_region_ids(region_dir, require_all_pairs=True)


def test_candidate_fingerprint_mismatch_is_rejected(tmp_path):
    cfg, _, region_dir, _, _, image_id = _write_native_fixture(tmp_path)
    metadata_path = region_dir / f"{image_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["candidate_cache_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(InputValidationError, match="fingerprint mismatch"):
        load_native_region_directory(cfg)


def test_region_candidate_reordering_is_rejected(tmp_path):
    cfg, _, region_dir, _, _, image_id = _write_native_fixture(tmp_path)
    data_path = region_dir / f"{image_id}.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["candidate_indices"] = np.asarray([1, 0], dtype=np.int32)
    np.savez_compressed(data_path, **arrays)
    with pytest.raises(InputValidationError, match="ordering is invalid"):
        load_native_region_directory(cfg)


def test_text_selected_protocol_is_rejected(tmp_path):
    cfg, _, _, _, protocol, _ = _write_native_fixture(tmp_path)
    _write_protocol(protocol, Path(cfg["paths"]["remoteclip_checkpoint"]), text_selection=True)
    with pytest.raises(InputValidationError, match="selection_used_text_prediction=false"):
        load_native_region_directory(cfg)


def test_seeded_source_class_reservoir_is_reproducible(tmp_path):
    cfg, _, _, _, _, _ = _write_native_fixture(tmp_path)
    cfg["region_input"]["max_regions_per_class"] = 1
    first = load_native_region_directory(cfg)
    second = load_native_region_directory(cfg)
    assert first.metadata["selected_records"] == second.metadata["selected_records"]
    assert first.metadata["ordered_record_key_sha256"] == second.metadata["ordered_record_key_sha256"]
    assert first.features.shape == (2, 512)


def test_candidate_source_outside_active_classes_is_rejected(tmp_path):
    cfg, _, region_dir, _, _, image_id = _write_native_fixture(tmp_path)
    data_path = region_dir / f"{image_id}.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["active_class_ids"] = np.asarray([1], dtype=np.int16)
    arrays["predicted_class_ids"] = np.asarray([1, 1], dtype=np.int16)
    arrays["margins"] = np.asarray([np.nan, np.nan], dtype=np.float32)
    np.savez_compressed(data_path, **arrays)
    metadata_path = region_dir / f"{image_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["active_class_ids"] = [1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(InputValidationError, match="absent from the image active classes"):
        load_native_region_directory(cfg)


def test_zero_region_feature_is_rejected(tmp_path):
    cfg, _, region_dir, _, _, image_id = _write_native_fixture(tmp_path)
    data_path = region_dir / f"{image_id}.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["region_features"][0] = 0
    np.savez_compressed(data_path, **arrays)
    with pytest.raises(InputValidationError, match="zero vectors"):
        load_native_region_directory(cfg)


def test_generic_unkeyed_region_bundle_is_disabled(tmp_path):
    cfg = load_config(ROOT / "configs" / "region_probe_v0.yaml", ROOT)
    path = tmp_path / "unkeyed.npz"
    np.savez_compressed(path, features=np.eye(2, 512, dtype=np.float32))
    with pytest.raises(InputValidationError, match="Generic formal region bundles are disabled"):
        load_region_bundle(str(path), None, cfg)


def test_output_root_cannot_escape_second_project(tmp_path):
    payload = yaml.safe_load((ROOT / "configs" / "region_probe_v0.yaml").read_text(encoding="utf-8"))
    payload["paths"]["output_root"] = str(tmp_path / "outside")
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(InputValidationError, match="output_root must be a named subdirectory"):
        load_config(config_path, ROOT)
