"""Unit tests for the blind LoveDA GT evaluation module (synthetic, no GT access)."""

from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # local Windows torch/OpenMP clash; server unaffected

from pathlib import Path

import numpy as np
import pytest

from ov_probe.io import InputValidationError, sha256_file
from ov_probe.loveda_blind_gt import (
    _classification_metrics,
    _region_gt_from_label,
    load_loveda_blind_gt_config,
    run_loveda_blind_gt_evaluate,
    run_loveda_blind_gt_predict,
)

CLASSES = ["building", "road", "water", "barren", "forest", "agriculture"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _make_protocol() -> dict:
    return {
        "format_version": 1,
        "status": "frozen_pre_result",
        "scientific_evidence": True,
        "blind": {"gt_read_before_predictions": False, "declaration": "synthetic test"},
        "dataset": "LoveDA",
        "split": "train",
        "classes": CLASSES,
        "pixel_pack": {"bundle_id": "x" * 64, "record_count": 12, "image_count": 6, "ordered_record_key_sha256": "y" * 64},
        "split_inputs": {
            "manifest_sha256": "a" * 64,
            "development_records_sha256": "b" * 64,
            "heldout_records_sha256": "c" * 64,
            "development_image_count": 4,
            "heldout_image_count": 2,
            "image_disjoint": True,
            "coverage": "synthetic",
        },
        "feature_cache": {
            "manifest_sha256": "d" * 64,
            "features_sha256": "e" * 64,
            "row_partitions_sha256": "f" * 64,
            "feature_shape": [12, 512],
            "feature_dtype": "float16",
        },
        "model": {
            "identity": "OpenAI CLIP ViT-B/32 quick-GELU",
            "architecture": "ViT-B-32-quickgelu",
            "checkpoint_sha256": "g" * 64,
            "feature_dimension": 512,
            "open_clip_version": "3.3.0",
            "load_policy": "weights_only=True, strict=True, eval mode",
        },
        "software": {},
        "prompts": {"group_a_templates": ["{class}"]},
        "strategy": {
            "visual_prototype": "synthetic",
            "text_prototype": "synthetic",
            "text_only_score": "cosine",
            "visual_only_score": "cosine",
            "fused_score": "0.5 text + 0.5 visual",
            "fixed_text_weight": 0.5,
            "fixed_visual_weight": 0.5,
            "prediction_rule": "argmax over 6 classes, no threshold",
            "development_label_use": "sam3_source_label",
        },
        "ground_truth": {
            "color_map": {
                "building": [255, 0, 0],
                "road": [255, 255, 0],
                "water": [0, 0, 255],
                "barren": [159, 129, 183],
                "forest": [0, 255, 0],
                "agriculture": [255, 255, 255],
            },
            "background_color": [0, 0, 0],
            "ignore_color": [255, 0, 255],
            "region_label_rule": "majority vote",
            "unlabeled_rule": "excluded with count disclosed",
        },
        "evaluation": {"region_level": True, "metrics": ["accuracy", "macro_f1", "per_class_f1", "confusion_matrix"], "comparisons": ["text_only", "visual_only", "fused_text_visual"]},
        "constraints": {
            "training": False, "adapter_or_new_network": False, "alpha_tuning": False,
            "prompt_tuning": False, "threshold_tuning": False, "grid_search": False,
            "model_selection_or_decision": False, "region_reencoding": False,
            "remoteclip_feature_or_text_reuse": False, "gt_before_predictions": False,
            "overwrite": False,
        },
    }


def _synthetic_project(tmp_path: Path, config_name: str = "deploy.yaml") -> tuple[dict, dict]:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "inputs").mkdir()
    pixel_pack = root / "inputs" / "pixel_pack"
    (pixel_pack / "shards").mkdir(parents=True)
    # 12 rows: 6 development images, 6 heldout images, one region each
    rng = np.random.default_rng(0)
    class_centers = rng.standard_normal((6, 512), dtype=np.float64)
    class_centers /= np.linalg.norm(class_centers, axis=1, keepdims=True)
    features = np.empty((12, 512), dtype=np.float32)
    for index in range(12):
        center = class_centers[index % 6]
        noise = rng.standard_normal(512, dtype=np.float64) * 0.05
        features[index] = (center + noise).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    rows: list[dict] = []
    shard_row_indices: list[int] = []
    shard_shapes: list[tuple[int, int]] = []
    shard_boxes: list[list[int]] = []
    shard_masks: list[bytes] = []
    for index in range(12):
        image_id = f"loveda_train_rural_{index}"
        label = CLASSES[index % 6]
        crop_h, crop_w = 32, 32
        mask = np.zeros((crop_h, crop_w), dtype=bool)
        mask[8:24, 8:24] = True
        box = [100, 100, 132, 132]
        rows.append({
            "row_index": index, "image_id": image_id, "candidate_index": 0,
            "sam3_source_label": label, "cam_label": label,
            "image_shape": [1024, 1024], "crop_shape": [crop_h, crop_w],
            "crop_box": box, "mask_area": int(mask.sum()), "mask_fraction": 0.25,
            "image_sha256": _sha(f"img{index}"), "candidate_cache_sha256": _sha(f"cand{index}"),
            "region_cache_sha256": _sha(f"reg{index}"), "context_sha256": _sha(f"ctx{index}"),
            "crop_mask_sha256": _sha(f"mask{index}"), "masked_view_sha256": _sha(f"mv{index}"),
            "shard": "shards/part-0000.npz",
        })
        shard_row_indices.append(index)
        shard_shapes.append((crop_h, crop_w))
        shard_boxes.append(box)
        shard_masks.append(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
    _write_jsonl(pixel_pack / "records.jsonl", rows)
    with (pixel_pack / "shards" / "part-0000.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray([1], dtype=np.int16),
            row_indices=np.asarray(shard_row_indices, dtype=np.int32),
            crop_shapes=np.asarray(shard_shapes, dtype=np.int32),
            crop_boxes=np.asarray(shard_boxes, dtype=np.int32),
            crop_rgb_offsets=np.asarray([0] * (len(rows) + 1), dtype=np.int64),
            crop_rgb_flat=np.zeros(1, dtype=np.uint8),
            crop_mask_offsets=np.asarray([0, *np.cumsum([len(m) for m in shard_masks])], dtype=np.int64),
            crop_mask_bits=np.concatenate([np.frombuffer(m, dtype=np.uint8) for m in shard_masks]),
        )
    # split: development rows 0-5, heldout rows 6-11
    development = [rows[i] for i in range(6)]
    heldout = [rows[i] for i in range(6, 12)]
    split_dir = root / "outputs" / "split"
    split_dir.mkdir()
    _write_jsonl(split_dir / "development_records.jsonl", development)
    _write_jsonl(split_dir / "heldout_records.jsonl", heldout)
    partitions = [
        {"row_index": i, "image_id": rows[i]["image_id"], "candidate_index": 0, "partition": "development" if i < 6 else "heldout"}
        for i in range(12)
    ]
    _write_jsonl(split_dir / "row_partitions.jsonl", partitions)
    cache_dir = root / "outputs" / "cache"
    cache_dir.mkdir()
    (cache_dir / "features_openai_clip.npy").write_bytes(b"")  # replaced below
    np.save(cache_dir / "features_openai_clip.npy", features.astype(np.float16), allow_pickle=False)
    cache_manifest = {
        "status": "completed",
        "preprocess": "synthetic",
        "outputs": {
            "features_openai_clip": {"sha256": sha256_file(cache_dir / "features_openai_clip.npy")},
            "row_partitions": {"sha256": sha256_file(split_dir / "row_partitions.jsonl")},
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(cache_manifest), encoding="utf-8")
    split_manifest = {
        "status": "completed",
        "development": {"records_sha256": sha256_file(split_dir / "development_records.jsonl")},
        "heldout": {"records_sha256": sha256_file(split_dir / "heldout_records.jsonl")},
    }
    (split_dir / "manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    checkpoint = root / "inputs" / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint-bytes")
    protocol = _make_protocol()
    protocol["split_inputs"]["manifest_sha256"] = sha256_file(split_dir / "manifest.json")
    protocol["split_inputs"]["development_records_sha256"] = sha256_file(split_dir / "development_records.jsonl")
    protocol["split_inputs"]["heldout_records_sha256"] = sha256_file(split_dir / "heldout_records.jsonl")
    protocol["feature_cache"]["manifest_sha256"] = sha256_file(cache_dir / "manifest.json")
    protocol["feature_cache"]["features_sha256"] = sha256_file(cache_dir / "features_openai_clip.npy")
    protocol["feature_cache"]["row_partitions_sha256"] = sha256_file(split_dir / "row_partitions.jsonl")
    protocol["model"]["checkpoint_sha256"] = sha256_file(checkpoint)
    (root / "configs" / "loveda_blind_gt_protocol_v0.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    output_root = root / "outputs" / "run"
    config = {
        "experiment": {"name": "loveda_blind_gt_v0", "overwrite": False},
        "paths": {
            "pixel_pack": str(pixel_pack),
            "split_manifest": str(split_dir / "manifest.json"),
            "development_records": str(split_dir / "development_records.jsonl"),
            "heldout_records": str(split_dir / "heldout_records.jsonl"),
            "feature_cache_manifest": str(cache_dir / "manifest.json"),
            "features": str(cache_dir / "features_openai_clip.npy"),
            "row_partitions": str(split_dir / "row_partitions.jsonl"),
            "openai_clip_checkpoint": str(checkpoint),
            "loveda_label_dir": None,
            "protocol_file": str(root / "configs" / "loveda_blind_gt_protocol_v0.json"),
            "output_root": str(output_root),
        },
        "runtime": {"device": "cpu", "batch_regions": 8},
    }
    import yaml
    (root / "configs" / config_name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config


def test_config_loader_rejects_missing_keys(tmp_path):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "outputs").mkdir()
    import yaml
    cfg_path = root / "configs" / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump({"experiment": {"name": "x", "overwrite": False}, "paths": {"output_root": str(root / "outputs" / "r")}}), encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_loveda_blind_gt_config(cfg_path, root)


def test_predict_phase_forbids_label_dir(tmp_path):
    root, config = _synthetic_project(tmp_path)
    import yaml
    cfg, protocol = load_loveda_blind_gt_config(root / "configs" / "deploy.yaml", root)
    cfg["paths"]["loveda_label_dir"] = str(root / "inputs")
    with pytest.raises(InputValidationError, match="must not configure a GT label directory"):
        run_loveda_blind_gt_predict(cfg, protocol)


def test_predict_and_evaluate_roundtrip(tmp_path, monkeypatch):
    root, config = _synthetic_project(tmp_path)
    import yaml
    cfg, protocol = load_loveda_blind_gt_config(root / "configs" / "deploy.yaml", root)

    def fake_text_prototypes(protocol, checkpoint, device):
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((6, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors, "deadbeef"

    monkeypatch.setattr("ov_probe.loveda_blind_gt._text_prototypes", fake_text_prototypes)
    manifest = run_loveda_blind_gt_predict(cfg, protocol)
    assert manifest["phase"] == "predict"
    assert manifest["status"] == "completed"
    output_dir = Path(cfg["paths"]["output_root"])
    assert (output_dir / "predictions.npz").is_file()
    assert (output_dir / "manifest.json").is_file()

    # evaluate requires label dir; build synthetic GT labels now (post-predict by design)
    from PIL import Image
    label_dir = root / "inputs" / "labels"
    label_dir.mkdir(exist_ok=True)
    color_map = protocol["ground_truth"]["color_map"]
    for index in range(6, 12):
        label = CLASSES[index % 6]
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[100:132, 100:132] = np.asarray(color_map[label], dtype=np.uint8)
        Image.fromarray(image, "RGB").save(label_dir / f"loveda_train_rural_{index}_label.png")
    cfg["paths"]["loveda_label_dir"] = str(label_dir)
    summary = run_loveda_blind_gt_evaluate(cfg, protocol, output_dir)
    assert summary["status"] == "completed"
    assert summary["heldout"]["record_count"] == 6
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "per_class_f1.csv").is_file()
    for method in ("text_only", "visual_only", "fused_text_visual"):
        assert (output_dir / f"confusion_matrix_{method}.csv").is_file()
    for method, result in summary["metrics"].items():
        assert "accuracy" in result and "macro_f1" in result and "confusion_matrix" in result
    # Features are class-separated, so a correctly indexed visual-only prediction must
    # recover near-perfect GT accuracy; this guards against row-index misalignment.
    assert summary["metrics"]["visual_only"]["accuracy"] > 0.8
    assert summary["metrics"]["fused_text_visual"]["accuracy"] > 0.8


def test_region_gt_majority_vote():
    color_map = {
        "building": [255, 0, 0],
        "road": [255, 255, 0],
        "water": [0, 0, 255],
        "barren": [159, 129, 183],
        "forest": [0, 255, 0],
        "agriculture": [255, 255, 255],
    }
    label = np.zeros((64, 64, 3), dtype=np.uint8)
    label[8:24, 8:24] = np.asarray([255, 0, 0], dtype=np.uint8)
    label[24:40, 24:40] = np.asarray([255, 255, 0], dtype=np.uint8)
    label[0:8, 0:8] = np.asarray([255, 0, 255], dtype=np.uint8)  # ignore
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:16, 0:16] = True  # mostly ignore + building
    info = {"crop_box": (8, 8, 40, 40), "crop_shape": (32, 32), "mask": mask}
    result, total, area = _region_gt_from_label(info, label, color_map)
    assert result == "building"
    assert total == 256
    assert area == 256


def test_metrics_classification():
    predictions = np.asarray([0, 0, 1, 2])
    labels = np.asarray([0, 1, 1, 2])
    result = _classification_metrics(predictions, labels, CLASSES)
    assert result["accuracy"] == pytest.approx(0.75)
    assert "confusion_matrix" in result
    assert set(result["per_class_f1"]) == set(CLASSES)


def test_evaluate_rejects_missing_predict_manifest(tmp_path):
    root, config = _synthetic_project(tmp_path)
    import yaml
    cfg, protocol = load_loveda_blind_gt_config(root / "configs" / "deploy.yaml", root)
    cfg["paths"]["loveda_label_dir"] = str(root / "inputs")
    output_dir = Path(cfg["paths"]["output_root"])
    output_dir.mkdir(parents=True)
    with pytest.raises(InputValidationError, match="Predict-phase manifest is missing"):
        run_loveda_blind_gt_evaluate(cfg, protocol, output_dir)


def test_evaluate_detects_prediction_tampering(tmp_path, monkeypatch):
    root, config = _synthetic_project(tmp_path)
    import yaml
    cfg, protocol = load_loveda_blind_gt_config(root / "configs" / "deploy.yaml", root)
    monkeypatch.setattr(
        "ov_probe.loveda_blind_gt._text_prototypes",
        lambda protocol, checkpoint, device: (np.eye(6, 512, dtype=np.float32), "beef"),
    )
    run_loveda_blind_gt_predict(cfg, protocol)
    output_dir = Path(cfg["paths"]["output_root"])
    (output_dir / "predictions.npz").write_bytes(b"tampered")
    from PIL import Image
    label_dir = root / "inputs" / "labels"
    label_dir.mkdir(exist_ok=True)
    cfg["paths"]["loveda_label_dir"] = str(label_dir)
    with pytest.raises(InputValidationError, match="changed since the predict phase"):
        run_loveda_blind_gt_evaluate(cfg, protocol, output_dir)
