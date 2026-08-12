import json
from pathlib import Path

import numpy as np

from ov_probe.io import load_config, load_feature_bundle


ROOT = Path(__file__).resolve().parents[1]


def _metadata(data_file: str, multi: bool) -> dict:
    classes = ["building", "road", "water", "barren", "forest", "agriculture"]
    return {
        "format_version": 2 if multi else 1,
        "data_file": data_file,
        "protocol": {
            "method": "adaptive robust CAM-consistent multi-prototypes" if multi else "robust CAM-consistent candidate visual prototypes",
            "pixel_gt_used": False,
            "love_da_val_used": False,
            "seed_rule": "SAM3 source class equals active-class CAM top1",
        },
        "inputs": {
            "config": "/data/LoveDA/train/loveda_config.json",
            "labels_csv": "/data/LoveDA/train/image_level_labels_train.csv",
        },
        "feature_dimension": 512,
        "classes": {
            name: {
                "class_id": index + 1,
                "retained_seeds": 10,
                "cluster_sizes": [4, 6],
            }
            for index, name in enumerate(classes)
        },
    }


def test_load_native_single_calibration(tmp_path):
    cfg = load_config(ROOT / "configs" / "ov_probe_v0.yaml", ROOT)
    features = np.eye(6, 512, dtype=np.float32)
    np.savez_compressed(tmp_path / "prototypes.npz", format_version=np.array([1]), class_ids=np.arange(1, 7), prototypes=features)
    (tmp_path / "prototypes.json").write_text(json.dumps(_metadata("prototypes.npz", False)), encoding="utf-8")
    bundle = load_feature_bundle(str(tmp_path / "prototypes.json"), cfg, "single")
    assert bundle.features.shape == (6, 512)
    assert bundle.class_names[0] == "building"
    assert bundle.metadata["native_first_paper_format"] is True


def test_load_native_multi_calibration(tmp_path):
    cfg = load_config(ROOT / "configs" / "ov_probe_v0.yaml", ROOT)
    features = np.eye(12, 512, dtype=np.float32)
    prototype_class_ids = np.repeat(np.arange(1, 7), 2)
    np.savez_compressed(
        tmp_path / "multi.npz",
        format_version=np.array([2]),
        class_ids=np.arange(1, 7),
        prototype_class_ids=prototype_class_ids,
        prototypes=features,
    )
    (tmp_path / "multi.json").write_text(json.dumps(_metadata("multi.npz", True)), encoding="utf-8")
    bundle = load_feature_bundle(str(tmp_path / "multi.json"), cfg, "multi")
    assert bundle.features.shape == (12, 512)
    assert bundle.class_names[:2] == ["building", "building"]
    assert bundle.cluster_sizes.tolist() == [4, 6] * 6
