from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ov_probe.io import InputValidationError
from ov_probe.voc2012_openai_clip_zeroshot import (
    _GROUP_A,
    _VOC_CLASSES,
    _read_image_manifest,
    build_text_prototypes,
    cosine_scores,
    load_voc2012_openai_clip_zeroshot_config,
)
from ov_probe.io import sha256_file


def test_frozen_voc_class_and_prompt_constants() -> None:
    assert len(_VOC_CLASSES) == 20
    assert len(_GROUP_A) == 8
    assert _VOC_CLASSES[0] == "aeroplane"
    assert _VOC_CLASSES[-1] == "tv monitor"


def test_prototypes_and_scores_are_normalized_and_deterministic() -> None:
    prompts = np.zeros((160, 3), dtype=np.float32)
    for index in range(20):
        prompts[index * 8:(index + 1) * 8, index % 3] = 2.0
    text = build_text_prototypes(prompts)
    images = np.asarray([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    scores = cosine_scores(images, text)
    assert text.shape == (20, 3)
    assert scores.shape == (2, 20)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), np.ones(20), atol=1e-6)
    np.testing.assert_allclose(scores[0, 0], 1.0, atol=1e-6)


def test_config_rejects_manifest_escape_and_protocol_binding(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    protocol_path = root / "configs" / "voc2012_openai_clip_zeroshot_protocol_v1.json"
    protocol_path.write_bytes((source / "configs" / "voc2012_openai_clip_zeroshot_protocol_v1.json").read_bytes())
    config = yaml.safe_load((source / "configs" / "voc2012_openai_clip_zeroshot_v1.yaml").read_text(encoding="utf-8"))
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, protocol = load_voc2012_openai_clip_zeroshot_config(config_path, root)
    assert loaded["paths"]["dataset_manifest"] is None
    assert protocol["dataset"]["val_image_count"] == 1449
    config["paths"]["dataset_manifest"] = "../escape.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(InputValidationError, match="escapes"):
        load_voc2012_openai_clip_zeroshot_config(config_path, root)


def test_protocol_is_explicitly_label_free_and_non_evaluative() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "voc2012_openai_clip_zeroshot_protocol_v1.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((root / "configs" / "voc2012_openai_clip_zeroshot_v1.yaml").read_text(encoding="utf-8"))
    assert protocol["scientific_evidence"] is False
    assert protocol["scoring"]["shape"] == [1449, 20]
    assert all(protocol["constraints"][key] is False for key in ("semantic_label_read_or_decode", "predictions", "metrics", "thresholds"))
    assert cfg["experiment"]["overwrite"] is False
    assert all(not Path(str(value)).is_absolute() for value in cfg["paths"].values() if value is not None)


def test_manifest_binding_and_label_schema_are_rejected_before_image_access(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"labels": ["forbidden"]}), encoding="utf-8")
    protocol = {"dataset": {"manifest_sha256": "0" * 64, "archive_md5": "6cd6e144f989b92b3379bac3b3de84fd", "val_image_count": 1449}}
    with pytest.raises(InputValidationError, match="hash"):
        _read_image_manifest(manifest_path, tmp_path, protocol)
    protocol["dataset"]["manifest_sha256"] = sha256_file(manifest_path)
    with pytest.raises(InputValidationError, match="schema"):
        _read_image_manifest(manifest_path, tmp_path, protocol)
