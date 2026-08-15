from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ov_probe.io import InputValidationError, sha256_file
from ov_probe.voc2012_openai_clip_presence_eval import (
    _IMAGE_ID_RE,
    _VOC_CLASSES,
    _load_fixed_score_cache,
    _mask_path,
    integral_average_precision,
    load_voc2012_openai_clip_presence_eval_config,
    presence_from_mask,
)


def test_integral_average_precision_is_exact_and_groups_equal_scores() -> None:
    assert integral_average_precision(np.asarray([0.9, 0.8, 0.7]), np.asarray([1, 0, 1])) == pytest.approx(5.0 / 6.0)
    assert integral_average_precision(np.asarray([0.9, 0.9]), np.asarray([1, 0])) == pytest.approx(0.5)
    with pytest.raises(InputValidationError, match="zero-support"):
        integral_average_precision(np.asarray([0.2, 0.1]), np.asarray([0, 0]))


def test_presence_ignores_background_and_void_and_rejects_unknown_ids() -> None:
    mask = np.asarray([[0, 255, 1], [2, 0, 255]], dtype=np.uint8)
    result = presence_from_mask(mask)
    assert result.shape == (20,)
    assert result[0] and result[1] and not result[2]
    assert not presence_from_mask(np.asarray([[0, 255]], dtype=np.uint8)).any()
    with pytest.raises(InputValidationError, match="invalid class ID"):
        presence_from_mask(np.asarray([[21]], dtype=np.uint8))


def test_registered_protocol_is_presence_ap_not_threshold_or_miou() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "voc2012_openai_clip_presence_eval_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["scientific_evidence"] is False
    assert protocol["evaluation_evidence"] is True
    assert protocol["metric"]["name"] == "uninterpolated integral average precision"
    assert protocol["constraints"]["threshold_selection"] is False
    assert protocol["constraints"]["segmentation_miou"] is False
    assert protocol["target"]["name"] == "image-level class presence"


def test_config_rejects_escape_and_binds_canonical_protocol(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "voc2012_openai_clip_presence_eval_protocol_v1.json").write_bytes((source / "configs" / "voc2012_openai_clip_presence_eval_protocol_v1.json").read_bytes())
    config = yaml.safe_load((source / "configs" / "voc2012_openai_clip_presence_eval_v1.yaml").read_text(encoding="utf-8"))
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, protocol = load_voc2012_openai_clip_presence_eval_config(config_path, root)
    assert loaded["paths"]["score_cache_manifest"] is None
    assert protocol["classes"] == list(_VOC_CLASSES)
    config["paths"]["raw_mask_root"] = "../escape"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(InputValidationError, match="escapes"):
        load_voc2012_openai_clip_presence_eval_config(config_path, root)


def test_score_cache_binding_rejects_wrong_hash_before_loading_arrays(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "completed",
        "scientific_evidence": False,
        "image_count": 1449,
        "role": "PASCAL VOC 2012 validation-image OpenAI-CLIP zero-shot score cache; input-only and no evaluation",
        "outputs": {},
    }), encoding="utf-8")
    protocol = {"score_cache": {"manifest_sha256": "0" * 64}}
    with pytest.raises(InputValidationError, match="SHA-256"):
        _load_fixed_score_cache(manifest_path, protocol)
    protocol["score_cache"]["manifest_sha256"] = sha256_file(manifest_path)
    with pytest.raises(InputValidationError, match="output schema"):
        _load_fixed_score_cache(manifest_path, protocol)


def test_image_ids_and_direct_mask_paths_must_be_exact(tmp_path: Path) -> None:
    assert _IMAGE_ID_RE.fullmatch("2007_000123")
    assert not _IMAGE_ID_RE.fullmatch("../2007_000123")
    assert not _IMAGE_ID_RE.fullmatch("2007_000123.png")
    mask_root = tmp_path / "SegmentationClass"
    mask_root.mkdir()
    expected = mask_root / "2007_000123.png"
    expected.write_bytes(b"not-decoded-by-this-test")
    assert _mask_path(mask_root, "2007_000123") == expected.resolve()
    with pytest.raises(InputValidationError, match="invalid image ID"):
        _mask_path(mask_root, "2007_000123.png")
