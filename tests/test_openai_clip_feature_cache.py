from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ov_probe.io import InputValidationError
from ov_probe.openai_clip_feature_cache import validate_split_partition_mapping


def _pixel_rows() -> list[dict[str, object]]:
    return [
        {"row_index": 0, "image_id": "image-a", "candidate_index": 0},
        {"row_index": 1, "image_id": "image-a", "candidate_index": 1},
        {"row_index": 2, "image_id": "image-b", "candidate_index": 0},
        {"row_index": 3, "image_id": "image-b", "candidate_index": 1},
    ]


def test_partition_mapping_is_complete_ordered_and_label_free() -> None:
    source = _pixel_rows()
    development = [source[1], source[0]]
    heldout = [source[3], source[2]]

    result = validate_split_partition_mapping(source, development, heldout)

    assert result == [
        {"row_index": 0, "image_id": "image-a", "candidate_index": 0, "partition": "development"},
        {"row_index": 1, "image_id": "image-a", "candidate_index": 1, "partition": "development"},
        {"row_index": 2, "image_id": "image-b", "candidate_index": 0, "partition": "heldout"},
        {"row_index": 3, "image_id": "image-b", "candidate_index": 1, "partition": "heldout"},
    ]
    assert all(set(row) == {"row_index", "image_id", "candidate_index", "partition"} for row in result)


def test_partition_mapping_rejects_image_overlap() -> None:
    source = _pixel_rows()
    with pytest.raises(InputValidationError, match="image-disjoint"):
        validate_split_partition_mapping(source, [source[0]], [source[1], source[2], source[3]])


def test_partition_mapping_rejects_missing_source_key() -> None:
    source = _pixel_rows()
    with pytest.raises(InputValidationError, match="cover every"):
        validate_split_partition_mapping(source, [source[0], source[1]], [source[2]])


def test_tracked_feature_cache_config_requires_deployment_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "openai_clip_feature_cache_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["experiment"]["overwrite"] is False
    assert cfg["paths"]["openai_clip_checkpoint"] is None
    assert cfg["paths"]["output_root"] == "outputs/openai_clip_feature_cache_v1"
    assert all(not Path(str(value)).is_absolute() for value in cfg["paths"].values() if value is not None)


def test_feature_cache_protocol_is_infrastructure_only_and_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs" / "openai_clip_feature_cache_protocol_v1.json").read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_pre_result"
    assert protocol["role"] == "infrastructure cache; no strategy evaluation"
    assert protocol["scientific_evidence"] is False
    assert protocol["pixel_pack"]["record_count"] == 6000
    assert protocol["pixel_pack"]["image_count"] == 2058
    assert protocol["model"]["architecture"] == "ViT-B-32-quickgelu"
    assert protocol["model"]["feature_dimension"] == 512
    assert protocol["model"]["open_clip_version"] == "3.3.0"
    assert protocol["output"]["feature_shape"] == [6000, 512]
    assert protocol["output"]["feature_dtype"] == "float16"
    assert protocol["output"]["row_partition_fields"] == [
        "row_index", "image_id", "candidate_index", "partition"
    ]
    for field in (
        "sam3_rerun", "training", "pixel_gt", "validation_split",
        "weak_labels_used_for_cache", "predictions", "similarity_scores", "metrics",
        "model_selection_or_decision", "overwrite",
    ):
        assert protocol["constraints"][field] is False
