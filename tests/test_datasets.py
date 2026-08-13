import json
from pathlib import Path

from ov_probe.datasets import load_dataset_registry
from ov_probe.voc_sbd import EXPECTED_MD5


def test_registered_datasets_have_expected_class_counts_and_sparse_coco_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_dataset_registry(root / "configs" / "datasets")
    assert set(registry) == {"loveda", "voc2012_sbd", "coco2014"}
    assert len(registry["loveda"].classes) == 6
    assert len(registry["voc2012_sbd"].classes) == 20
    assert len(registry["coco2014"].classes) == 80
    assert registry["voc2012_sbd"].metadata["augmentation"]["required_split"] == "train_noval"
    assert registry["coco2014"].dataset_ids != tuple(range(1, 81))
    assert registry["coco2014"].metadata["semantic_conversion"]["status"] == "blocked_pending_frozen_policy"


def test_voc_sbd_preparation_protocol_matches_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs" / "voc_sbd_preparation_protocol_v0.json").read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_before_extraction"
    assert {
        name: record["md5"] for name, record in protocol["artifacts"].items()
    } == EXPECTED_MD5
    preparation = protocol["preparation"]
    assert preparation["require_train_noval_voc2012_val_overlap"] == 0
    assert preparation["read_pixel_annotation_values"] is False
    assert preparation["run_sam_or_proposal_model"] is False
    assert preparation["run_training"] is False
    assert protocol["voc_image_level_tags"]["difficult_policy"] == "positive_presence"
    assert protocol["voc_image_level_tags"]["segmentation_masks_read"] is False
