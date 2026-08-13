from pathlib import Path

from ov_probe.datasets import load_dataset_registry


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

