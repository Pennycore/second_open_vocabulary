import json
from pathlib import Path


def test_stage1_leave_one_class_out_folds_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "stage1_ov_protocol_v0.json").read_text())
    classes = set(protocol["classes"])
    folds = protocol["folds"]
    assert len(folds) == len(classes) == 6
    held_out = []
    for fold in folds:
        assert len(fold["unseen"]) == 1
        assert set(fold["seen"]).isdisjoint(fold["unseen"])
        assert set(fold["seen"]) | set(fold["unseen"]) == classes
        held_out.extend(fold["unseen"])
    assert set(held_out) == classes

