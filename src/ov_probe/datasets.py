from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import InputValidationError


@dataclass(frozen=True)
class DatasetClass:
    dataset_id: int
    canonical_name: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    domain: str
    task: str
    classes: tuple[DatasetClass, ...]
    background_id: int
    ignore_id: int
    splits: dict[str, str | None]
    weak_label_origin: str
    metadata: dict[str, Any]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.canonical_name for item in self.classes)

    @property
    def dataset_ids(self) -> tuple[int, ...]:
        return tuple(item.dataset_id for item in self.classes)


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise InputValidationError("Unsupported dataset-spec schema.")
    classes_raw = payload.get("classes")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise InputValidationError("Dataset spec must contain at least one foreground class.")
    classes = tuple(
        DatasetClass(int(item["dataset_id"]), str(item["canonical_name"]))
        for item in classes_raw
    )
    ids = [item.dataset_id for item in classes]
    names = [item.canonical_name for item in classes]
    if len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise InputValidationError("Foreground dataset IDs must be unique positive integers.")
    if len(names) != len(set(names)) or any(not value.strip() for value in names):
        raise InputValidationError("Canonical class names must be unique and non-empty.")
    background_id = int(payload["background"]["canonical_id"])
    ignore_id = int(payload["ignore"]["canonical_id"])
    if background_id == ignore_id or background_id in ids or ignore_id in ids:
        raise InputValidationError("Background, ignore, and foreground IDs must not overlap.")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits.get("weak_train") or not splits.get("final_evaluation"):
        raise InputValidationError("Dataset spec requires weak_train and final_evaluation splits.")
    if splits["weak_train"] == splits["final_evaluation"]:
        raise InputValidationError("Weak-train and final-evaluation splits must differ.")
    if payload.get("pixel_gt_policy", {}).get("weak_train_direct_access") != "forbidden":
        raise InputValidationError("Direct weak-train pixel-GT access must remain forbidden.")
    known = {
        "schema_version", "dataset_id", "domain", "task", "classes", "background",
        "ignore", "splits", "weak_label_origin",
    }
    return DatasetSpec(
        dataset_id=str(payload["dataset_id"]),
        domain=str(payload["domain"]),
        task=str(payload["task"]),
        classes=classes,
        background_id=background_id,
        ignore_id=ignore_id,
        splits={str(key): None if value is None else str(value) for key, value in splits.items()},
        weak_label_origin=str(payload["weak_label_origin"]),
        metadata={key: value for key, value in payload.items() if key not in known},
    )


def load_dataset_registry(directory: str | Path) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        spec = load_dataset_spec(path)
        if spec.dataset_id in specs:
            raise InputValidationError(f"Duplicate dataset ID in registry: {spec.dataset_id}")
        specs[spec.dataset_id] = spec
    if not specs:
        raise InputValidationError("Dataset registry is empty.")
    return specs

