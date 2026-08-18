"""Unit tests for the Vaihingen blind evaluation module (synthetic, no GT access)."""

from __future__ import annotations

import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.vaihingen_blind import (
    CLASSES,
    GT_COLOR_MAP,
    TRAIN_AREAS,
    TEST_AREAS,
    _area_from_id,
    _crop_view,
    _region_gt,
)


def test_area_from_id():
    assert _area_from_id("vaih_area1") == 1
    assert _area_from_id("vaih_area37") == 37
    with pytest.raises(InputValidationError):
        _area_from_id("other")
    with pytest.raises(InputValidationError):
        _area_from_id("vaih_areaX")


def test_split_disjoint_and_complete():
    assert set(TRAIN_AREAS).isdisjoint(set(TEST_AREAS))
    assert len(TRAIN_AREAS) == 11 and len(TEST_AREAS) == 5
    assert set(TRAIN_AREAS) | set(TEST_AREAS) == {1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37}


def test_crop_view_geometry():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[40:60, 40:60] = 200
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    view = _crop_view(image, mask, x0=40, y0=40)
    assert view["crop_mask"].shape == view["context"].shape[:2]
    assert view["crop_mask"].sum() == mask.sum()
    assert view["mask_fraction"] > 0
    assert view["masked"].dtype == np.uint8
    # masked region keeps original values
    assert view["masked"][view["crop_mask"]].max() == 200
    # background retained at 25%
    assert view["masked"][~view["crop_mask"]].max() <= 51


def test_crop_view_centered_bounds():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, :] = True
    view = _crop_view(image, mask, x0=0, y0=0)
    x1, y1, x2, y2 = view["crop_box"]
    assert x2 - x1 == view["context"].shape[1]
    assert y2 - y1 == view["context"].shape[0]
    assert x1 >= 0 and y1 >= 0 and x2 <= 64 and y2 <= 64


def test_region_gt_majority_vote():
    label = np.zeros((64, 64, 3), dtype=np.uint8)
    label[0:32, 0:32] = np.asarray([255, 255, 255], dtype=np.uint8)  # impervious
    label[32:64, 32:64] = np.asarray([0, 0, 255], dtype=np.uint8)    # building
    label[0:8, 0:8] = np.asarray([255, 0, 0], dtype=np.uint8)        # clutter (ignore)
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:16, 0:16] = True  # mostly impervious + some clutter
    gt = _region_gt([{"mask": mask, "x0": 0, "y0": 0}], label)
    assert gt == ["impervious_surface"]


def test_region_gt_unlabeled_when_no_class_pixels():
    label = np.zeros((32, 32, 3), dtype=np.uint8)
    label[:, :] = np.asarray([255, 0, 0], dtype=np.uint8)  # all clutter
    mask = np.ones((32, 32), dtype=bool)
    gt = _region_gt([{"mask": mask, "x0": 0, "y0": 0}], label)
    assert gt == [None]


def test_gt_color_map_consistent():
    assert set(GT_COLOR_MAP) == set(CLASSES)
    for color in GT_COLOR_MAP.values():
        assert len(color) == 3 and all(0 <= v <= 255 for v in color)
