"""Unit coverage for the fail-closed CTP-v1.1 development-only scaffold."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ov_probe.ctp_v11_tuning import (
    ALPHA_GRID,
    BASELINE,
    DEVELOPMENT_AREAS,
    REGISTERED_TEST_AREAS,
    TAU_GRID,
    anchored_prototypes,
    assert_exact_development_areas,
    assert_safe_source_path,
    canonical_grid,
    evaluate_class_predictions,
    full_support_per_class_rows,
    load_deployment_config,
    load_protocol,
    select_configuration,
    source_code_identity,
    support_subsets,
    validate_development_manifest,
    validate_tau,
    write_prediction_archives,
    required_output_hashes,
)
from ov_probe.io import InputValidationError


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "configs" / "ctp_v1_1_tuning_protocol.json"


def _manifest() -> dict:
    return {
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "records": [
            {"area_id": area, "image_id": f"vaih_area{area}", "candidate_index": 0}
            for area in DEVELOPMENT_AREAS
        ],
    }


def _selection_rows() -> list[dict]:
    rows = []
    for alpha, tau in canonical_grid():
        rows.append({
            "alpha": alpha,
            "tau_conflict": tau,
            "full_OA": 0.4,
            "full_MacroF1": 0.4,
            "full_mIoU": 0.4,
            "abstention_ratio": 0.2,
            "partial_mean_S_IoU": 0.5,
            "partial_mean_U_IoU": 0.4,
            "partial_mean_H_IoU": 0.5,
            "partial_min_U_IoU": 0.1,
            "collapse_subset_count": 0,
        })
    return rows


def _write_config(tmp_path: Path, *, area_ids: list[int] | None = None, candidate_dir: str | None = None) -> Path:
    cfg = {
        "experiment": {"name": "candidate", "overwrite": False},
        "paths": {
            "protocol_file": "configs/ctp_v1_1_tuning_protocol.json",
            "development_manifest": None,
            "feature_cache": None,
            "candidate_dir": candidate_dir,
            "image_dir": None,
            "openai_clip_checkpoint": None,
            "label_dir": None,
            "output_root": "outputs/ctp_tuning_v1_1",
        },
        "development": {"area_ids": area_ids or list(DEVELOPMENT_AREAS)},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_protocol_is_exactly_registered() -> None:
    protocol = load_protocol(PROTOCOL)
    assert tuple(protocol["registered_test_area_ids"]) == REGISTERED_TEST_AREAS
    assert tuple(protocol["development_area_ids"]) == DEVELOPMENT_AREAS
    assert tuple(protocol["allowed_parameters"]["alpha"]) == ALPHA_GRID
    assert tuple(protocol["allowed_parameters"]["tau_conflict"]) == TAU_GRID
    assert len(canonical_grid()) == 49
    assert BASELINE in canonical_grid()


def test_all_partial_subsets_are_complete_and_nontrivial() -> None:
    subsets = support_subsets(["a", "b", "c", "d", "e"])
    assert len(subsets) == 25
    assert {len(subset) for subset in subsets} == {2, 3, 4}
    assert len(canonical_grid()) * len(subsets) == 49 * 25


def test_test_id_and_nonexact_dev_lists_are_rejected() -> None:
    with pytest.raises(InputValidationError, match="intersects registered test"):
        assert_exact_development_areas([1, 3, 5, 7, 11, 13, 17, 21, 23, 26, 32])
    with pytest.raises(InputValidationError, match="exactly equal"):
        assert_exact_development_areas(DEVELOPMENT_AREAS[:-1])


def test_manifest_rejects_test_id_and_requires_complete_dev_coverage() -> None:
    valid = _manifest()
    assert validate_development_manifest(valid)["record_count"] == len(DEVELOPMENT_AREAS)
    invalid = _manifest()
    invalid["records"][0]["area_id"] = 11
    invalid["records"][0]["image_id"] = "vaih_area11"
    with pytest.raises(InputValidationError, match="non-development"):
        validate_development_manifest(invalid)
    incomplete = _manifest()
    incomplete["records"] = incomplete["records"][:-1]
    with pytest.raises(InputValidationError, match="cover every"):
        validate_development_manifest(incomplete)


def test_test_and_external_paths_are_rejected() -> None:
    for value in ("outputs/test_run", "inputs/vaih_area11", "outputs/segearth_artifacts", "inputs/Potsdam"):
        with pytest.raises(InputValidationError):
            assert_safe_source_path(value, "source")


def test_config_requires_canonical_output_and_no_test_source(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    cfg, protocol = load_deployment_config(path, ROOT)
    assert cfg["paths"]["label_dir"] is None
    assert protocol["name"] == "CTP-v1.1-tuning-candidate"
    unsafe = _write_config(tmp_path / "unsafe", candidate_dir="outputs/test_candidates")
    with pytest.raises(InputValidationError, match="forbidden"):
        load_deployment_config(unsafe, ROOT)


def test_alpha_tau_are_grid_bound_and_gt_isolation_is_structural() -> None:
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    visual = text.copy()
    assert anchored_prototypes(text, visual, 0.2).shape == (2, 2)
    with pytest.raises(InputValidationError, match="alpha"):
        anchored_prototypes(text, visual, 0.25)
    assert validate_tau(0.015) == pytest.approx(0.015)
    with pytest.raises(InputValidationError, match="tau_conflict"):
        validate_tau(0.017)
    import ov_probe.ctp_v11_tuning as module
    assert "gt" not in inspect.signature(module.build_visual_prototypes).parameters
    assert "gt" not in inspect.signature(module.score_region_features).parameters
    assert "gt" in inspect.signature(evaluate_class_predictions).parameters


def test_strict_selection_constraints_and_tie_breakers() -> None:
    rows = _selection_rows()
    baseline = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == BASELINE)
    baseline.update({"full_mIoU": 0.5, "partial_mean_H_IoU": 0.5, "abstention_ratio": 0.3})
    first = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == (0.2, 0.0))
    first.update({"full_mIoU": 0.7000, "partial_mean_H_IoU": 0.5, "partial_mean_U_IoU": 0.4, "abstention_ratio": 0.5})
    near = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == (0.3, 0.0))
    near.update({"full_mIoU": 0.6995, "partial_mean_H_IoU": 0.6, "partial_mean_U_IoU": 0.4, "abstention_ratio": 0.5})
    collapsed = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == (0.4, 0.0))
    collapsed.update({"full_mIoU": 0.9, "collapse_subset_count": 1, "partial_mean_H_IoU": 0.8})
    result = select_configuration(rows)
    assert (result["selected"]["alpha"], result["selected"]["tau_conflict"]) == (0.3, 0.0)
    assert result["baseline"]["selected"] is False
    assert sum(row["selected"] for row in result["grid"]) == 1
    assert next(row for row in result["grid"] if row["alpha"] == 0.4 and row["tau_conflict"] == 0.0)["feasible"] is False


def test_selection_rejects_incomplete_grid() -> None:
    with pytest.raises(InputValidationError, match="49"):
        select_configuration(_selection_rows()[:-1])


def test_prediction_archives_cover_all_alpha_and_25_subsets_without_gt(tmp_path: Path) -> None:
    classes = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
    rng = np.random.default_rng(3)
    features = rng.normal(size=(4, 3)).astype(np.float32)
    text = rng.normal(size=(5, 3)).astype(np.float32)
    visual = rng.normal(size=(5, 3)).astype(np.float32)
    info = write_prediction_archives(features, text, visual, classes, tmp_path / "region_predictions")
    assert len(info["archives"]) == 7
    assert info["subset_count"] == 26
    first = tmp_path / info["archives"]["0.200"]["path"]
    with np.load(first, allow_pickle=False) as archive:
        assert "full_prediction" in archive.files
        assert "subset_24_prediction" in archive.files
        assert archive["subset_24_prediction"].shape == (4,)


def test_cache_and_scoring_interfaces_cannot_receive_gt() -> None:
    import ov_probe.ctp_v11_tuning as module
    for function in (module.build_development_feature_cache, module.load_frozen_candidates, module.write_prediction_archives):
        assert "gt" not in inspect.signature(function).parameters
    source = inspect.getsource(module.build_development_feature_cache)
    assert "label_dir" not in source


def test_per_class_export_has_exactly_49_times_5_rows_and_hash_binding(tmp_path: Path) -> None:
    classes = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
    rows = _selection_rows()
    for row in rows:
        row["per_class_iou"] = {name: (index + 1) / 10 for index, name in enumerate(classes)}
    per_class = full_support_per_class_rows(rows, classes)
    assert len(per_class) == 49 * 5
    assert {(row["alpha"], row["tau_conflict"], row["class"]) for row in per_class} == {
        (alpha, tau, name) for alpha, tau in canonical_grid() for name in classes
    }
    output = tmp_path / "full_support_per_class.csv"
    output.write_text("alpha,tau_conflict,class,IoU\n", encoding="utf-8")
    binding = required_output_hashes(tmp_path, ("full_support_per_class.csv",))
    assert binding["full_support_per_class.csv"]
    identity = source_code_identity()
    assert set(identity) == {"repo_head", "module_sha256", "runner_sha256"}
