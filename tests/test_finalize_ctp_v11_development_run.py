"""Tests for the metadata-only CTP-v1.1 development finalizer."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from ov_probe.ctp_v11_tuning import BASELINE, DEVELOPMENT_AREAS, REGISTERED_TEST_AREAS, canonical_grid, load_protocol, select_configuration
from ov_probe.io import InputValidationError, sha256_file


ROOT = Path(__file__).resolve().parent.parent
FINALIZER_PATH = ROOT / "scripts" / "finalize_ctp_v11_development_run.py"
SPEC = importlib.util.spec_from_file_location("ctp_v11_finalizer", FINALIZER_PATH)
assert SPEC and SPEC.loader
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


def _selection_rows() -> list[dict]:
    rows = []
    for alpha, tau in canonical_grid():
        rows.append({
            "alpha": alpha, "tau_conflict": tau,
            "full_OA": 0.4, "full_MacroF1": 0.4, "full_mIoU": 0.4,
            "abstention_ratio": 0.2, "partial_mean_S_IoU": 0.5,
            "partial_mean_U_IoU": 0.4, "partial_mean_H_IoU": 0.5,
            "partial_min_U_IoU": 0.1, "collapse_subset_count": 0,
        })
    baseline = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == BASELINE)
    baseline["full_mIoU"] = 0.5
    winner = next(row for row in rows if (row["alpha"], row["tau_conflict"]) == (0.2, 0.0))
    winner.update({"full_mIoU": 0.7, "partial_mean_H_IoU": 0.6})
    return select_configuration(rows)["grid"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    shutil.copy2(ROOT / "configs" / "ctp_v1_1_tuning_protocol.json", root / "configs")
    run = root / "outputs" / "ctp_tuning_v1_1" / "run_synthetic"
    run.mkdir(parents=True)
    protocol = load_protocol(root / "configs" / "ctp_v1_1_tuning_protocol.json")
    prediction = {
        "phase": "predict", "status": "completed", "gt_read": False,
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "protocol_sha256": protocol["sha256"],
        "cache_prototype_text_binding": {"feature_cache_sha256": "a" * 64},
        "source_code": {"repo_head": "b" * 40},
    }
    (run / "prediction_manifest.json").write_text(json.dumps(prediction), encoding="utf-8")
    grid = _selection_rows()
    _write_csv(run / "grid_search_full.csv", grid)
    partial = []
    for alpha, tau in canonical_grid():
        for index in range(25):
            partial.append({
                "alpha": alpha, "tau_conflict": tau, "subset_index": index,
                "k": 2, "supported": "a|b", "unsupported": "c|d|e", "OA": 0.4,
            })
    _write_csv(run / "partial_all_subsets.csv", partial)
    accounting = [{"alpha": alpha, "tau_conflict": tau, "pixels_total": 1.0} for alpha, tau in canonical_grid()]
    _write_csv(run / "full_support_accounting.csv", accounting)
    per_class = [
        {"alpha": alpha, "tau_conflict": tau, "class": name, "IoU": 0.3}
        for alpha, tau in canonical_grid() for name in protocol["classes"]
    ]
    _write_csv(run / "full_support_per_class.csv", per_class)
    return root, run


def test_finalizes_complete_dev_only_run_and_hashes_all_csvs(tmp_path: Path) -> None:
    root, run = _make_run(tmp_path)
    result = FINALIZER.finalize_development_run("outputs/ctp_tuning_v1_1/run_synthetic", project_root=root)
    assert result["selected"] == {"alpha": 0.2, "tau_conflict": 0.0}
    manifest = json.loads((run / "development_evaluation_manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads((run / "ctp_v1_1_tuned_candidate.json").read_text(encoding="utf-8"))
    assert manifest["posthoc_finalization"] is True
    assert manifest["computation_rerun"] is False
    assert candidate["new_prediction_or_gt_read"] is False
    assert candidate["baseline_config"] == {"alpha": 0.5, "tau_conflict": 0.03}
    assert candidate["csv_sha256"]["grid_search_full.csv"] == sha256_file(run / "grid_search_full.csv")


def test_refuses_test_ids_in_prediction_manifest(tmp_path: Path) -> None:
    root, run = _make_run(tmp_path)
    path = run / "prediction_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["development_area_ids"][0] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputValidationError, match="development IDs"):
        FINALIZER.finalize_development_run("outputs/ctp_tuning_v1_1/run_synthetic", project_root=root)
    assert not (run / "development_evaluation_manifest.json").exists()


def test_refuses_missing_csv_and_never_overwrites(tmp_path: Path) -> None:
    root, run = _make_run(tmp_path)
    (run / "partial_all_subsets.csv").unlink()
    with pytest.raises(InputValidationError, match="Cannot read required output"):
        FINALIZER.finalize_development_run("outputs/ctp_tuning_v1_1/run_synthetic", project_root=root)
    _root, repaired = _make_run(tmp_path / "other")
    FINALIZER.finalize_development_run("outputs/ctp_tuning_v1_1/run_synthetic", project_root=_root)
    with pytest.raises(InputValidationError, match="overwrite"):
        FINALIZER.finalize_development_run("outputs/ctp_tuning_v1_1/run_synthetic", project_root=_root)
    assert (repaired / "ctp_v1_1_tuned_candidate.json").is_file()


def test_refuses_noncanonical_run_path(tmp_path: Path) -> None:
    root, _run = _make_run(tmp_path)
    with pytest.raises(InputValidationError, match="only outputs/ctp_tuning"):
        FINALIZER.finalize_development_run("outputs/other/run_synthetic", project_root=root)
