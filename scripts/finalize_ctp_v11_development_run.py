"""Finalize provenance for one already-completed CTP-v1.1 dev-only run.

This is intentionally a metadata finalizer.  It never loads image data, label
data, features, masks, prediction archives, or any non-Vaihingen-development
asset.  It validates the four completed CSV tables and the sealed prediction
manifest, then writes two *new* provenance records with exclusive creation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.ctp_v11_tuning import (  # noqa: E402
    BASELINE,
    DEVELOPMENT_AREAS,
    REGISTERED_TEST_AREAS,
    canonical_grid,
    load_protocol,
    select_configuration,
    source_code_identity,
)
from ov_probe.io import InputValidationError, sha256_file  # noqa: E402


REQUIRED_CSVS = {
    "grid_search_full.csv": 49,
    "partial_all_subsets.csv": 1225,
    "full_support_accounting.csv": 49,
    "full_support_per_class.csv": 245,
}
_GRID_NUMERIC = (
    "alpha", "tau_conflict", "full_OA", "full_MacroF1", "full_mIoU",
    "abstention_ratio", "partial_mean_S_IoU", "partial_mean_U_IoU",
    "partial_mean_H_IoU", "partial_min_U_IoU", "collapse_subset_count",
)
_CANONICAL_PREFIX = ("outputs", "ctp_tuning_v1_1")


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _exact_run_dir(project_root: Path, relative_run_dir: str | Path) -> Path:
    raw = Path(relative_run_dir)
    if raw.is_absolute() or len(raw.parts) != 3 or tuple(raw.parts[:2]) != _CANONICAL_PREFIX:
        raise InputValidationError(
            "Finalizer accepts only outputs/ctp_tuning_v1_1/<single-run-directory>."
        )
    if raw.parts[2] in {"", ".", ".."} or "test" in raw.parts[2].lower():
        raise InputValidationError("Finalizer run directory is not an allowed development-only run path.")
    run_dir = (project_root / raw).resolve()
    output_root = (project_root / Path(*_CANONICAL_PREFIX)).resolve()
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise InputValidationError("Finalizer run directory escapes the CTP-v1.1 output root.") from exc
    if not run_dir.is_dir():
        raise InputValidationError("Finalizer run directory does not exist.")
    return run_dir


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Cannot read {label}.") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must be a JSON object.")
    return value


def _exact_ids(value: Any, expected: tuple[int, ...], label: str) -> None:
    try:
        actual = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{label} must be integer IDs.") from exc
    if actual != expected:
        raise InputValidationError(f"{label} differs from the immutable registry.")


def _validate_prediction_manifest(path: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    manifest = _read_json(path, "prediction_manifest")
    if manifest.get("phase") != "predict" or manifest.get("status") != "completed":
        raise InputValidationError("prediction_manifest is not a completed prediction phase.")
    if manifest.get("gt_read") is not False:
        raise InputValidationError("prediction_manifest does not prove GT-free prediction.")
    _exact_ids(manifest.get("development_area_ids"), DEVELOPMENT_AREAS, "prediction development IDs")
    _exact_ids(manifest.get("registered_test_area_ids"), REGISTERED_TEST_AREAS, "prediction test registry")
    if manifest.get("protocol_sha256") != protocol["sha256"]:
        raise InputValidationError("prediction_manifest protocol hash does not bind the canonical protocol.")
    binding = manifest.get("cache_prototype_text_binding")
    if not isinstance(binding, Mapping) or not binding:
        raise InputValidationError("prediction_manifest lacks cache/prototype/text binding.")
    if not isinstance(manifest.get("source_code"), Mapping):
        raise InputValidationError("prediction_manifest lacks source-code identity.")
    return manifest, sha256_file(path)


def _parse_finite(value: Any, field: str, filename: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{filename} has non-numeric {field}.") from exc
    if not math.isfinite(numeric):
        raise InputValidationError(f"{filename} has non-finite {field}.")
    return numeric


def _read_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise InputValidationError(f"{path.name} has no CSV header.")
            rows = list(reader)
    except OSError as exc:
        raise InputValidationError(f"Cannot read required output {path.name}.") from exc
    if len(rows) != expected_rows:
        raise InputValidationError(f"{path.name} row count is {len(rows)}, expected {expected_rows}.")
    for row in rows:
        if any(value is None for value in row.values()):
            raise InputValidationError(f"{path.name} has a malformed CSV row.")
        for field, value in row.items():
            if field in {"supported", "unsupported", "class", "feasible", "selected"}:
                continue
            _parse_finite(value, field, path.name)
    return rows


def _parse_flag(value: str, field: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise InputValidationError(f"grid_search_full.csv has invalid boolean {field}.")


def _validate_grid(rows: Iterable[Mapping[str, str]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if any(field not in row for field in (*_GRID_NUMERIC, "feasible", "selected")):
            raise InputValidationError("grid_search_full.csv lacks required selection fields.")
        item = {field: _parse_finite(row[field], field, "grid_search_full.csv") for field in _GRID_NUMERIC}
        item["collapse_subset_count"] = int(item["collapse_subset_count"])
        if item["collapse_subset_count"] < 0 or item["collapse_subset_count"] != float(row["collapse_subset_count"]):
            raise InputValidationError("grid_search_full.csv has invalid collapse_subset_count.")
        item["feasible"] = _parse_flag(row["feasible"], "feasible")
        item["selected"] = _parse_flag(row["selected"], "selected")
        parsed.append(item)
    observed = {(item["alpha"], item["tau_conflict"]) for item in parsed}
    if observed != set(canonical_grid()) or len(observed) != 49:
        raise InputValidationError("grid_search_full.csv does not cover the exact registered 49-cell grid.")
    selection = select_configuration(parsed)
    expected_grid = {(row["alpha"], row["tau_conflict"]): row for row in selection["grid"]}
    for item in parsed:
        expected = expected_grid[(item["alpha"], item["tau_conflict"])]
        if item["feasible"] != bool(expected["feasible"]):
            raise InputValidationError("grid_search_full.csv feasibility does not match the registered rule.")
        if item["selected"] != bool(expected["selected"]):
            raise InputValidationError("grid_search_full.csv selection does not match the registered rule.")
    selected_rows = [item for item in parsed if item["selected"]]
    if len(selected_rows) != 1 or not selected_rows[0]["feasible"]:
        raise InputValidationError("grid_search_full.csv must contain exactly one feasible selected row.")
    baseline_rows = [item for item in parsed if (item["alpha"], item["tau_conflict"]) == BASELINE]
    if len(baseline_rows) != 1:
        raise InputValidationError("grid_search_full.csv lacks the immutable CTP-v1 baseline (0.5, 0.03).")
    return selected_rows[0], baseline_rows[0], selection["selected"]


def _validate_cross_tables(
    partial: list[dict[str, str]], accounting: list[dict[str, str]], per_class: list[dict[str, str]],
    protocol: Mapping[str, Any],
) -> None:
    grid = set(canonical_grid())
    partial_keys = {(float(row["alpha"]), float(row["tau_conflict"]), int(float(row["subset_index"]))) for row in partial}
    if partial_keys != {(alpha, tau, index) for alpha, tau in grid for index in range(25)}:
        raise InputValidationError("partial_all_subsets.csv does not contain exactly 25 subsets for every grid cell.")
    accounting_keys = {(float(row["alpha"]), float(row["tau_conflict"])) for row in accounting}
    if accounting_keys != grid or len(accounting_keys) != 49:
        raise InputValidationError("full_support_accounting.csv does not bind every grid cell exactly once.")
    classes = tuple(protocol.get("classes", ()))
    if len(classes) != 5:
        raise InputValidationError("Canonical protocol has invalid class registry.")
    per_class_keys = {(float(row["alpha"]), float(row["tau_conflict"]), str(row["class"])) for row in per_class}
    expected = {(alpha, tau, name) for alpha, tau in grid for name in classes}
    if per_class_keys != expected:
        raise InputValidationError("full_support_per_class.csv does not bind every class/grid cell exactly once.")


def finalize_development_run(relative_run_dir: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    """Validate one completed dev-only run and x-create its two provenance files."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent.parent
    run_dir = _exact_run_dir(root, relative_run_dir)
    final_manifest_path = run_dir / "development_evaluation_manifest.json"
    candidate_path = run_dir / "ctp_v1_1_tuned_candidate.json"
    if final_manifest_path.exists() or candidate_path.exists():
        raise InputValidationError("Finalization output already exists; overwrite is forbidden.")
    protocol = load_protocol(root / "configs" / "ctp_v1_1_tuning_protocol.json")
    prediction_manifest, prediction_hash = _validate_prediction_manifest(run_dir / "prediction_manifest.json", protocol)
    tables = {name: _read_csv(run_dir / name, count) for name, count in REQUIRED_CSVS.items()}
    selected, baseline, expected_selected = _validate_grid(tables["grid_search_full.csv"])
    _validate_cross_tables(
        tables["partial_all_subsets.csv"], tables["full_support_accounting.csv"],
        tables["full_support_per_class.csv"], protocol,
    )
    csv_hashes = {name: sha256_file(run_dir / name) for name in REQUIRED_CSVS}
    source_code = {
        **source_code_identity(),
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
    }
    candidate = {
        "format_version": 1,
        "name": "CTP-v1.1-tuned-candidate",
        "status": "development_selected_pending_final_test",
        "development_only": True,
        "posthoc_finalization": True,
        "computation_rerun": False,
        "new_prediction_or_gt_read": False,
        "alpha": selected["alpha"],
        "tau_conflict": selected["tau_conflict"],
        "selection_rule": protocol["selection"],
        "selection_metrics": selected,
        "baseline_config": {"alpha": BASELINE[0], "tau_conflict": BASELINE[1]},
        "baseline_metrics": baseline,
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "protocol_sha256": protocol["sha256"],
        "source_prediction_manifest_sha256": prediction_hash,
        "csv_sha256": csv_hashes,
        "cache_prototype_text_binding": dict(prediction_manifest["cache_prototype_text_binding"]),
        "source_code": source_code,
    }
    development_manifest = {
        "format_version": 1,
        "phase": "development_evaluation_posthoc_finalization",
        "status": "completed",
        "development_only": True,
        "registered_test_evaluation": False,
        "posthoc_finalization": True,
        "computation_rerun": False,
        "new_prediction_or_gt_read": False,
        "source_prediction_manifest_sha256": prediction_hash,
        "csv_sha256": csv_hashes,
        "protocol_sha256": protocol["sha256"],
        "development_area_ids": list(DEVELOPMENT_AREAS),
        "registered_test_area_ids": list(REGISTERED_TEST_AREAS),
        "selected": candidate,
        "selection_recomputed": {
            "alpha": expected_selected["alpha"],
            "tau_conflict": expected_selected["tau_conflict"],
        },
        "cache_prototype_text_binding": dict(prediction_manifest["cache_prototype_text_binding"]),
        "source_code": source_code,
    }
    _exclusive_json(candidate_path, candidate)
    _exclusive_json(final_manifest_path, development_manifest)
    return {
        "run_dir": str(run_dir),
        "candidate": str(candidate_path),
        "development_evaluation_manifest": str(final_manifest_path),
        "selected": {"alpha": selected["alpha"], "tau_conflict": selected["tau_conflict"]},
        "prediction_manifest_sha256": prediction_hash,
        "csv_sha256": csv_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize one completed CTP-v1.1 development-only run.")
    parser.add_argument("--run-dir", required=True, help="exact relative path outputs/ctp_tuning_v1_1/<run>")
    args = parser.parse_args()
    result = finalize_development_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputValidationError as exc:
        print(f"CTP-v1.1 finalizer rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
