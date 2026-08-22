"""Offline, strict fixed-support CTP-v1 versus SegEarth-OV evaluator.

This program intentionally has no model, CLIP, SAM3, candidate, or inference
code.  It validates immutable artifacts before opening any semantic GT file,
then evaluates serialized maps only.  It never writes maps or predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ov_probe.common_support import (  # noqa: E402
    CLASSES, IGNORE, TEST_AREAS, bootstrap_area_deltas, metrics, mutual_valid_mask, strict_counts,
)


CTP_MANIFEST_SHA = "b064984cd2a3baf7f70835ec8a8c8d767477066223ad7874ddcbfaeab51b0309"
SEG_MANIFEST_SHA = "f1f8f4c7d070b0640f41718d1da75f6a29572e933e2e499af693016eb0ade264"
OMEGA_MANIFEST_SHA = "2a41f5826ff882d62b0a3bcf8d6f1313a51066ed4714141198bf5622e723a25f"
TILE_MANIFEST_SHA = "2a0fab569982a87a78b13ee0353b32115c983e8bc964968873d6fbb73769b634"
SEG_METRICS_SHA = "f1993014231a506969b654cdb1b55587184aaa58106dfc49a022c5bbe26e28ea"
SEG_AGGREGATE_SHA = "c6bd7bbc0223d65877879f90e7303693247bfe70af17ce65cc14ce1583d0f1ab"
OFFICIAL_COMMIT = "3e22a969b32c6d751bdbba64a88a0b670e630f55"
CTP_HASHES = {
    11: "8cf956ba5006b5813f3153727f1392b5581cfdbbabe1a67a14b987648a88bf37",
    15: "11c2c6e094918d9fa16972925673cfe788f5e0f6d7b3dc7010555e0fa94f8b38",
    28: "63bc5243e7c0b4203f855f8873621002a97f74d8ecf27e2d2e3fe7b56d22775f",
    30: "06446ae3b962fdd66bc3382aec0ad9de09cd101f0d0bef59ce341319365cb003",
    34: "13ecd3e601a6070008f5bd2db0f89eda143e9d91b8dc6e43b0ed6cd05494732e",
}
RGB = {
    "impervious_surface": (255, 255, 255), "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255), "tree": (0, 255, 0), "car": (255, 255, 0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.link(temp, path); temp.unlink()


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Frozen hash mismatch for {path}: {actual} != {expected}")


def load_json_bound(path: Path, expected: str) -> dict[str, Any]:
    require_hash(path, expected)
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    """Hash every frozen artifact before any GT label is opened."""
    ctp = load_json_bound(args.ctp_dir / "manifest.json", CTP_MANIFEST_SHA)
    seg = load_json_bound(args.segearth_run / "prediction_manifest.json", SEG_MANIFEST_SHA)
    omega = load_json_bound(args.omega_dir.parent / "omega_candidate_manifest.json", OMEGA_MANIFEST_SHA)
    require_hash(args.omega_dir.parent / "tile_manifest.json", TILE_MANIFEST_SHA)
    require_hash(args.segearth_run / "metrics.json", SEG_METRICS_SHA)
    if ctp.get("test_areas") != list(TEST_AREAS) or seg.get("test_areas") != list(TEST_AREAS):
        raise RuntimeError("CTP or SegEarth test areas do not equal the frozen five-area split.")
    if tuple(ctp.get("support", {}).get("classes", ())) != CLASSES:
        raise RuntimeError("Formal CTP class mapping differs from frozen five-class mapping.")
    if ctp.get("fusion", {}).get("ignore_index") != IGNORE or ctp.get("fusion", {}).get("uncovered_label") != IGNORE:
        raise RuntimeError("Formal CTP ignore policy is not 255.")
    if seg.get("official_commit") != OFFICIAL_COMMIT or seg.get("prediction_aggregate_sha256") != SEG_AGGREGATE_SHA:
        raise RuntimeError("SegEarth source identity differs from frozen baseline.")
    if seg.get("prepared", {}).get("omega_manifest_sha256") != OMEGA_MANIFEST_SHA or seg.get("prepared", {}).get("tile_manifest_sha256") != TILE_MANIFEST_SHA:
        raise RuntimeError("SegEarth prepared-manifest binding differs from frozen Omega/tile manifests.")
    ctp_artifacts = ctp.get("artifacts", {})
    seg_artifacts = {item["image_id"]: item for item in seg.get("artifacts", [])}
    omega_rows = {item["image_id"]: item for item in omega.get("areas", [])}
    if set(seg_artifacts) != {f"vaih_area{area}" for area in TEST_AREAS} or set(omega_rows) != set(seg_artifacts):
        raise RuntimeError("SegEarth/Omega area mapping is not exactly the frozen five areas.")
    hashes: dict[str, str] = {
        "ctp_manifest": CTP_MANIFEST_SHA, "segearth_prediction_manifest": SEG_MANIFEST_SHA,
        "omega_manifest": OMEGA_MANIFEST_SHA, "tile_manifest": TILE_MANIFEST_SHA,
        "segearth_metrics": SEG_METRICS_SHA,
    }
    for area in TEST_AREAS:
        image_id = f"vaih_area{area}"
        ctp_path = args.ctp_dir / f"CTP_{image_id}_semantic.npz"
        if ctp_artifacts.get(ctp_path.name) != CTP_HASHES[area]:
            raise RuntimeError(f"CTP archive manifest binding differs for {ctp_path.name}")
        require_hash(ctp_path, CTP_HASHES[area]); hashes[f"ctp_{image_id}"] = CTP_HASHES[area]
        seg_path = args.segearth_run / "semantic_predictions" / f"{image_id}_semantic.npz"
        expected_seg = str(seg_artifacts[image_id].get("sha256"))
        require_hash(seg_path, expected_seg); hashes[f"segearth_{image_id}"] = expected_seg
        omega_path = args.omega_dir / f"{image_id}_omega_candidate.npz"
        expected_omega = str(omega_rows[image_id].get("mask_sha256"))
        require_hash(omega_path, expected_omega); hashes[f"omega_{image_id}"] = expected_omega
    return {"ctp": ctp, "seg": seg, "omega": omega, "hashes": hashes}


def load_gt(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
    for index, name in enumerate(CLASSES):
        result[np.all(rgb == np.asarray(RGB[name], dtype=np.uint8), axis=-1)] = index
    known = {tuple(value) for value in RGB.values()} | {(255, 0, 0)}
    colors = {tuple(int(v) for v in row) for row in np.unique(rgb.reshape(-1, 3), axis=0)}
    if colors - known:
        raise RuntimeError(f"Unknown frozen Vaihingen GT colors: {sorted(colors - known)}")
    return result


def load_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ({"label_map"} if key == "label_map" else {"prediction", "tile_hit_count"} if key == "prediction" else {"omega"}):
            raise RuntimeError(f"Unexpected NPZ schema: {path}")
        return np.asarray(archive[key])


def row(method: str, scope: str, value: dict[str, Any], *, coverage: float | None = None, clutter: int | None = None, ignored: int | None = None) -> dict[str, Any]:
    return {
        "method": method, "scope": scope, "OA": value["OA"], "Macro_F1": value["Macro_F1"], "mIoU": value["mIoU"],
        "denominator": value["denominator"], "correct": value["correct"], "semantic_predictions": value["semantic_predictions"],
        "nonsemantic_predictions": value["nonsemantic_predictions"], "ctp_abstentions": value["ctp_abstentions"],
        "ctp_abstention_ratio": value["ctp_abstention_ratio"], "coverage_of_fixed_omega": coverage,
        "segearth_clutter_predictions": clutter, "segearth_ignore_predictions": ignored,
    }


def run(args: argparse.Namespace) -> Path:
    if args.bootstrap_seed != 42 or args.bootstrap_repeats != 5000:
        raise RuntimeError("Frozen bootstrap requires seed=42 and repeats=5000.")
    bindings = validate_sources(args)  # No GT access before this line succeeds.
    run_dir = args.output_root / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pre = {
        "format_version": 1, "status": "pre_gt_hash_validated", "gt_read": False,
        "model_or_feature_inference": False, "new_prediction_or_map_written": False,
        "frozen_assets_unchanged_attestation": "pending_post_evaluation_rehash",
        "input_hashes": bindings["hashes"], "test_areas": list(TEST_AREAS), "classes": list(CLASSES),
        "ignore_index": IGNORE, "evaluator_sha256": sha256_file(Path(__file__)), "code_commit": git_commit(),
    }
    write_json_exclusive(run_dir / "common_support_evaluation_manifest.json", pre)
    fixed_ctp: dict[int, Any] = {}; fixed_seg: dict[int, Any] = {}
    mutual_ctp: dict[int, Any] = {}; mutual_seg: dict[int, Any] = {}
    fixed_rows: list[dict[str, Any]] = []; mutual_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []; class_rows: list[dict[str, Any]] = []
    try:
        for area in TEST_AREAS:
            image_id = f"vaih_area{area}"
            gt = load_gt(args.labels_dir / f"{image_id}_label.tif")
            ctp = load_array(args.ctp_dir / f"CTP_{image_id}_semantic.npz", "label_map")
            seg = load_array(args.segearth_run / "semantic_predictions" / f"{image_id}_semantic.npz", "prediction")
            omega = load_array(args.omega_dir / f"{image_id}_omega_candidate.npz", "omega").astype(bool, copy=False)
            if gt.shape != ctp.shape or gt.shape != seg.shape or gt.shape != omega.shape:
                raise RuntimeError(f"Shape mismatch on {image_id}")
            fixed_ctp[area] = strict_counts(omega, gt, ctp, method="CTP-v1")
            fixed_seg[area] = strict_counts(omega, gt, seg, method="SegEarth-OV")
            mutual = mutual_valid_mask(omega, ctp, seg)
            mutual_ctp[area] = strict_counts(mutual, gt, ctp, method="CTP-v1")
            mutual_seg[area] = strict_counts(mutual, gt, seg, method="SegEarth-OV")
            for scope, ctp_count, seg_count, mask in (("fixed_omega_strict", fixed_ctp[area], fixed_seg[area], omega), ("mutual_valid_diagnostic", mutual_ctp[area], mutual_seg[area], mutual)):
                ctp_m, seg_m = metrics(ctp_count), metrics(seg_count)
                coverage = float(mask.sum() / omega.sum()) if omega.sum() else 0.0
                fixed = scope == "fixed_omega_strict"
                clutter = int(np.count_nonzero(seg[mask] == 5))
                ignored = int(np.count_nonzero(seg[mask] == IGNORE))
                target = fixed_rows if fixed else mutual_rows
                target.extend([row("CTP-v1", scope, ctp_m, coverage=coverage), row("SegEarth-OV", scope, seg_m, coverage=coverage, clutter=clutter, ignored=ignored)])
                area_rows.append({"area": area, "image_id": image_id, "scope": scope, "omega_pixels": int(omega.sum()), "evaluated_pixels": int(mask.sum()), "coverage_of_fixed_omega": coverage, "SegEarth_OA": seg_m["OA"], "CTP_OA": ctp_m["OA"], "Delta_OA_CTP_minus_SegEarth": ctp_m["OA"] - seg_m["OA"], "SegEarth_Macro_F1": seg_m["Macro_F1"], "CTP_Macro_F1": ctp_m["Macro_F1"], "Delta_Macro_F1_CTP_minus_SegEarth": ctp_m["Macro_F1"] - seg_m["Macro_F1"], "SegEarth_mIoU": seg_m["mIoU"], "CTP_mIoU": ctp_m["mIoU"], "Delta_mIoU_CTP_minus_SegEarth": ctp_m["mIoU"] - seg_m["mIoU"], "CTP_abstention_ratio": ctp_m["ctp_abstention_ratio"], "SegEarth_clutter_predictions": clutter, "SegEarth_ignore_predictions": ignored})
                for class_name in CLASSES:
                    c, s = ctp_m["per_class"][class_name], seg_m["per_class"][class_name]
                    class_rows.append({"scope": scope, "class": class_name, "SegEarth_Precision": s["Precision"], "CTP_Precision": c["Precision"], "SegEarth_Recall": s["Recall"], "CTP_Recall": c["Recall"], "SegEarth_F1": s["F1"], "CTP_F1": c["F1"], "SegEarth_IoU": s["IoU"], "CTP_IoU": c["IoU"], "Delta_IoU_CTP_minus_SegEarth": c["IoU"] - s["IoU"]})
        def total(items: dict[int, Any]) -> Any:
            first = items[TEST_AREAS[0]]
            result = type(first)(np.zeros_like(first.confusion), np.zeros_like(first.fn_extra), 0, 0, 0, 0, 0)
            for area in TEST_AREAS: result = result.add(items[area])
            return result
        fixed_ctp_m, fixed_seg_m = metrics(total(fixed_ctp)), metrics(total(fixed_seg))
        mutual_ctp_m, mutual_seg_m = metrics(total(mutual_ctp)), metrics(total(mutual_seg))
        fixed_rows.extend([row("CTP-v1", "fixed_omega_strict", fixed_ctp_m, coverage=1.0), row("SegEarth-OV", "fixed_omega_strict", fixed_seg_m, coverage=1.0)])
        mutual_coverage = float(mutual_ctp_m["denominator"] / fixed_ctp_m["denominator"])
        mutual_rows.extend([row("CTP-v1", "mutual_valid_diagnostic", mutual_ctp_m, coverage=mutual_coverage), row("SegEarth-OV", "mutual_valid_diagnostic", mutual_seg_m, coverage=mutual_coverage)])
        bootstrap = bootstrap_area_deltas(fixed_ctp, fixed_seg, seed=args.bootstrap_seed, repeats=args.bootstrap_repeats)
        write_csv_exclusive(run_dir / "segearth_ctp_fixed_support_metrics.csv", fixed_rows)
        write_csv_exclusive(run_dir / "segearth_ctp_mutual_valid_metrics.csv", mutual_rows)
        write_csv_exclusive(run_dir / "segearth_ctp_per_area.csv", area_rows)
        write_csv_exclusive(run_dir / "segearth_ctp_per_class.csv", class_rows)
        write_json_exclusive(run_dir / "segearth_ctp_common_support_bootstrap.json", bootstrap)
        for name, frozen_hash in bindings["hashes"].items():
            if name.startswith("ctp_") or name.startswith("segearth_") or name.startswith("omega_"):
                path = None
                if name.startswith("ctp_vaih_area"): path = args.ctp_dir / f"CTP_{name.removeprefix('ctp_')}_semantic.npz"
                elif name.startswith("segearth_vaih_area"): path = args.segearth_run / "semantic_predictions" / f"{name.removeprefix('segearth_')}_semantic.npz"
                elif name.startswith("omega_vaih_area"): path = args.omega_dir / f"{name.removeprefix('omega_')}_omega_candidate.npz"
                if path is not None: require_hash(path, frozen_hash)
        final = dict(pre)
        final.update({"status": "completed", "gt_read": True, "frozen_assets_unchanged_attestation": "all source NPZ hashes were revalidated after offline scoring; evaluator contains no inference code", "outputs": {p.name: sha256_file(p) for p in run_dir.iterdir() if p.name != "common_support_evaluation_manifest.json"}, "fixed_metrics": {"CTP-v1": fixed_ctp_m, "SegEarth-OV": fixed_seg_m}, "mutual_valid_metrics": {"CTP-v1": mutual_ctp_m, "SegEarth-OV": mutual_seg_m}, "mutual_valid_coverage": mutual_coverage})
        (run_dir / "common_support_evaluation_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    except Exception as exc:
        raise RuntimeError(f"Offline evaluation failed after unique output creation: {type(exc).__name__}: {exc}") from exc
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctp-dir", type=Path, required=True)
    parser.add_argument("--segearth-run", type=Path, required=True)
    parser.add_argument("--omega-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()
    try:
        output = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True)); return 2
    print(json.dumps({"status": "completed", "output_dir": str(output)}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
