"""Runner for the exhaustive partial-support benchmark (Phases J/K/L/M).

Usage:

    python scripts/run_loveda_partial_support.py \\
        --config <evaluate deployment yaml> --run-dir <p0 run dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError  # noqa: E402
from ov_probe.loveda_blind_gt import (  # noqa: E402
    _CLASSES,
    _load_record_masks,
    _region_gt_from_label,
    load_loveda_blind_gt_config,
)
from ov_probe.loveda_partial_support import benchmark_all_subsets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustive partial-support benchmark.")
    parser.add_argument("--config", required=True, help="Evaluate-phase deployment YAML.")
    parser.add_argument("--run-dir", required=True, help="Run directory with predictions.npz / heldout_keys.jsonl.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
    run_dir = Path(args.run_dir).resolve()

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        text_protos = archive["text_prototypes"].astype(np.float32)
        visual_protos = archive["visual_prototypes"].astype(np.float32)
        text_pred_frozen = archive["text_only"].astype(np.int64)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    records = [json.loads(line) for line in (run_dir / "heldout_keys.jsonl").open(encoding="utf-8")]
    label_dir = cfg["paths"]["loveda_label_dir"]
    color_map = {name: tuple(int(v) for v in color) for name, color in protocol["ground_truth"]["color_map"].items()}
    mask_info = _load_record_masks(cfg["paths"]["pixel_pack"])
    from PIL import Image
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            with Image.open(Path(label_dir) / f"{image_id}_label.png") as image:
                label_cache[image_id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        info = mask_info.get(int(row["row_index"]))
        if info is None:
            raise InputValidationError(f"No pixel-pack mask for row {row['row_index']}.")
        label, _, _ = _region_gt_from_label(info, label_cache[image_id], color_map)
        gt_labels.append(label)
    labeled = np.asarray([gt is not None for gt in gt_labels], dtype=bool)
    gt_index = np.asarray([_CLASSES.index(gt) for gt in gt_labels if gt is not None], dtype=np.int64)
    hold_labeled = hold_indices[labeled]

    subset_rows, k_rows, checks = benchmark_all_subsets(
        text_scores, visual_scores, text_protos, visual_protos,
        text_pred_frozen, gt_index, hold_labeled, _CLASSES,
    )

    method_names = ["text_only", "C1", "C2", "SCC", "guard"]
    header = ["subset_index", "k", "supported", "unsupported"]
    for name in method_names:
        header += [f"{name}_S", f"{name}_U", f"{name}_H", f"{name}_acc", f"{name}_macro_f1", f"{name}_macro_iou"]
    with (run_dir / "partial_support_all_subsets.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(header) + "\n")
        for row in subset_rows:
            handle.write(",".join("nan" if isinstance(v, float) and np.isnan(v) else str(v) for v in row) + "\n")
    with (run_dir / "partial_support_by_k.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(",".join(str(v) for v in row) for row in k_rows) + "\n")
    with (run_dir / "partial_support_checks.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(checks, handle, indent=2, sort_keys=True)

    print(f"SCC k=0 == Text-only: {checks['scc_k0_equals_text']}")
    print(f"SCC k=6 == C2: {checks['scc_k6_equals_c2']}")
    print("\n=== mean S/U/H by support count k ===")
    for row in k_rows:
        print("  " + "  ".join(str(v) for v in row))
    print("\nwrote", run_dir / "partial_support_all_subsets.csv", run_dir / "partial_support_by_k.csv", run_dir / "partial_support_checks.json")
    return 0 if (checks["scc_k0_equals_text"] and checks["scc_k6_equals_c2"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
