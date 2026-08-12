from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import (  # noqa: E402
    FeatureBundle,
    build_input_manifest,
    create_run_dir,
    environment_text,
    load_config,
    seed_everything,
    write_json,
    write_yaml,
)
from ov_probe.metrics import l2_normalize  # noqa: E402
from ov_probe.native_region import (  # noqa: E402
    inspect_native_region_inputs,
    load_native_region_directory,
)
from ov_probe.probe import run_region_probe  # noqa: E402
from ov_probe.prompts import (  # noqa: E402
    CachedTextEncoder,
    HashTextEncoder,
    build_prompt_bank,
    encode_prompt_group,
)
from ov_probe.visualization import save_region_agreement_plot  # noqa: E402


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("region_probe")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", mode="x", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def synthetic_region_bundle(
    cfg: dict[str, Any], bank: dict[str, Any]
) -> tuple[FeatureBundle, HashTextEncoder]:
    seed = int(cfg["experiment"]["seed"])
    rng = np.random.default_rng(seed)
    encoder = HashTextEncoder(int(cfg["model"]["feature_dim"]), seed)
    class_names, text, _ = encode_prompt_group(encoder, bank, "A", "closed")
    features = []
    cam_labels = []
    sam3_labels = []
    selected_records = []
    for class_index, name in enumerate(class_names):
        for region_index in range(8):
            features.append(text[class_index] + rng.normal(0, 0.045, text[class_index].shape))
            sam3_labels.append(name)
            cam_labels.append(
                name if region_index < 6 else class_names[(class_index + 1) % len(class_names)]
            )
            selected_records.append({
                "image_id": f"synthetic_{class_index:02d}",
                "candidate_index": region_index,
            })
    array = l2_normalize(np.asarray(features, dtype=np.float32))
    keys = [f"{item['image_id']}:{item['candidate_index']}" for item in selected_records]
    import hashlib

    metadata = {
        "synthetic": True,
        "scientific_evidence": False,
        "shape": list(array.shape),
        "selected_records": selected_records,
        "ordered_record_key_sha256": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
        "stored_remoteclip_predicted_class_ids_used_as_labels": False,
        "stored_remoteclip_scores_used_as_targets": False,
        "sampling": {"method": "synthetic fixture", "seed": seed},
    }
    return FeatureBundle(
        features=array,
        class_names=["unknown"] * len(array),
        metadata=metadata,
        cam_labels=cam_labels,
        sam3_source_labels=sam3_labels,
    ), encoder


def compact_region_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampling": results["sampling"],
        "interpretation_limit": results["interpretation_limit"],
        **{
            group: {
                vocabulary: {
                    key: value
                    for key, value in results[group][vocabulary].items()
                    if key != "per_region"
                }
                for vocabulary in ("closed", "expanded")
            }
            for group in ("A", "B")
        },
    }


def write_selected_records(
    path: Path,
    bundle: FeatureBundle,
    results: dict[str, Any],
) -> None:
    keys = bundle.metadata.get("selected_records", [])
    primary = results["A"]["closed"]["per_region"]
    if len(keys) != len(primary):
        raise ValueError("Selected row identities do not align with region predictions.")
    with path.open("x", encoding="utf-8") as handle:
        for index, (identity, metrics) in enumerate(zip(keys, primary)):
            row = {
                "row_index": index,
                **identity,
                "cam_label": bundle.cam_labels[index],
                "sam3_source_label": bundle.sam3_source_labels[index],
                "group_a_closed_text_prediction": metrics["predicted_class"],
                "max_similarity": metrics["max_similarity"],
                "top1_top2_margin": metrics["top1_top2_margin"],
                "normalized_entropy": metrics["normalized_entropy"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the read-only E0.5 region–text probe")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a tiny deterministic synthetic bundle; never scientific evidence.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, PROJECT_ROOT)
    if not str(cfg["experiment"]["name"]).startswith("region_probe_"):
        raise ValueError("The region runner requires an experiment.name beginning with region_probe_.")
    seed_everything(int(cfg["experiment"]["seed"]))
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    logger = configure_logging(run_dir)
    write_yaml(run_dir / "config_resolved.yaml", cfg)
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    manifest = build_input_manifest(cfg)
    manifest["run_mode"] = "synthetic_dry_run" if args.dry_run else "formal_native_region"
    write_json(run_dir / "input_manifest.json", manifest)
    bank = build_prompt_bank(cfg)
    write_json(run_dir / "prompt_bank.json", bank)
    write_json(
        run_dir / "class_mapping.json",
        {str(item["name"]): int(item["id"]) for item in cfg["data"]["classes"]},
    )
    logger.info("Allocated non-overwriting run directory: %s", run_dir)
    try:
        if args.dry_run:
            bundle, encoder = synthetic_region_bundle(cfg, bank)
            scientific = False
            readiness = {"ready": True, "mode": "synthetic_dry_run"}
            logger.info("Using 48 deterministic synthetic region features; metrics are not evidence.")
        else:
            readiness = inspect_native_region_inputs(cfg)
            text_cache = cfg["paths"].get("text_feature_cache")
            checkpoint = cfg["paths"].get("remoteclip_checkpoint")
            if not checkpoint or not Path(checkpoint).is_file():
                readiness["ready"] = False
                readiness.setdefault("items", {})["remoteclip_checkpoint"] = {
                    "path": checkpoint,
                    "status": "missing_required",
                }
            if not text_cache or not Path(text_cache).is_file() or not Path(text_cache).with_suffix(".json").is_file():
                readiness["ready"] = False
                readiness.setdefault("items", {})["text_feature_cache"] = {
                    "path": text_cache,
                    "status": "missing_required",
                }
            if not readiness["ready"]:
                summary = {
                    "status": "blocked",
                    "scientific_evidence": False,
                    "input_readiness": readiness,
                }
                write_json(run_dir / "summary_metrics.json", summary)
                logger.error("Formal native region probe is blocked by missing/invalid inputs.")
                print(json.dumps({"status": "blocked", "run_dir": str(run_dir)}, ensure_ascii=False))
                return 2
            bundle = load_native_region_directory(cfg)
            encoder = CachedTextEncoder(text_cache, cfg)
            scientific = True
        results = run_region_probe(bundle, encoder, bank, cfg)
        write_json(run_dir / "region_level_results.json", results)
        write_json(run_dir / "validated_region_input.json", bundle.metadata)
        write_selected_records(run_dir / "selected_region_records.jsonl", bundle, results)
        save_region_agreement_plot(results, run_dir / "region_weak_agreement.png")
        summary = {
            "status": "completed",
            "scientific_evidence": scientific,
            "synthetic_dry_run": bool(args.dry_run),
            "input_readiness": readiness,
            "region": compact_region_results(results),
            "row_identity": {
                "count": len(bundle.features),
                "ordered_record_key_sha256": bundle.metadata["ordered_record_key_sha256"],
            },
            "interpretation_limits": [
                "Weak-label agreement is not pixel-level segmentation accuracy.",
                "Expanded-vocabulary ranking is not unseen-class segmentation.",
                "Synthetic dry-run metrics are never scientific evidence."
                if args.dry_run
                else "The inherited image-level weak tags were simulated from LoveDA Train masks.",
            ],
        }
        write_json(run_dir / "summary_metrics.json", summary)
        logger.info("Region probe completed. scientific_evidence=%s", scientific)
        print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.error("Region probe failed: %s", exc)
        logger.debug(traceback.format_exc())
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        if not (run_dir / "summary_metrics.json").exists():
            write_json(run_dir / "summary_metrics.json", failure)
        print(json.dumps({"run_dir": str(run_dir), **failure}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
