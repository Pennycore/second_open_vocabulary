"""Run the frozen RemoteCLIP Potsdam baseline in isolated predict/evaluate phases."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ov_probe.io import InputValidationError, sha256_file
from ov_probe.pixel_ovss import assemble_semantic_map, load_candidate_masks
from ov_probe.remoteclip_potsdam_baseline import (
    CLASSES,
    COLORS,
    METHODS,
    _aggregate,
    _load_model,
    _normalize,
    crop_views,
    directory_sha256,
    load_config,
    pixel_confusion_fast,
    score_methods,
    text_prototypes,
)

def _cfg(path):
    cfg, protocol = load_config(path)
    return cfg, protocol, Path(path).resolve().parents[1]


def _source_manifest(paths, *, include_labels: bool) -> dict:
    candidates_hash, candidate_count = directory_sha256(Path(paths["candidates_dir"]), "*.npz")
    images_hash, image_count = directory_sha256(Path(paths["image_dir"]), "*_RGB.tif")
    result = {
        "candidates": {"sha256": candidates_hash, "count": candidate_count},
        "images": {"sha256": images_hash, "count": image_count},
        "checkpoint": {"sha256": sha256_file(Path(paths["checkpoint"]))},
    }
    if include_labels:
        labels_hash, label_count = directory_sha256(Path(paths["label_dir"]), "*_label.tif")
        result["labels"] = {"sha256": labels_hash, "count": label_count}
    return result


def _validate_source_manifest(cfg, actual: dict, *, include_labels: bool) -> None:
    expected = cfg.get("source_hashes", {})
    needed = ["candidates", "images", "checkpoint"] + (["labels"] if include_labels else [])
    for key in needed:
        if expected.get(key) != actual[key]["sha256"]:
            raise InputValidationError(f"Strict source hash mismatch for {key}.")


def _validate_image_mapping(paths: dict, candidate_paths: list[Path]) -> None:
    image_dir = Path(paths["image_dir"])
    for candidate_path in candidate_paths:
        image_id = candidate_path.stem
        if not (image_dir / f"{image_id}_RGB.tif").is_file():
            raise InputValidationError(f"Candidate/image mapping missing for {image_id}.")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=["predict", "evaluate"])
    args = ap.parse_args()
    cfg, protocol, _ = _cfg(args.config)
    paths = cfg["paths"]
    out = Path(paths["output_root"]).resolve()
    if args.phase == "predict":
        candidate_paths = sorted(Path(paths["candidates_dir"]).glob("*.npz"))
        _validate_image_mapping(paths, candidate_paths)
        sources = _source_manifest(paths, include_labels=False)
        _validate_source_manifest(cfg, sources, include_labels=False)
        if sources["checkpoint"]["sha256"] != protocol["model"]["checkpoint_sha256"]:
            raise InputValidationError("RemoteCLIP checkpoint does not match the frozen protocol.")
        out.mkdir(parents=True, exist_ok=False)
        import tifffile, torch
        from PIL import Image
        dev = "cuda" if cfg.get("runtime", {}).get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu"
        model, preprocess, tokenizer, torch = _load_model(Path(paths["checkpoint"]), protocol, dev)
        text_proto, token_hash = text_prototypes(model, tokenizer, protocol, dev, torch)
        records, all_features, rows_by_image = [], [], {}
        cand, imgdir = Path(paths["candidates_dir"]), Path(paths["image_dir"])
        for npz in candidate_paths:
            image_id = npz.stem
            shape, regions = load_candidate_masks(cand, image_id)
            if not regions:
                continue
            arr = tifffile.imread(imgdir / f"{image_id}_RGB.tif")
            rgb = arr[:, :, :3] if arr.ndim == 3 else arr
            if rgb.dtype != np.uint8:
                lo, hi = float(np.percentile(rgb, 1)), float(np.percentile(rgb, 99))
                rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            feats = []
            batch_size = int(cfg.get("runtime", {}).get("image_batch", 32))
            for start in range(0, len(regions), batch_size):
                batch = []
                for c in regions[start:start + batch_size]:
                    context, masked, _ = crop_views(rgb, c["mask"], c["x0"], c["y0"])
                    batch.extend([preprocess(Image.fromarray(context)), preprocess(Image.fromarray(masked))])
                with torch.inference_mode():
                    e = model.encode_image(torch.stack(batch).to(dev)).float()
                    e = e / e.norm(dim=1, keepdim=True)
                    e = e.reshape(-1, 2, e.shape[-1]).mean(dim=1)
                    e = e / e.norm(dim=1, keepdim=True)
                feats.append(e.cpu().numpy())
            feats = np.concatenate(feats).astype(np.float32)
            all_features.append(feats)
            local = []
            for i, c in enumerate(regions):
                r = {"row_index": len(records), "image_id": image_id, "candidate_index": i, "sam3_source_label": c["class_name"], "x0": c["x0"], "y0": c["y0"]}
                records.append(r)
                local.append(r)
            rows_by_image[image_id] = (shape, regions, local)
        features = _normalize(np.concatenate(all_features))
        visual = np.asarray([_normalize(features[[r["row_index"] for r in records if r["sam3_source_label"] == c]].mean(axis=0, keepdims=True))[0] for c in CLASSES])
        text_scores, visual_scores = features @ text_proto.T, features @ visual.T
        pred, scores = score_methods(text_scores, visual_scores, text_proto, visual)
        for image_id, (shape, regions, local) in rows_by_image.items():
            for method in METHODS:
                p = np.asarray([pred[method][r["row_index"]] for r in local])
                s = np.asarray([scores[method][r["row_index"], p[j]] for j, r in enumerate(local)])
                label_map, _ = assemble_semantic_map(shape, regions, p, s, CLASSES)
                with (out / f"{method}_{image_id}_semantic.npz").open("xb") as h:
                    np.savez_compressed(h, label_map=label_map)
        with (out / "records.jsonl").open("x", encoding="utf-8") as h:
            for r in records:
                h.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
        with (out / "predictions.npz").open("xb") as h:
            np.savez_compressed(h, features=features.astype(np.float16), text_scores=text_scores.astype(np.float16), visual_scores=visual_scores.astype(np.float16), text_prototypes=text_proto.astype(np.float16), visual_prototypes=visual.astype(np.float16), text_pred=pred["text_only"])
        artifacts = {p.name: sha256_file(p) for p in sorted(out.glob("*_semantic.npz"))}
        manifest = {"format_version": 1, "phase": "predict", "status": "completed", "scientific_evidence": False, "dataset": "Potsdam", "methods": METHODS, "record_count": len(records), "image_count": len(rows_by_image), "protocol_sha256": hashlib.sha256(Path(paths["protocol_file"]).read_bytes()).hexdigest(), "sources": sources, "checkpoint_sha256": sources["checkpoint"]["sha256"], "token_sha256": token_hash, "predictions_npz_sha256": sha256_file(out / "predictions.npz"), "artifacts": artifacts, "gt_read": False}
        with (out / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return 0
    import tifffile
    man = json.loads((out / "manifest.json").read_text())
    if man.get("phase") != "predict" or man.get("status") != "completed":
        raise InputValidationError("Predict manifest missing.")
    for name, expected in man["artifacts"].items():
        if sha256_file(out / name) != expected:
            raise InputValidationError(f"Prediction artifact changed since predict: {name}")
    if sha256_file(out / "predictions.npz") != man["predictions_npz_sha256"]:
        raise InputValidationError("predictions.npz changed since predict.")
    sources = _source_manifest(paths, include_labels=True)
    _validate_source_manifest(cfg, sources, include_labels=True)
    if sources["candidates"] != man["sources"]["candidates"] or sources["images"] != man["sources"]["images"] or sources["checkpoint"] != man["sources"]["checkpoint"]:
        raise InputValidationError("Non-GT sources changed since predict.")
    records = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
    ids = sorted({r["image_id"] for r in records})
    results = {m: [] for m in METHODS}
    labeldir = Path(paths["label_dir"])
    for image_id in ids:
        lab = tifffile.imread(labeldir / f"{image_id}_label.tif")
        lab = lab[:, :, :3] if lab.ndim == 3 else lab
        gt = np.full(lab.shape[:2], 255, dtype=np.int64)
        for i, c in enumerate(CLASSES):
            gt[np.all(lab == np.asarray(COLORS[c], dtype=np.uint8), axis=-1)] = i
        for method in METHODS:
            with np.load(out / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as a:
                pred = a["label_map"].astype(np.int64)
            results[method].append(pixel_confusion_fast(pred, gt, CLASSES))
    overall = {method: _aggregate(results[method]) for method in METHODS}
    with (out / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)
    man.update({"phase": "evaluate", "scientific_evidence": True, "gt_read": True, "sources": sources, "metrics_sha256": sha256_file(out / "metrics.json")})
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
