from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .encoder_bridge import _iter_pixel_views, _normalize
from .io import InputValidationError, sha256_file
from .pixel_pack import validate_region_pixel_pack


def load_compare_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Comparison config must be a mapping with overwrite=false.")
    for key, value in cfg["paths"].items():
        candidate = Path(str(value))
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise InputValidationError(f"Comparison path must remain inside project root: {key}") from exc
        cfg["paths"][key] = str(resolved)
    if Path(cfg["paths"]["output_root"]).parent != (root / "outputs").resolve():
        raise InputValidationError("Comparison output must be directly under project outputs/.")
    protocol_path = Path(cfg["paths"]["protocol_file"])
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol["path"] = str(protocol_path)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_compare_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != expected_commit:
        raise InputValidationError("Comparison code commit differs from the approved commit.")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if dirty.strip():
        raise InputValidationError("Tracked comparison worktree must be clean.")
    actual = sha256_file(root / "configs" / "encoder_compare_run_protocol_v0.json")
    if actual != expected_protocol_sha256:
        raise InputValidationError("Comparison protocol differs from the approved SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def _load_records(package: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (package / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
        raise InputValidationError("Comparison records are not in frozen row order.")
    return rows


def _encode_model(
    package: Path, checkpoint: Path, architecture: str, batch_regions: int, device: str
) -> tuple[np.ndarray, Any, Any, str]:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(architecture)
    count = json.loads((package / "manifest.json").read_text(encoding="utf-8"))["record_count"]
    features = np.empty((int(count), 512), dtype=np.float32)
    with torch.inference_mode():
        for rows, contexts, masked_views in _iter_pixel_views(package):
            for start in range(0, len(rows), batch_regions):
                stop = min(start + batch_regions, len(rows))
                tensors = []
                for context, masked in zip(contexts[start:stop], masked_views[start:stop]):
                    tensors.extend([preprocess(Image.fromarray(context)), preprocess(Image.fromarray(masked))])
                encoded = model.encode_image(torch.stack(tensors).to(device)).float()
                encoded = encoded / encoded.norm(dim=1, keepdim=True)
                encoded = encoded.reshape(-1, 2, 512).mean(dim=1)
                encoded = encoded / encoded.norm(dim=1, keepdim=True)
                features[rows[start:stop]] = encoded.cpu().numpy()
    return _normalize(features), model, tokenizer, repr(preprocess)


def _prompt_texts(protocol: dict[str, Any], group: str, vocabulary: str) -> tuple[list[str], dict[str, list[str]]]:
    names = list(protocol["classes"])
    if vocabulary == "expanded":
        names += list(protocol["prompts"]["distractors"])
    prompts: dict[str, list[str]] = {}
    for name in names:
        if group == "A":
            prompts[name] = [value.format(**{"class": name}) for value in protocol["prompts"]["group_a_templates"]]
        else:
            aliases = protocol["prompts"]["aliases"].get(name, [name])
            prompts[name] = [template.format(alias=alias) for alias in aliases for template in protocol["prompts"]["group_b_templates"]]
    return names, prompts


def _text_prototypes(model: Any, tokenizer: Any, protocol: dict[str, Any], group: str, vocabulary: str, device: str) -> tuple[list[str], np.ndarray, str]:
    import torch

    names, prompts = _prompt_texts(protocol, group, vocabulary)
    flat = [text for name in names for text in prompts[name]]
    token_ids = tokenizer(flat)
    token_hash = hashlib.sha256(token_ids.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(flat), 256):
            encoded = model.encode_text(token_ids[start : start + 256].to(device)).float()
            encoded = encoded / encoded.norm(dim=1, keepdim=True)
            vectors.append(encoded.cpu().numpy())
    all_vectors = np.concatenate(vectors)
    prototypes = []
    offset = 0
    for name in names:
        count = len(prompts[name])
        prototypes.append(_normalize(all_vectors[offset : offset + count].mean(axis=0, keepdims=True))[0])
        offset += count
    return names, np.asarray(prototypes, dtype=np.float32), token_hash


def _metrics(predictions: np.ndarray, names: list[str], records: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    predicted = np.asarray([names[int(index)] for index in predictions])
    sam3 = np.asarray([row["sam3_source_label"] for row in records])
    cam = np.asarray([row["cam_label"] for row in records])
    per_class = {name: float(np.mean(predicted[sam3 == name] == name)) for name in classes}
    return {
        "sam3_agreement": float(np.mean(predicted == sam3)),
        "macro_sam3_agreement": float(np.mean(list(per_class.values()))),
        "cam_agreement": float(np.mean(predicted == cam)),
        "three_way_agreement": float(np.mean((predicted == sam3) & (predicted == cam))),
        "distractor_rate": float(np.mean(~np.isin(predicted, classes))),
        "per_class_sam3_agreement": per_class,
        "prediction_counts": {name: int(np.sum(predicted == name)) for name in names},
    }


def clustered_macro_bootstrap(
    remote_predictions: np.ndarray,
    clip_predictions: np.ndarray,
    records: list[dict[str, Any]],
    classes: list[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    image_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        image_rows[str(row["image_id"])].append(index)
    domains: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(image_rows):
        domain = "rural" if "_rural_" in image_id else "urban"
        domains[domain].append(image_id)
    sam3 = np.asarray([row["sam3_source_label"] for row in records])
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected: list[int] = []
        for image_ids in domains.values():
            sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
            for image_id in sampled:
                selected.extend(image_rows[str(image_id)])
        indices = np.asarray(selected, dtype=np.int64)
        remote_macro = np.mean([np.mean(remote_predictions[indices][sam3[indices] == name] == name) for name in classes])
        clip_macro = np.mean([np.mean(clip_predictions[indices][sam3[indices] == name] == name) for name in classes])
        deltas[replicate] = clip_macro - remote_macro
    low, high = np.quantile(deltas, [0.025, 0.975])
    decision = "openai_clip_better" if low > 0 else ("remoteclip_better" if high < 0 else "inconclusive")
    return {
        "replicates": replicates,
        "seed": seed,
        "strata": {name: len(values) for name, values in domains.items()},
        "mean_delta": float(deltas.mean()),
        "ci95": [float(low), float(high)],
        "decision": decision,
    }


def run_encoder_comparison(cfg: dict[str, Any], protocol: dict[str, Any], anchor: dict[str, str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import open_clip
    import torch

    if open_clip.__version__ != protocol["software"]["open_clip"]:
        raise InputValidationError("OpenCLIP version differs from registered comparison software.")
    package = Path(cfg["paths"]["pixel_pack"])
    validation = validate_region_pixel_pack(package, package / "encoder_compare_protocol_v0.json")
    for key in ("bundle_id", "record_count", "image_count", "ordered_record_key_sha256"):
        if validation[key] != protocol["pixel_pack"][key]:
            raise InputValidationError(f"Pixel pack differs from comparison registration: {key}")
    bridge_path = Path(cfg["paths"]["remoteclip_bridge_summary"])
    gate = protocol["remoteclip_bridge_gate"]
    if sha256_file(bridge_path) != gate["summary_sha256"]:
        raise InputValidationError("RemoteCLIP bridge summary differs from comparison registration.")
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if bridge.get("status") != gate["required_status"]:
        raise InputValidationError("RemoteCLIP bridge gate did not pass.")
    if bridge.get("pixel_pack_validation", {}).get("bundle_id") != gate["required_bundle_id"]:
        raise InputValidationError("RemoteCLIP bridge used a different pixel pack.")
    records = _load_records(package)
    requested = str(cfg["runtime"]["device"])
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    model_paths = {
        "remoteclip": Path(cfg["paths"]["remoteclip_checkpoint"]),
        "openai_clip": Path(cfg["paths"]["openai_clip_checkpoint"]),
    }
    for model_name, path in model_paths.items():
        if sha256_file(path) != protocol["models"][model_name]["checkpoint_sha256"]:
            raise InputValidationError(f"Checkpoint differs from comparison registration: {model_name}")
    metrics: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    features: dict[str, np.ndarray] = {}
    token_hashes: dict[str, dict[str, str]] = {}
    preprocesses: dict[str, str] = {}
    for model_name in ("remoteclip", "openai_clip"):
        spec = protocol["models"][model_name]
        feature, model, tokenizer, preprocess = _encode_model(
            package, model_paths[model_name], spec["architecture"], int(cfg["runtime"]["batch_regions"]), device
        )
        features[model_name] = feature
        predictions[model_name] = {}
        metrics[model_name] = {}
        token_hashes[model_name] = {}
        preprocesses[model_name] = preprocess
        for evaluation in protocol["evaluations"]:
            group, vocabulary = evaluation["prompt_group"], evaluation["vocabulary"]
            key = f"group_{group}_{vocabulary}"
            names, text_features, token_hash = _text_prototypes(model, tokenizer, protocol, group, vocabulary, device)
            scores = feature @ text_features.T
            predicted_names = np.asarray(names)[np.argmax(scores, axis=1)]
            predictions[model_name][key] = predicted_names
            token_hashes[model_name][key] = token_hash
            metrics[model_name][key] = _metrics(np.argmax(scores, axis=1), names, records, protocol["classes"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    primary_key = "group_A_closed"
    bootstrap = clustered_macro_bootstrap(
        predictions["remoteclip"][primary_key], predictions["openai_clip"][primary_key], records,
        protocol["classes"], int(protocol["primary_endpoint"]["bootstrap_replicates"]), int(protocol["seed"])
    )
    summary = {
        "status": "completed",
        "scientific_evidence": True,
        "scope": "paired weak-label region-text diagnostic; not true accuracy or segmentation",
        "repository_anchor": anchor,
        "pixel_pack_validation": validation,
        "device": device,
        "models": protocol["models"],
        "preprocesses": preprocesses,
        "prompt_token_sha256": token_hashes,
        "metrics": metrics,
        "primary_endpoint": {**protocol["primary_endpoint"], "result": bootstrap},
        "constraints": protocol["constraints"],
    }
    arrays = {f"features_{name}": value.astype(np.float16) for name, value in features.items()}
    for model_name, groups in predictions.items():
        for key, value in groups.items():
            arrays[f"predictions_{model_name}_{key}"] = value
    return summary, arrays
