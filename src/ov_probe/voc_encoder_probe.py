from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import InputValidationError, sha256_file


IMAGE_ID_PATTERN = re.compile(r"^\d{4}_\d{6}$")


def load_voc_probe_config(path: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("VOC probe config must have overwrite=false.")
    for key, value in cfg["paths"].items():
        candidate = Path(str(value))
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise InputValidationError(f"VOC probe path escapes project root: {key}") from exc
        cfg["paths"][key] = str(resolved)
    output = Path(cfg["paths"]["output_root"])
    if output.parent != (root / "outputs").resolve():
        raise InputValidationError("VOC probe output must be directly under project outputs/.")
    protocol_path = Path(cfg["paths"]["protocol_file"])
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_voc_probe_anchor(project_root: Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = project_root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip()
    if commit != expected_commit or dirty:
        raise InputValidationError("VOC probe requires the approved clean code commit.")
    protocol_path = root / "configs" / "voc_encoder_probe_protocol_v0.json"
    actual = sha256_file(protocol_path)
    if actual != expected_protocol_sha256:
        raise InputValidationError("VOC probe protocol differs from the approved hash.")
    return {"code_commit": commit, "protocol_sha256": actual}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or values.shape != truth.shape or not np.isfinite(values).all():
        raise InputValidationError("Invalid arrays for average precision.")
    positives = int(truth.sum())
    if positives == 0:
        raise InputValidationError("Average precision is undefined without positives.")
    order = np.argsort(-values, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def macro_average_precision(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise InputValidationError("VOC label and score matrices must have the same 2-D shape.")
    per_class = np.asarray(
        [average_precision(labels[:, index], scores[:, index]) for index in range(labels.shape[1])],
        dtype=np.float64,
    )
    return float(per_class.mean()), per_class


def _load_registered_rows(tag_bundle: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    manifest_path = tag_bundle / "manifest.json"
    records_path = tag_bundle / "records.jsonl"
    if sha256_file(manifest_path) != protocol["inputs"]["tag_manifest_sha256"]:
        raise InputValidationError("VOC tag manifest hash mismatch.")
    if sha256_file(records_path) != protocol["inputs"]["tag_records_sha256"]:
        raise InputValidationError("VOC tag records hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("segmentation_masks_read") is not False or manifest.get("voc_val_read") is not False:
        raise InputValidationError("VOC tag bundle violates the registered supervision boundary.")
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != int(protocol["record_count"]):
        raise InputValidationError("VOC tag record count mismatch.")
    if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
        raise InputValidationError("VOC tag rows are not in registered order.")
    classes = list(protocol["class_text"])
    if manifest.get("class_names") != classes or int(manifest.get("record_count", -1)) != len(rows):
        raise InputValidationError("VOC tag manifest ontology or count mismatch.")
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = np.zeros((len(rows), len(classes)), dtype=bool)
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        image_id = str(row["image_id"])
        if not IMAGE_ID_PATTERN.fullmatch(image_id) or image_id in seen:
            raise InputValidationError("Unsafe or duplicate VOC image ID.")
        seen.add(image_id)
        names = row.get("class_names")
        if not isinstance(names, list) or not names:
            raise InputValidationError("VOC tag row has no labels.")
        for name in names:
            if name not in class_to_index:
                raise InputValidationError(f"Unknown VOC class in tag row: {name}")
            labels[row_index, class_to_index[name]] = True
    return rows, labels, classes


def _source_inventory(image_dir: Path, rows: list[dict[str, Any]]) -> tuple[str, list[Path]]:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for row in rows:
        path = image_dir / f"{row['image_id']}.jpg"
        if not path.is_file() or path.is_symlink():
            raise InputValidationError(f"Missing regular VOC train image: {path}")
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{sha256_file(path)}\n".encode("utf-8"))
        paths.append(path)
    return digest.hexdigest(), paths


def _load_model(checkpoint: Path, architecture: str, device: str) -> tuple[Any, Any, Any, str]:
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, open_clip.get_tokenizer(architecture), preprocess, repr(preprocess)


def _encode_images(model: Any, preprocess: Any, paths: list[Path], batch_size: int, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    output = np.empty((len(paths), 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            stop = min(start + batch_size, len(paths))
            tensors = []
            for path in paths[start:stop]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            encoded = model.encode_image(torch.stack(tensors).to(device)).float()
            encoded = encoded / encoded.norm(dim=1, keepdim=True)
            output[start:stop] = encoded.cpu().numpy()
    return output


def _text_features(model: Any, tokenizer: Any, protocol: dict[str, Any], device: str) -> tuple[np.ndarray, str]:
    import torch

    prompts = [
        template.format(**{"class": protocol["class_text"][name]})
        for name in protocol["class_text"]
        for template in protocol["prompt_templates"]
    ]
    token_ids = tokenizer(prompts)
    token_hash = hashlib.sha256(token_ids.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    with torch.inference_mode():
        encoded = model.encode_text(token_ids.to(device)).float()
        encoded = encoded / encoded.norm(dim=1, keepdim=True)
        encoded = encoded.reshape(len(protocol["class_text"]), len(protocol["prompt_templates"]), -1).mean(dim=1)
        encoded = encoded / encoded.norm(dim=1, keepdim=True)
    return encoded.cpu().numpy(), token_hash


def paired_bootstrap(
    labels: np.ndarray,
    remote_scores: np.ndarray,
    clip_scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    index = 0
    rejected = 0
    while index < replicates:
        sample = rng.integers(0, labels.shape[0], size=labels.shape[0])
        if np.any(labels[sample].sum(axis=0) == 0):
            rejected += 1
            if rejected > replicates * 20:
                raise InputValidationError("Too many bootstrap samples lacked a positive class.")
            continue
        remote, _ = macro_average_precision(labels[sample], remote_scores[sample])
        clip, _ = macro_average_precision(labels[sample], clip_scores[sample])
        deltas[index] = clip - remote
        index += 1
    low, high = np.quantile(deltas, [0.025, 0.975])
    decision = "openai_clip_better" if low > 0 else ("remoteclip_better" if high < 0 else "inconclusive")
    return {
        "replicates": replicates,
        "seed": seed,
        "rejected_resamples_without_all_classes": rejected,
        "mean_delta": float(deltas.mean()),
        "ci95": [float(low), float(high)],
        "decision": decision,
    }


def run_voc_encoder_probe(
    cfg: dict[str, Any], protocol: dict[str, Any], anchor: dict[str, str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import open_clip
    import torch

    if open_clip.__version__ != "3.3.0":
        raise InputValidationError("VOC probe requires OpenCLIP 3.3.0.")
    if sha256_file(Path(cfg["paths"]["dataset_manifest"])) != protocol["inputs"]["dataset_manifest_sha256"]:
        raise InputValidationError("VOC dataset manifest hash mismatch.")
    rows, labels, classes = _load_registered_rows(Path(cfg["paths"]["tag_bundle"]), protocol)
    image_dir = Path(cfg["paths"]["image_dir"])
    inventory_before, image_paths = _source_inventory(image_dir, rows)
    requested = str(cfg["runtime"]["device"])
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    batch_size = int(cfg["runtime"]["batch_images"])
    if batch_size <= 0:
        raise InputValidationError("VOC probe batch size must be positive.")
    scores_by_model: dict[str, np.ndarray] = {}
    features_by_model: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for model_name in ("remoteclip", "openai_clip"):
        spec = protocol["models"][model_name]
        checkpoint = Path(cfg["paths"][f"{model_name}_checkpoint"])
        if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
            raise InputValidationError(f"VOC probe checkpoint hash mismatch: {model_name}")
        model, tokenizer, preprocess, preprocess_repr = _load_model(checkpoint, spec["architecture"], device)
        image_features = _encode_images(model, preprocess, image_paths, batch_size, device)
        text_features, token_hash = _text_features(model, tokenizer, protocol, device)
        scores = image_features @ text_features.T
        macro_ap, per_class = macro_average_precision(labels, scores)
        topk_recall = []
        for row_index in range(len(rows)):
            k = int(labels[row_index].sum())
            selected = np.argsort(-scores[row_index], kind="stable")[:k]
            topk_recall.append(float(labels[row_index, selected].sum() / k))
        metrics[model_name] = {
            "macro_average_precision": macro_ap,
            "per_class_average_precision": {
                name: float(per_class[index]) for index, name in enumerate(classes)
            },
            "mean_recall_at_true_tag_count": float(np.mean(topk_recall)),
        }
        identities[model_name] = {**spec, "prompt_token_sha256": token_hash, "preprocess": preprocess_repr}
        scores_by_model[model_name] = scores.astype(np.float32)
        features_by_model[model_name] = image_features.astype(np.float32)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bootstrap = paired_bootstrap(
        labels,
        scores_by_model["remoteclip"],
        scores_by_model["openai_clip"],
        replicates=int(protocol["primary_endpoint"]["replicates"]),
        seed=int(protocol["seed"]),
    )
    inventory_after, _ = _source_inventory(image_dir, rows)
    if inventory_after != inventory_before:
        raise InputValidationError("VOC image inventory changed during the probe.")
    summary = {
        "status": "completed",
        "scientific_evidence": True,
        "scope": "whole-image paired encoder sanity check on VOC train image tags; not segmentation",
        "repository_anchor": anchor,
        "record_count": len(rows),
        "class_count": len(classes),
        "device": device,
        "source_inventory_sha256": inventory_before,
        "models": identities,
        "metrics": metrics,
        "primary_endpoint": {**protocol["primary_endpoint"], "result": bootstrap},
        "constraints": protocol["constraints"],
    }
    arrays = {
        "labels": labels.astype(np.uint8),
        "remoteclip_scores": scores_by_model["remoteclip"],
        "openai_clip_scores": scores_by_model["openai_clip"],
        "remoteclip_image_features": features_by_model["remoteclip"],
        "openai_clip_image_features": features_by_model["openai_clip"],
    }
    return summary, arrays
