from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .encoder_compare import _encode_model, _load_records, _metrics, _text_prototypes
from .io import InputValidationError, sha256_file
from .pixel_pack import validate_region_pixel_pack


_PROTOCOL_NAME = "openai_clip_region_probe_protocol_v1.json"


def load_openai_clip_region_config(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("OpenAI CLIP region config must set overwrite=false.")
    for key, value in cfg["paths"].items():
        candidate = Path(str(value))
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise InputValidationError(f"OpenAI CLIP region path escapes project root: {key}") from exc
        cfg["paths"][key] = str(resolved)
    if Path(cfg["paths"]["output_root"]).parent != (root / "outputs").resolve():
        raise InputValidationError("OpenAI CLIP region output must be directly under outputs/.")
    protocol_path = Path(cfg["paths"]["protocol_file"])
    if protocol_path.resolve() != (root / "configs" / _PROTOCOL_NAME).resolve():
        raise InputValidationError("OpenAI CLIP region protocol must be the committed canonical file.")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol["path"] = str(protocol_path)
    protocol["sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    return cfg, protocol


def verify_openai_clip_region_anchor(project_root: str | Path, expected_commit: str, expected_protocol_sha256: str) -> dict[str, str]:
    root = Path(project_root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    actual = sha256_file(root / "configs" / _PROTOCOL_NAME)
    if commit != expected_commit or dirty or actual != expected_protocol_sha256:
        raise InputValidationError("OpenAI CLIP region run requires the approved clean commit and protocol SHA-256.")
    return {"code_commit": commit, "protocol_sha256": actual}


def run_openai_clip_region_probe(cfg: dict[str, Any], protocol: dict[str, Any], anchor: dict[str, str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import open_clip
    import torch

    if open_clip.__version__ != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the registered protocol.")
    package = Path(cfg["paths"]["pixel_pack"])
    validation = validate_region_pixel_pack(package, package / "encoder_compare_protocol_v0.json")
    for key, expected in protocol["pixel_pack"].items():
        if validation[key] != expected:
            raise InputValidationError(f"Pixel package differs from OpenAI CLIP registration: {key}")
    checkpoint = Path(cfg["paths"]["openai_clip_checkpoint"])
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("OpenAI CLIP checkpoint differs from the registered artifact.")
    requested = str(cfg["runtime"].get("device", "auto"))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    features, model, tokenizer, preprocess = _encode_model(
        package, checkpoint, protocol["model"]["architecture"], int(cfg["runtime"]["batch_regions"]), device
    )
    records = _load_records(package)
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    token_hashes: dict[str, str] = {}
    for evaluation in protocol["evaluations"]:
        group, vocabulary = evaluation["prompt_group"], evaluation["vocabulary"]
        key = f"group_{group}_{vocabulary}"
        names, text_features, token_hash = _text_prototypes(model, tokenizer, protocol, group, vocabulary, device)
        scores = features @ text_features.T
        indices = np.argmax(scores, axis=1)
        metrics[key] = _metrics(indices, names, records, protocol["classes"])
        predictions[key] = np.asarray(names)[indices]
        token_hashes[key] = token_hash
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    primary_key = "group_A_closed"
    summary = {
        "status": "completed",
        "scientific_evidence": True,
        "scope": "OpenAI-CLIP-only weak-label region-text diagnostic; not true accuracy or segmentation",
        "repository_anchor": anchor,
        "pixel_pack_validation": validation,
        "model": protocol["model"],
        "device": device,
        "preprocess": preprocess,
        "prompt_token_sha256": token_hashes,
        "metrics": metrics,
        "primary_endpoint": {**protocol["primary_endpoint"], "result": metrics[primary_key]},
        "constraints": protocol["constraints"],
    }
    arrays = {"features_openai_clip": features.astype(np.float16)}
    arrays.update({f"predictions_{key}": value for key, value in predictions.items()})
    return summary, arrays
