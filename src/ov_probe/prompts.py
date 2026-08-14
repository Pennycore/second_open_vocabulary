from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .metrics import l2_normalize
from .io import InputValidationError, sha256_file


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def build_prompt_bank(cfg: dict[str, Any]) -> dict[str, Any]:
    class_names = [str(item["name"]) for item in cfg["data"]["classes"]]
    distractors = [str(item) for item in cfg["prompts"]["distractors"]]
    aliases = cfg["prompts"]["aliases"]
    group_a_templates = [str(item) for item in cfg["prompts"]["group_a_templates"]]
    group_b_templates = [str(item) for item in cfg["prompts"]["group_b_templates"]]
    bank: dict[str, Any] = {
        "class_names": class_names,
        "distractors": distractors,
        "group_a_templates": group_a_templates,
        "group_b_templates": group_b_templates,
        "aliases": aliases,
        "groups": {"A": {}, "B": {}},
    }
    for name in class_names + distractors:
        bank["groups"]["A"][name] = [template.format(**{"class": name}) for template in group_a_templates]
        name_aliases = aliases.get(name, [name])
        bank["groups"]["B"][name] = [
            template.format(alias=alias)
            for alias in name_aliases
            for template in group_b_templates
        ]
    return bank


def encode_prompt_group(encoder: TextEncoder, bank: dict[str, Any], group: str, vocabulary: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    names = list(bank["class_names"])
    if vocabulary == "expanded":
        names += list(bank["distractors"])
    prompts_by_class = bank["groups"][group]
    flat = [text for name in names for text in prompts_by_class[name]]
    encoded = l2_normalize(encoder.encode(flat))
    class_vectors = []
    per_prompt: dict[str, np.ndarray] = {}
    offset = 0
    for name in names:
        count = len(prompts_by_class[name])
        values = encoded[offset : offset + count]
        per_prompt[name] = values
        class_vectors.append(l2_normalize(values.mean(axis=0, keepdims=True))[0])
        offset += count
    return names, np.asarray(class_vectors, dtype=np.float32), per_prompt


class HashTextEncoder:
    """Deterministic synthetic encoder used only by --dry-run and tests."""

    def __init__(self, feature_dim: int = 512, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.seed = seed

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(f"{self.seed}:{text}".encode("utf-8")).digest()
            local_seed = int.from_bytes(digest[:8], "little")
            rows.append(np.random.default_rng(local_seed).normal(size=self.feature_dim))
        return np.asarray(rows, dtype=np.float32)


class CachedTextEncoder:
    """Read a fixed prompt-feature cache produced by the configured checkpoint."""

    def __init__(self, cache_path: str | Path, cfg: dict[str, Any]) -> None:
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(f"Text feature cache not found: {path}")
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise InputValidationError(f"Text feature cache metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as archive:
            prompts = [str(value) for value in archive["prompts"].reshape(-1)]
            features = np.asarray(archive["features"], dtype=np.float32)
        expected_dim = int(cfg["model"]["feature_dim"])
        if features.shape != (len(prompts), expected_dim) or not np.isfinite(features).all():
            raise InputValidationError("Text feature cache has invalid shape or non-finite values.")
        if str(metadata.get("model_name")) != str(cfg["model"]["model_name"]):
            raise InputValidationError("Text cache model name differs from the configured model.")
        checkpoint = Path(cfg["paths"]["remoteclip_checkpoint"])
        if str(metadata.get("checkpoint_sha256", "")).lower() != sha256_file(checkpoint).lower():
            raise InputValidationError("Text cache checkpoint hash differs from the configured checkpoint.")
        normalized = l2_normalize(features)
        self._features: dict[str, np.ndarray] = {}
        for prompt, feature in zip(prompts, normalized):
            if prompt in self._features and not np.allclose(self._features[prompt], feature, atol=1e-6):
                raise InputValidationError(f"Text cache has inconsistent duplicate prompt: {prompt!r}")
            self._features[prompt] = feature

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in texts if text not in self._features]
        if missing:
            raise InputValidationError(f"Text feature cache lacks {len(missing)} prompts; first={missing[0]!r}")
        return np.stack([self._features[text] for text in texts]).astype(np.float32)


class RemoteCLIPTextEncoder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        source_root = cfg["paths"].get("remoteclip_source_root")
        if source_root and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            import open_clip
            import torch
        except Exception as exc:
            raise RuntimeError("open_clip is not installed in this Python environment; no download was attempted.") from exc
        checkpoint = Path(cfg["paths"]["remoteclip_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {checkpoint}")
        requested = str(cfg["model"].get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.torch = torch
        self.model = open_clip.create_model(str(cfg["model"]["model_name"]), pretrained=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint/model mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(str(cfg["model"]["model_name"]))
        self.batch_size = int(cfg["model"].get("text_batch_size", 128))

    def encode(self, texts: list[str]) -> np.ndarray:
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[start : start + self.batch_size]).to(self.device)
                outputs.append(self.model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


class OpenAIClipTextEncoder:
    """Frozen OpenAI CLIP text tower packaged as an OpenCLIP state dictionary.

    This is the forward architecture for the second paper.  It deliberately
    requires the quick-GELU ViT-B/32 variant: loading this checkpoint into the
    default GELU variant can succeed structurally while changing the model.
    """

    architecture = "ViT-B-32-quickgelu"
    checkpoint_sha256 = "9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e"

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import open_clip
            import torch
        except Exception as exc:
            raise RuntimeError("open_clip is not installed in this Python environment; no download was attempted.") from exc
        model_cfg = cfg.get("model", {})
        configured_architecture = str(model_cfg.get("architecture", self.architecture))
        if configured_architecture != self.architecture:
            raise InputValidationError(
                f"OpenAI CLIP requires {self.architecture}, not {configured_architecture}."
            )
        checkpoint_value = cfg.get("paths", {}).get("openai_clip_checkpoint")
        if not checkpoint_value:
            raise InputValidationError("OpenAI CLIP requires paths.openai_clip_checkpoint.")
        checkpoint = Path(str(checkpoint_value))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"OpenAI CLIP checkpoint not found: {checkpoint}")
        if sha256_file(checkpoint).lower() != self.checkpoint_sha256:
            raise InputValidationError("OpenAI CLIP checkpoint hash differs from the registered artifact.")
        expected_dim = int(model_cfg.get("feature_dim", 512))
        if expected_dim != 512:
            raise InputValidationError("OpenAI CLIP ViT-B/32 quick-GELU has a 512-dimensional embedding space.")
        requested = str(model_cfg.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.torch = torch
        self.model = open_clip.create_model(self.architecture, pretrained=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.architecture)
        self.batch_size = int(model_cfg.get("text_batch_size", 128))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[start : start + self.batch_size]).to(self.device)
                outputs.append(self.model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)
