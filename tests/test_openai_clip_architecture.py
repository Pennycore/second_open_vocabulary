import json
from pathlib import Path

from ov_probe.prompts import OpenAIClipTextEncoder


def test_forward_architecture_is_openai_clip_and_rejects_cross_encoder_reuse() -> None:
    root = Path(__file__).resolve().parents[1]
    architecture = json.loads((root / "configs" / "architecture_v1.json").read_text(encoding="utf-8"))
    assert architecture["primary_encoder"]["family"] == "OpenAI CLIP"
    assert architecture["primary_encoder"]["architecture"] == OpenAIClipTextEncoder.architecture
    assert architecture["primary_encoder"]["checkpoint_sha256"] == OpenAIClipTextEncoder.checkpoint_sha256
    assert architecture["legacy_remoteclip"]["cross_encoder_feature_reuse_allowed"] is False


def test_stage1_v1_uses_openai_clip_only_feature_space() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "stage1_ov_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["encoder_decision"]["primary"] == "OpenAI CLIP ViT-B/32 quick-GELU"
    assert protocol["encoder_decision"]["frozen_checkpoint_sha256"] == OpenAIClipTextEncoder.checkpoint_sha256
    assert protocol["feature_space_rule"]["remoteclip_feature_or_prototype_reuse"] is False
    assert protocol["execution_gate"]["current_state"] == "blocked_pending_openai_clip_pixel_package"
