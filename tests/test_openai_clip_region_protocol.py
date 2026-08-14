import json
from pathlib import Path


def test_openai_clip_region_protocol_is_single_encoder_and_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "openai_clip_region_probe_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_pre_result"
    assert protocol["model"]["architecture"] == "ViT-B-32-quickgelu"
    assert protocol["model"]["feature_dimension"] == 512
    assert protocol["pixel_pack"]["record_count"] == 6000
    assert protocol["constraints"]["remoteclip_feature_or_text_reuse"] is False
    assert protocol["constraints"]["sam3_rerun"] is False
    assert protocol["constraints"]["training"] is False
