"""Contract tests for the frozen Vaihingen SAM3 candidate-cache generator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.vaihingen_sam3_candidates import (
    CHECKPOINT_PLACEHOLDER,
    DTYPE_GUARD,
    _build_backend,
    _load_config,
    _validate_candidate_cache_schema,
    _write_json_exclusive,
    enumerate_tiles,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tile_enumeration_uses_frozen_shift_at_edge_rule():
    tiles = enumerate_tiles(width=1000, height=600, tile_size=512, overlap=128)
    assert [(tile.x0, tile.y0, tile.x1, tile.y1) for tile in tiles] == [
        (0, 0, 512, 512), (384, 0, 896, 512), (488, 0, 1000, 512),
        (0, 88, 512, 600), (384, 88, 896, 600), (488, 88, 1000, 600),
    ]


def test_runtime_reuses_first_paper_fp32_input_hook_after_backend_construction(tmp_path: Path):
    events: list[tuple[str, object]] = []

    class FakeBackend:
        def __init__(self, repo, checkpoint, *, device, confidence_threshold):
            events.append(("backend", (repo, checkpoint, device, confidence_threshold)))
            self.model = object()

    def fake_hook(model):
        events.append(("hook", model))

    paths = {"sam3_repo": tmp_path / "sam3", "sam3_checkpoint": tmp_path / "sam3.pt"}
    proposal = {"device": "cuda", "score_threshold": 0.55}
    backend = _build_backend(
        {"SAM3ImageBackend": FakeBackend, "install_fp32_dtype_hooks": fake_hook}, paths, proposal,
    )
    assert events == [
        ("backend", (paths["sam3_repo"], paths["sam3_checkpoint"], "cuda", 0.55)),
        ("hook", backend.model),
    ]
    assert DTYPE_GUARD == "first_paper_fp32_input_hooks"


def test_prompt_and_class_schema_remain_first_paper_vaihingen_contract():
    protocol = json.loads((ROOT / "configs" / "vaihingen_sam3_candidate_protocol_v0.json").read_text(encoding="utf-8"))
    assert protocol["proposal"]["tile_size"] == 512
    assert protocol["proposal"]["tile_overlap"] == 128
    assert protocol["proposal"]["score_threshold"] == 0.55
    assert protocol["prompting"] == {"style": "remoteclip_b2c", "include_manual_prompts": True, "max_prompts_per_class": 4}
    assert [item["name"] for item in protocol["classes"]] == ["impervious_surface", "building", "low_vegetation", "tree", "car"]
    assert all(len(item["prompts"]) == 4 for item in protocol["classes"])
    assert protocol["proposal"]["checkpoint_sha256"] == CHECKPOINT_PLACEHOLDER


def test_candidate_cache_schema_validator_accepts_first_paper_v1_pair(tmp_path: Path):
    image_id, shape = "vaih_area1", (4, 5)
    np.savez_compressed(
        tmp_path / f"{image_id}.npz",
        format_version=np.asarray([1], dtype=np.int16), image_shape=np.asarray(shape, dtype=np.int32),
        packed_masks=np.asarray([9], dtype=np.uint8), offsets=np.asarray([0, 1], dtype=np.int64),
        shapes=np.asarray([[2, 2]], dtype=np.int32), origins=np.asarray([[1, 1]], dtype=np.int32),
        boxes=np.asarray([[1, 1, 3, 3]], dtype=np.int32), areas=np.asarray([2], dtype=np.int64),
        scores=np.asarray([0.8], dtype=np.float32), class_ids=np.asarray([2], dtype=np.int16), prompt_ids=np.asarray([0], dtype=np.int16),
    )
    (tmp_path / f"{image_id}.json").write_text(json.dumps({
        "format_version": 1, "image_id": image_id, "image_shape": list(shape), "candidate_count": 1,
        "data_file": f"{image_id}.npz", "prompts": [{"id": 0, "class_id": 2, "class_name": "building", "prompt": "building"}],
    }), encoding="utf-8")
    result = _validate_candidate_cache_schema(tmp_path, image_id, shape)
    assert result["candidate_count"] == 1 and len(result["npz_sha256"]) == 64


def test_paths_are_portable_and_absolute_or_existing_outputs_are_rejected(tmp_path: Path):
    raw = (ROOT / "configs" / "vaihingen_sam3_candidate_v0.yaml").read_text(encoding="utf-8")
    assert "/home/" not in raw and "C:\\" not in raw
    cfg, _, _ = _load_config(ROOT / "configs" / "vaihingen_sam3_candidate_v0.yaml")
    assert cfg["experiment"]["overwrite"] is False
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    bad = config_dir / "bad.yaml"
    bad.write_text(raw.replace("inputs/vaihingen/images", "C:/absolute/not-allowed"), encoding="utf-8")
    protocol = tmp_path / "vaihingen_sam3_candidate_protocol_v0.json"
    protocol.write_text((ROOT / "configs" / "vaihingen_sam3_candidate_protocol_v0.json").read_text(encoding="utf-8"), encoding="utf-8")
    bad.write_text(bad.read_text(encoding="utf-8").replace("configs/vaihingen_sam3_candidate_protocol_v0.json", protocol.name), encoding="utf-8")
    with pytest.raises(InputValidationError, match="project-relative"):
        _load_config(bad)
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(InputValidationError, match="Refusing to overwrite"):
        _write_json_exclusive(existing, {"new": "content"})
