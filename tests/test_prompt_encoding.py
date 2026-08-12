from pathlib import Path

import numpy as np

from ov_probe.io import load_config
from ov_probe.prompts import HashTextEncoder, build_prompt_bank, encode_prompt_group


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_groups_are_complete_and_separate():
    cfg = load_config(ROOT / "configs" / "ov_probe_v0.yaml", ROOT)
    bank = build_prompt_bank(cfg)
    assert len(bank["group_a_templates"]) == 8
    assert len(bank["groups"]["A"]["building"]) == 8
    assert len(bank["groups"]["B"]["building"]) == 12
    assert bank["groups"]["A"]["building"] != bank["groups"]["B"]["building"]
    assert len(bank["distractors"]) == 15


def test_hash_encoder_shapes_and_normalization():
    cfg = load_config(ROOT / "configs" / "ov_probe_v0.yaml", ROOT)
    bank = build_prompt_bank(cfg)
    encoder = HashTextEncoder(512, 42)
    names, vectors, per_prompt = encode_prompt_group(encoder, bank, "A", "expanded")
    assert len(names) == 21
    assert vectors.shape == (21, 512)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert per_prompt["water"].shape == (8, 512)
