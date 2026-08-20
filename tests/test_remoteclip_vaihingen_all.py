"""Contract tests for the 3090v2 one-shot RemoteCLIP Vaihingen entry point."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("remoteclip_vaihingen_all", ROOT / "scripts" / "run_remoteclip_vaihingen_all.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_deployment_config_is_portable_and_frozen():
    raw = (ROOT / "configs" / "remoteclip_vaihingen_all_v0.yaml").read_text(encoding="utf-8")
    cfg, _ = MODULE._load_config(ROOT / "configs" / "remoteclip_vaihingen_all_v0.yaml")
    assert "/home/" not in raw and "C:\\" not in raw
    assert cfg["experiment"]["overwrite"] is False
    assert cfg["matrix"]["full_support_methods"] == ["text_only", "C2", "CTP"]
    assert cfg["matrix"]["partial_support_counts"] == [2, 3, 4]


def test_missing_inputs_block_before_outputs_are_created(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "remoteclip_protocol.json").write_text('{"classes": [], "model": {}}', encoding="utf-8")
    (config_dir / "ctp.json").write_text('{"status": "frozen"}', encoding="utf-8")
    config = config_dir / "one.yaml"
    config.write_text(
        """experiment:\n  overwrite: false\npaths:\n  image_dir: inputs/images\n  label_dir: inputs/labels\n  candidates_dir: inputs/candidates\n  sam3_python_root: inputs/sam3/src\n  remoteclip_checkpoint: inputs/remoteclip.pt\n  remoteclip_protocol_file: configs/remoteclip_protocol.json\n  ctp_frozen_file: configs/ctp.json\n  output_root: outputs/remoteclip\nruntime:\n  required_gpu_substring: RTX 3090\n  image_batch: 1\nsplit:\n  train_areas: [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]\n  test_areas: [11, 15, 28, 30, 34]\nmatrix:\n  full_support_methods: [text_only, C2, CTP]\n  partial_support_methods: [text_only, C2, CTP]\n  partial_support_counts: [2, 3, 4]\n  partial_subset_policy: all_deterministic_bitmasks\nintegrity:\n  remoteclip_checkpoint_sha256: 0000000000000000000000000000000000000000000000000000000000000000\n  required_open_clip_version: 3.3.0\n  required_feature_dimension: 512\n""",
        encoding="utf-8",
    )
    result = MODULE._preflight(config)
    assert result.status == "blocked"
    assert any("Required directory missing" in error for error in result.errors)
    assert not (tmp_path / "outputs").exists()


def test_partial_metrics_use_per_subset_harmonic_mean():
    matrix = np.array([[8, 1, 0, 0, 0], [1, 6, 0, 0, 0], [0, 0, 5, 1, 0], [0, 0, 1, 4, 0], [0, 0, 0, 0, 3]])
    row = MODULE._metrics_for_subset(matrix, ["impervious_surface", "building"], ["low_vegetation", "tree", "car"])
    assert min(row["S_F1"], row["U_F1"]) <= row["H_F1"] <= max(row["S_F1"], row["U_F1"])
    assert min(row["S_IoU"], row["U_IoU"]) <= row["H_IoU"] <= max(row["S_IoU"], row["U_IoU"])
