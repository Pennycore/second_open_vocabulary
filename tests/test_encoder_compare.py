from __future__ import annotations

import numpy as np

from ov_probe.encoder_compare import clustered_macro_bootstrap


def test_clustered_bootstrap_is_deterministic_and_paired() -> None:
    records = [
        {"image_id": "loveda_train_rural_0", "sam3_source_label": "building"},
        {"image_id": "loveda_train_rural_0", "sam3_source_label": "road"},
        {"image_id": "loveda_train_urban_1", "sam3_source_label": "building"},
        {"image_id": "loveda_train_urban_1", "sam3_source_label": "road"},
    ]
    remote = np.asarray(["road", "road", "road", "road"])
    clip = np.asarray(["building", "road", "building", "road"])
    first = clustered_macro_bootstrap(remote, clip, records, ["building", "road"], 100, 42)
    second = clustered_macro_bootstrap(remote, clip, records, ["building", "road"], 100, 42)
    assert first == second
    assert first["decision"] == "openai_clip_better"
    assert first["ci95"] == [0.5, 0.5]

