from ov_probe.compare_analysis import classify_pair


def test_classify_pair_covers_registered_categories() -> None:
    assert classify_pair("road", "road", "road") == "both_correct"
    assert classify_pair("road", "road", "water") == "remoteclip_only_correct"
    assert classify_pair("road", "water", "road") == "openai_clip_only_correct"
    assert classify_pair("road", "water", "water") == "both_wrong_same"
    assert classify_pair("road", "water", "forest") == "both_wrong_different"

