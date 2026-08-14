"""Create one protected, label-free VOC2012 OpenAI-CLIP score cache."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.voc2012_openai_clip_zeroshot import (  # noqa: E402
    load_voc2012_openai_clip_zeroshot_config,
    run_voc2012_openai_clip_zeroshot,
    verify_voc2012_openai_clip_zeroshot_anchor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen label-free VOC2012 OpenAI-CLIP score cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_code_commit) or not re.fullmatch(r"[0-9a-f]{64}", args.expected_protocol_sha256):
        raise InputValidationError("Expected commit/protocol anchors must be lowercase SHA-1/SHA-256 values.")
    anchor = verify_voc2012_openai_clip_zeroshot_anchor(PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256)
    cfg, protocol = load_voc2012_openai_clip_zeroshot_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise InputValidationError("Loaded VOC zero-shot protocol differs from the approved repository anchor.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    cfg["repository_anchor"] = anchor
    manifest = run_voc2012_openai_clip_zeroshot(cfg, protocol, run_dir)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
