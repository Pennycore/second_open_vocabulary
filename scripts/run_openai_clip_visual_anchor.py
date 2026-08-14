"""Run the frozen post-hoc OpenAI-CLIP visual-anchor diagnostic exactly once."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.openai_clip_visual_anchor import (  # noqa: E402
    load_openai_clip_visual_anchor_config,
    run_openai_clip_visual_anchor,
    verify_openai_clip_visual_anchor_anchor,
)


_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen post-hoc OpenAI-CLIP visual-anchor diagnostic.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    if not _SHA1.fullmatch(args.expected_code_commit) or not _SHA256.fullmatch(args.expected_protocol_sha256):
        raise InputValidationError("Expected commit/protocol anchors must be lowercase SHA-1/SHA-256 values.")
    # Anchor verification completes before a protected unique output directory is allocated.
    anchor = verify_openai_clip_visual_anchor_anchor(
        PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256
    )
    cfg, protocol = load_openai_clip_visual_anchor_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise InputValidationError("Loaded visual-anchor protocol differs from the approved repository anchor.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    cfg["repository_anchor"] = anchor
    manifest = run_openai_clip_visual_anchor(cfg, protocol, run_dir)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
