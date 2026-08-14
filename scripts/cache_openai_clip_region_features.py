"""Create a uniquely named frozen OpenAI-CLIP feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "openai_clip_feature_cache_protocol_v1.json"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.openai_clip_feature_cache import (  # noqa: E402
    create_openai_clip_feature_cache,
    load_openai_clip_feature_cache_config,
    verify_openai_clip_feature_cache_anchor,
)


def _validate_anchor_values(expected_code_commit: str, expected_protocol_sha256: str) -> None:
    if not _SHA1_RE.fullmatch(expected_code_commit):
        raise InputValidationError("--expected-code-commit must be a 40-character lowercase SHA-1.")
    if not _SHA256_RE.fullmatch(expected_protocol_sha256):
        raise InputValidationError("--expected-protocol-sha256 must be a 64-character lowercase SHA-256.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a frozen OpenAI-CLIP region feature cache without evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    _validate_anchor_values(args.expected_code_commit, args.expected_protocol_sha256)
    anchor = verify_openai_clip_feature_cache_anchor(
        PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256
    )
    cfg, protocol = load_openai_clip_feature_cache_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise InputValidationError("Loaded feature-cache protocol differs from the approved repository anchor.")
    # The run directory is allocated only after the repository and canonical protocol are frozen.
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    cfg["repository_anchor"] = anchor
    manifest = create_openai_clip_feature_cache(cfg, protocol, run_dir)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
