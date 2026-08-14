"""Create one protected, image-only PASCAL VOC 2012 external-data manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.voc2012_external_data import (  # noqa: E402
    create_voc2012_external_data_manifest,
    load_voc2012_external_data_config,
    verify_voc2012_external_data_anchor,
)


def _validate_anchor_values(commit: str, protocol_sha256: str) -> None:
    if not _SHA1_RE.fullmatch(commit):
        raise InputValidationError("--expected-code-commit must be a 40-character lowercase SHA-1.")
    if not _SHA256_RE.fullmatch(protocol_sha256):
        raise InputValidationError("--expected-protocol-sha256 must be a 64-character lowercase SHA-256.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local VOC2012 val JPEG inputs without reading labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    _validate_anchor_values(args.expected_code_commit, args.expected_protocol_sha256)
    anchor = verify_voc2012_external_data_anchor(
        PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256
    )
    cfg, protocol = load_voc2012_external_data_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise InputValidationError("Loaded VOC protocol differs from the approved repository anchor.")
    # Allocate a unique protected run only after the repository and protocol anchors pass.
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    cfg["repository_anchor"] = anchor
    manifest = create_voc2012_external_data_manifest(cfg, protocol, run_dir, PROJECT_ROOT)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
