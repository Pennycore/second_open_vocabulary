from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.pixel_pack import validate_region_pixel_pack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen region pixel pack")
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    protocol = PROJECT_ROOT / "configs" / "encoder_compare_protocol_v0.json"
    actual = __import__("hashlib").sha256(protocol.read_bytes()).hexdigest()
    if actual != args.expected_protocol_sha256:
        raise RuntimeError("Canonical protocol differs from the externally approved SHA-256.")
    print(json.dumps(validate_region_pixel_pack(args.package, protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
