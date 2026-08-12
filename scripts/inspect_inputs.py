from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import inspect_configured_inputs, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only OV probe input audit")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, PROJECT_ROOT)
    result = inspect_configured_inputs(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
