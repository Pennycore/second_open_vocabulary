from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import create_run_dir, write_json  # noqa: E402
from ov_probe.openai_clip_region_probe import (  # noqa: E402
    load_openai_clip_region_config,
    run_openai_clip_region_probe,
    verify_openai_clip_region_anchor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen OpenAI-CLIP-only LoveDA region diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    anchor = verify_openai_clip_region_anchor(PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256)
    cfg, protocol = load_openai_clip_region_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise RuntimeError("Loaded protocol differs from approved repository anchor.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    try:
        summary, arrays = run_openai_clip_region_probe(cfg, protocol, anchor)
        np.savez_compressed(run_dir / "openai_clip_region_arrays.npz", **arrays)
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))
        return 0
    except Exception as exc:
        write_json(run_dir / "summary.json", {"status": "failed", "scientific_evidence": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
