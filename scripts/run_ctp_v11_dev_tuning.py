"""CTP-v1.1 development-only deployment gate.

The runner has three fail-closed phases: `preflight` reads metadata only;
`cache` creates a new OpenAI-CLIP feature cache with `label_dir=null`; `run`
seals GT-free region predictions before it opens development GT for the fixed
49x25 evaluation. It never falls back to a historical test runner.

Usage:
    python scripts/run_ctp_v11_dev_tuning.py --config deployment.yaml --phase preflight|cache|run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.ctp_v11_tuning import (  # noqa: E402
    build_development_feature_cache,
    load_deployment_config,
    preflight_status,
    run_development_grid,
)
from ov_probe.io import InputValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed CTP-v1.1 development-only preflight.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=["preflight", "cache", "run"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_deployment_config(args.config, project_root)
    if args.phase == "preflight":
        print(json.dumps(preflight_status(cfg, protocol), indent=2, sort_keys=True))
        return 0
    paths = cfg["paths"]
    if args.phase == "cache":
        if paths["label_dir"] is not None:
            raise InputValidationError("Cache phase must set label_dir=null; GT is forbidden.")
        if paths["feature_cache"] is not None or paths["development_manifest"] is not None:
            raise InputValidationError("Cache phase requires a fresh deployment config with no cache/manifest input.")
        run_dir = build_development_feature_cache(
            paths["image_dir"], paths["candidate_dir"], paths["openai_clip_checkpoint"], protocol,
            paths["output_root"], batch_size=args.batch_size,
        )
        print(json.dumps({"phase": "cache", "run_dir": str(run_dir), "gt_read": False}, indent=2))
        return 0
    if paths["label_dir"] is None:
        raise InputValidationError("Run phase requires a development-only label_dir after prediction sealing.")
    if paths["feature_cache"] is None or paths["development_manifest"] is None:
        raise InputValidationError("Run phase requires a completed cache and its cache_manifest.json.")
    run_dir = run_development_grid(
        paths["feature_cache"], paths["development_manifest"], paths["candidate_dir"], paths["label_dir"],
        protocol, paths["output_root"],
    )
    print(json.dumps({"phase": "run", "run_dir": str(run_dir), "registered_test_evaluation": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputValidationError as exc:
        print(f"CTP-v1.1 preflight rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
