"""Entry point for the frozen, non-overwriting Vaihingen SAM3 cache generator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ov_probe.vaihingen_sam3_candidates import _preflight, _run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen Vaihingen SAM3 candidate caches without GT access.")
    parser.add_argument("--config", type=Path, default=Path("configs/vaihingen_sam3_candidate_v0.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate only and print the GT-free manifest; no SAM3 is imported and no output is written.")
    args = parser.parse_args()
    try:
        preflight = _preflight(args.config, require_checkpoint_hash=True)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": f"configuration error: {type(exc).__name__}: {exc}", "outputs_created": False}, sort_keys=True))
        return 3
    if args.dry_run or preflight.status != "ready":
        print(json.dumps({**preflight.manifest, "outputs_created": False}, sort_keys=True))
        return 0 if preflight.status == "ready" else 3
    try:
        run_dir = _run(preflight)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "outputs_created": True}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "outputs_created": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
