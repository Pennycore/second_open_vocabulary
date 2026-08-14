from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import create_run_dir, environment_text, write_json, write_yaml  # noqa: E402
from ov_probe.voc_encoder_probe import (  # noqa: E402
    load_voc_probe_config,
    run_voc_encoder_probe,
    verify_voc_probe_anchor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen mask-free VOC encoder sanity check.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    anchor = verify_voc_probe_anchor(PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256)
    cfg, protocol = load_voc_probe_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise RuntimeError("Loaded VOC probe protocol differs from approved hash.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    try:
        summary, arrays = run_voc_encoder_probe(cfg, protocol, anchor)
        write_yaml(run_dir / "config_resolved.yaml", cfg)
        (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
        np.savez_compressed(run_dir / "encoder_outputs.npz", **arrays)
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"run_dir": str(run_dir), "primary": summary["primary_endpoint"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "scientific_evidence": False, "error_type": type(exc).__name__, "error": str(exc)}
        if not (run_dir / "summary.json").exists():
            write_json(run_dir / "summary.json", failure)
        print(json.dumps({"run_dir": str(run_dir), **failure}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
