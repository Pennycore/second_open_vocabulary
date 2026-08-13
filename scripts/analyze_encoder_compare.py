from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.compare_analysis import (  # noqa: E402
    load_analysis_config,
    render_analysis_report,
    run_descriptive_analysis,
    verify_analysis_anchor,
)
from ov_probe.io import create_run_dir, environment_text, write_json, write_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create post-result descriptive encoder analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    anchor = verify_analysis_anchor(PROJECT_ROOT, args.expected_code_commit, args.expected_protocol_sha256)
    cfg, protocol = load_analysis_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != anchor["protocol_sha256"]:
        raise RuntimeError("Loaded analysis protocol differs from approved anchor.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    try:
        result = run_descriptive_analysis(cfg, protocol, run_dir)
        comparison_summary = json.loads(
            (Path(cfg["paths"]["comparison_run"]) / "summary.json").read_text(encoding="utf-8")
        )
        (run_dir / "report.md").write_text(
            render_analysis_report(result, comparison_summary), encoding="utf-8"
        )
        write_yaml(run_dir / "config_resolved.yaml", cfg)
        (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
        write_json(run_dir / "run_summary.json", {"status": "completed", "scientific_evidence": False, "repository_anchor": anchor})
        print(json.dumps({"run_dir": str(run_dir), "status": "completed"}))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "scientific_evidence": False, "error_type": type(exc).__name__, "error": str(exc)}
        if not (run_dir / "run_summary.json").exists():
            write_json(run_dir / "run_summary.json", failure)
        print(json.dumps({"run_dir": str(run_dir), **failure}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

