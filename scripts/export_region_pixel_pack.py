from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import create_run_dir  # noqa: E402
from ov_probe.pixel_pack import (  # noqa: E402
    export_region_pixel_pack,
    load_export_config,
    sync_pixel_pack_tree,
    validate_region_pixel_pack,
    verify_repository_anchor,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the frozen 6000-region lossless RGB/mask comparison pack"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    repository_anchor = verify_repository_anchor(
        PROJECT_ROOT,
        args.expected_code_commit,
        args.expected_protocol_sha256,
    )
    cfg, protocol = load_export_config(args.config, PROJECT_ROOT)
    if protocol["sha256"] != repository_anchor["protocol_sha256"]:
        raise RuntimeError("Loaded protocol differs from the approved repository anchor.")
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    package_dir = run_dir / "pixel_pack"
    staging_dir = run_dir / f".pixel_pack.staging-{uuid.uuid4().hex}"
    try:
        manifest = export_region_pixel_pack(
            cfg, protocol, staging_dir, repository_anchor
        )
        validation = validate_region_pixel_pack(staging_dir, protocol["path"])
        sync_pixel_pack_tree(staging_dir)
        if package_dir.exists():
            raise FileExistsError(f"Final package directory already exists: {package_dir}")
        os.replace(staging_dir, package_dir)
        summary = {
            "status": "completed",
            "scientific_evidence": False,
            "role": "frozen pixel transport package; no encoder comparison executed",
            "repository_anchor": repository_anchor,
            "manifest": manifest,
            "validation": validation,
        }
        _atomic_json(run_dir / "summary.json", summary)
        print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))
        return 0
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        failure = {
            "status": "failed",
            "scientific_evidence": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not (run_dir / "summary.json").exists():
            _atomic_json(run_dir / "summary.json", failure)
        print(json.dumps({"run_dir": str(run_dir), **failure}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
