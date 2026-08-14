from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.openai_clip_holdout_split import create_holdout_split  # noqa: E402


def _load_config(path: str | Path) -> tuple[dict, dict]:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Holdout split config must set experiment.overwrite=false.")
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise InputValidationError("Holdout split config requires paths.")
    for key in ("pixel_pack", "protocol_file", "output_root"):
        if key not in paths:
            raise InputValidationError(f"Holdout split config requires paths.{key}.")
        candidate = Path(str(paths[key]))
        resolved = (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise InputValidationError(f"Holdout split path escapes project root: {key}") from exc
        paths[key] = str(resolved)
    if Path(paths["output_root"]).parent != (PROJECT_ROOT / "outputs").resolve():
        raise InputValidationError("Holdout split output_root must be directly under project outputs/.")
    expected_protocol = (PROJECT_ROOT / "configs" / "openai_clip_holdout_split_protocol_v1.json").resolve()
    if Path(paths["protocol_file"]) != expected_protocol:
        raise InputValidationError("Holdout split protocol_file must be the committed canonical protocol.")
    protocol = json.loads(expected_protocol.read_text(encoding="utf-8"))
    return cfg, protocol


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the frozen image-disjoint OpenAI-CLIP holdout split.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg, protocol = _load_config(args.config)
    run_dir = create_run_dir(cfg["paths"]["output_root"])
    try:
        manifest = create_holdout_split(cfg["paths"]["pixel_pack"], protocol, run_dir)
    except Exception:
        # A failed protected run directory is intentionally retained and never reused.
        raise
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
