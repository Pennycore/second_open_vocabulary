from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "openai_clip_holdout_split_protocol_v1.json"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import InputValidationError, create_run_dir  # noqa: E402
from ov_probe.openai_clip_holdout_split import create_holdout_split  # noqa: E402


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_anchor_values(expected_code_commit: str, expected_protocol_sha256: str) -> None:
    if not _SHA1_RE.fullmatch(expected_code_commit):
        raise InputValidationError("--expected-code-commit must be a 40-character lowercase SHA-1.")
    if not _SHA256_RE.fullmatch(expected_protocol_sha256):
        raise InputValidationError("--expected-protocol-sha256 must be a 64-character lowercase SHA-256.")


def _run_git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as exc:
        raise InputValidationError("Repository anchor verification could not run git.") from exc
    if completed.returncode != 0:
        raise InputValidationError("Repository anchor verification could not read git state.")
    return completed.stdout


def _verify_repository_anchor(expected_code_commit: str, expected_protocol_sha256: str) -> None:
    """Fail closed unless code, tracked worktree, and canonical protocol are frozen as declared."""
    _validate_anchor_values(expected_code_commit, expected_protocol_sha256)
    if _run_git("rev-parse", "HEAD").strip() != expected_code_commit:
        raise InputValidationError("Repository HEAD does not match --expected-code-commit.")
    if _run_git("status", "--porcelain", "--untracked-files=no") != "":
        raise InputValidationError("Repository has tracked modifications; holdout creation is refused.")
    try:
        actual_protocol_sha256 = _sha256_file(CANONICAL_PROTOCOL_PATH)
    except OSError as exc:
        raise InputValidationError("Canonical holdout protocol is unavailable for anchor verification.") from exc
    if actual_protocol_sha256 != expected_protocol_sha256:
        raise InputValidationError("Canonical protocol hash does not match --expected-protocol-sha256.")


def _load_config(path: str | Path, expected_protocol_sha256: str) -> tuple[dict, dict]:
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
    expected_protocol = CANONICAL_PROTOCOL_PATH.resolve()
    if Path(paths["protocol_file"]) != expected_protocol:
        raise InputValidationError("Holdout split protocol_file must be the committed canonical protocol.")
    if _sha256_file(expected_protocol) != expected_protocol_sha256:
        raise InputValidationError("Loaded canonical protocol hash does not match --expected-protocol-sha256.")
    protocol = json.loads(expected_protocol.read_text(encoding="utf-8"))
    return cfg, protocol


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the frozen image-disjoint OpenAI-CLIP holdout split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    # All anchors are checked before reading a config or allocating a protected run directory.
    _verify_repository_anchor(args.expected_code_commit, args.expected_protocol_sha256)
    cfg, protocol = _load_config(args.config, args.expected_protocol_sha256)
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
