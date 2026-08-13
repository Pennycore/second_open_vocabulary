from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ov_probe.voc_sbd import prepare_voc_sbd


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely prepare and audit VOC2012 + SBD.")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    protocol = project_root / "configs" / "voc_sbd_preparation_protocol_v0.json"
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = prepare_voc_sbd(
        args.dataset_root,
        protocol_path=protocol,
        code_commit=code_commit,
    )
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
