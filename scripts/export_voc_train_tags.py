from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ov_probe.datasets import load_dataset_spec
from ov_probe.voc_sbd import export_voc_segmentation_train_tags


def main() -> int:
    parser = argparse.ArgumentParser(description="Export mask-free VOC segmentation-train image tags.")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    spec = load_dataset_spec(project_root / "configs" / "datasets" / "voc2012.yaml")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = export_voc_segmentation_train_tags(
        args.dataset_root,
        spec.class_names,
        code_commit=commit,
        protocol_path=project_root / "configs" / "voc_train_tag_protocol_v0.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
