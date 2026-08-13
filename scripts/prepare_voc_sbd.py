from __future__ import annotations

import argparse
import json
from pathlib import Path

from ov_probe.voc_sbd import prepare_voc_sbd


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely prepare and audit VOC2012 + SBD.")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    manifest = prepare_voc_sbd(args.dataset_root)
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
