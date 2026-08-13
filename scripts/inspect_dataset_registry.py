from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.datasets import load_dataset_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset-independent benchmark registry")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "datasets"))
    args = parser.parse_args()
    registry = load_dataset_registry(args.registry)
    output = {
        name: {
            "domain": spec.domain,
            "task": spec.task,
            "class_count": len(spec.classes),
            "dataset_ids_are_contiguous": spec.dataset_ids == tuple(range(1, len(spec.classes) + 1)),
            "weak_train": spec.splits["weak_train"],
            "final_evaluation": spec.splits["final_evaluation"],
            "weak_label_origin": spec.weak_label_origin,
        }
        for name, spec in registry.items()
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

