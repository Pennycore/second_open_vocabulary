"""Runner for the frozen blind LoveDA Train GT evaluation.

Usage (project root):

    python scripts/run_loveda_blind_gt.py --config <deployment.yaml> --phase predict
    python scripts/run_loveda_blind_gt.py --config <deployment.yaml> --phase evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError, environment_text, seed_everything  # noqa: E402
from ov_probe.loveda_blind_gt import (  # noqa: E402
    load_loveda_blind_gt_config,
    run_loveda_blind_gt_evaluate,
    run_loveda_blind_gt_predict,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind LoveDA GT evaluation for frozen OpenAI-CLIP region recognition.")
    parser.add_argument("--config", required=True, help="Deployment YAML config path.")
    parser.add_argument("--phase", required=True, choices=["predict", "evaluate"], help="predict never opens GT; evaluate opens GT only after predictions are verified.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for any random operation.")
    parser.add_argument("--support-fraction", type=float, default=1.0, help="Per-class support subset fraction for prototype stability probes (predict phase only).")
    parser.add_argument("--support-seed", type=int, default=42, help="Seed for the support subsample (predict phase only).")
    parser.add_argument("--output-subdir", default=None, help="Optional subdirectory under paths.output_root for this run (stability probes).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    project_root = Path(__file__).resolve().parent.parent
    try:
        cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
        if args.output_subdir:
            import re as _re
            if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_subdir):
                raise InputValidationError("output-subdir contains invalid characters.")
            cfg["paths"]["output_root"] = str(Path(cfg["paths"]["output_root"]) / args.output_subdir)
        if args.phase == "predict":
            manifest = run_loveda_blind_gt_predict(
                cfg, protocol, support_fraction=args.support_fraction, support_seed=args.support_seed
            )
        else:
            output_root = cfg["paths"]["output_root"]
            manifest = run_loveda_blind_gt_evaluate(cfg, protocol, output_root)
        logging.info("phase=%s completed; outputs under %s", args.phase, cfg["paths"]["output_root"])
        print("ENVIRONMENT")
        print(environment_text())
        print("MANIFEST")
        import json
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except InputValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
