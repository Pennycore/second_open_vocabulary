"""Build the four Phase-I CSVs from final-audit JSONs.

Usage:
    python scripts/build_final_audit_csvs.py --audit-dir <dir-with-json> [--out-dir outputs/final_audit]

Expected inputs (produced by scripts/run_final_audit.py and
scripts/run_loveda_cluster_bootstrap.py):
    common_pixel_metrics_{vaihingen,potsdam}.json
    five_method_metrics_{vaihingen,potsdam}.json
    cluster_bootstrap_{vaihingen,potsdam,loveda}.json
Outputs:
    common_pixel_metrics.csv
    guard_pixel_partial_support.csv
    cluster_bootstrap_summary.csv
    per_cluster_deltas.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ["OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU"]
METHODS = ["text_only", "C2", "SCC", "CTP", "guard"]
REFS = ["text_only", "C2", "SCC", "guard"]
DELTA = ["OA", "macro_f1", "mIoU", "H_F1", "H_IoU"]
DATASETS = ("vaihingen", "potsdam", "loveda")


def load(audit_dir: Path, name: str):
    p = audit_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def subset_label(r: dict) -> str:
    return str(r.get("subset"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final-audit CSVs from JSONs.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/final_audit")
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in DATASETS:
        d = load(audit_dir, f"common_pixel_metrics_{ds}.json")
        if not d:
            continue
        for r in d:
            base = {"dataset": ds, "subset": subset_label(r), "k": r.get("k"), "ratio": r.get("ratio"),
                    "seed": r.get("seed"), "supported": r.get("supported"), "unsupported": r.get("unsupported"),
                    "common_valid_pixels": r["common_valid_pixels"]}
            for m in METHODS:
                for met in METRICS + ["valid"]:
                    base[f"{m}_{met}"] = r.get(f"{m}_{met}")
                base[f"{m}_coverage_ratio"] = r.get(f"{m}_coverage_ratio")
            rows.append(base)
    if rows:
        with (out / "common_pixel_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote common_pixel_metrics.csv: {len(rows)} rows")

    rows = []
    for ds in DATASETS:
        d = load(audit_dir, f"five_method_metrics_{ds}.json")
        if not d:
            continue
        for r in d:
            base = {"dataset": ds, "subset": subset_label(r), "k": r.get("k"), "ratio": r.get("ratio"),
                    "seed": r.get("seed"), "supported": r.get("supported"), "unsupported": r.get("unsupported")}
            for m in METHODS:
                for met in METRICS:
                    base[f"{m}_orig_{met}"] = r.get(f"{m}_orig_{met}")
                    base[f"{m}_common_{met}"] = r.get(f"{m}_common_{met}")
                base[f"{m}_orig_valid"] = r.get(f"{m}_orig_valid")
                base[f"{m}_common_valid"] = r.get(f"{m}_common_valid")
            rows.append(base)
    if rows:
        with (out / "guard_pixel_partial_support.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote guard_pixel_partial_support.csv: {len(rows)} rows")

    rows = []
    for ds in DATASETS:
        d = load(audit_dir, f"cluster_bootstrap_{ds}.json")
        if not d:
            continue
        for r in d:
            for ref in REFS:
                row = {"dataset": ds, "subset": subset_label(r), "k": r.get("k"), "ratio": r.get("ratio"),
                       "seed": r.get("seed"), "cluster_unit": r.get("cluster_unit"),
                       "n_clusters": len(r.get("clusters", [])), "ref": ref}
                for met in DELTA:
                    h = r[f"d{met}_vs_{ref}"]
                    row[f"d{met}_point"] = h["point"]
                    row[f"d{met}_mean"] = h["mean"]
                    row[f"d{met}_ci95_low"] = h["ci95_low"]
                    row[f"d{met}_ci95_high"] = h["ci95_high"]
                    row[f"d{met}_sign_consistency"] = h["pct_sign"]
                rows.append(row)
    if rows:
        with (out / "cluster_bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote cluster_bootstrap_summary.csv: {len(rows)} rows")

    rows = []
    for ds in DATASETS:
        d = load(audit_dir, f"cluster_bootstrap_{ds}.json")
        if not d:
            continue
        for r in d:
            for ref in REFS:
                for cluster, val in r["per_cluster_dH_IoU_vs"][ref].items():
                    rows.append({"dataset": ds, "subset": subset_label(r), "ref": ref,
                                 "cluster": cluster, "dH_IoU_CTP_minus_ref": val})
    if rows:
        with (out / "per_cluster_deltas.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote per_cluster_deltas.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
