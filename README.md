# OV-WSSS Stage 0: RemoteCLIP Visual–Text Alignment Probe

This project tests one narrow question: whether **existing, Train-only, weakly supervised** RemoteCLIP region features or visual prototypes from the first paper align with RemoteCLIP text embeddings. It does not run SAM3, produce pseudo-labels, train a student, tune RemoteCLIP, or claim open-vocabulary segmentation.

All external inputs are opened read-only. Every run receives a unique directory such as `outputs/ov_probe_v0/run_20260812_001`; existing runs are never silently replaced. Random operations use seed 42 and every cosine comparison L2-normalizes both sides.

## Input contract

Set external resources in `configs/ov_probe_v0.yaml` or pass an alternate config. Do not put external paths into source code.

Recommended `.npz` layouts:

- Single prototypes: `features [C,512]`, `class_names [C]`, optional `sample_counts [C]`, and provenance fields.
- Multi prototypes: `features [P,512]`, `class_names [P]`, optional `prototype_ids [P]`, `cluster_sizes [P]`, and provenance fields.
- Region cache: `features [N,512]`, `cam_labels [N]` and/or `sam3_source_labels [N]`; an optional separate weak-label file may provide these fields.

Provenance may be embedded as `provenance_json` or supplied in a sidecar named `<input>.provenance.json`. A formal input must state:

```json
{
  "dataset": "LoveDA",
  "split": "train",
  "uses_pixel_gt": false,
  "uses_oracle": false,
  "construction": "description of the first-paper Train-only weak seed rule"
}
```

Files with missing provenance, Val split, pixel GT, or oracle use are rejected by default. `--dry-run` is the only mode that permits explicitly synthetic inputs.

## Commands

From the project root:

```powershell
$env:PYTHONPATH = "src"
python scripts/inspect_inputs.py --config configs/ov_probe_v0.yaml
python scripts/run_ov_probe.py --config configs/ov_probe_v0.yaml --dry-run
python scripts/run_ov_probe.py --config configs/ov_probe_v0.yaml
python -m pytest -q
```

For `ov_probe_v1`, the four small native prototype artifacts are preserved in a
versioned local input snapshot. Its text cache can be reproduced without
installing OpenCLIP locally and without creating files on the first-paper
server:

```powershell
python scripts/cache_remoteclip_text_features.py `
  --config configs/ov_probe_v1.yaml `
  --identity-file "$env:USERPROFILE\.ssh\codex_sam3_remote"
python scripts/run_ov_probe.py --config configs/ov_probe_v1.yaml
```

The audit command reports exactly which files are ready or missing and does not create features. The dry run checks the complete metrics/plot/report pipeline with clearly marked synthetic vectors; its numbers are not scientific evidence. A full run requires real prototypes and a locally installed OpenCLIP package compatible with the configured checkpoint. No model or dataset is downloaded automatically.

## Results

Each run contains resolved config, environment, input manifest, full prompt bank and class map, CSV/JSON metrics, plots, and `run.log`. The project-level status report is `reports/ov_probe_v0_report.md`. Interpret expanded-vocabulary ranking only as semantic selectivity: it is not unseen-class segmentation.

The initial resource audit is in `reports/project_inventory.md`. After the correct
first-paper root was supplied, the preserved follow-up configuration is
`configs/ov_probe_v1.yaml`; it directly accepts the first paper's native
prototype JSON+NPZ pairs and writes only under `outputs/ov_probe_v1`.
