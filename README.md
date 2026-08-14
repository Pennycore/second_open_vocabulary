# OpenAI CLIP OV-WSSS

The forward architecture for this second-paper project is **OpenAI CLIP
ViT-B/32 quick-GELU**, with encoder-matched image and text towers. It does not
rerun SAM3, produce pseudo-labels, train a student, or fine-tune the encoder.

The existing Stage 0 RemoteCLIP probe is retained as an immutable historical
audit. Its feature vectors and prototypes are incompatible with OpenAI CLIP and
may only be used to reproduce the original RemoteCLIP-space results. New LoveDA
work must re-encode the existing read-only RGB/mask candidate views in OpenAI
CLIP space after the v1 pixel-package gate is registered. See
`configs/architecture_v1.json` and `configs/stage1_ov_protocol_v1.json`.

All external inputs are opened read-only. Every run receives a unique directory such as `outputs/ov_probe_v0/run_20260812_001`; existing runs are never silently replaced. Random operations use seed 42 and every cosine comparison L2-normalizes both sides.

## Input contract

Set external resources in `configs/ov_probe_v0.yaml` or pass an alternate config. Do not put external paths into source code.

Prototype inputs may use the portable `.npz` layouts below or the first paper's
native JSON+NPZ pairs:

- Single prototypes: `features [C,512]`, `class_names [C]`, optional `sample_counts [C]`, and provenance fields.
- Multi prototypes: `features [P,512]`, `class_names [P]`, optional `prototype_ids [P]`, `cluster_sizes [P]`, and provenance fields.

Formal region inputs deliberately do **not** accept an unkeyed aggregate file.
They must use the first paper's three native, read-only directories:

- `region_features` from each region-score JSON/NPZ pair;
- SAM3 source class from the matching candidate JSON/NPZ pair;
- CAM top-1 recomputed from the candidate mask and matching CAM NPZ.

The adapter joins rows by `(image_id, candidate_index)`, verifies the candidate
bundle SHA-256, and never uses the region cache's existing
`predicted_class_ids` as weak truth. This prevents a RemoteCLIP prediction from
being evaluated against itself.

Prototype provenance may be embedded as `provenance_json` or supplied in a
sidecar named `<input>.provenance.json`. A formal prototype input must state:

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

For region work, copy `configs/region_probe_v0.yaml` to an ignored file such as
`configs/region_probe_v0.local.yaml` and fill only that local copy. External
paths, credentials, datasets, checkpoints and run outputs are excluded by
`.gitignore`.

## Commands

From the project root:

```powershell
$env:PYTHONPATH = "src"
python scripts/inspect_inputs.py --config configs/ov_probe_v0.yaml
python scripts/run_ov_probe.py --config configs/ov_probe_v0.yaml --dry-run
python scripts/run_ov_probe.py --config configs/ov_probe_v0.yaml
python scripts/run_region_probe.py --config configs/region_probe_v0.yaml --dry-run
python -m pytest -q
```

For `ov_probe_v1`, the four small native prototype artifacts are preserved in a
local, Git-ignored input snapshot. A compatible text cache may be produced by
the helper without creating a file in the first-paper project or server:

```powershell
python scripts/cache_remoteclip_text_features.py `
  --config configs/ov_probe_v1.yaml `
  --identity-file "<path-to-your-read-only-ssh-identity>"
python scripts/run_ov_probe.py --config configs/ov_probe_v1.yaml
```

The audit command reports exactly which files are ready or missing and does not
create features. Dry runs check the metrics/plot/report pipeline with clearly
marked synthetic vectors; their numbers are not scientific evidence. Formal
runs require real inputs and either a compatible cached text feature file or a
compatible local OpenCLIP environment. No model or dataset is downloaded
automatically.

The local region dry run is:

```powershell
python scripts/run_region_probe.py `
  --config configs/region_probe_v0.yaml `
  --dry-run
```

It creates 48 deterministic synthetic regions and exercises Group A/B,
closed/expanded vocabularies, weak-label agreement, margin, entropy, keyed row
records and non-overwriting output allocation. The native candidate/mask/CAM
adapter is exercised by the unit-test fixtures without reading the large server
cache.

After filling an ignored local config with the registered first-paper Train
cache paths, a real-cache pilot can be run with bounded inputs:

```powershell
python scripts/run_region_probe.py `
  --config configs/region_probe_v0.local.yaml `
  --limit-images 5 `
  --max-regions-per-class 50 `
  --allow-partial-classes
```

Pilot runs are always recorded as `pilot_native_region` with
`scientific_evidence=false`. The registered formal command has no pilot
overrides:

```powershell
python scripts/run_region_probe.py `
  --config configs/region_probe_v0.local.yaml
```

Formal evidence is accepted only when the resolved config matches the committed
protocol and the loader validates all 2,522 Train image bundles, all 270,641
candidates, the registered per-class counts, 1,000 selected records per class,
and unchanged source-file stat inventories.

## Results

Each run contains resolved config, environment, input manifest, full prompt bank and class map, CSV/JSON metrics, plots, and `run.log`. The completed E0.1–E0.5 report is
`reports/ov_probe_stage0_complete_20260812.md`. Interpret expanded-vocabulary
ranking only as semantic selectivity: it is not unseen-class segmentation.

Region dry runs are written under `outputs/region_probe_v0/run_*` and additionally
contain `validated_region_input.json`, `selected_region_records.jsonl`,
`region_level_results.json` and `region_weak_agreement.png`.

The initial resource audit is in `reports/project_inventory.md`. After the correct
first-paper root was supplied, the preserved follow-up configuration is
`configs/ov_probe_v1.yaml`; it directly accepts the first paper's native
prototype JSON+NPZ pairs and writes only under `outputs/ov_probe_v1`.
