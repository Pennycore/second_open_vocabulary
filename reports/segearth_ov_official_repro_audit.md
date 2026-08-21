# SegEarth-OV official reproducibility audit (2026-08-21)

## Verdict

**NO-GO — no official inference or metric was produced.** The official code,
fixed source assets, Vaihingen split geometry, checkpoints, and isolated CUDA
environment were verified. Two prediction-only technical launches then failed
before SegEarth model construction, weight loading into the model, semantic
prediction creation, prediction-manifest sealing, or semantic-GT reading.

This is a reproducibility and protocol gate, not a result-based decision. The
frozen CTP-v1 method, CTP data/evaluation protocol, RemoteCLIP runs, SAM3
candidate caches, and the first-paper project were not modified.

## Official source and inference contract

| Item | Evidence |
|---|---|
| Repository | `https://github.com/likyoo/SegEarth-OV` |
| Fixed commit | `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| Isolated checkout on 3090 | `/home/zhongsz/second_open_vocabulary/external_baselines/segearth_ov` |
| Official entry point | `eval.py --config configs/cfg_vaihingen.py --work-dir ...` |
| Model status | The README describes SegEarth-OV/SimFeatUp inference as training-free; no training was invoked. |
| Official Vaihingen config | `clip_type=CLIP`, `vit_type=ViT-B/16`, `model_type=SegEarth`, 448px resize, five foreground classes plus clutter. |
| Official requirements | Python 3.9 suggested; `torch==2.1.2`, `torchvision==0.16.2`, `mmcv==2.1.0`, `mmengine==0.10.4`, `mmsegmentation==1.2.2`, `numpy==2.0.0`, `opencv_python_headless==4.8.0.76`. |

The official `cfg_vaihingen.py` vocabulary is uniquely compatible with the
frozen CTP five-class vocabulary: impervious surface, building, low vegetation,
tree, and car. Its clutter output is treated as a distinct prediction state;
the frozen CTP red clutter GT rule remains ignore for ground truth.

## Checkpoint audit

| Checkpoint | Source / path | SHA-256 | Status |
|---|---|---:|---|
| SimFeatUp `xclip_jbu_one_million_aid.ckpt` | Tracked in the official fixed commit at `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt` | `cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8` | Present; 5.5 MiB |
| OpenAI CLIP ViT-B/16 | Official vendored OpenCLIP resolver uses the OpenAI public URL encoded in `open_clip/pretrained.py`; downloaded from that exact URL only after environment gate passed | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` | Present at `runtime_py39/clip_cache_home/.cache/clip/ViT-B-16.pt`; 335 MiB |

The official source's OpenAI resolver validates the URL-embedded SHA prefix.
All model assets are locally bound; the OpenAI file was not loaded into a model
because both launches stopped first.

## Vaihingen protocol and data gate

The frozen CTP test areas are exactly `[11, 15, 28, 30, 34]`. On the 3090,
each corresponding full-resolution image and label TIFF exists and matches the
frozen SAM3 candidate cache geometry:

| Area | Image shape (H×W) | Frozen SAM3 candidates | GT-valid pixels | Ω_candidate pixels |
|---:|---:|---:|---:|---:|
| 11 | 2566×1893 | 1713 | 4,857,438 | 3,185,737 |
| 15 | 2565×1919 | 1317 | 4,915,052 | 3,616,894 |
| 28 | 2567×1917 | 1359 | 4,912,533 | 3,224,461 |
| 30 | 2563×1934 | 1740 | 4,956,842 | 3,356,686 |
| 34 | 2555×1388 | 1286 | 3,546,340 | 2,509,587 |

An allowed external-only tiling/output adapter was prepared. It uses the
official documented 512px tile / 256px stride convention with shifted final
tiles and freezes mean unscaled-logit stitching before inference. It does not
change the SegEarth model, prompts, attention, SimFeatUp, classifier, or
post-processing. Annotation loading is removed only in the prediction phase so
that semantic GT cannot be read before prediction sealing.

Pre-inference artifacts on the 3090:

```
/home/zhongsz/second_open_vocabulary/outputs/external_baselines/segearth_ov/vaihingen_prepared/run_20260821T080951Z_b192d3ba
```

- 325 frozen input tiles
- `tile_manifest.json` SHA-256:
  `2a0fab569982a87a78b13ee0353b32115c983e8bc964968873d6fbb73769b634`
- `omega_candidate_manifest.json` SHA-256:
  `2a41f5826ff882d62b0a3bcf8d6f1313a51066ed4714141198bf5622e723a25f`

`Ω_candidate = GT-valid ∩ frozen SAM3 candidate-covered geometry`. During its
construction, label values were used only to identify the pre-registered red
clutter/ignore color; no class statistics, metrics, model settings, or
predictions informed the mask.

## Isolated environment gate

An independent Python 3.9.21 runtime was installed at:

```
/home/zhongsz/second_open_vocabulary/external_baselines/segearth_ov/runtime_py39/conda
```

The project Python environment and all first-paper paths were untouched. The
official pinned CUDA stack was installed only in this prefix: PyTorch
2.1.2+cu118, torchvision 0.16.2+cu118, MMCV 2.1.0, MMEngine 0.10.4, MMSeg
1.2.2, OpenCV 4.6.0.66, and OpenCV-headless 4.8.0.76.

The literal official `numpy==2.0.0` requirement failed its import/CUDA gate
before model construction:

```
torch: Failed to initialize NumPy: _ARRAY_API not found
cv2: ImportError: numpy.core.multiarray failed to import
mmcv import: AttributeError: _ARRAY_API not found
```

This is the NumPy-2 ABI incompatibility of the pinned PyTorch 2.1.2 and
OpenCV/MMCV binaries. The exact initial log is retained as:

```
/home/zhongsz/second_open_vocabulary/external_baselines/segearth_ov/runtime_py39/pip_install_20260821_attempt3.log
```

With explicit one-time authorization, only NumPy was changed to `1.26.4`; no
PyTorch/MMCV/MMEngine/MMSeg version, model source, model algorithm, data,
prompt, or CTP component changed. The runtime then passed `pip check` and the
import/CUDA gate:

```
Python 3.9.21
torch 2.1.2+cu118, CUDA 11.8, RTX 3090
numpy 1.26.4                 # sole official-requirements deviation
opencv-python 4.6.0.66
opencv-python-headless 4.8.0.76
mmcv 2.1.0; mmengine 0.10.4; mmsegmentation 1.2.2
```

The compatibility/install records are retained under `runtime_py39/`:
`numpy_compat_fix_20260821.log`,
`opencv_official_stack_repair_20260821.log`, and
`opencv_metadata_alignment_20260821.log`.

## Prediction-launch record

Two unique prediction-only directories/logs were retained. Neither yielded a
prediction artifact, and no retry remains authorized.

1. The first launch failed before model construction because the adapter was
   invoked from `integration/`, leaving the official checkout absent from
   `sys.path` (`ModuleNotFoundError: segearth_segmentor`). The empty run
   directory and `runtime_py39/vaihingen_official_prediction_20260821.log` are
   retained.
2. After a Sol-approved adapter-only path correction and full repeat of the
   official commit/environment/checkpoint/tile/Omega hashes, the second unique
   launch failed before model construction because the MMSeg transform registry
   had not been bootstrapped (`KeyError: PackSegInputs is not in the
   mmengine::transform registry`). Its log is
   `runtime_py39/vaihingen_official_prediction_relaunch_20260821.log`.

The second non-model startup failure triggers the frozen stop condition. The
adapter was not changed again; no official core source, algorithm, model
setting, class mapping, or input protocol was modified. There is no
prediction manifest, semantic map, whole-image metric, common-support metric,
per-scene metric, or GT-derived result to report.

## Potsdam gate

Potsdam remains **NO-GO** independently of the environment failure.

- The incoming Potsdam RGB/IRRG trees are present (1,764 images, labels, and
  DSMs each), and all 14 CTP parent IDs occur.
- They provide 49 896px patches per registered parent (686 target patches),
  whereas the frozen CTP protocol uses 3,584 512px patches (256 per parent).
  They are not the same pixel geometry, so an identical frozen
  `Ω_candidate` cannot be established from the current assets.
- SegEarth's Potsdam vocabulary merges `road,parking lot`, while frozen CTP
  uses `impervious_surface`. That mapping is not uniquely equivalent and must
  not be selected from results.

No Potsdam adaptation, inference, or metric computation was performed.

## Consequence for paper reporting

SegEarth-OV must not be reported as a numerical external baseline from this
round. The scientifically accurate statement is that an official fixed-commit,
isolated-environment reproduction was prepared with bound checkpoints and
frozen Vaihingen common support, but stopped before model construction and
inference after two preserved adapter-bootstrap failures. The Vaihingen
data/common-support preparations and all associated hashes are retained for a
future, explicitly approved compatibility study.
