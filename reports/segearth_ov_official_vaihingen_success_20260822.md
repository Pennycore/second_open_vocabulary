# SegEarth-OV official Vaihingen external baseline (2026-08-22)

## Result

**Completed.** The fixed official SegEarth-OV pipeline produced a sealed
prediction-only Vaihingen result and was evaluated only afterwards.  This is
an **external, protocol-different baseline**; it is not a controlled CTP
comparison and must not be presented as one.

| Scope | OA | Macro F1 | mIoU |
|---|---:|---:|---:|
| Whole image, five semantic classes, clutter ignored | 0.560106011 | 0.449311234 | 0.318387433 |
| Frozen `Omega_candidate` common support, five classes, clutter ignored | 0.683463382 | 0.533887544 | 0.407630696 |

`Omega_candidate` is the pre-inference intersection of valid non-clutter GT
pixels and the frozen SAM3 candidate geometry.  It is included as a
common-support diagnostic, not as evidence that SegEarth-OV used CTP or SAM3.

## Frozen evidence

| Item | Bound value |
|---|---|
| Official SegEarth-OV | commit `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| Prediction adapter | SHA-256 `d888a61d55ae4a0a594911c99fd2fa4941b6709f1ab71efe3739ceaeb98bb87e` (only the prior checkpoint-path canonicalization) |
| SimFeatUp source | official commit `78a0ba70b1d6ea7283684a88c98ce338af4593ca` |
| SimFeatUp source archive | SHA-256 `246131c82cab5321c7e9297da9573166060cb9502be64099d89fcc52a317ef94` |
| Built `adaptive_conv_cuda_impl` | SHA-256 `a0bb81657c7054bc98862e4539ed2aef1d4faff6c5ca11f0b8774926d444d04f` |
| OpenAI CLIP ViT-B/16 | SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| SimFeatUp checkpoint | SHA-256 `cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8` |
| Frozen Vaihingen tile manifest | SHA-256 `2a0fab569982a87a78b13ee0353b32115c983e8bc964968873d6fbb73769b634` |
| Frozen `Omega_candidate` manifest | SHA-256 `2a41f5826ff882d62b0a3bcf8d6f1313a51066ed4714141198bf5622e723a25f` |

The isolated Python runtime remained torch `2.1.2+cu118`, MMCV `2.1.0`,
MMSeg `1.2.2`, and NumPy `1.26.4`; `pip check` remained clean.  The shared
`py310` environment and system CUDA were not changed.  The dedicated runtime
received the NVIDIA `cuda-toolkit=11.8.0` complete development toolchain from
the exact `nvidia/label/cuda-11.8.0` channel (including cuSOLVER), with
`nvcc` `11.8.89`.  The first unconstrained-channel dry-run was rejected because
it resolved CUDA 12.x dependencies; it was never installed.  The accepted
Toolkit installation log has SHA-256
`bb4124c582aa97eb5c32c7df3cee183d90b13453a36465ff2aaab5bf543132b8`.

## Gate sequence and result locations

1. GPU preflight found no compute process.  The official checkout, adapters,
   checkpoints, tile/Omega manifests, and isolated runtime were hash-checked.
2. The official SimFeatUp CUDA extensions compiled in the dedicated runtime;
   no official SegEarth code was changed.
3. A no-GT smoke imported a non-null `AdaptiveConv`, strictly built the
   official model, and forwarded frozen tile `vaih_area11_x0_y0.png`.  Its
   finite logits had shape `[6, 512, 512]`.  The first smoke preflight used an
   incorrect root-level checkpoint check and stopped before model construction;
   the one permitted execution-path retry used the official config path and
   passed.  The successful smoke log is
   `/data/second_open_vocabulary_storage/external_baselines/segearth_ov/runtime_py39/simfeatup_adaptiveconv_smoke_retry_pathfix_20260821T184408Z.log`.
4. The single formal prediction-only run used all five areas and 325 frozen
   tiles.  It completed successfully at:

   ```text
   /data/second_open_vocabulary_storage/outputs/external_baselines/segearth_ov/
   vaihingen_official_fulltoolkit_20260821T184600Z_9f4cb8d1/
   run_20260821T184608Z_3ba6a6b7/
   ```

   Its sealed `prediction_manifest.json` has SHA-256
   `f1f8f4c7d070b0640f41718d1da75f6a29572e933e2e499af693016eb0ade264`,
   and its five-map prediction aggregate has SHA-256
   `c6bd7bbc0223d65877879f90e7303693247bfe70af17ce65cc14ce1583d0f1ab`.
5. Only after that manifest was sealed was semantic GT read.  `metrics.json`
   has SHA-256 `f1993014231a506969b654cdb1b55587184aaa58106dfc49a022c5bbe26e28ea`.
   It records the RGB-to-five-class mapping, per-class counts, whole-image
   confusion matrix, and frozen common-support confusion matrix.

No Potsdam, ReAttnCLIP, training, parameter tuning, CTP/SCC/C2/Guard change,
prompt/prototype change, SAM3 rerun, or first-project action occurred.

## Interpretation boundary

SegEarth-OV supplies its own OpenAI CLIP ViT-B/16, SimFeatUp, prompt list, and
official model forward pass.  For compatible image handling, the external
adapter uses frozen 512/256 tiles and a mean-logit stitch, while CTP results
use the frozen SAM3 candidate/FusionCanvas pipeline.  Therefore these results
belong in an **external-method comparison** table with the proposal, backbone,
and fusion differences explicitly disclosed, not in the controlled OpenAI
CLIP/RemoteCLIP CTP table.
