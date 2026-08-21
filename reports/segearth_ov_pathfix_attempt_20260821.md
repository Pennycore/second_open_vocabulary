# SegEarth-OV Vaihingen path-fix attempt (2026-08-21)

## Verdict

**Stopped without a reportable baseline.** This was the one explicitly
authorized post-migration Vaihingen retry. It fixed only the integration
adapter's checkpoint-path canonicalization before `chdir`; the official
SegEarth-OV source, checkpoints, frozen inputs, class mapping, CTP-v1, SAM3,
and evaluation protocol were unchanged. The first frozen tile reached genuine
prediction-only inference, then the official SimFeatUp path stopped at an
unavailable `AdaptiveConv` implementation. No repair or further launch was
made.

## Frozen gate evidence

| Item | Value |
|---|---|
| Official source commit | `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| OpenAI CLIP ViT-B/16 SHA-256 | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| SimFeatUp SHA-256 | `cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8` |
| Tile manifest SHA-256 | `2a0fab569982a87a78b13ee0353b32115c983e8bc964968873d6fbb73769b634` |
| Omega manifest SHA-256 | `2a41f5826ff882d62b0a3bcf8d6f1313a51066ed4714141198bf5622e723a25f` |
| Adapter SHA-256 | `d888a61d55ae4a0a594911c99fd2fa4941b6709f1ab71efe3739ceaeb98bb87e` |
| Runtime | Python 3.9; torch 2.1.2+cu118; MMCV 2.1.0; MMEngine 0.10.4; MMSeg 1.2.2; NumPy 1.26.4; RTX 3090 |

After storage migration, `inputs` and the external-baseline `outputs`, runtime,
and integration paths resolved under `/data/second_open_vocabulary_storage/`.
All listed hashes matched their pre-migration frozen values and `nvidia-smi`
reported no compute process before launch.

## Unique run record

- Launch root:
  `/data/second_open_vocabulary_storage/outputs/external_baselines/segearth_ov/vaihingen_official_pathfix_launch_20260821T163700Z`
- Unique run:
  `run_20260821T132524Z_1efb41b0`
- Run-manifest SHA-256:
  `daab299047ead1caaa2ac0c9d049fa7142c40d0895e6c71f3a1157547155e30c`
- Log SHA-256:
  `b129d7691cbd1ce7002a45e83144721578faffb3e7c0d80148d08a9390b454f0`

The prediction-only manifest was written before tile inference. The first tile
then stopped in the unmodified official SimFeatUp dependency:

```
simfeatup_dev/upsamplers.py:274, in forward
    result = AdaptiveConv.apply(hr_source_padded, combined_kernel)
AttributeError: 'NoneType' object has no attribute 'apply'
```

The unique run contains only `run_manifest.json`; its prediction directory is
empty. There is no prediction seal, semantic map, metric file, or evaluation
output. Semantic GT was not read, and `nvidia-smi` was empty after the stopped
process.

## Reporting boundary

SegEarth-OV remains a failed reproducibility attempt, not a numerical external
baseline. It must not be presented as a controlled comparison with CTP-v1 or
as evidence for any metric claim. Potsdam and every other external baseline
remain out of scope for this attempt.
