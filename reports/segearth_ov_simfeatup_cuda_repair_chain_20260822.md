# SegEarth-OV SimFeatUp CUDA repair chain (2026-08-22)

## Verdict

**STOP / NO-GO.** An explicitly authorized execution-layer retry corrected the
relative Python-path launcher mistake and then reached real `nvcc` compilation
of the fixed official SimFeatUp source. The isolated CUDA 11.8 toolkit was
augmented only with the two subsequently authorized development-component
families, cuSPARSE and cuBLAS. The final allowed compilation then stopped on a
third distinct missing CUDA development header, `cusolverDn.h`. The maximum of
two automatic component additions was reached, so cuSOLVER was not installed
and no further retry was made.

No `AdaptiveConv` extension artifact, import smoke, strict SegEarth model
smoke, tile forward, formal Vaihingen run, prediction, semantic-GT read, or
metric was produced.

## Immutable experiment boundary

All changes were limited to:

```
/data/second_open_vocabulary_storage/external_baselines/segearth_ov/runtime_py39/
```

The first-paper project, shared `py310` environment, official SegEarth source,
CTP-v1, adapter protocol, SAM3 caches, data splits, frozen tile/Omega manifests,
and pre-existing outputs were not changed. The main isolated runtime remained:
torch `2.1.2+cu118`, torchvision `0.16.2+cu118`, MMCV `2.1.0`, MMEngine
`0.10.4`, MMSeg `1.2.2`, and NumPy `1.26.4`; `pip check` remained clean after
each completed toolkit transaction.

## Fixed sources and toolkit

| Item | Fixed version / SHA-256 |
|---|---|
| SegEarth-OV | commit `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| SimFeatUp source | official commit `78a0ba70b1d6ea7283684a88c98ce338af4593ca` |
| SimFeatUp source archive | `246131c82cab5321c7e9297da9573166060cb9502be64099d89fcc52a317ef94` |
| CUDA compiler | NVIDIA `cuda-nvcc` `11.8.89-0`; package SHA-256 `23ee509485627c7e402d0d6c4567e02641430a37581f1da0a4d5478dd5cecc0e` |
| CUDA runtime/dev/CCCL | `cuda-cudart`, `cuda-cudart-dev`, `cuda-cccl` `11.8.89-0`; SHA-256 `f8cf96ae45acf1bef5ff0be3e849d87e3543144ec8c0075db235f4933113a3b0`, `46f31a6b45ebdb09e03f3e0ec8a3cba13ceef53805de87de5ab0056b7ff69d80`, `cc68223476b91e15de718d4e31470ac9166e86eb123528bb61ec0b83c2ea1474` |
| cuSPARSE runtime/dev | `libcusparse`, `libcusparse-dev` `11.7.5.86-0`; SHA-256 `61f9bee3a0ed675a8978815071b63a6bbfe3ea141c2d12613ba1d1cc65c584d2`, `f8f48bd3ffad7bd915a68fab259826cc589ff084cdcb462e4da5250aee62621c` |
| cuBLAS runtime/dev | `libcublas`, `libcublas-dev` `11.11.3.6-0`; SHA-256 `c03d5d7ee8b279808f8bd11ccbb5b6ced263f56dcb87e9268cb590a705729b6f`, `202e37cd0210a726da327d0d380bac3d9f320e6a71ff69dd5a902587c5580b0c` |

## Bounded repair chain

1. The original build launcher stopped before Python started because its runtime
   path became relative after `cd`. One execution-only retry canonicalized the
   runtime Python and `CUDA_HOME` paths.
2. The corrected build invoked the official source's `setup.py build_ext
   --inplace` with `TORCH_CUDA_ARCH_LIST=8.6` and reached `nvcc`. It stopped at
   missing `cusparse.h`; exact CUDA-11.8 cuSPARSE runtime/dev components were
   added only to the dedicated toolkit.
3. The next real compile stopped at missing `cublas_v2.h`; exact CUDA-11.8
   cuBLAS runtime/dev components were added only to the dedicated toolkit.
4. The final allowed compile stopped at missing `cusolverDn.h`. This is a third
   independent CUDA development component. It was not installed, preserving the
   predefined two-component repair limit.

The final build log is:

```
/data/second_open_vocabulary_storage/external_baselines/segearth_ov/runtime_py39/simfeatup_adaptiveconv_build_retry_cublas_20260822T002901Z.log
```

Its SHA-256 is
`d61611404216d2da1255ad081610ca0aa9ca6164050558aedfe20c1a9861ae4f`.
At close, `nvidia-smi` reported no compute process and the source tree contained
no built `adaptive_conv_cuda_impl` or `adaptive_conv*.so` artifact.

## Next authorization boundary

Proceeding requires explicit approval to add the matching CUDA-11.8 cuSOLVER
development component and reopen a single compilation gate. It must not be
treated as a baseline result until the extension import, strict-model build, and
one prediction-only frozen-tile smoke gate all pass.
