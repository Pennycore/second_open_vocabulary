# CTP-v1 / SegEarth-OV Vaihingen common-support asset audit

Date: 2026-08-22
Status: **BLOCKED — do not evaluate**

## Scope and immutable boundary

This audit was limited to locating and byte-validating frozen inputs for the
requested offline CTP-v1 versus SegEarth-OV comparison.  No model, SAM3,
OpenAI CLIP, feature extraction, prediction reconstruction, GT evaluator, or
GPU computation was started.  No method, class mapping, split, prompt,
prototype, alpha, FusionCanvas, or `Omega_candidate` definition was changed.

## SegEarth-OV frozen inputs

The following sealed SegEarth-OV identity was re-hashed on the active 3090
storage volume and **passed** its frozen values:

| Item | Required SHA-256 / identity |
|---|---|
| Official source commit | `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| Prediction manifest | `f1f8f4c7d070b0640f41718d1da75f6a29572e933e2e499af693016eb0ade264` |
| Five-map aggregate | `c6bd7bbc0223d65877879f90e7303693247bfe70af17ce65cc14ce1583d0f1ab` |
| Metrics artifact | `f1993014231a506969b654cdb1b55587184aaa58106dfc49a022c5bbe26e28ea` |
| Tile manifest | `2a0fab569982a87a78b13ee0353b32115c983e8bc964968873d6fbb73769b634` |
| `Omega_candidate` manifest | `2a41f5826ff882d62b0a3bcf8d6f1313a51066ed4714141198bf5622e723a25f` |

The five recorded SegEarth maps are present, have areas `[11, 15, 28, 30, 34]`,
and their prediction-manifest aggregate is the required
`c6bd7bbc0223d65877879f90e7303693247bfe70af17ce65cc14ce1583d0f1ab`.

The sealed SegEarth semantic maps and frozen Omega files are present on the
3090 storage volume.  This is not sufficient to score the requested paired
comparison without the matching formal CTP maps.

## Formal CTP-v1 archive located

The formal pixel-level CTP archive is identified by the checked-in manifest:

`outputs/pixel_ovss_vaihingen_v0/manifest.json`

It declares the exact required Vaihingen test areas `[11, 15, 28, 30, 34]`,
five-class order `impervious_surface`, `building`, `low_vegetation`, `tree`,
`car`, and `ignore_index=255` / `uncovered_label=255`.  It is bound to:

| Item | SHA-256 |
|---|---|
| Formal pixel manifest present locally | `b064984cd2a3baf7f70835ec8a8c8d767477066223ad7874ddcbfaeab51b0309` |
| Pixel protocol | `8fd0f020bd13ede994c99320ab0b025b68b4b825e3ec1c07eb7c06c593d5c358` |
| Source region predictions | `96be715f5796a7877985c6a2a2a644acb28c3c1adc74ef395fcea5786712f39a` |
| Source records | `8bd1b180a5dcaf1c1b2008a977e9ec08b8579d0cdfef4ed5447d2a06d88bf298` |
| Frozen CTP configuration | `788f1962d497022fbd5cacd7b63eaedddecd0343104aa726ee80afcdf1b37430` |

The manifest lists the expected CTP semantic-map hashes for each of the five
areas.  However, it does **not** contain the arrays themselves.

## Blocking evidence

1. `outputs/pixel_ovss_vaihingen_v0/` contains only `manifest.json`,
   `pixel_overall.json`, and `pixel_stats.json`; no `*_semantic.npz` files.
2. A recursive search under `C:\Users\28457\Desktop` found zero
   `CTP_vaih_area*_semantic.npz` files.
3. The active 3090 project storage contains sealed RemoteCLIP CTP maps, but
   those maps are a different backbone/run and cannot be substituted for the
   OpenAI-CLIP formal CTP-v1 archive.
4. The historical dual-2080-Ti host alias `P` (`172.18.56.240`) timed out on
   2026-08-22, so its old run directory cannot presently be audited or copied.

Because the five CTP byte arrays are absent, their expected SHA-256 values
cannot be verified, their geometry cannot be compared to the frozen Omega
masks, and CTP `255` abstentions cannot be strictly counted.  Reconstructing
maps from region scores would be a new derived prediction artifact and is
explicitly prohibited by this task.

## Decision

**NO-GO.** Do not create the common-support evaluator, do not open GT, do not
write metrics/CSV/bootstrap outputs, and do not start inference.  The requested
comparison can resume only after the original five formal CTP semantic maps and
their original prediction manifest are recovered intact and their hashes match
the values declared by the formal pixel manifest.  The existing SegEarth-OV
assets must remain unchanged while this recovery is pending.
