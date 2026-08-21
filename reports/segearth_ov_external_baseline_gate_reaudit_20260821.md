# SegEarth-OV external whole-method baseline：执行门禁复核（2026-08-21）

## Material Passport

| Field | Value |
|---|---|
| Scope | SegEarth-OV only; protocol-different external whole-method baseline |
| Frozen-method boundary | CTP-v1, SCC, C2, Guard, SAM3 candidates, FusionCanvas, prompts, alpha and all splits were not edited |
| Official source identity | `likyoo/SegEarth-OV` at `3e22a969b32c6d751bdbba64a88a0b670e630f55` |
| Prior source audit | `reports/segearth_ov_feasibility_audit_20260821.md` |
| Runtime inspected | 3090v2, `/home/zhongsz/second_open_vocabulary`, read-only inventory on 2026-08-21 |
| Result | **NO-GO — do not download, deploy, or run SegEarth-OV in this round** |

## Decision

SegEarth-OV remains eligible only as a **protocol-different external whole-method baseline**. It is not a controlled same-pipeline baseline, cannot participate in the CTP partial-support protocol, and is not a CTP plug-in candidate.

The new task permits an external comparison only when all of the following are established before execution: pinned official runtime and weights, same selected test images, unambiguous five-class/ignore GT mapping, a frozen whole-image evaluator, and the fixed common support

`Ω_candidate = {GT-valid pixels} ∩ {pixels covered by the frozen CTP SAM3 candidates}`.

The read-only 3090v2 inventory does not yet establish all of those inputs. Therefore the required all-gates rule evaluates to **NO-GO**, and no source clone, checkpoint download, environment installation, data conversion, or inference was started.

## Gate ledger

| Gate | Evidence | Status |
|---|---|---|
| Pinned official code | Official repository and immutable commit are recorded in the prior audit. | PASS (source identity only) |
| Official inference runtime | The existing project has the RemoteCLIP environment, not the official Python/mmcv/mmseg/torch lock. No isolated SegEarth workspace or environment exists on 3090v2. | FAIL / not deployed |
| SimFeatUp asset | The fixed repository tree contains the declared SimFeatUp checkpoint, but no downloaded-file SHA-256 has been bound in an isolated runtime. | FAIL / not bound |
| OpenAI CLIP asset | SegEarth-OV requests OpenAI CLIP `ViT-B/16`; no local resolved asset, source URL/cache provenance, or SHA-256 is bound. | FAIL / not bound |
| Frozen Vaihingen inputs | The server has 16 RGB areas and labels, including frozen test areas 11/15/28/30/34. | PASS (asset presence) |
| Frozen Vaihingen candidate support | The completed SAM3 candidate cache and RemoteCLIP semantic maps are present, so `Ω_candidate` is constructible after a signed coverage manifest is generated. No such external-evaluation manifest exists yet. | CONDITIONAL |
| Same five-class GT mapping | Official Vaihingen is six-class and uses `clutter`; CTP scores five classes and ignores clutter. The needed rule — valid five-class GT only, but SegEarth clutter predictions count as errors on those valid pixels — is specified in the prior audit but not frozen as executable input/evaluator hashes. | FAIL / not bound |
| Same Potsdam inputs and GT | No Potsdam input or label directory was found in the project-local 3090v2 inventory. | FAIL |
| Whole-image interpretation | A wrapper is conceptually allowed, but it must be explicitly labelled `protocol-different external whole-method comparison`; it is not yet registered with image/GT/class/coverage hashes. | FAIL / not pre-registered |
| Partial-support | SegEarth-OV has no visual support set or supported/unsupported semantic mechanism. | NOT APPLICABLE / incompatible |
| CTP plug-in | SegEarth logits do not expose the CTP-required visual support score or support partition. | NO-GO |

## Direct server evidence

The 3090v2 project head was `35155e8`. Read-only listing confirmed:

- `inputs/vaihingen/images/` and `inputs/vaihingen/labels/` contain 16 Vaihingen areas, including 11, 15, 28, 30, and 34;
- frozen candidate cache exists at `outputs/proposals/vaihingen_sam3_v0/run_20260820T145952Z_5beba872/`;
- sealed RemoteCLIP Vaihingen maps exist at `outputs/baselines/remoteclip/vaihingen_v0/run_20260820T152937Z_1afc6939/`;
- no `SegEarth-OV` directory was found under `/home/zhongsz` in the bounded inventory;
- no project-local Potsdam or SegEarth input directory was found.

These observations establish that a future Vaihingen-only adapter could be prepared without touching CTP. They do **not** prove that official SegEarth inference has run, or that a Vaihingen/Potsdam external result is available.

## Fairness classification

If the gates are later passed, report two distinct tables, never a merged SOTA table:

1. **External whole-image result:** SegEarth-OV's native dense pipeline, evaluated on the fixed five-class GT with OA, Macro-F1, mIoU, per-class IoU, valid/ignored pixels, and its complete prediction coverage.
2. **Common-support semantic result:** both frozen CTP output and SegEarth prediction evaluated only on `Ω_candidate`; report the same metrics and the support cardinality/ratio. SegEarth retains its native dense prediction; CTP retains its native candidate/FusionCanvas procedure.

This does not make the pipelines controlled-equivalent. It only fixes the evaluated GT pixels, so the table must retain the label **“protocol-different external whole-method comparison.”**

## Re-entry conditions (analysis only; do not execute automatically)

A later execution request may reopen the gate only after a single pre-registered external-evaluation manifest binds:

1. the supplied Vaihingen and Potsdam image/GT paths and SHA-256 lists;
2. frozen five-class mapping and ignore policy, including the treatment of a SegEarth `clutter` prediction;
3. exact `Ω_candidate` mask construction, per-image hashes, and CTP prediction references;
4. official commit, separate environment lock, SimFeatUp SHA-256, resolved OpenAI CLIP asset SHA-256, preprocessing and patch-reconstruction details;
5. output map/logit, prediction hash, and evaluator version requirements.

Only after that manifest and an isolated official-runtime smoke inference succeed may the order be Vaihingen first and Potsdam second. No retraining, tuning, core-method edit, prompt change, partial-support emulation, or CTP insertion is permitted.

## Final statement for the current round

`SegEarthOV_external_baseline = NO-GO`.

The reason is incomplete reproducibility and evaluation binding, not a model-quality judgment. This is a scientifically transparent stop condition: fabricating a class mapping, silently using unverified weights, or reporting a result without the common-support mask would weaken rather than strengthen the TGRS evidence package.
