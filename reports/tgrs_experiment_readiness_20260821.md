# TGRS experiment-readiness snapshot

Date: 2026-08-21
Status: evidence package updated; no new method development

## Evidence ready

1. The CTP-v1 method and its protocol remain frozen.
2. The RemoteCLIP Vaihingen controlled replacement now has complete float32
   partial-support statistical assets: 27 score archives, 390 semantic maps,
   390 confusion matrices, 390 per-area metric rows, 390 pixel-accounting rows,
   78 subset metrics, and 25 stored bootstrap records.
3. Prediction-before-GT gating is recorded in the complete-run manifest.
4. The original RemoteCLIP full-support metrics are reproduced exactly.

## Claims ready for review

- A bounded robustness claim: CTP-v1 retained a positive controlled effect with
  RemoteCLIP on the frozen Vaihingen protocol.
- A partial-support claim may use the registered per-area/subset statistics and
  explicit valid/assigned/conflict-ignore/uncovered accounting.
- Every such statement must state the dataset, protocol, support subsets, and
  tested backbone; it must not be generalized to all backbones.

## External-baseline status

SegEarth-OV is **NO-GO** for this round as an external whole-method baseline.
The re-audit is at
`reports/segearth_ov_external_baseline_gate_reaudit_20260821.md`. There is no
isolated official runtime, no SHA-bound local SimFeatUp or OpenAI CLIP asset,
no frozen five-class/ignore plus common-support manifest, and no Potsdam input/
GT binding on 3090v2. It must not be put in the controlled table or used as a
CTP plug-in.

## Next step (analysis only)

Prepare manuscript tables from sealed artifacts, with a controlled
RemoteCLIP-backbone table and a separate external-method gate table. Reopen a
SegEarth experiment only after all stated reproducibility and fairness gates
are pre-registered and satisfied.
