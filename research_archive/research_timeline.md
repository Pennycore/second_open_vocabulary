# Evidence-backed research timeline

Dates below come from Git history, frozen protocols, reports, and output manifests. “Recorded” means an artifact exists; it does not automatically make a result confirmatory.

| Date | Milestone | Evidence |
|---|---|---|
| 2026-08-12 | Native region probes and formal provenance gates established. | commits `0cf243f` through `ec1a661`; `reports/ov_probe_stage0_complete_20260812.md` |
| 2026-08-13 | VOC/SBD handling, benchmark registry, RemoteCLIP bridge gate, and region pixel-pack export were frozen. | commits `403fe4a` through `fa47595` |
| 2026-08-14 | Forward architecture switched to OpenAI CLIP; fixed feature cache, image-disjoint split, visual-anchor exploratory study, and VOC quarantine were added. | commits `6ab562e` through `b5a4294` |
| 2026-08-15 | VOC image-level zero-shot cache and presence evaluation were recorded. | commits `4afe7b0`, `51ec149` |
| 2026-08-18 | LoveDA GT-isolated evaluation, prototype stability, score audit, calibration, partial-support analysis, and SCC freeze were recorded. | commits `6d90a93` through `8151c2a`; corresponding reports |
| 2026-08-19 | Vaihingen external confirmation, CTP freeze, pixel-level semantic-map experiments, score-scale ablation, and Potsdam protocol/prediction locks were recorded. | commits `056059a` through `2704f99` |
| 2026-08-20 | Final common-pixel, cluster-bootstrap, Guard, and qualitative audits were recorded. | commit `ab64c0c`; final audit reports |
| 2026-08-20 | Controlled RemoteCLIP backbone-replacement baseline completed on Potsdam full support; partial-support evaluation was intentionally not completed. | commits `cef2c49`, `9cb17c2`; remote workspace HEAD `7aecc73`; RemoteCLIP run manifest and report |
| 2026-08-20 | Controlled RemoteCLIP backbone-replacement baseline completed on Vaihingen with full support and frozen partial-support subsets k=2/3/4. | runner commit `7fbbaaa`; run `run_20260820T152937Z_1afc6939`; candidate run `run_20260820T145952Z_5beba872`; `reports/remoteclip_vaihingen_baseline_20260820.md` |
| 2026-08-20 | This local traceability index and dated machine snapshots were created. | `research_archive/` and dated artifacts |
| 2026-08-21 | RemoteCLIP Vaihingen sealed-output, partial-support aggregate, and five-area bootstrap audit were completed without new inference. | source run `run_20260820T152937Z_1afc6939`; cache-only audit `run_20260821T044500Z_35155e8`; code commit `35155e8`; `reports/remoteclip_vaihingen_partial_support_20260821.md` |
| 2026-08-21 | SegEarth-OV reproducibility and fairness audit concluded NO-GO for the current controlled comparison; no SegEarth environment or inference was started. | `reports/segearth_ov_feasibility_audit_20260821.md` |
| 2026-08-21 | Artifact-complete RemoteCLIP Vaihingen controlled run preserved float32 scores, prediction-before-GT gating, complete partial-support per-area statistics/accounting, and area-cluster bootstrap assets. | run `run_20260821T070310Z_ab6e429b`; commit `3564ea7`; `reports/remoteclip_vaihingen_partial_support_complete_20260821.md` |
| 2026-08-21 | SegEarth-OV external baseline gate was re-audited as NO-GO; no official deployment, asset download, or inference was initiated. | commit `48678ec`; `reports/segearth_ov_external_baseline_gate_reaudit_20260821.md` |
| 2026-08-21 | A bounded fixed-commit SegEarth-OV Vaihingen reproduction attempt established an isolated Python 3.9 CUDA environment, bound SimFeatUp/OpenAI weights, and froze 325-tile/Omega inputs. Two prediction-only adapter bootstrap failures occurred before model construction; an explicitly authorized final registry bootstrap passed but then failed on a relative checkpoint-hash path after model build and before prediction output or semantic GT access. The final stop condition was applied. | `reports/segearth_ov_official_repro_audit.md`; server logs and manifests cited therein |
| 2026-08-21 | After storage migration, one explicitly authorized path-only attempt canonicalized the OpenAI checkpoint path and reached the first frozen tile of official SegEarth-OV prediction. The unmodified SimFeatUp dependency failed because `AdaptiveConv` was unavailable. No prediction, GT read, or metric was produced; no retry was made. | `reports/segearth_ov_pathfix_attempt_20260821.md` |
| 2026-08-22 | The SimFeatUp Phase-A installation gate stopped before any install: the isolated runtime was intact and the RTX 3090 idle, but the host lacked `nvcc`/CUDA Toolkit required by the official `AdaptiveConv` CUDA extension. No clone, build, smoke, inference, GT read, or metric occurred. | `reports/segearth_ov_simfeatup_phase_a_gate_20260822.md` |

## Interpretation boundary

The earlier OpenAI CLIP zero-shot baseline inspected all 6,000 LoveDA regions before the later visual-anchor holdout was created. Therefore that visual-anchor holdout is **post-hoc exploratory**, not a blind confirmatory test. Later LoveDA GT-isolated and external Vaihingen/Potsdam protocols document their own prediction/GT ordering and must be cited at their own level of evidence.

No full conversational transcript was exported. This chronology reconstructs milestones from persisted artifacts only.
