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
| 2026-08-20 | This local traceability index and dated machine snapshots were created. | `research_archive/` and dated artifacts |

## Interpretation boundary

The earlier OpenAI CLIP zero-shot baseline inspected all 6,000 LoveDA regions before the later visual-anchor holdout was created. Therefore that visual-anchor holdout is **post-hoc exploratory**, not a blind confirmatory test. Later LoveDA GT-isolated and external Vaihingen/Potsdam protocols document their own prediction/GT ordering and must be cited at their own level of evidence.

No full conversational transcript was exported. This chronology reconstructs milestones from persisted artifacts only.
