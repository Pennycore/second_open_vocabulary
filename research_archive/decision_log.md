# Decision log and scientific boundaries

| Decision | Rationale and boundary | Provenance |
|---|---|---|
| Use OpenAI CLIP as the forward feature space. | The forward pipeline uses OpenAI CLIP ViT-B/32 quick-GELU and 512-D normalized features. RemoteCLIP outputs remain legacy evidence only. **RemoteCLIP and OpenAI CLIP feature spaces were never mixed.** | `configs/architecture_v1.json`; `reports/openai_clip_architecture_v1.md`; commit `6ab562e` |
| Preserve SAM3 candidates without rerunning SAM3. | Existing candidate regions and inherited weak tags are treated as inputs. No SAM3 rerun or large model training is part of this project phase. | frozen protocols and manifests |
| Treat inherited weak labels cautiously. | First-paper inherited weak tags originated from a Train pixel-mask presence simulation. They are weak supervision, not independent semantic ground truth. | region-probe reports and input manifests |
| Separate prediction from GT evaluation. | Prediction hashes/manifests are frozen before GT reading where a protocol says so. Results are evaluated at the declared level only. | LoveDA, Vaihingen, Potsdam protocols and reports |
| Report VOC correctly. | VOC presence mAP is image-level multi-label presence evaluation; it is **not** segmentation mIoU and must not be described as such. | `outputs/voc2012_openai_clip_presence_eval_v1/run_20260815_001/metrics.json`; commit `51ec149` |
| Disclose visual-anchor chronology. | The first all-6,000 baseline predates the later holdout; visual-anchor heldout findings are exploratory/post-hoc and cannot be used as blind confirmation. | visual-anchor protocol and timeline |
| Retain raw-material references rather than duplicate them. | Large datasets and weights are referenced through manifests/hashes; dated snapshots retain code/config/report/output evidence without replicating raw material. | material passport and manifests |

## Reproducibility caveats

- Conversation content was not fully exported, so decisions recorded only in chat may be absent.
- LLM reasoning is not byte-reproducible; persisted protocols and commits are the reviewable record.
- Some server experiments were carried out from uncommitted workspace files. Those files are preserved in the dated server source snapshot but have no asserted commit identity.
