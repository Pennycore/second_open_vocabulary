# CTP-v1.1 development-only tuning and freeze

## A. Status and interpretation boundary

`CTP-v1.1-tuned` is frozen with status **`frozen_development_only_pending_final_test`**. This was a **post-hoc development-only performance optimization with frozen test-evaluation rules**. It does not constitute a prospective blind tuning study.

The selected configuration has **not** been evaluated on the registered Vaihingen test areas, Potsdam, RemoteCLIP, or SegEarth. Consequently, this record makes no claim of test-set improvement.

## B. Frozen method and permitted search

CTP-v1 remains unchanged: its formula, SCC, Guard, prompts, visual-prototype construction, SAM3 candidates, and evaluation protocol were not modified. The only searchable pre-existing global protocol parameters were visual-anchor weight `alpha` and FusionCanvas conflict threshold `tau_conflict`.

The registered 49-cell grid was:

| Parameter | Values |
|---|---|
| `alpha` | 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 |
| `tau_conflict` | 0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030 |

The immutable untuned reference was CTP-v1-untuned: `alpha=0.5`, `tau_conflict=0.03`.

## C. Development/test isolation

Development used exactly Vaihingen areas `[1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]`. The registered test areas `[11, 15, 28, 30, 34]` were excluded from tuning.

Partial-support evaluation enumerated all 25 lexicographic nontrivial support subsets: 10 at `k=2`, 10 at `k=3`, and 5 at `k=4`, yielding 1,225 rows over the 49 grid cells. No seed-selected subset was used as a selection shortcut.

## D. Selection rule

A configuration was feasible only if no partial-support subset had zero U-IoU and its mean partial H-IoU was at least the untuned baseline. Among feasible configurations, full development mIoU was maximized. If full mIoU differed by less than 0.001, tie-breaking was mean H-IoU, then mean U-IoU, then lower abstention, then proximity to `(0.5, 0.03)`.

## E. Development-only result

The selected feasible point is:

| Item | Value |
|---|---:|
| `alpha*` | 0.5 |
| `tau_conflict*` | 0.03 |
| Full development mIoU | 0.41999849322575217 |
| Mean partial H-IoU | 0.14479417086838733 |
| Abstention ratio | 0.1162064099832935 |
| Partial-support collapse count | 0 |

The selection equals CTP-v1-untuned, so this tuning pass selected **no parameter change**. This is a development-only selection outcome, not evidence about any registered test result.

## F. Provenance and integrity binding

The canonical protocol file has SHA-256 `cc6ad09029e229713b1882dd07c9a4b327b00d587eefdaf4f53e84723060ecbc`; the recorded code commit is `63ac207ec00af7081dc5f60f2c6baa5fac3c9bdd`.

The prediction manifest was sealed before development GT evaluation: `41c2e099d40ae2e95fe6f76d52318bc0aabbccaa33df11899c1860c1c4af199a`. The development-evaluation manifest was a **post-hoc finalization**; it did not rerun computation, inference, model loading, prediction, or GT access, and is bound to the completed CSVs and the sealed prediction manifest. Its SHA-256 is `3f76cee29dd501ffd9be1c5809dd7e6fd6729ca03ad942570dd7348e75e72fa6`. The bound tuned-candidate record has SHA-256 `45e8e367f4434706f1664bea9ada30666b72972508d8996ec71f8a8a786d80af`.

| Completed table | SHA-256 |
|---|---|
| `grid_search_full.csv` | `e9b0963c633898ad1355e5da65e2c6accee5e03bad5b69a1dc3d292357a3f181` |
| `partial_all_subsets.csv` | `60de0657cd9ef3709e9f02e0c6cb3e8505ce06a155c8e3ddd7159fab14b4128a` |
| `full_support_accounting.csv` | `d4097d735071ebcd485a93398aaae4d73ab088fd7dbf8f8b4fbf89aa2f542345` |
| `full_support_per_class.csv` | `d69194c13694b1b186a047abefc54ba074d21fd15259a15c9b9e4bfb79712dac` |

## G. Freeze decision

The frozen record is [`configs/ctp_v1_1_tuned_frozen.json`](../configs/ctp_v1_1_tuned_frozen.json). It preserves the same values as CTP-v1-untuned and records the development-only selection and its integrity bindings. It contains no server paths, host identifiers, or hidden test results.

## H. Stop condition and next action

This phase stops here. No final test was run, no new model/grid/inference computation was run, and no further tuning is authorized by this record. A final test evaluation, if later approved, must be one-shot with this exact frozen configuration and must separately report the post-hoc development-only nature of this selection.
