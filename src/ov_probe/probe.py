from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .io import FeatureBundle
from .metrics import (
    aggregate_subprototype_scores,
    alignment_metrics,
    cosine_similarity,
    long_similarity_frame,
    region_statistics,
)
from .prompts import TextEncoder, encode_prompt_group


def validate_single_bundle(bundle: FeatureBundle, class_names: list[str]) -> None:
    counts = Counter(bundle.class_names)
    if set(counts) != set(class_names) or any(counts[name] != 1 for name in class_names):
        raise ValueError(f"Single-prototype input must contain exactly one row per configured class; counts={dict(counts)}")


def reorder_single(bundle: FeatureBundle, class_names: list[str]) -> np.ndarray:
    return np.stack([bundle.features[bundle.class_names.index(name)] for name in class_names])


def run_single_probe(
    bundle: FeatureBundle,
    encoder: TextEncoder,
    bank: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    class_names = list(bank["class_names"])
    validate_single_bundle(bundle, class_names)
    visual = reorder_single(bundle, class_names)
    all_long = []
    details: dict[str, pd.DataFrame] = {}
    matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
    summary: dict[str, Any] = {}
    prompt_rows = []
    for group in ("A", "B"):
        summary[group] = {}
        for vocabulary in cfg["evaluation"]["vocabularies"]:
            candidates, text, per_prompt = encode_prompt_group(encoder, bank, group, vocabulary)
            scores = cosine_similarity(visual, text)
            metrics, detail = alignment_metrics(scores, class_names, candidates, cfg["evaluation"]["topk"])
            key = f"{group}_{vocabulary}"
            summary[group][vocabulary] = metrics
            details[key] = detail.assign(prompt_group=group, vocabulary=vocabulary)
            matrices[key] = (scores, candidates)
            all_long.append(long_similarity_frame(scores, class_names, candidates, prompt_group=group, vocabulary=vocabulary, row_type="single"))
            # Per-class prompt sensitivity: replace that class's ensemble text
            # vector with one fixed prompt while all competitors remain fixed.
            for visual_index, name in enumerate(class_names):
                correct_index = candidates.index(name)
                for prompt_index, prompt_feature in enumerate(per_prompt[name]):
                    trial_text = text.copy()
                    trial_text[correct_index] = prompt_feature
                    trial_scores = cosine_similarity(visual[visual_index : visual_index + 1], trial_text)
                    _, prompt_detail = alignment_metrics(
                        trial_scores,
                        [name],
                        candidates,
                        cfg["evaluation"]["topk"],
                    )
                    prompt_detail["prompt_group"] = group
                    prompt_detail["prompt_index"] = prompt_index
                    prompt_detail["prompt"] = bank["groups"][group][name][prompt_index]
                    prompt_detail["vocabulary"] = vocabulary
                    prompt_rows.append(prompt_detail)
    return {
        "summary": summary,
        "long_frame": pd.concat(all_long, ignore_index=True),
        "details": details,
        "matrices": matrices,
        "prompt_stability": pd.concat(prompt_rows, ignore_index=True),
    }


def run_multi_probe(
    bundle: FeatureBundle,
    encoder: TextEncoder,
    bank: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    class_names = list(bank["class_names"])
    missing = sorted(set(class_names) - set(bundle.class_names))
    if missing:
        raise ValueError(f"Multi-prototype input is missing classes: {missing}")
    frames = []
    summaries: dict[str, Any] = {}
    primary_matrix = None
    subprototype_analysis: dict[str, Any] = {}
    for group in ("A", "B"):
        summaries[group] = {}
        for vocabulary in cfg["evaluation"]["vocabularies"]:
            candidates, text, _ = encode_prompt_group(encoder, bank, group, vocabulary)
            raw = cosine_similarity(bundle.features, text)
            raw_frame = long_similarity_frame(
                raw,
                bundle.class_names,
                candidates,
                prompt_group=group,
                vocabulary=vocabulary,
                row_type="subprototype",
                aggregation="none",
            )
            repeats = len(candidates)
            raw_frame["prototype_index"] = np.repeat(np.arange(len(bundle.features)), repeats)
            raw_frame["prototype_id"] = np.repeat(
                bundle.prototype_ids if bundle.prototype_ids else [str(i) for i in range(len(bundle.features))], repeats
            )
            raw_frame["cluster_size"] = np.repeat(bundle.cluster_sizes, repeats) if bundle.cluster_sizes is not None else None
            best_indices = np.argmax(raw, axis=1)
            raw_frame["nearest_text_class"] = np.repeat([candidates[int(i)] for i in best_indices], repeats)
            raw_frame["correct_is_top1"] = np.repeat(
                [int(candidates[int(best_indices[i])] == bundle.class_names[i]) for i in range(len(bundle.features))], repeats
            )
            frames.append(raw_frame)
            class_score_rows: dict[str, list[np.ndarray]] = {name: [] for name in class_names}
            for name in class_names:
                indices = [i for i, value in enumerate(bundle.class_names) if value == name]
                class_score_rows[name] = [raw[i] for i in indices]
            summaries[group][vocabulary] = {}
            for method in [item for item in cfg["evaluation"]["aggregation"] if item != "single"]:
                aggregated = np.stack([
                    aggregate_subprototype_scores(np.stack(class_score_rows[name]), method, float(cfg["evaluation"]["logsumexp_temperature"]))
                    for name in class_names
                ])
                metrics, detail = alignment_metrics(aggregated, class_names, candidates, cfg["evaluation"]["topk"])
                summaries[group][vocabulary][method] = metrics
                frames.append(long_similarity_frame(aggregated, class_names, candidates, prompt_group=group, vocabulary=vocabulary, row_type="aggregate", aggregation=method))
                if group == "A" and vocabulary == "closed" and method == "mean":
                    primary_matrix = aggregated
            if group == "A" and vocabulary == "closed":
                for name in class_names:
                    indices = [i for i, value in enumerate(bundle.class_names) if value == name]
                    predicted = [candidates[int(np.argmax(raw[i]))] for i in indices]
                    correct_fraction = float(np.mean([value == name for value in predicted]))
                    subprototype_analysis[name] = {
                        "num_subprototypes": len(indices),
                        "correct_alignment_fraction": correct_fraction,
                        "predicted_text_counts": dict(Counter(predicted)),
                        "semantic_consistency": float(max(Counter(predicted).values()) / len(predicted)),
                        "cluster_sizes": [float(bundle.cluster_sizes[i]) for i in indices] if bundle.cluster_sizes is not None else None,
                    }
    return {
        "summary": summaries,
        "frame": pd.concat(frames, ignore_index=True),
        "primary_matrix": primary_matrix,
        "subprototype_analysis": subprototype_analysis,
    }


def run_region_probe(
    bundle: FeatureBundle,
    encoder: TextEncoder,
    bank: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if bundle.cam_labels is None or bundle.sam3_source_labels is None:
        raise ValueError(
            "The registered region probe requires both independently derived CAM and SAM3-source labels."
        )
    features = bundle.features
    cam_labels = bundle.cam_labels
    sam3_labels = bundle.sam3_source_labels
    if len(cam_labels) != len(features) or len(sam3_labels) != len(features):
        raise ValueError("Region feature and weak-label row counts differ.")
    allowed = set(bank["class_names"])
    for field, values in (("cam_labels", cam_labels), ("sam3_source_labels", sam3_labels)):
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"{field} contains classes outside the registered allowlist: {invalid}")
    sample_cap = cfg["evaluation"].get("region_sample_per_class")
    sampling: dict[str, Any] = {"enabled": False, "seed": int(cfg["experiment"]["seed"]), "original_count": len(features)}
    if sample_cap is not None:
        if int(sample_cap) <= 0:
            raise ValueError("evaluation.region_sample_per_class must be null or positive.")
        reference = sam3_labels
        rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
        selected: list[int] = []
        per_class: dict[str, int] = {}
        for name in bank["class_names"]:
            indices = np.asarray([i for i, value in enumerate(reference) if value == name], dtype=int)
            if len(indices) > int(sample_cap):
                indices = np.sort(rng.choice(indices, size=int(sample_cap), replace=False))
            selected.extend(indices.tolist())
            per_class[name] = len(indices)
        selected = sorted(selected)
        features = features[selected]
        cam_labels = [cam_labels[i] for i in selected] if cam_labels else None
        sam3_labels = [sam3_labels[i] for i in selected] if sam3_labels else None
        sampling.update({
            "enabled": True,
            "reference": "sam3_source_labels",
            "cap_per_class": int(sample_cap),
            "selected_count": len(selected),
            "per_class": per_class,
        })
    results = {"sampling": sampling}
    for group in ("A", "B"):
        results[group] = {}
        for vocabulary in cfg["evaluation"]["vocabularies"]:
            candidates, text, _ = encode_prompt_group(encoder, bank, group, vocabulary)
            scores = cosine_similarity(features, text)
            results[group][vocabulary] = region_statistics(
                scores,
                candidates,
                cam_labels,
                sam3_labels,
                float(cfg["evaluation"]["logsumexp_temperature"]),
            )
    results["interpretation_limit"] = "Agreement is measured against weak labels, not true classification accuracy."
    return results


def predeclared_error_analysis(single_result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    scores, candidates = single_result["matrices"]["A_expanded"]
    class_names = [str(item["name"]) for item in cfg["data"]["classes"]]
    registered = cfg["evaluation"]["predeclared_confusions"]
    result = {}
    for row_index, name in enumerate(class_names):
        confusions = []
        correct = float(scores[row_index, candidates.index(name)])
        for candidate in registered[name]:
            if candidate in candidates:
                similarity = float(scores[row_index, candidates.index(candidate)])
                confusions.append({"class": candidate, "similarity": similarity, "gap_from_correct": correct - similarity})
        result[name] = {"correct_similarity": correct, "predeclared_confusions": confusions}
    return result
