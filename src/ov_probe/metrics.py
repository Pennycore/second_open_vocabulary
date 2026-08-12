from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot L2-normalize a zero or near-zero feature vector.")
    return array / norms


def cosine_similarity(visual: np.ndarray, text: np.ndarray) -> np.ndarray:
    return l2_normalize(visual) @ l2_normalize(text).T


def alignment_metrics(
    scores: np.ndarray,
    true_names: list[str],
    candidate_names: list[str],
    topk: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(true_names), len(candidate_names)):
        raise ValueError("Score matrix shape does not match labels.")
    rows = []
    ranks = []
    for index, true_name in enumerate(true_names):
        if true_name not in candidate_names:
            raise ValueError(f"True class {true_name!r} is absent from candidate vocabulary.")
        correct_index = candidate_names.index(true_name)
        order = np.argsort(-values[index], kind="stable")
        rank = int(np.flatnonzero(order == correct_index)[0]) + 1
        incorrect = np.delete(values[index], correct_index)
        max_wrong_index = int(np.argmax(np.where(np.arange(len(candidate_names)) == correct_index, -np.inf, values[index])))
        correct_score = float(values[index, correct_index])
        max_wrong = float(incorrect.max()) if len(incorrect) else float("nan")
        row = {
            "visual_class": true_name,
            "predicted_class": candidate_names[int(order[0])],
            "correct_rank": rank,
            "correct_similarity": correct_score,
            "max_wrong_class": candidate_names[max_wrong_index] if len(candidate_names) > 1 else None,
            "max_wrong_similarity": max_wrong,
            "positive_negative_margin": correct_score - max_wrong if len(incorrect) else float("nan"),
        }
        for k in topk:
            row[f"top_{k}"] = int(correct_index in order[: min(k, len(order))])
        rows.append(row)
        ranks.append(rank)
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "num_visual_classes": len(true_names),
        "vocabulary_size": len(candidate_names),
        "mean_correct_rank": float(np.mean(ranks)),
        "mean_correct_similarity": float(frame["correct_similarity"].mean()),
        "mean_max_wrong_similarity": float(frame["max_wrong_similarity"].mean()),
        "mean_positive_negative_margin": float(frame["positive_negative_margin"].mean()),
        "positive_margin_fraction": float((frame["positive_negative_margin"] > 0).mean()),
    }
    for k in topk:
        summary[f"top_{k}_accuracy"] = float(frame[f"top_{k}"].mean())
    return summary, frame


def long_similarity_frame(
    scores: np.ndarray,
    row_names: list[str],
    candidate_names: list[str],
    **context: Any,
) -> pd.DataFrame:
    records = []
    for row_index, row_name in enumerate(row_names):
        for col_index, candidate in enumerate(candidate_names):
            records.append({
                **context,
                "visual_class": row_name,
                "text_class": candidate,
                "similarity": float(scores[row_index, col_index]),
                "is_correct": int(row_name == candidate),
            })
    return pd.DataFrame(records)


def aggregate_subprototype_scores(scores: np.ndarray, method: str, temperature: float = 0.07) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if method == "max":
        return values.max(axis=0)
    if method == "mean":
        return values.mean(axis=0)
    if method == "logsumexp":
        scaled = values / temperature
        maximum = scaled.max(axis=0)
        return temperature * (maximum + np.log(np.exp(scaled - maximum).mean(axis=0)))
    raise ValueError(f"Unknown aggregation method: {method}")


def normalized_entropy(scores: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    denominator = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    return entropy / denominator


def region_statistics(
    scores: np.ndarray,
    candidate_names: list[str],
    cam_labels: list[str] | None,
    sam3_labels: list[str] | None,
    temperature: float = 0.07,
) -> dict[str, Any]:
    values = np.asarray(scores)
    order = np.argsort(-values, axis=1, kind="stable")
    predictions = [candidate_names[index] for index in order[:, 0]]
    top1 = values[np.arange(len(values)), order[:, 0]]
    top2 = values[np.arange(len(values)), order[:, 1]] if values.shape[1] > 1 else np.full(len(values), np.nan)
    entropy = normalized_entropy(values, temperature)
    records = []
    for i, prediction in enumerate(predictions):
        records.append({
            "region_index": i,
            "predicted_class": prediction,
            "max_similarity": float(top1[i]),
            "top1_top2_margin": float(top1[i] - top2[i]),
            "normalized_entropy": float(entropy[i]),
            "normalized_confidence": float(1.0 - entropy[i]),
            "cam_label": cam_labels[i] if cam_labels else None,
            "sam3_source_label": sam3_labels[i] if sam3_labels else None,
        })
    result: dict[str, Any] = {
        "num_regions": len(values),
        "mean_max_similarity": float(np.mean(top1)),
        "mean_top1_top2_margin": float(np.mean(top1 - top2)),
        "mean_normalized_entropy": float(np.mean(entropy)),
        "prediction_counts": dict(sorted(pd.Series(predictions).value_counts().to_dict().items())),
        "per_region": records,
    }
    if cam_labels:
        result["cam_text_agreement"] = float(np.mean([a == b for a, b in zip(cam_labels, predictions)]))
    if sam3_labels:
        result["sam3_text_agreement"] = float(np.mean([a == b for a, b in zip(sam3_labels, predictions)]))
    if cam_labels and sam3_labels:
        result["cam_sam3_agreement"] = float(
            np.mean([a == b for a, b in zip(cam_labels, sam3_labels)])
        )
        result["cam_sam3_text_three_way_agreement"] = float(np.mean([a == b == c for a, b, c in zip(cam_labels, sam3_labels, predictions)]))
    def summarize_by_reference(reference: list[str]) -> dict[str, Any]:
        per_class = {}
        for name in sorted(set(reference)):
            indices = [i for i, value in enumerate(reference) if value == name]
            per_class[name] = {
                "count": len(indices),
                "mean_margin": float(np.mean((top1 - top2)[indices])),
                "text_agreement": float(np.mean([predictions[i] == name for i in indices])),
            }
        return per_class
    if cam_labels:
        result["per_class_by_cam"] = summarize_by_reference(cam_labels)
    if sam3_labels:
        result["per_class_by_sam3_source"] = summarize_by_reference(sam3_labels)
    return result
