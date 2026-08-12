from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_heatmap(scores: np.ndarray, rows: list[str], columns: list[str], path: Path, title: str) -> None:
    width = max(7.0, 0.48 * len(columns))
    height = max(4.5, 0.5 * len(rows))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(scores, xticklabels=columns, yticklabels=rows, cmap="vlag", center=0, annot=len(columns) <= 8, fmt=".2f", ax=ax)
    ax.set_xlabel("Text class")
    ax.set_ylabel("Visual class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_margin_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a9d8f" if value > 0 else "#e76f51" for value in frame["positive_negative_margin"]]
    ax.bar(frame["visual_class"], frame["positive_negative_margin"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Positive-negative margin")
    ax.set_title("Per-class visual–text margin (Group A, closed vocabulary)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rank_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["visual_class"], frame["correct_rank"], color="#457b9d")
    ax.set_ylabel("Correct text rank (lower is better)")
    ax.set_yticks(range(1, int(frame["correct_rank"].max()) + 1))
    ax.set_title("Per-class correct text rank (Group A, closed vocabulary)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prompt_stability(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    summary = frame.groupby("visual_class", sort=False)["correct_similarity"].agg(["mean", "std", "min", "max"])
    x = np.arange(len(summary))
    axes[0].errorbar(x, summary["mean"], yerr=summary["std"].fillna(0), fmt="o", capsize=4)
    axes[0].vlines(x, summary["min"], summary["max"], alpha=0.5)
    axes[0].set_xticks(x, summary.index, rotation=25)
    axes[0].set_ylabel("Correct similarity")
    axes[0].set_title("Prompt similarity range")
    rank_pivot = frame.pivot(index="visual_class", columns="template_index", values="correct_rank")
    sns.heatmap(rank_pivot, cmap="crest_r", annot=True, fmt=".0f", ax=axes[1])
    axes[1].set_title("Correct rank by template")
    axes[1].set_xlabel("Template index")
    axes[1].set_ylabel("Visual class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_region_agreement_plot(results: dict, path: Path) -> None:
    rows = []
    for group in ("A", "B"):
        for vocabulary in ("closed", "expanded"):
            values = results[group][vocabulary]
            rows.append({
                "setting": f"{group}-{vocabulary}",
                "CAM–text": values["cam_text_agreement"],
                "SAM3–text": values["sam3_text_agreement"],
                "CAM–SAM3–text": values["cam_sam3_text_three_way_agreement"],
            })
    frame = pd.DataFrame(rows).set_index("setting")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    frame.plot(kind="bar", ylim=(0, 1), color=["#2a9d8f", "#457b9d", "#e9c46a"], ax=ax)
    ax.set_ylabel("Weak-label agreement")
    ax.set_xlabel("Prompt group and vocabulary")
    ax.set_title("Region–text agreement (not pixel-level accuracy)")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
