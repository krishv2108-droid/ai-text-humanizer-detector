"""Detector figures: ROC curves and confusion matrices (light surface, validated palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from hd.eval.style import AXIS, BLUE_RAMP, GRID, INK, INK2, MUTED, SERIES, SURFACE

GENRE_ORDER = ["essay", "casual", "story"]


def _genres(df: pd.DataFrame) -> list[str]:
    return [g for g in GENRE_ORDER if g in set(df["genre"])]


def plot_roc(models: dict[str, object], labels: dict[str, str], df: pd.DataFrame, path: Path, title: str) -> None:
    """One panel per genre, one line per model (3 max: an all-pairs-safe series count)."""
    genres = _genres(df)
    fig, axes = plt.subplots(1, len(genres), figsize=(4.6 * len(genres), 4.9), facecolor=SURFACE, squeeze=False)
    for ax, genre in zip(axes[0], genres):
        g = df[df["genre"] == genre]
        y = g["label"].to_numpy()
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)  # chance
        for color, (name, model) in zip(SERIES, models.items()):
            p = model.predict_proba(g)[:, 1]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{labels[name]}  (AUC {roc_auc_score(y, p):.3f})")  # fmt: skip
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.set_title(genre, loc="left", color=INK, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("False positive rate (human flagged as AI)", color=INK2, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.6)
    axes[0][0].set_ylabel("True positive rate (AI caught)", color=INK2, fontsize=9)
    fig.text(0.01, 0.985, title, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.12, wspace=0.18)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(model, df: pd.DataFrame, path: Path, title: str, threshold: float = 0.5) -> None:
    """Row-normalized confusion matrices, one per genre, in a one-hue sequential ramp."""
    genres = _genres(df)
    fig, axes = plt.subplots(1, len(genres), figsize=(3.9 * len(genres), 4.4), facecolor=SURFACE, squeeze=False)
    cmap = matplotlib_cmap()
    for ax, genre in zip(axes[0], genres):
        g = df[df["genre"] == genre]
        pred = (model.predict_proba(g)[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(g["label"], pred, labels=[0, 1])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=12,
                        color="#ffffff" if share[i, j] > 0.5 else INK, fontweight="bold")  # fmt: skip
        # 2px surface gap between cells
        for k in (0.5,):
            ax.axhline(k, color=SURFACE, linewidth=3)
            ax.axvline(k, color=SURFACE, linewidth=3)
        ax.set_xticks([0, 1], ["predicted\nhuman", "predicted\nAI"], color=INK2, fontsize=9)
        ax.set_yticks([0, 1], ["actually\nhuman", "actually\nAI"], color=INK2, fontsize=9)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(genre, loc="left", color=INK, fontsize=11, fontweight="bold", pad=8)
    fig.text(0.01, 0.985, title, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.12, wspace=0.35)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def matplotlib_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
