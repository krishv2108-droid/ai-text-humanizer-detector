"""Which features separate human from AI text? Effect sizes + plots, on the TRAINING split only.

The test split is never read here: looking at it to decide what to plot or keep would leak it into
our choices. Univariate ROC-AUC reads as "if I scored texts by this one feature alone, how often would
a random AI text outrank a random human text?" (0.5 = no signal, 1.0 = AI always higher, 0.0 = human
always higher).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

from hd.config import resolve_path
from hd.data.splits import load_split
from hd.features.stylometric import FEATURE_GROUPS, feature_names

# Chart tokens (light surface). Series colors are categorical slots 1-3 of the validated palette.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
HUMAN_C, AI_C = SERIES[0], SERIES[1]

GENRE_ORDER = ["essay", "casual", "story"]
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def stratified_auc(x: np.ndarray, y: np.ndarray, n_words: np.ndarray, bins: int = 16, min_class: int = 10) -> float:
    """ROC-AUC computed *within* text-length bands, then averaged (weighted by human*AI pair count).

    Human and AI texts differ in length in our data, and many features (like language-model
    perplexity) also drift with length. Comparing only texts of similar length means length can't be
    what drives the number. Bins where either class has < `min_class` texts are skipped.

    This reduces the length confound but does not eliminate it: texts inside one band still differ a
    little in length by label. In a worst-case simulation (a feature that *is* length, 2-SD length gap)
    it returns 0.50-0.60 instead of the raw 0.85+. Treat small values near 0.5 as "no signal", not as
    exact zeros, and prefer features whose stratified AUC is far from 0.5.
    """
    edges = np.unique(np.quantile(n_words, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, n_words, side="right") - 1, 0, len(edges) - 2)
    num = den = 0.0
    for b in range(len(edges) - 1):
        m = which == b
        n0, n1 = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        if n0 < min_class or n1 < min_class:
            continue
        num += n0 * n1 * roc_auc_score(y[m], x[m])
        den += n0 * n1
    return num / den if den else np.nan


def effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (genre, feature): length-matched AUC (main), raw AUC, Cohen's d, length correlation."""
    rows = []
    for genre, g in df.groupby("genre"):
        y, nw = g["label"].to_numpy(), g["n_words"].to_numpy(dtype=float)
        for feat in feature_names():
            x = g[feat].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            if ok.sum() < 20 or len(set(y[ok])) < 2 or np.nanstd(x) == 0:
                continue
            h, a = x[ok & (y == 0)], x[ok & (y == 1)]
            pooled = np.sqrt((h.var(ddof=1) + a.var(ddof=1)) / 2)
            rows.append(
                {
                    "genre": genre,
                    "feature": feat,
                    "group": FEATURE_TO_GROUP[feat],
                    "auc": stratified_auc(x[ok], y[ok], nw[ok]),
                    "auc_raw": roc_auc_score(y[ok], x[ok]),
                    "cohens_d": (a.mean() - h.mean()) / pooled if pooled else np.nan,
                    "length_rho": pd.Series(x[ok]).corr(pd.Series(nw[ok]), method="spearman"),
                    "n": int(ok.sum()),
                }
            )
        # Reference: how well does raw length alone do? Not a feature, but the bar to beat.
        ref = roc_auc_score(y, nw)
        rows.append({"genre": genre, "feature": "(raw text length)", "group": "reference", "auc": ref,
                     "auc_raw": ref, "cohens_d": np.nan, "length_rho": 1.0, "n": len(g)})  # fmt: skip
    out = pd.DataFrame(rows)
    out["direction"] = np.where(out["auc"] >= 0.5, "higher in AI", "higher in human")
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _pretty(name: str) -> str:
    return name.replace("_per100w", " /100w").replace("_frac", " share").replace("_", " ")


def plot_auc_dumbbell(effects: pd.DataFrame, path: Path, top: int = 14) -> list[str]:
    genres = [g for g in GENRE_ORDER if g in set(effects["genre"])]
    feats = effects[effects["feature"] != "(raw text length)"]
    wide = feats.pivot(index="feature", columns="genre", values="auc")[genres]
    order = (wide - 0.5).abs().mean(axis=1).sort_values(ascending=False).head(top).index.tolist()
    ref = effects[effects["feature"] == "(raw text length)"].set_index("genre")["auc"]

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * top + 2.2), facecolor=SURFACE)
    _style(ax)
    ys = np.arange(len(order))[::-1]
    for y, feat in zip(ys, order):
        vals = wide.loc[feat].dropna()
        ax.plot([vals.min(), vals.max()], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        for i, genre in enumerate(genres):
            if genre in vals:  # small vertical offset keeps near-identical values from hiding each other
                offset = (i - (len(genres) - 1) / 2) * 0.22
                ax.plot(vals[genre], y + offset, "o", markersize=9, color=SERIES[i], markeredgecolor=SURFACE,
                        markeredgewidth=1.8, zorder=3, clip_on=False)  # fmt: skip
    ax.axvline(0.5, color=INK2, linewidth=1, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([_pretty(f) for f in order], color=INK2, fontsize=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Length-matched ROC-AUC, training split (0.5 = no signal)", color=INK2, fontsize=10, labelpad=8)
    ax.text(0.49, len(order) - 0.35, "higher in human ←", ha="right", color=MUTED, fontsize=9)
    ax.text(0.51, len(order) - 0.35, "→ higher in AI", ha="left", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=SERIES[i], markeredgecolor=SURFACE, label=g) for i, g in enumerate(genres)]  # fmt: skip
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(genres), frameon=False, labelcolor=INK2, fontsize=10)  # fmt: skip
    ref_txt = ", ".join(f"{g} {ref[g]:.2f}" for g in genres)
    fig.text(0.015, 0.97, "What gives AI text away? Strongest single features", color=INK, fontsize=13, fontweight="bold", ha="left", va="top")  # fmt: skip
    fig.text(0.015, 0.012, "AUC is computed within text-length bands, so length cannot drive it.   "
             f"For reference, raw length alone scores {ref_txt}.", color=MUTED, fontsize=8, ha="left", va="bottom")  # fmt: skip
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return order


def plot_distributions(df: pd.DataFrame, features: list[str], path: Path) -> None:
    genres = [g for g in GENRE_ORDER if g in set(df["genre"])]
    fig, axes = plt.subplots(len(genres), len(features), figsize=(2.9 * len(features), 2.3 * len(genres) + 0.9),
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    for r, genre in enumerate(genres):
        g = df[df["genre"] == genre]
        for c, feat in enumerate(features):
            ax = axes[r, c]
            _style(ax)
            ax.grid(False)
            ax.set_yticks([])
            pooled = g[feat].dropna()
            lo, hi = pooled.quantile(0.01), pooled.quantile(0.99)
            xs = np.linspace(lo, hi, 200)
            for label, color in ((0, HUMAN_C), (1, AI_C)):
                vals = g.loc[g["label"] == label, feat].dropna()
                vals = vals[(vals >= lo) & (vals <= hi)]
                if len(vals) < 5 or vals.std() == 0:
                    continue
                dens = gaussian_kde(vals)(xs)
                ax.fill_between(xs, dens, color=color, alpha=0.10, linewidth=0)
                ax.plot(xs, dens, color=color, linewidth=2, solid_capstyle="round")
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(_pretty(feat), color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=8)
            if c == 0:
                ax.set_ylabel(genre, color=INK2, fontsize=10.5, labelpad=8)
    handles = [plt.Line2D([], [], color=HUMAN_C, linewidth=2, label="human"), plt.Line2D([], [], color=AI_C, linewidth=2, label="AI")]  # fmt: skip
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, labelcolor=INK2, fontsize=10, bbox_to_anchor=(0.99, 0.995))  # fmt: skip
    fig.text(0.01, 0.985, "How the top features distribute: human vs AI (training split)", color=INK, fontsize=13, fontweight="bold", ha="left", va="top")  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84 if len(genres) == 2 else 0.88, bottom=0.07, wspace=0.12, hspace=0.45)  # fmt: skip
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pick_showcase(df: pd.DataFrame, ranked: list[str], per_kind: int = 2, max_corr: float = 0.9) -> list[str]:
    """Top features of each kind (GPT-2 vs stylometric), skipping near-duplicates of ones already picked."""
    corr = df[ranked].corr(method="spearman").abs()
    perplexity = set(FEATURE_GROUPS["perplexity"])
    chosen: list[str] = []
    for is_ppl in (True, False):
        taken = 0
        for feat in ranked:
            if (feat in perplexity) != is_ppl or taken == per_kind:
                continue
            if any(corr.loc[feat, c] > max_corr for c in chosen):
                continue
            chosen.append(feat)
            taken += 1
    return chosen


def run_analysis(cfg: dict[str, Any]) -> pd.DataFrame:
    processed, results = resolve_path(cfg, "processed"), resolve_path(cfg, "results")
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(processed / "corpus.parquet")
    corpus = corpus[corpus["has_pair"]]
    train = load_split(corpus, "train")  # the test split stays locked
    feats = pd.read_parquet(processed / "features.parquet")
    df = train.merge(feats, on="id", how="inner")

    effects = effect_table(df)
    effects.round(4).to_csv(results / "feature_effects.csv", index=False)
    top = plot_auc_dumbbell(effects, figures / "feature_auc.png")
    plot_distributions(df, pick_showcase(df, top), figures / "feature_distributions.png")

    show = effects[effects["feature"].isin(top[:8])].pivot(index="feature", columns="genre", values="auc").loc[top[:8]]
    print("Top features (univariate AUC, train split):\n", show.round(3).to_string())
    print("\nRaw-length AUC:", effects[effects.feature == "(raw text length)"].set_index("genre")["auc"].round(3).to_dict())
    return effects
