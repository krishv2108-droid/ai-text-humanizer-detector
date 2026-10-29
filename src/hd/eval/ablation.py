"""Feature-group ablation for the detector: which groups of linguistic cues does detection depend on?

For each genre separately (genres are never pooled), a boosted-tree model is evaluated with prompt-grouped
cross-validation under three conditions:
  all         every feature group the detector uses;
  drop:<g>    all groups except <g>   (how much does detection lose without it?);
  only:<g>    just group <g>          (how much can it do on its own?).
Splits used: train + validation. The held-out test split is never read here.
Many groups carry overlapping information (e.g. GPT-2 predictability and sentence rhythm), so a small drop does
not mean a group is useless: another group may be covering for it. "only" shows the reverse view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from hd.config import resolve_path
from hd.data.splits import load_split
from hd.detector.baseline import FULL_GROUPS, model_features
from hd.eval.metrics import grouped_bootstrap_ci
from hd.eval.style import AI_C, GRID, HUMAN_C, INK, INK2, SURFACE, style_axes


def _cv_auc(df: pd.DataFrame, cols: list[str], seed: int, folds: int, n_boot: int) -> tuple[float, float, float]:
    y, groups = df["label"].to_numpy(), df["prompt_id"].to_numpy()
    oof = np.zeros(len(df))
    for train, test in GroupKFold(n_splits=folds).split(df, y, groups):
        model = HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=8, max_iter=200, early_stopping=False, random_state=seed)
        model.fit(df.iloc[train][cols], y[train])
        oof[test] = model.predict_proba(df.iloc[test][cols])[:, 1]
    auc = float(roc_auc_score(y, oof))
    lo, hi = grouped_bootstrap_ci(y, oof, groups, roc_auc_score, n_boot=n_boot, seed=seed) if n_boot else (np.nan, np.nan)
    return auc, lo, hi


def group_ablation(df: pd.DataFrame, groups: list[str] | None = None, seed: int = 42, folds: int = 5, n_boot: int = 200) -> pd.DataFrame:
    """`df` needs columns genre, label, prompt_id and the feature columns."""
    groups = list(groups or FULL_GROUPS)
    rows = []
    for genre, part in df.groupby("genre"):
        part = part.reset_index(drop=True)
        everything = model_features(groups)
        conditions = [("all", everything)]
        conditions += [(f"drop:{g}", model_features([x for x in groups if x != g])) for g in groups]
        conditions += [(f"only:{g}", model_features([g])) for g in groups]
        base_auc = None
        for name, cols in conditions:
            auc, lo, hi = _cv_auc(part, cols, seed, folds, n_boot)
            base_auc = auc if name == "all" else base_auc
            rows.append({"genre": genre, "condition": name, "n_features": len(cols), "n_texts": len(part), "auc": auc, "auc_lo": lo,
                         "auc_hi": hi, "auc_change_vs_all": auc - base_auc})  # fmt: skip
    return pd.DataFrame(rows)


def load_ablation_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Paired train + validation texts with features. The test split is not touched."""
    processed = resolve_path(cfg, "processed")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    paired = corpus[corpus["has_pair"]]
    parts = [load_split(paired, name) for name in ("train", "val")]
    return pd.concat(parts, ignore_index=True).merge(pd.read_parquet(processed / "features.parquet"), on="id", how="inner")


def plot_group_drop(table: pd.DataFrame, path: Path, title: str) -> None:
    """For each genre, how much AUC is lost when one group is removed (bars to the left = the group mattered)."""
    import matplotlib.pyplot as plt

    drops = table[table["condition"].str.startswith("drop:")].copy()
    drops["group"] = drops["condition"].str.removeprefix("drop:")
    genres = list(dict.fromkeys(drops["genre"]))
    order = drops.groupby("group")["auc_change_vs_all"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(genres), figsize=(3.9 * len(genres), 0.36 * len(order) + 1.8), facecolor=SURFACE, sharey=True, sharex=True, squeeze=False)
    for ax, genre in zip(axes[0], genres):
        style_axes(ax)
        g = drops[drops["genre"] == genre].set_index("group").loc[order]
        ax.barh(range(len(order)), g["auc_change_vs_all"], color=[AI_C if v < -0.005 else HUMAN_C for v in g["auc_change_vs_all"]], height=0.62, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(range(len(order)), order, fontsize=9, color=INK2)
        ax.set_title(f"{genre}  (all features: AUC {table[(table.genre == genre) & (table.condition == 'all')].auc.iloc[0]:.3f})", loc="left", color=INK, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Change in AUC when the group is removed", color=INK2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=1)
    fig.text(0.01, 0.985, title, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.12, right=0.98, top=1 - 0.85 / fig.get_figheight(), bottom=0.7 / fig.get_figheight() + 0.03, wspace=0.12)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
