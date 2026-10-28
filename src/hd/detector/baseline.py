"""Layer-1 detector: interpretable features + a linear or boosted-tree classifier.

Protocol (so the numbers can be trusted):
  * hyperparameters are chosen by prompt-grouped cross-validation on the TRAIN split;
  * the VAL split is then a clean holdout we report on;
  * the TEST split is never touched here (`load_split` would raise); it is for the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hd.config import resolve_path
from hd.data.splits import load_split
from hd.eval.metrics import classification_metrics, grouped_bootstrap_ci, stratified_auc
from hd.features.stylometric import FEATURE_GROUPS, STYLOMETRIC_GROUPS, feature_names

# Layout (line breaks) is partly an artifact of where each side was collected, so it is off by default.
EXCLUDED_BY_DEFAULT = ["format"]
# burstiness == (cv - 1) / (cv + 1), a monotone *transform* of sent_len_cv. Giving a linear model both
# lets it bend a curve (worth ~0.02 validation AUC for the stylometric model) but makes their
# coefficients huge and opposite (+6 / -6), which is useless for explanation. This layer exists to be
# interpretable, so only one goes in; the boosted trees capture such curves anyway.
REDUNDANT_FEATURES = {"burstiness"}
STYLE_GROUPS = [g for g in STYLOMETRIC_GROUPS if g not in EXCLUDED_BY_DEFAULT]
FULL_GROUPS = STYLE_GROUPS + ["perplexity"]


@dataclass(frozen=True)
class Variant:
    label: str
    model: str  # "lr" | "gbm"
    groups: tuple[str, ...]
    pos_ngrams: bool = False


VARIANTS: dict[str, Variant] = {
    "lr_style": Variant("Stylometric, logistic", "lr", tuple(STYLE_GROUPS)),
    "lr_style_pos": Variant("Stylometric + POS n-grams, logistic", "lr", tuple(STYLE_GROUPS), True),
    "lr_full": Variant("Stylometric + GPT-2, logistic", "lr", tuple(FULL_GROUPS)),
    "lr_full_pos": Variant("Stylometric + GPT-2 + POS n-grams, logistic", "lr", tuple(FULL_GROUPS), True),
    "gbm_full": Variant("Stylometric + GPT-2, boosted trees", "gbm", tuple(FULL_GROUPS)),
}


def model_features(groups: list[str] | tuple[str, ...]) -> list[str]:
    return [f for f in feature_names(list(groups)) if f not in REDUNDANT_FEATURES]


def build_pipeline(v: Variant, seed: int) -> tuple[Pipeline, dict[str, list]]:
    cols = model_features(v.groups)
    if v.model == "lr":
        # median-impute (+ a "was missing" flag, e.g. no contraction opportunity), then standardize so
        # coefficients are comparable across features
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
        clf = LogisticRegression(max_iter=5000)
        grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
    else:
        numeric = "passthrough"  # histogram GBM handles missing values natively
        clf = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed)
        grid = {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [8, 31], "clf__l2_regularization": [0.0, 1.0]}
    blocks: list = [("num", numeric, cols)]
    if v.pos_ngrams:
        if v.model != "lr":
            raise ValueError("POS n-grams are sparse; use the logistic variants")
        # POS n-gram *rates* (l1-normalized), so text length doesn't leak in; the vocabulary is
        # learned from training folds only
        blocks.append(("pos", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(2, 3),
                                              use_idf=False, norm="l1", min_df=20, max_features=300), "pos_seq"))  # fmt: skip
    return Pipeline([("features", ColumnTransformer(blocks)), ("clf", clf)]), grid


def load_split_features(cfg: dict[str, Any], split: str, *, unlock_test: bool = False) -> pd.DataFrame:
    """Paired texts of one split joined with their features. The test split stays locked by default."""
    processed = resolve_path(cfg, "processed")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    part = load_split(corpus[corpus["has_pair"]], split, unlock_test=unlock_test)
    return part.merge(pd.read_parquet(processed / "features.parquet"), on="id", how="inner")


def fit_variant(name: str, train: pd.DataFrame, seed: int, cv_folds: int = 5) -> tuple[Pipeline, dict[str, Any]]:
    v = VARIANTS[name]
    pipe, grid = build_pipeline(v, seed)
    search = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=cv_folds), scoring="roc_auc", refit=True, n_jobs=1)
    search.fit(train, train["label"], groups=train["prompt_id"])
    return search.best_estimator_, {"cv_auc": float(search.best_score_), "params": search.best_params_}


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df)[:, 1]


def evaluate(model: Pipeline, df: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """Metrics overall and per genre, with prompt-grouped bootstrap intervals and length-matched AUC."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    p = predict_proba(model, df)
    rows = []
    for genre, idx in [("all", np.arange(len(df)))] + [(g, np.flatnonzero(df["genre"] == g)) for g in sorted(df["genre"].unique())]:
        y, pg, groups = df["label"].to_numpy()[idx], p[idx], df["prompt_id"].to_numpy()[idx]
        m = classification_metrics(y, pg)
        nan = (float("nan"), float("nan"))
        auc_lo, auc_hi = grouped_bootstrap_ci(y, pg, groups, roc_auc_score, n_boot, seed) if n_boot else nan
        acc_lo, acc_hi = grouped_bootstrap_ci(y, pg, groups, lambda a, b: accuracy_score(a, b >= 0.5), n_boot, seed) if n_boot else nan
        rows.append({"genre": genre, "n": len(idx), **m, "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
                     "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
                     "auc_length_matched": stratified_auc(pg, y, df["n_words"].to_numpy(dtype=float)[idx],
                                                            bins=int(np.clip(len(idx) // 50, 4, 16)))})  # fmt: skip
    return rows


def linear_coefficients(model: Pipeline) -> pd.DataFrame:
    """Standardized logistic coefficients: positive pushes toward 'AI', negative toward 'human'."""
    names = model.named_steps["features"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def permutation_table(model: Pipeline, val: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    """Drop in validation AUC when one feature's values are shuffled (model-agnostic importance).

    Only valid for variants whose inputs are exactly `features` (no POS n-gram text column).
    """
    res = permutation_importance(model, val[features], val["label"], scoring="roc_auc", n_repeats=5, random_state=seed)
    out = pd.DataFrame({"feature": features, "auc_drop": res.importances_mean, "auc_drop_std": res.importances_std})
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def run_baseline(cfg: dict[str, Any], variants: list[str] | None = None) -> pd.DataFrame:
    seed = cfg["seed"]
    results, models_dir = resolve_path(cfg, "results"), Path(resolve_path(cfg, "results")).parent / "models"
    models_dir.mkdir(exist_ok=True)
    train, val = load_split_features(cfg, "train"), load_split_features(cfg, "val")
    print(f"train {len(train)} texts, val {len(val)} texts (genres: {sorted(train['genre'].unique())})")

    rows, fitted = [], {}
    for name in variants or list(VARIANTS):
        model, info = fit_variant(name, train, seed)
        fitted[name] = model
        joblib.dump({"name": name, "model": model, "variant": VARIANTS[name]}, models_dir / f"{name}.joblib")
        for split, df in (("train", train), ("val", val)):
            for r in evaluate(model, df, n_boot=500 if split == "val" else 0, seed=seed):
                rows.append({"variant": name, "label": VARIANTS[name].label, "split": split, "cv_auc": info["cv_auc"],
                             "params": str(info["params"]), **r})  # fmt: skip
        v = [r for r in rows if r["variant"] == name and r["split"] == "val" and r["genre"] == "all"][0]
        print(f"{name:13s} cv_auc={info['cv_auc']:.3f}  val acc={v['accuracy']:.3f} auc={v['roc_auc']:.3f} "
              f"(length-matched {v['auc_length_matched']:.3f})  params={info['params']}")  # fmt: skip

    table = pd.DataFrame(rows)
    table.round(4).to_csv(results / "baseline_metrics.csv", index=False)

    for name in ("lr_style", "lr_full"):
        if name in fitted:
            linear_coefficients(fitted[name]).round(4).to_csv(results / f"coefficients_{name}.csv", index=False)
    _make_figures(fitted, table, val, results / "figures")
    if "gbm_full" in fitted:
        feats = model_features(VARIANTS["gbm_full"].groups)
        permutation_table(fitted["gbm_full"], val, feats, seed).round(4).to_csv(results / "importance_gbm_full.csv", index=False)
    return table


def _make_figures(fitted: dict[str, Pipeline], table: pd.DataFrame, val: pd.DataFrame, figures: Path) -> None:
    from hd.eval.detector_plots import plot_confusion, plot_roc

    figures.mkdir(parents=True, exist_ok=True)
    shown = {k: fitted[k] for k in ("lr_style", "lr_full", "gbm_full") if k in fitted}
    labels = {k: VARIANTS[k].label for k in shown}
    plot_roc(shown, labels, val, figures / "baseline_roc.png", "Baseline detector: ROC on the validation split")
    best = table[(table["split"] == "val") & (table["genre"] == "all")].sort_values("roc_auc").iloc[-1]["variant"]
    plot_confusion(fitted[best], val, figures / "baseline_confusion.png",
                   f"Confusion matrix, validation split: {VARIANTS[best].label}")  # fmt: skip
