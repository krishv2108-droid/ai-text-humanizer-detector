"""Score human vs AI texts with the independent detector and compare with our own detector.

Validation split only (the test split stays locked until the final evaluation).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hd.config import resolve_path
from hd.eval.metrics import threshold_at_fpr


TARGET_FPR = 0.05


def detection_summary(df: pd.DataFrame, score_col: str, calibrated: float, threshold: float = 0.5) -> dict[str, float]:
    """Detection at the default 0.5 cutoff and at `calibrated` (the 5%-false-positive cutoff)."""
    ai, human = df[df["label"] == 1][score_col], df[df["label"] == 0][score_col]
    return {
        "n": len(df),
        "roc_auc": roc_auc_score(df["label"], df[score_col]),
        "ai_detected_at_0.5": float((ai >= threshold).mean()),  # recall on AI text
        "human_flagged_at_0.5": float((human >= threshold).mean()),  # false-positive rate on human text
        "calibrated_threshold": calibrated,
        "ai_detected_at_fpr5": float((ai > calibrated).mean()),
        "human_flagged_at_fpr5": float((human > calibrated).mean()),
        "mean_score_ai": float(ai.mean()),
        "mean_score_human": float(human.mean()),
    }


def run_external_eval(cfg: dict[str, Any]) -> pd.DataFrame:
    from hd.detector.baseline import load_split_features, predict_proba
    from hd.detector.external import DesklibDetector
    import joblib

    processed, results = resolve_path(cfg, "processed"), resolve_path(cfg, "results")
    val = load_split_features(cfg, "val")
    ours = joblib.load(resolve_path(cfg, "results").parent / "models" / "gbm_full.joblib")["model"]
    val = val.assign(ours=predict_proba(ours, val))

    cache = processed / "external_val_scores.parquet"
    cached = pd.read_parquet(cache) if cache.exists() else None
    if cached is not None and set(val["id"]) <= set(cached["id"]):
        val = val.merge(cached[["id", "external"]], on="id")
    else:
        ext = DesklibDetector()
        print(f"scoring {len(val)} validation texts with {ext.name} on {ext.device}")
        val = val.assign(external=ext.predict_proba(val["text"].tolist()))
    val[["id", "genre", "label", "ours", "external"]].to_parquet(cache, index=False)

    # 5%-false-positive cutoffs, chosen on validation HUMAN texts per genre; later evaluations on
    # unseen data reuse these frozen numbers (so their false-positive rate there is an honest result).
    names = {"ours": "ours (gbm_full)", "external": "desklib"}
    thresholds = {n: {genre: threshold_at_fpr(g.loc[g["label"] == 0, n], TARGET_FPR) for genre, g in val.groupby("genre")} for n in names}
    for n in names:
        thresholds[n]["all"] = threshold_at_fpr(val.loc[val["label"] == 0, n], TARGET_FPR)
    (results / "detector_thresholds.json").write_text(json.dumps({"target_fpr": TARGET_FPR, "source": "validation human texts", **thresholds}, indent=2))

    rows = []
    for genre, g in [("all", val)] + list(val.groupby("genre")):
        for name, label in names.items():
            rows.append({"genre": genre, "detector": label, **detection_summary(g, name, thresholds[name][genre])})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(results / "external_metrics.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("score correlation (Spearman) ours vs external:", round(val["ours"].corr(val["external"], method="spearman"), 3))

    from hd.eval.detector_plots import plot_score_distributions

    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    plot_score_distributions(val, "external", figures / "external_scores.png", "Independent detector (desklib): score by true label, validation split")
    return out
