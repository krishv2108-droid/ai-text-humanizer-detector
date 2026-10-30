"""Adversarial retraining, round 1: does training on humanizer rewrites make our detector catch them?

Our detector misses most rewrites that desklib, fakespot and ZeroGPT catch. The classic defense is to add the attacker's
output to the training data as AI examples. This module does exactly that and measures it honestly:

  * TRAIN: the train split as before, plus every rewrite found in TRAIN-split loop logs (labelled AI; they inherit the prompt
    of their source story, so cross-validation never puts a story and its rewrite on opposite sides).
  * EVALUATE on the VALIDATION split only: human stories, raw AI stories, and every rewrite in the validation loop logs. The
    5%-false-positive cutoff is set on validation human stories, once per model, exactly as for the original detector.
  * The original model is never overwritten; the retrained one is saved as models/gbm_full_adv.joblib.
  * The test split is not touched. Validation rewrites were made against the OLD detector, so this measures a defense against
    past attacks; a humanizer that adapts to the new detector (round 2) is a separate experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hd.config import resolve_path
from hd.detector.baseline import VARIANTS, fit_variant, load_split_features, model_features, predict_proba
from hd.eval.loop_report import wilson_ci
from hd.eval.metrics import threshold_at_fpr

VARIANT = "gbm_full"


def collect_rewrites(logs: list[Path], corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct rewrite text in the logs: id, prompt_id, genre, label 1, text, n_words, kept, source."""
    meta = corpus.set_index("id")[["prompt_id", "genre"]]
    rows: dict[str, dict[str, Any]] = {}
    for log in logs:
        for line in open(log):
            rec = json.loads(line)
            if rec["id"] not in meta.index:
                continue
            for step in rec["steps"][1:]:
                text = step["text"]
                rid = "rw:" + hashlib.sha1(text.encode()).hexdigest()[:14]
                kept = rec["final"] == step["iteration"]
                if rid in rows:
                    rows[rid]["kept"] = rows[rid]["kept"] or kept
                    continue
                rows[rid] = {"id": rid, "source_id": rec["id"], "prompt_id": meta.loc[rec["id"], "prompt_id"], "genre": meta.loc[rec["id"], "genre"],
                             "label": 1, "text": text, "n_words": len(text.split()), "kept": kept, "source": log.stem,
                             "external": step.get("external"), "ext_cutoff": rec.get("ext_cutoff"),
                             "judge": step.get("judge"), "judge_cutoff": rec.get("judge_cutoff")}  # fmt: skip
    return pd.DataFrame(list(rows.values()))


def with_features(rewrites: pd.DataFrame, cache: Path, scorer) -> pd.DataFrame:
    from hd.features.pipeline import compute_features

    feats = compute_features(rewrites, cache_path=cache, scorer=scorer)
    return rewrites.merge(feats, on="id", how="inner")


def _flag_stats(scores: np.ndarray, cutoff: float) -> dict[str, Any]:
    k = int((scores > cutoff).sum())
    lo, hi = wilson_ci(k, len(scores))
    return {"n": len(scores), "flagged": k, "rate": k / len(scores) if len(scores) else float("nan"), "lo": lo, "hi": hi}


def run_adversarial(cfg: dict[str, Any], processed: Path | None = None) -> pd.DataFrame:
    from hd.features.perplexity import GPT2Scorer

    processed = processed or resolve_path(cfg, "processed")
    seed = cfg["seed"]
    corpus = pd.read_parquet(processed / "corpus.parquet")
    scorer = GPT2Scorer()
    logs = {s: sorted(processed.glob(f"loop_{s}_story_*qwen*.jsonl")) for s in ("train", "val")}
    train_rw = with_features(collect_rewrites(logs["train"], corpus), processed / "features_rewrites_train.parquet", scorer)
    val_rw = with_features(collect_rewrites(logs["val"], corpus), processed / "features_rewrites_val.parquet", scorer)
    print(f"train rewrites {len(train_rw)} (from {train_rw['source_id'].nunique()} stories); val rewrites {len(val_rw)} (from {val_rw['source_id'].nunique()} stories)")

    train, val = load_split_features(cfg, "train"), load_split_features(cfg, "val")
    cols = list(train.columns)
    augmented = pd.concat([train, train_rw[[c for c in cols if c in train_rw.columns]]], ignore_index=True)

    models_dir = Path(resolve_path(cfg, "results")).parent / "models"
    old = joblib.load(models_dir / f"{VARIANT}.joblib")["model"]
    new, info = fit_variant(VARIANT, augmented, seed)
    joblib.dump({"name": VARIANT + "_adv", "model": new, "variant": VARIANTS[VARIANT]}, models_dir / f"{VARIANT}_adv.joblib")
    print(f"retrained on {len(augmented)} texts ({len(train_rw)} rewrites); cv_auc {info['cv_auc']:.4f}")

    story = val[val["genre"] == "story"]
    human, raw = story[story["label"] == 0], story[story["label"] == 1]
    rows = []
    for name, model in (("original", old), ("retrained", new)):
        cut = threshold_at_fpr(predict_proba(model, human), 0.05)
        groups = {"raw AI stories": raw, "all rewrites": val_rw, "kept rewrites (passed guardrails)": val_rw[val_rw["kept"]]}
        for gname, df in groups.items():
            rows.append({"detector": name, "cutoff": cut, "texts": gname, **_flag_stats(predict_proba(model, df), cut)})
        rows.append({"detector": name, "cutoff": cut, "texts": "human stories (false positives)", **_flag_stats(predict_proba(model, human), cut)})
    # reference: the independent detectors on the same rewrites, from the loop logs (their own frozen cutoffs)
    for det, score, cutcol in (("desklib", "external", "ext_cutoff"), ("fakespot", "judge", "judge_cutoff")):
        have = val_rw.dropna(subset=[score, cutcol])
        for gname, df in {"all rewrites": have, "kept rewrites (passed guardrails)": have[have["kept"]]}.items():
            k = int((df[score] > df[cutcol]).sum())
            lo, hi = wilson_ci(k, len(df))
            rows.append({"detector": det, "cutoff": float("nan"), "texts": gname, "n": len(df), "flagged": k, "rate": k / len(df) if len(df) else float("nan"), "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)
    results = resolve_path(cfg, "results")
    table.round(4).to_csv(results / "adversarial_round1.csv", index=False)
    return table
