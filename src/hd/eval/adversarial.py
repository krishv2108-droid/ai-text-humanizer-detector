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


# ---- the one-time look at the test split ---------------------------------------------------------------------------------

TEST_MARK = "adversarial-test"


def run_adversarial_test(cfg: dict[str, Any], loop_logs: list[Path], *, again: bool = False) -> pd.DataFrame:
    """Score the retrained detector (and the original, for comparison) on the TEST split, once.

    Everything was fixed before this ran: the model is the one `run_adversarial` saved (trained on train data only) and each
    model's cutoff is the 5%-false-positive point on VALIDATION human stories. Refuses a second run unless `again`; the log
    records every unlock. This is the second time the test split is read (the first was the main final evaluation), and
    FINDINGS.md says so.
    """
    from hd.data.splits import load_split
    from hd.eval.final import access_log_path, log_test_access
    from hd.features.perplexity import GPT2Scorer

    log = access_log_path(cfg)
    runs = sum(TEST_MARK in line for line in log.read_text().splitlines()) if log.exists() else 0
    if runs and not again:
        raise RuntimeError(f"the retrained detector was already scored on the test split ({runs} time(s), see {log.name}); pass --again only to repeat on purpose")
    processed, models_dir = resolve_path(cfg, "processed"), Path(resolve_path(cfg, "results")).parent / "models"
    old = joblib.load(models_dir / f"{VARIANT}.joblib")["model"]
    new = joblib.load(models_dir / f"{VARIANT}_adv.joblib")["model"]
    val = load_split_features(cfg, "val")
    human_val = val[(val["genre"] == "story") & (val["label"] == 0)]
    cutoffs = {n: threshold_at_fpr(predict_proba(m, human_val), 0.05) for n, m in (("original", old), ("retrained", new))}

    corpus = pd.read_parquet(processed / "corpus.parquet")
    log_test_access(cfg, f"{TEST_MARK}{' (repeat, --again)' if runs else ''}")
    test = load_split(corpus[corpus["has_pair"]], "test", unlock_test=True).merge(pd.read_parquet(processed / "features.parquet"), on="id", how="inner")
    story = test[test["genre"] == "story"]
    human, raw = story[story["label"] == 0], story[story["label"] == 1]
    rewrites = with_features(collect_rewrites(loop_logs, corpus), processed / "features_rewrites_test.parquet", GPT2Scorer())
    print(f"test: {len(human)} human / {len(raw)} raw AI stories, {len(rewrites)} rewrites of {rewrites['source_id'].nunique()} stories; cutoffs {cutoffs}")

    rows = []
    for name, model in (("original", old), ("retrained", new)):
        cut = cutoffs[name]
        for gname, df in {"raw AI stories": raw, "all rewrites": rewrites, "kept rewrites (passed guardrails)": rewrites[rewrites["kept"]],
                          "human stories (false positives)": human}.items():  # fmt: skip
            rows.append({"detector": name, "cutoff": cut, "texts": gname, **_flag_stats(predict_proba(model, df), cut)})
        # what the user actually receives per story: the kept rewrite, or the original if no rewrite passed the guardrails
        recs = [json.loads(line) for p in loop_logs for line in open(p)]
        kept = rewrites[rewrites["kept"]].drop_duplicates("source_id").set_index("source_id")
        delivered = pd.concat([kept.loc[[r["id"]]] if r["id"] in kept.index else raw[raw["id"] == r["id"]].set_index("id") for r in recs])
        rows.append({"detector": name, "cutoff": cut, "texts": "delivered text per story (kept rewrite, else original)", **_flag_stats(predict_proba(model, delivered), cut)})
    for det, score, cutcol in (("desklib", "external", "ext_cutoff"), ("fakespot", "judge", "judge_cutoff")):
        have = rewrites.dropna(subset=[score, cutcol])
        for gname, df in {"all rewrites": have, "kept rewrites (passed guardrails)": have[have["kept"]]}.items():
            k = int((df[score] > df[cutcol]).sum())
            lo, hi = wilson_ci(k, len(df))
            rows.append({"detector": det, "cutoff": float("nan"), "texts": gname, "n": len(df), "flagged": k, "rate": k / len(df) if len(df) else float("nan"), "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)
    out = resolve_path(cfg, "results") / "final"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "adversarial_test.csv", index=False)
    return table
