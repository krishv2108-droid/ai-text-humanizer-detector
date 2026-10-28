"""Sentence-level AI-likeness: which sentences of a text look most machine-written?

Each sentence is described by (a) how predictable GPT-2 finds its words *given the text before it*,
and (b) cheap local cues (transition opener, hedges, stock phrases, contractions, length relative to the
text's average). A small logistic regression turns those into P(AI | sentence).

CAVEAT (weak supervision): a sentence inherits the label of its document, so every sentence of an AI
text counts as "AI" and every sentence of a human text as "human". Plenty of human sentences are
plainly predictable and some AI sentences are quirky, so this model cannot be near-perfect and its
scores are for ranking sentences within a text, not for certifying any single sentence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hd.features.lexicons import FIRST_PERSON
from hd.features.stylometric import count_contractions, count_hedges, count_stock_phrases, starts_with_transition
from hd.features.text import is_word, sentences

SENTENCE_FEATURES = [
    "s_mean_surprisal",  # average GPT-2 surprise of the sentence's tokens, in context
    "s_max_surprisal",  # its single most surprising token
    "s_top10_frac",  # share of tokens inside GPT-2's top-10 guesses
    "s_top1_frac",
    "s_len_words",
    "s_len_rel",  # length / the text's average sentence length (AI rhythm is more uniform)
    "s_starts_transition",
    "s_hedges",
    "s_stock",
    "s_has_contraction",
    "s_commas",
    "s_first_person",
    "s_excl_q",
]


def sentence_rows(text: str, doc, stats: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Features for each sentence of `text`. `stats` is GPT-2's `token_stats` entry for the same text.

    Only the first 512 GPT-2 tokens are scored, so sentences beyond that get NaN GPT-2 features.
    """
    spans = sentences(doc)
    if not spans:
        return []
    lens = [sum(1 for t in s if is_word(t)) for s in spans]
    mean_len = float(np.mean(lens))
    last_char = stats["offsets"][:, 1] - 1  # last character of each scored token
    rows = []
    for k, (span, n_words) in enumerate(zip(spans, lens)):
        sel = (last_char >= span.start_char) & (last_char < span.end_char)
        if sel.any():
            surprisal, rank = stats["surprisal"][sel], stats["rank"][sel]
            gpt = {"s_mean_surprisal": float(surprisal.mean()), "s_max_surprisal": float(surprisal.max()),
                   "s_top10_frac": float((rank <= 10).mean()), "s_top1_frac": float((rank == 1).mean())}  # fmt: skip
        else:
            gpt = {k_: math.nan for k_ in ("s_mean_surprisal", "s_max_surprisal", "s_top10_frac", "s_top1_frac")}
        words = [t.text.lower() for t in span if is_word(t)]
        rows.append(
            {
                "sent_idx": k,
                "start": span.start_char,
                "end": span.end_char,
                "text": span.text.strip(),
                **gpt,
                "s_len_words": n_words,
                "s_len_rel": n_words / mean_len,
                "s_starts_transition": float(starts_with_transition(span.text)),
                "s_hedges": count_hedges(span.text),
                "s_stock": count_stock_phrases(span.text),
                "s_has_contraction": float(count_contractions(span) > 0),
                "s_commas": span.text.count(","),
                "s_first_person": float(any(w in FIRST_PERSON for w in words)),
                "s_excl_q": span.text.count("!") + span.text.count("?"),
            }
        )
    return rows


def build_sentence_table(df: pd.DataFrame, scorer, nlp, meta_cols=("id", "prompt_id", "label", "genre", "split")) -> pd.DataFrame:
    """One row per sentence for every text in `df` (needs the GPT-2 scorer; ~3 min for 5k texts)."""
    texts = df["text"].tolist()
    all_stats = scorer.token_stats(texts)
    rows = []
    for (_, meta), text, doc, stats in zip(df[list(meta_cols)].iterrows(), texts, nlp.pipe(texts, batch_size=64), all_stats):
        for r in sentence_rows(text, doc, stats):
            rows.append({**meta.to_dict(), **r})
    return pd.DataFrame(rows)


def fit_sentence_model(train: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict[str, Any]]:
    train = train[train["s_mean_surprisal"].notna()]
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])  # fmt: skip
    search = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, cv=GroupKFold(n_splits=5), scoring="roc_auc")
    search.fit(train[SENTENCE_FEATURES], train["label"], groups=train["prompt_id"])
    return search.best_estimator_, {"cv_auc": float(search.best_score_), "params": search.best_params_}


def sentence_coefficients(model: Pipeline) -> pd.DataFrame:
    coefs = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": SENTENCE_FEATURES, "coef": coefs})
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def run_sentence(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the sentence table (train + val only), fit the model, report on val sentences."""
    import joblib
    from sklearn.metrics import roc_auc_score

    from hd.config import resolve_path
    from hd.data.splits import load_split
    from hd.features.perplexity import GPT2Scorer
    from hd.features.text import get_nlp

    processed, results = resolve_path(cfg, "processed"), resolve_path(cfg, "results")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    corpus = corpus[corpus["has_pair"]]
    texts = pd.concat([load_split(corpus, "train"), load_split(corpus, "val")])  # test stays locked

    table_path = processed / "sentences.parquet"
    if table_path.exists() and set(pd.read_parquet(table_path, columns=["id"])["id"]) >= set(texts["id"]):
        table = pd.read_parquet(table_path)
    else:
        scorer = GPT2Scorer()
        print(f"scoring sentences of {len(texts)} texts with GPT-2 on {scorer.device}")
        table = build_sentence_table(texts, scorer, get_nlp())
        table.to_parquet(table_path, index=False)

    train, val = table[table["split"] == "train"], table[table["split"] == "val"]
    val = val[val["s_mean_surprisal"].notna()]
    model, info = fit_sentence_model(train)
    joblib.dump({"model": model, "features": SENTENCE_FEATURES}, resolve_path(cfg, "results").parent / "models" / "sentence_lr.joblib")

    val = val.assign(p=model.predict_proba(val[SENTENCE_FEATURES])[:, 1])
    rows = []
    for genre, g in [("all", val)] + list(val.groupby("genre")):
        doc_p = g.groupby("id").agg(p=("p", "mean"), label=("label", "first"))
        rows.append({"genre": genre, "sentences": len(g), "sentence_auc": roc_auc_score(g["label"], g["p"]),
                     "doc_auc_from_mean_sentence_p": roc_auc_score(doc_p["label"], doc_p["p"]), "cv_auc": info["cv_auc"]})  # fmt: skip
    out = pd.DataFrame(rows)
    out.round(4).to_csv(results / "sentence_metrics.csv", index=False)
    sentence_coefficients(model).round(3).to_csv(results / "coefficients_sentence.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(sentence_coefficients(model).round(2).head(8).to_string(index=False))
    return out
