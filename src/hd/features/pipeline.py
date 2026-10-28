"""Compute features for a corpus table, caching by text id so reruns only do new rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hd.features.stylometric import extract_features, feature_names, pos_sequence
from hd.features.text import get_nlp


def stylometric_table(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    nlp = get_nlp()
    rows = []
    for text, doc in zip(df["text"], nlp.pipe(df["text"].tolist(), batch_size=batch_size)):
        rows.append({**extract_features(doc, text), "pos_seq": pos_sequence(doc)})
    return pd.DataFrame(rows, index=df.index)


def compute_features(df: pd.DataFrame, cache_path: Path | None = None, scorer=None) -> pd.DataFrame:
    """Return one row per `df` row: id, stylometric features, pos_seq, and (if `scorer`) GPT-2 features.

    Rows already in `cache_path` (same id) are reused. Perplexity is computed only when a scorer is
    given, and only for rows missing those columns.
    """
    cached = pd.read_parquet(cache_path) if cache_path and cache_path.exists() else None
    done = set(cached["id"]) if cached is not None else set()

    new = df[~df["id"].isin(done)]
    parts = [cached] if cached is not None else []
    if len(new):
        table = stylometric_table(new)
        table.insert(0, "id", new["id"].values)
        parts.append(table)
    feats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"id": []})

    if scorer is not None:
        ppl_cols = feature_names(["perplexity"])
        has_ppl = feats[ppl_cols[0]].notna() if ppl_cols[0] in feats else pd.Series(False, index=feats.index)
        todo_ids = feats.loc[~has_ppl, "id"]
        if len(todo_ids):
            texts = df.set_index("id").loc[todo_ids, "text"].tolist()
            ppl = pd.DataFrame(scorer.features(texts), index=feats.index[~has_ppl])
            for col in ppl_cols:
                if col not in feats:
                    feats[col] = float("nan")
                feats.loc[ppl.index, col] = ppl[col]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(cache_path, index=False)
    return feats[feats["id"].isin(df["id"])].reset_index(drop=True)
