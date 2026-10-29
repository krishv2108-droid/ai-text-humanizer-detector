"""The final held-out evaluation (M7). The test split is unlocked here and nowhere else.

Rules this module enforces:
  * every unlock of the test split is appended to `results/test_access_log.txt` with its purpose;
  * the final evaluation itself runs once. A second run needs an explicit `again=True` and says so in the log, because
    re-running until the numbers look good turns the test set into a second validation set;
  * thresholds are the ones frozen on validation human text (`results/detector_thresholds.json`); nothing is retuned.
Loop runs on test items are logged too (`log_test_access`), so a report can say how many attempts were made.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hd.config import resolve_path
from hd.data.splits import load_split
from hd.eval.loop_report import wilson_ci
from hd.eval.metrics import grouped_bootstrap_ci

LOG_NAME = "test_access_log.txt"
FINAL_MARK = "final-eval"


class TestSetAlreadyUsed(RuntimeError):
    __test__ = False  # not a pytest test class


def access_log_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "results") / LOG_NAME


def log_test_access(cfg: dict[str, Any], purpose: str) -> None:
    path = access_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} test split unlocked | {purpose}\n")


def final_eval_runs(cfg: dict[str, Any]) -> int:
    path = access_log_path(cfg)
    return sum(FINAL_MARK in line for line in path.read_text().splitlines()) if path.exists() else 0


def unlock_for_final(cfg: dict[str, Any], *, again: bool = False) -> pd.DataFrame:
    """The paired test texts with features. Refuses a repeat run unless `again`; logs the unlock either way."""
    runs = final_eval_runs(cfg)
    if runs and not again:
        raise TestSetAlreadyUsed(
            f"the final evaluation already ran {runs} time(s) (see results/{LOG_NAME}). Running it again and keeping the "
            "better result would make the test set a second validation set. Pass --again only to repeat on purpose; "
            "the log records it, and FINDINGS.md should say so."
        )
    processed = resolve_path(cfg, "processed")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    log_test_access(cfg, f"{FINAL_MARK}{' (repeat, --again)' if runs else ''}")
    part = load_split(corpus[corpus["has_pair"]], "test", unlock_test=True)
    return part.merge(pd.read_parquet(processed / "features.parquet"), on="id", how="inner")


def detector_test_table(scored: pd.DataFrame, thresholds: dict[str, dict[str, float]], n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    """Per genre (and 'all') and detector: AUC and the frozen-cutoff catch rate and false-positive rate, with intervals.

    `scored` needs genre, label, prompt_id and one score column per detector ('ours', 'external').
    """
    names = {"ours": "ours (gbm_full)", "external": "desklib"}
    rows = []
    for genre, part in [("all", scored)] + list(scored.groupby("genre")):
        y, groups = part["label"].to_numpy(), part["prompt_id"].to_numpy()
        for col, label in names.items():
            if col not in part or part[col].isna().all():
                continue
            s, cut = part[col].to_numpy(), thresholds[col][genre]
            flagged = s > cut
            n_ai, n_h = int((y == 1).sum()), int((y == 0).sum())
            k_ai, k_h = int(flagged[y == 1].sum()), int(flagged[y == 0].sum())
            lo, hi = grouped_bootstrap_ci(y, s, groups, roc_auc_score, n_boot=n_boot, seed=seed)
            rows.append({
                "genre": genre, "detector": label, "n_ai": n_ai, "n_human": n_h, "roc_auc": roc_auc_score(y, s), "auc_lo": lo, "auc_hi": hi,
                "cutoff": cut, "ai_caught": k_ai / n_ai, "caught_lo": wilson_ci(k_ai, n_ai)[0], "caught_hi": wilson_ci(k_ai, n_ai)[1],
                "human_flagged": k_h / n_h, "flagged_lo": wilson_ci(k_h, n_h)[0], "flagged_hi": wilson_ci(k_h, n_h)[1],
            })  # fmt: skip
    return pd.DataFrame(rows)


def run_final(cfg: dict[str, Any], loop_logs: list[Path], *, external: bool = True, again: bool = False) -> dict[str, Path]:
    """Detector table on the test split, plus a loop report per log of test-split runs. Writes to results/final/."""
    import joblib

    from hd.detector.baseline import predict_proba
    from hd.eval.loop_report import write_loop_report
    from hd.humanizer.pipeline import load_thresholds
    from hd.loop import load_results

    thresholds = load_thresholds(cfg)
    if not thresholds:
        raise RuntimeError("no frozen cutoffs: run `hd detector eval-external` first")
    test = unlock_for_final(cfg, again=again)
    ours = joblib.load(resolve_path(cfg, "results").parent / "models" / "gbm_full.joblib")["model"]
    scored = test.assign(ours=predict_proba(ours, test))
    if external:
        from hd.detector.external import DesklibDetector

        scored["external"] = DesklibDetector().predict_proba(scored["text"].tolist())
    out = resolve_path(cfg, "results") / "final"
    out.mkdir(parents=True, exist_ok=True)
    table = detector_test_table(scored, thresholds)
    written = {"detectors": out / "detector_test_metrics.csv"}
    table.round(4).to_csv(written["detectors"], index=False)
    (out / "thresholds_used.json").write_text(json.dumps(thresholds, indent=2))
    for log in loop_logs:
        records = load_results(log)
        tag = log.stem.removeprefix("loop_")
        written.update({f"{tag}:{k}": v for k, v in write_loop_report(records, out, tag, f"held-out test split, n={len(records)}").items()})
    return written
