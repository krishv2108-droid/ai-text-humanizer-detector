"""The adversarial loop: rewrite -> score -> feedback -> rewrite, with a per-iteration log.

Rules that keep the experiment honest:
  * Only OUR detector steers the loop (its analysis becomes the next prompt's feedback; it decides when to
    stop and which iteration to keep). The independent detector is scored at every step for the record but
    never influences a decision, so its numbers stay a fair, unoptimized measurement.
  * "Flagged" means above the frozen 5%-false-positive cutoff for that detector and genre, never 0.5.
  * Guardrails always compare against the ORIGINAL draft, so meaning can't drift a little each pass. A rewrite
    that fails them is discarded (the next attempt starts from the last accepted draft and is told why).
  * The kept rewrite is the guardrail-passing iteration with the lowest score from our detector. If none
    passes, the user gets no rewrite, and the summary counts the text as unchanged (still flagged if it was).
  * Every draft gets at least one rewrite, even one our detector already passes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from hd.data.splits import load_split, stable_hash01
from hd.llm import RateLimited


def cutoff_for(thresholds: dict[str, dict[str, float]], detector: str, genre: str) -> float:
    """The frozen 5%-FPR cutoff. A missing genre is an error: silently borrowing another genre's cutoff would
    quietly change what 'fooled the detector' means."""
    try:
        return float(thresholds[detector][genre])
    except KeyError:
        raise RuntimeError(
            f"no frozen {detector!r} cutoff for genre {genre!r} in results/detector_thresholds.json; "
            "rebuild the corpus and features, then run `hd detector eval-external`"
        ) from None


@dataclass
class Step:
    iteration: int  # 0 is the original AI draft
    text: str
    ours: float  # our detector's P(AI)
    external: float | None  # independent detector's P(AI); recorded, never used to decide anything
    guardrails_ok: bool | None = None  # None for the original
    flags: list[str] = field(default_factory=list)
    similarity: float | None = None
    length_ratio: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    retry_note: str = ""


@dataclass
class LoopResult:
    id: str
    genre: str
    strategy: str
    our_cutoff: float
    ext_cutoff: float | None
    steps: list[Step]
    final: int | None  # iteration kept, or None if no rewrite passed the guardrails
    stop_reason: str  # success | max_iterations | stalled | error: ...
    humanizer_model: str = ""  # which LLM wrote the rewrites; results from different models must not be pooled

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_final(steps: list[Step]) -> int | None:
    """The accepted rewrite our detector likes best (earliest on ties)."""
    ok = [s for s in steps[1:] if s.guardrails_ok]
    return min(ok, key=lambda s: (s.ours, s.iteration)).iteration if ok else None


def _retry_note(flags: list[str]) -> str:
    return ("Your previous attempt was rejected: " + "; ".join(flags) + ". Stay closer to the original: keep every event "
            "and idea, keep a similar length, and add nothing new.")  # fmt: skip


def run_loop(
    detector,
    humanizer,
    checker,
    text: str,
    genre: str,
    *,
    strategy: str = "full",
    our_cutoff: float,
    ext_cutoff: float | None = None,
    external=None,
    max_iterations: int = 4,
    item_id: str = "",
) -> LoopResult:
    def ext_score(t: str) -> float | None:
        return None if external is None else float(external.predict_proba([t])[0])

    base = detector.analyze(text)
    original = base.text
    steps = [Step(0, original, base.prob_ai, ext_score(original))]
    current, current_analysis, note = original, base, None  # the last ACCEPTED draft
    seen, stop = {original}, "max_iterations"

    for i in range(1, max_iterations + 1):
        try:
            rewrite = humanizer.humanize(current, genre, strategy, analysis=current_analysis, original=original, retry_note=note)
        except RateLimited:
            raise  # the batch runner stops cleanly; this text is retried (cheaply, via the cache) next run
        except RuntimeError as err:  # refusal, empty output, or cut-off: end this text's loop
            stop = f"error: {str(err)[:120]}"
            break
        if rewrite.text in seen:  # the model returned something we already scored: more iterations won't help
            stop = "stalled"
            break
        seen.add(rewrite.text)
        analysis = detector.analyze(rewrite.text)
        guard = checker.check(original, analysis.text)
        steps.append(Step(i, analysis.text, analysis.prob_ai, ext_score(analysis.text), guard.ok, list(guard.flags),
                          guard.similarity, guard.length_ratio, rewrite.input_tokens, rewrite.output_tokens,
                          rewrite.cached, note or ""))  # fmt: skip
        if guard.ok:
            current, current_analysis, note = analysis.text, analysis, None
            if analysis.prob_ai <= our_cutoff:
                stop = "success"
                break
        else:
            note = _retry_note(guard.flags)

    return LoopResult(item_id, genre, strategy, our_cutoff, ext_cutoff, steps, choose_final(steps), stop,
                      getattr(humanizer, "model", ""))  # fmt: skip


def select_items(corpus: pd.DataFrame, split: str, genre: str, n: int, seed: int, *, unlock_test: bool = False,
                 access_log: Path | None = None) -> pd.DataFrame:  # fmt: skip
    """A deterministic sample of AI texts: same seed and n always give the same items, and a larger n extends it."""
    part = load_split(corpus, split, unlock_test=unlock_test, access_log=access_log)
    part = part[(part["label"] == 1) & (part["genre"] == genre) & part["has_pair"]].copy()
    part["_order"] = [stable_hash01(seed, "loop", i) for i in part["id"]]
    return part.sort_values("_order").head(n).drop(columns="_order").reset_index(drop=True)


def run_batch(items: Iterable[dict[str, Any]], out_path: Path, run_one: Callable[[dict[str, Any]], LoopResult]) -> dict[str, Any]:
    """Run the loop over many texts, appending one JSON line each. Resumable; stops cleanly at a rate limit."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    stats: dict[str, Any] = {"written": 0, "skipped_done": 0, "stopped_by_limit": False, "message": ""}
    with open(out_path, "a") as out:
        for item in items:
            if item["id"] in done:
                stats["skipped_done"] += 1
                continue
            try:
                result = run_one(item)
            except RateLimited as err:
                stats["stopped_by_limit"] = True
                wait = f" (retry in about {err.retry_in_s / 60:.0f} min)" if err.retry_in_s else ""
                stats["message"] = f"stopped at a rate limit{wait}: {str(err)[:200]}"
                break
            out.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            out.flush()
            stats["written"] += 1
    return stats


def load_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path)]


def _after(rec: dict[str, Any], key: str) -> float | None:
    """The score a user would end up with: the kept rewrite's, or the unchanged original's if none was kept."""
    steps = rec["steps"]
    return steps[rec["final"] if rec["final"] is not None else 0][key]


def summarize(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (genre, strategy, humanizer model): detection rate before and after, per detector, plus acceptance."""
    rows = []
    frame = pd.DataFrame({"genre": [r["genre"] for r in records], "strategy": [r["strategy"] for r in records],
                          "model": [r.get("humanizer_model", "") for r in records]})  # fmt: skip
    for (genre, strategy, model), idx in frame.groupby(["genre", "strategy", "model"]).groups.items():
        recs = [records[i] for i in idx]
        row: dict[str, Any] = {"genre": genre, "strategy": strategy, "model": model, "n": len(recs)}
        row["ours_flagged_before"] = sum(r["steps"][0]["ours"] > r["our_cutoff"] for r in recs) / len(recs)
        row["ours_flagged_after"] = sum(_after(r, "ours") > r["our_cutoff"] for r in recs) / len(recs)
        if all(r["ext_cutoff"] is not None and r["steps"][0]["external"] is not None for r in recs):
            row["external_flagged_before"] = sum(r["steps"][0]["external"] > r["ext_cutoff"] for r in recs) / len(recs)
            row["external_flagged_after"] = sum(_after(r, "external") > r["ext_cutoff"] for r in recs) / len(recs)
        row["rewrite_accepted"] = sum(r["final"] is not None for r in recs) / len(recs)
        row["mean_rewrites_tried"] = sum(len(r["steps"]) - 1 for r in recs) / len(recs)
        kept = [r["steps"][r["final"]]["similarity"] for r in recs if r["final"] is not None]
        row["mean_similarity_kept"] = sum(kept) / len(kept) if kept else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def iteration_curve(records: list[dict[str, Any]], key: str = "ours") -> pd.DataFrame:
    """Mean score after k rewrites (k=0 is the original). A text that stopped early keeps its last score."""
    top = max(len(r["steps"]) for r in records)
    out = []
    for k in range(top):
        vals = [r["steps"][min(k, len(r["steps"]) - 1)][key] for r in records]
        vals = [v for v in vals if v is not None]
        out.append({"rewrites": k, f"mean_{key}": sum(vals) / len(vals) if vals else float("nan")})
    return pd.DataFrame(out)
