"""A kit for testing the rewrites by hand against commercial AI detectors (their APIs are paid, so this is manual).

Writes blind sample files (neutral names, shuffled), a separate answer key, and an empty log to fill in while pasting
each sample into each tool. Only AI-generated texts and their rewrites are exported. Human controls must be texts the
tester wrote themself: pasting other people's writing (our human stories are Reddit posts) into third-party services
is not ours to do.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

LOG_COLUMNS = ["sample", "tool", "tool_version_or_mode", "date", "ai_score_percent", "verdict_as_shown", "notes"]

PROTOCOL = """# Manual commercial-detector test log

**Purpose.** Our two detectors are open-source and research-grade. Real-world checkers (GPTZero, Originality.ai,
Copyleaks, ZeroGPT, Turnitin, ...) use different models and thresholds. This log records how a few of them score the
raw AI texts and our rewrites. It is a small, hand-run sanity check, not a benchmark.

**How to run it**
1. `uv run hd eval manual-kit --log <loop log> --n 5` writes `reports/manual_kit/sample-XX.txt` (blind, shuffled)
   and `answer_key.csv`. Do not open the answer key until you have recorded every score.
2. Write **3 short texts of your own** (a paragraph each, in the genre being tested) and save them as
   `reports/manual_kit/sample-own-1.txt` ... These are your human controls. Do not use other people's writing.
3. For every tool you can use (free tiers are fine): paste each sample unchanged and record the tool's AI score and
   its verdict **exactly as shown**, the date, and the tool's mode/version, in `reports/manual_detector_log.csv`
   (one row per sample per tool).
4. Afterwards open `answer_key.csv`, add a `kind` column (raw_ai / humanized / own_human) and summarise per tool:
   how often raw AI, humanized and your own text were flagged.

**Read it carefully**
- With 5 raw + 5 rewrites + 3 controls, one text moves a rate by 8-10 points. Report counts, not just percentages.
- Commercial tools change their models without notice; record the date.
- If a tool flags your own writing, say so: that is the false-positive problem these tools are known for.
- Do not tune the rewrites against these tools and then report the same numbers; that would repeat the mistake of
  optimizing against the measuring instrument.
"""


def build_manual_kit(records: list[dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> list[dict[str, str]]:
    """Export up to n raw/rewrite pairs as blind samples. Returns the answer key rows."""
    usable = [r for r in records if r["final"] is not None]
    rng = random.Random(seed)
    rng.shuffle(usable)
    texts: list[tuple[str, str, str]] = []  # (record id, kind, text)
    for r in usable[:n]:
        texts.append((r["id"], "raw_ai", r["steps"][0]["text"]))
        texts.append((r["id"], "humanized", r["steps"][r["final"]]["text"]))
    rng.shuffle(texts)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = []
    for i, (rid, kind, text) in enumerate(texts, 1):
        name = f"sample-{i:02d}"
        (out_dir / f"{name}.txt").write_text(text + "\n")
        key.append({"sample": name, "record_id": rid, "kind": kind, "genre": next(r["genre"] for r in records if r["id"] == rid)})
    with open(out_dir / "answer_key.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "record_id", "kind", "genre"])
        writer.writeheader()
        writer.writerows(key)
    return key


def write_log_template(path: Path, samples: list[str], own_controls: int = 3) -> None:
    """An empty CSV with one row per sample (plus the tester's own controls); the tool column is filled in by hand."""
    names = samples + [f"sample-own-{i}" for i in range(1, own_controls + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows({"sample": s} for s in names)
