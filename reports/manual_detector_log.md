# Manual commercial-detector test log

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
