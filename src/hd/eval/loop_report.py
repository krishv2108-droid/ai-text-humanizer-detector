"""Tables and figures from adversarial-loop logs (the JSONL written by `hd loop run`).

Everything here is a pure function of the log records, so results regenerate deterministically. Rates come
with Wilson 95% intervals: with a few dozen texts per cell, the interval is part of the result, not decoration.

Two outcome policies are reported side by side:
  kept       what a user would get: the guardrail-passing rewrite our detector likes best, or the unchanged
             original if no rewrite passed (so a failed rewrite is never credited as a success);
  unguarded  the rewrite our detector likes best whether or not it passed the guardrails. The gap between the two
             is the price of keeping the meaning.
Neither policy looks at the independent detector, so its numbers stay an unoptimized measurement.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hd.eval.style import AI_C, AXIS, GRID, HUMAN_C, INK, INK2, MUTED, SERIES, SURFACE, style_axes

# detector label -> (score key in a step, cutoff key in a record)
DETECTORS = {"ours": ("ours", "our_cutoff"), "desklib": ("external", "ext_cutoff")}
POLICIES = ("kept", "unguarded")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (well behaved at small n and at 0% / 100%)."""
    if n == 0:
        return (math.nan, math.nan)
    p, denom = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def outcome_step(rec: dict[str, Any], policy: str = "kept") -> dict[str, Any]:
    """The step whose scores represent this text after humanizing, under a policy."""
    steps = rec["steps"]
    if policy == "kept":
        return steps[rec["final"]] if rec["final"] is not None else steps[0]
    if policy == "unguarded":
        rewrites = steps[1:]
        return min(rewrites, key=lambda s: (s["ours"], s["iteration"])) if rewrites else steps[0]
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


def _flagged(rec: dict[str, Any], step: dict[str, Any], detector: str) -> bool | None:
    score_key, cut_key = DETECTORS[detector]
    if step[score_key] is None or rec[cut_key] is None:
        return None
    return bool(step[score_key] > rec[cut_key])  # flagged = strictly above the frozen cutoff


def outcome_table(records: list[dict[str, Any]], policy: str = "kept") -> pd.DataFrame:
    """One row per (genre, strategy, humanizer model, detector): flagged rate before and after, with intervals."""
    frame = pd.DataFrame({"genre": [r["genre"] for r in records], "strategy": [r["strategy"] for r in records],
                          "model": [r.get("humanizer_model", "") for r in records]})  # fmt: skip
    rows = []
    for (genre, strategy, model), idx in frame.groupby(["genre", "strategy", "model"]).groups.items():
        recs = [records[i] for i in idx]
        for det in DETECTORS:
            pairs = [(_flagged(r, r["steps"][0], det), _flagged(r, outcome_step(r, policy), det)) for r in recs]
            if any(b is None or a is None for b, a in pairs):
                continue  # this detector was not run for these records
            n = len(pairs)
            kb, ka = sum(b for b, _ in pairs), sum(a for _, a in pairs)
            detected = [(b, a) for b, a in pairs if b]
            kept = [r["steps"][r["final"]] for r in recs if r["final"] is not None]
            rows.append({
                "genre": genre, "strategy": strategy, "model": model, "policy": policy, "detector": det, "n": n,
                "flagged_before": kb / n, "before_lo": wilson_ci(kb, n)[0], "before_hi": wilson_ci(kb, n)[1],
                "flagged_after": ka / n, "after_lo": wilson_ci(ka, n)[0], "after_hi": wilson_ci(ka, n)[1],
                "n_detected_before": len(detected),
                "evaded_of_detected": (sum(not a for _, a in detected) / len(detected)) if detected else math.nan,
                "rewrite_accepted": sum(r["final"] is not None for r in recs) / n,
                "mean_similarity_kept": float(np.mean([s["similarity"] for s in kept])) if kept else math.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def first_pass_table(by_strategy: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Compare prompt strategies at equal budget: each text's FIRST rewrite, scored before any guardrail decision.

    The loop's later iterations depend on the guardrails, so the first rewrite is the one place where every
    strategy has had exactly one call per text.
    """
    rows = []
    for strategy, recs in by_strategy.items():
        firsts = [(r, r["steps"][1]) for r in recs if len(r["steps"]) > 1]
        if not firsts:
            continue
        n = len(firsts)
        row: dict[str, Any] = {"strategy": strategy, "n": n}
        for det, (score_key, cut_key) in DETECTORS.items():
            if any(s[score_key] is None or r[cut_key] is None for r, s in firsts):
                continue
            k = sum(s[score_key] > r[cut_key] for r, s in firsts)
            row[f"{det}_flagged"] = k / n
            row[f"{det}_lo"], row[f"{det}_hi"] = wilson_ci(k, n)
            row[f"{det}_mean_score"] = float(np.mean([s[score_key] for _, s in firsts]))
        k_ok = sum(bool(s["guardrails_ok"]) for _, s in firsts)
        row["guardrails_ok"], (row["guardrails_lo"], row["guardrails_hi"]) = k_ok / n, wilson_ci(k_ok, n)
        row["mean_similarity"] = float(np.mean([s["similarity"] for _, s in firsts]))
        grades = [s["fk_grade"] for _, s in firsts if s.get("fk_grade") is not None]
        row["mean_reading_grade"] = float(np.mean(grades)) if grades else math.nan
        row["mean_tokens_per_call"] = float(np.mean([s["input_tokens"] + s["output_tokens"] for _, s in firsts]))
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floats: str = "{:.2f}", percent: tuple[str, ...] = ()) -> str:
    """A GitHub-flavoured table without extra dependencies. Columns in `percent` are shown as 0-100%."""
    def cell(col: str, v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "-"
            return f"{v * 100:.0f}%" if col in percent else floats.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(cell(c, v) for c, v in row.items()) + " |" for row in df.to_dict("records")) + "\n"


def interval_label(rate: float, lo: float, hi: float) -> str:
    return f"{rate:.0%} ({lo:.0%}-{hi:.0%})"


# ---- figures ------------------------------------------------------------------------------------------------

def _canvas(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    return fig


def _title(fig, text: str, sub: str | None = None) -> None:
    fig.text(0.01, 0.985, text, color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    if sub:
        fig.text(0.01, 0.935, sub, color=INK2, fontsize=9, ha="left", va="top")


def plot_before_after(table: pd.DataFrame, path: Path, title: str, subtitle: str | None = None) -> None:
    """Flagged rate before vs after humanizing, one panel per detector, with 95% intervals.

    `table` is `outcome_table` output filtered to one genre x strategy x model per row group; one cluster of bars
    per row, so pass one row per condition you want to compare.
    """
    dets = [d for d in DETECTORS if d in set(table["detector"])]
    fig = _canvas(4.6 * len(dets) + 0.6, 4.6)
    labels = [f"{r.genre}\n{r.strategy}" + (f"\n{r.model.split('/')[-1]}" if r.model else "") for r in table.drop_duplicates(["genre", "strategy", "model"]).itertuples()]
    groups = list(table.drop_duplicates(["genre", "strategy", "model"])[["genre", "strategy", "model"]].itertuples(index=False, name=None))
    for i, det in enumerate(dets):
        ax = fig.add_subplot(1, len(dets), i + 1)
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.grid(axis="x", visible=False)
        sub = table[table["detector"] == det]
        x = np.arange(len(groups))
        for off, key, color, name in ((-0.2, "before", AI_C, "raw AI"), (0.2, "after", HUMAN_C, "after loop")):
            vals, lo, hi = [], [], []
            for g in groups:
                row = sub[(sub["genre"] == g[0]) & (sub["strategy"] == g[1]) & (sub["model"] == g[2])].iloc[0]
                vals.append(row[f"flagged_{key}"]); lo.append(row[f"flagged_{key}"] - row[f"{key}_lo"]); hi.append(row[f"{key}_hi"] - row[f"flagged_{key}"])
            ax.bar(x + off, vals, width=0.36, color=color, label=name, zorder=3)
            ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
            for xi, v in zip(x + off, vals):
                ax.text(xi, min(v, 0.97) + 0.015 if v < 0.9 else v - 0.07, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK2 if v < 0.9 else SURFACE, zorder=5)
        ax.set_xticks(x, labels, fontsize=8, color=INK2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title({"ours": "Our detector (steers the loop)", "desklib": "Independent detector (never steers)"}[det], loc="left", color=INK, fontsize=10, fontweight="bold", pad=8)
        if i == 0:
            ax.set_ylabel("Texts flagged as AI", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.2, wspace=0.18)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_iteration_curves(records: list[dict[str, Any]], path: Path, title: str, subtitle: str | None = None) -> None:
    """Mean P(AI) after k rewrites for both detectors, with each detector's frozen cutoff as a dotted line."""
    top = max(len(r["steps"]) for r in records)
    fig = _canvas(6.4, 4.6)
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    for color, (det, (score_key, cut_key)) in zip(SERIES[:2], DETECTORS.items()):
        if any(r["steps"][0][score_key] is None for r in records):
            continue
        ys = [np.mean([r["steps"][min(k, len(r["steps"]) - 1)][score_key] for r in records]) for k in range(top)]
        ax.plot(range(top), ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round")
        cut = float(np.mean([r[cut_key] for r in records]))
        ax.axhline(cut, color=color, linewidth=1, linestyle=(0, (1, 2)))
        ax.text(top - 1 + 0.05, ys[-1], {"ours": " our detector", "desklib": " desklib"}[det], color=INK2, fontsize=9, va="center")
        ax.text(1.35, cut + 0.012, f"{'ours' if det == 'ours' else 'desklib'} cutoff (flags 5% of human text)", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_xticks(range(top))
    ax.set_xlim(-0.15, top - 1 + 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rewrites so far (0 = the raw AI text)", color=INK2, fontsize=9)
    ax.set_ylabel("Mean P(AI)", color=INK2, fontsize=9)
    _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.82, bottom=0.14)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_strategy_ablation(table: pd.DataFrame, path: Path, title: str, subtitle: str | None = None) -> None:
    """First-pass flagged rate by prompt strategy and detector, with the guardrail pass rate as a third bar."""
    series = [("ours_flagged", "ours_lo", "ours_hi", SERIES[0], "flagged by our detector"),
              ("desklib_flagged", "desklib_lo", "desklib_hi", SERIES[1], "flagged by desklib"),
              ("guardrails_ok", "guardrails_lo", "guardrails_hi", SERIES[2], "kept the meaning (guardrails pass)")]  # fmt: skip
    series = [s for s in series if s[0] in table.columns]
    fig = _canvas(7.4, 4.8)
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    x, w = np.arange(len(table)), 0.8 / len(series)
    for j, (key, lo, hi, color, name) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * w
        v = table[key].to_numpy()
        ax.bar(x + off, v, width=w * 0.92, color=color, label=name, zorder=3)
        ax.errorbar(x + off, v, yerr=[v - table[lo].to_numpy(), table[hi].to_numpy() - v], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
        for xi, vi, hi_i in zip(x + off, v, table[hi].to_numpy()):
            ax.text(xi, min(hi_i, 1.0) + 0.02, f"{vi:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=5)
    ax.set_xticks(x, table["strategy"], fontsize=9, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of texts (95% interval)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=len(series), bbox_to_anchor=(0.5, 1.01))
    _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.12)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---- what actually changed in the text ----------------------------------------------------------------------

def human_spread(profile, genre: str, feature: str) -> float:
    """A robust 'standard deviation' of a feature among human texts, from the profile's percentiles.

    Takes the largest of three normal-equivalent estimates (IQR, 10th-90th, median-to-99th) because sparse features
    (em dashes, stock phrases) are exactly 0 for most humans, which would make any single estimate collapse to zero.
    """
    q = {k: profile.q(genre, feature, k) for k in ("p10", "p25", "p50", "p75", "p90", "p99")}
    return max((q["p75"] - q["p25"]) / 1.349, (q["p90"] - q["p10"]) / 2.563, (q["p99"] - q["p50"]) / 2.326, 1e-9)


def feature_shift(records: list[dict[str, Any]], features_of: Callable[[str], dict[str, float]], profile, genre: str,
                  features: list[str]) -> pd.DataFrame:  # fmt: skip
    """Median feature values of raw AI text vs kept rewrites, in robust human standard deviations (`human_spread`).

    0 means "at the typical human value", +1 means one human standard deviation above it. `features_of(text)` returns the feature
    dict (the detector's `analyze(text).features`); kept rewrites only, so the table describes what a user would get.
    """
    raw, new = [], []
    for r in records:
        if r["final"] is None:
            continue
        raw.append(features_of(r["steps"][0]["text"]))
        new.append(features_of(r["steps"][r["final"]]["text"]))
    rows = []
    for f in features:
        p50, spread = profile.q(genre, f, "p50"), human_spread(profile, genre, f)
        rows.append({"feature": f, "human_median": p50, "raw_ai_median": float(np.median([d[f] for d in raw])) if raw else math.nan,
                     "rewrite_median": float(np.median([d[f] for d in new])) if new else math.nan, "n": len(new)})  # fmt: skip
        rows[-1]["raw_ai_gap"] = (rows[-1]["raw_ai_median"] - p50) / spread
        rows[-1]["rewrite_gap"] = (rows[-1]["rewrite_median"] - p50) / spread
    return pd.DataFrame(rows)


def plot_feature_shift(shift: pd.DataFrame, path: Path, title: str, subtitle: str | None = None) -> None:
    """Dumbbell per feature: raw AI vs rewrite, distance from the human median in human standard deviations."""
    shift = shift.assign(_sort=shift["raw_ai_gap"].abs()).sort_values("_sort")
    fig = _canvas(7.6, 0.42 * len(shift) + 1.9)
    ax = fig.add_subplot(1, 1, 1)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=1)
    y = np.arange(len(shift))
    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    for yi, a, b in zip(y, shift["raw_ai_gap"], shift["rewrite_gap"]):
        ax.plot([a, b], [yi, yi], color=AXIS, linewidth=2, zorder=2, solid_capstyle="round")
    ax.scatter(shift["raw_ai_gap"], y, s=46, color=AI_C, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="raw AI")
    ax.scatter(shift["rewrite_gap"], y, s=46, color=HUMAN_C, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="kept rewrite")
    ax.set_yticks(y, [f.replace("_per100w", "/100w").replace("_", " ") for f in shift["feature"]], fontsize=9, color=INK2)
    ax.set_xlabel("Distance from the typical human value, in human standard deviations (0 = human median)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    _title(fig, title, subtitle)
    fig.subplots_adjust(left=0.22, right=0.97, top=1 - 1.1 / fig.get_figheight(), bottom=0.9 / fig.get_figheight() + 0.02)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def write_loop_report(records: list[dict[str, Any]], out_dir: Path, tag: str, subtitle: str, shift: pd.DataFrame | None = None) -> dict[str, Path]:
    """Write the outcome tables (both policies), the before/after and iteration figures, and optionally the feature shift."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    written: dict[str, Path] = {}
    tables = [outcome_table(records, policy) for policy in POLICIES]
    full = pd.concat(tables, ignore_index=True)
    written["outcomes"] = out_dir / f"outcomes_{tag}.csv"
    full.round(4).to_csv(written["outcomes"], index=False)
    shown = full.assign(before=[interval_label(a, b, c) for a, b, c in zip(full.flagged_before, full.before_lo, full.before_hi)],
                        after=[interval_label(a, b, c) for a, b, c in zip(full.flagged_after, full.after_lo, full.after_hi)])  # fmt: skip
    shown = shown[["genre", "strategy", "model", "policy", "detector", "n", "before", "after", "evaded_of_detected", "rewrite_accepted", "mean_similarity_kept"]]
    written["outcomes_md"] = out_dir / f"outcomes_{tag}.md"
    written["outcomes_md"].write_text(to_markdown(shown, percent=("evaded_of_detected", "rewrite_accepted")))
    kept = tables[0]
    plot_before_after(kept, figures / f"before_after_{tag}.png", "Flagged as AI before and after the loop", subtitle)
    written["before_after"] = figures / f"before_after_{tag}.png"
    plot_iteration_curves(records, figures / f"iterations_{tag}.png", "Detector score by number of rewrites", subtitle)
    written["iterations"] = figures / f"iterations_{tag}.png"
    if shift is not None and len(shift):
        shift.round(4).to_csv(out_dir / f"feature_shift_{tag}.csv", index=False)
        plot_feature_shift(shift, figures / f"feature_shift_{tag}.png", "What the rewrites changed", subtitle)
        written["feature_shift"] = figures / f"feature_shift_{tag}.png"
    return written
