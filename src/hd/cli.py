"""Command-line entry point. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import sys

from hd import __version__
from hd.config import load_config, resolve_path, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hd", description="AI-text detector + humanizer")
    parser.add_argument("--version", action="version", version=f"hd {__version__}")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    sub = parser.add_subparsers(dest="command")

    data = sub.add_parser("data", help="dataset commands")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("download", help="download approved raw datasets into data/raw/")
    data_sub.add_parser("build", help="clean, pair and split into data/processed/corpus.parquet")
    gen = data_sub.add_parser("generate-stories", help="write LLM stories for the story prompts (resumable; stops cleanly at a rate limit)")
    gen.add_argument("--limit", type=int, default=None, help="max stories to write this run")

    feats = sub.add_parser("features", help="feature extraction and analysis")
    feats_sub = feats.add_subparsers(dest="features_command", required=True)
    build = feats_sub.add_parser("build", help="compute features for all paired texts (cached)")
    build.add_argument("--no-perplexity", action="store_true", help="skip the GPT-2 features")
    feats_sub.add_parser("analyze", help="per-feature effect sizes + distribution plots (train split)")

    det = sub.add_parser("detector", help="detector commands")
    det_sub = det.add_subparsers(dest="detector_command", required=True)
    base = det_sub.add_parser("train-baseline", help="fit the feature-based detectors; report on the validation split")
    base.add_argument("--variants", nargs="*", default=None, help="subset of variants (default: all)")
    det_sub.add_parser("train-sentence", help="fit the per-sentence model (needs GPT-2; ~3 min the first time)")
    det_sub.add_parser("eval-external", help="score validation texts with the independent detector")

    sub.add_parser("llm", help="LLM provider commands").add_subparsers(dest="llm_command", required=True).add_parser(
        "check", help="verify the key and models; show your account's rate limits")

    hum = sub.add_parser("humanizer", help="humanizer commands")
    hum_sub = hum.add_subparsers(dest="humanizer_command", required=True)
    hum_sub.add_parser("calibrate-guardrails", help="simulate rewrite failures; measure how the guardrails respond")

    humanize = sub.add_parser("humanize", help="rewrite AI text to read as human; show before/after scores")
    humanize.add_argument("text", nargs="?", help="the text (or use --file / stdin)")
    humanize.add_argument("--file", help="read the text from a file")
    humanize.add_argument("--genre", choices=["story", "essay", "casual"], default="story")
    humanize.add_argument("--strategy", choices=["naive", "craft", "feedback", "full"], default="full")
    humanize.add_argument("--dry-run", action="store_true", help="show the prompt that would be sent; call no API")
    humanize.add_argument("--external", action="store_true", help="also score before/after with the independent detector")

    detect = sub.add_parser("detect", help="P(AI), per-sentence breakdown and drivers for a text")
    detect.add_argument("text", nargs="?", help="the text (or use --file / stdin)")
    detect.add_argument("--file", help="read the text from a file")
    detect.add_argument("--external", action="store_true", help="also score with the independent detector")
    detect.add_argument("--top", type=int, default=3, help="how many most-AI-like sentences to show")
    detect.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    set_seed(cfg["seed"])

    if args.command in ("data", "features", "detector", "detect", "humanizer", "humanize", "llm"):
        runners = {"data": _run_data, "features": _run_features, "detector": _run_detector, "detect": _run_detect,
                   "humanizer": _run_humanizer, "humanize": _run_humanize, "llm": _run_llm}  # fmt: skip
        try:
            return runners[args.command](args, cfg)
        except RuntimeError as err:  # e.g. missing API key: show the message, not a traceback
            print(f"error: {err}", file=sys.stderr)
            return 1
    print("hd: pick a command, e.g. `hd data build`. See `hd --help`.")
    return 0


def _run_data(args: argparse.Namespace, cfg: dict) -> int:
    if args.data_command == "download":
        from hd.data.download import download_all

        download_all(resolve_path(cfg, "raw"))
    elif args.data_command == "build":
        from hd.data.build import build_corpus, summarize

        corpus = build_corpus(cfg)
        summarize(corpus).to_csv(resolve_path(cfg, "results") / "corpus_summary.csv", index=False)
        print(f"wrote {len(corpus)} texts to {resolve_path(cfg, 'processed') / 'corpus.parquet'}")
    elif args.data_command == "generate-stories":
        from hd.data.generate import generate_stories
        from hd.llm import make_client

        processed = resolve_path(cfg, "processed")
        client = make_client(cfg, resolve_path(cfg, "cache"))
        stats = generate_stories(
            client,
            processed / "story_prompts.jsonl",
            processed / "ai_stories.jsonl",
            model=cfg["llm"]["generation_model"],
            max_tokens=cfg["llm"]["generation_max_tokens"],
            effort=cfg["llm"]["generation_effort"],
            limit=args.limit,
        )
        print({k: v for k, v in stats.items() if k != "message"})
        if stats["message"]:
            print(stats["message"] + "\nWork so far is saved; run the same command later to continue.")
    return 0


def _run_features(args: argparse.Namespace, cfg: dict) -> int:
    import pandas as pd

    processed = resolve_path(cfg, "processed")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    paired = corpus[corpus["has_pair"]].reset_index(drop=True)

    if args.features_command == "build":
        from hd.features.pipeline import compute_features

        scorer = None
        if not args.no_perplexity:
            from hd.features.perplexity import GPT2Scorer

            scorer = GPT2Scorer()
            print(f"GPT-2 on {scorer.device}")
        feats = compute_features(paired, processed / "features.parquet", scorer)
        print(f"features for {len(feats)} texts -> {processed / 'features.parquet'}")
    elif args.features_command == "analyze":
        from hd.eval.feature_analysis import run_analysis

        run_analysis(cfg)
    return 0


def _run_detector(args: argparse.Namespace, cfg: dict) -> int:
    if args.detector_command == "train-baseline":
        from hd.detector.baseline import run_baseline

        run_baseline(cfg, args.variants)
    elif args.detector_command == "train-sentence":
        from hd.detector.sentence import run_sentence

        run_sentence(cfg)
    elif args.detector_command == "eval-external":
        from hd.eval.external_eval import run_external_eval

        run_external_eval(cfg)
    return 0


def _run_llm(args: argparse.Namespace, cfg: dict) -> int:
    from hd.llm import PROVIDERS, OpenAICompatClient, api_key

    llm = cfg["llm"]
    provider = llm["provider"]
    key = api_key(provider)  # raises a clear error if missing
    print(f"provider: {provider}  (key found, {len(key)} characters)")
    if provider == "anthropic":
        print("limits are not probed for Anthropic; make a call to test the key.")
        return 0
    client = OpenAICompatClient(PROVIDERS[provider]["base_url"], key, max_retries=2, seed=cfg.get("seed"))
    for role in ("generation_model", "humanizer_model"):
        model = llm[role]
        if role == "humanizer_model" and model == llm["generation_model"]:
            print(f"{role}: same model as generation_model")
            continue
        info = client.probe(model)
        print(f"\n{role}: {model}")
        for k, v in info.items():
            print(f"  {k}: {v}")
        tpm = float(info["limit_tokens_per_minute"] or 0)
        if tpm:
            print(f"  -> per minute, roughly {tpm / 1000:.0f}k tokens: about {int(tpm // 900)} stories or {int(tpm // 2800)} humanizer calls")
    print("\nDaily token limits are not in the response headers; see https://console.groq.com/settings/limits")
    return 0


def _run_humanizer(args: argparse.Namespace, cfg: dict) -> int:
    if args.humanizer_command == "calibrate-guardrails":
        import pandas as pd

        from hd.humanizer.calibrate import run_calibration

        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        print(run_calibration(cfg).T.to_string())
    return 0


def _read_text(args: argparse.Namespace) -> str:
    if args.file:
        return open(args.file).read()
    return args.text if args.text else sys.stdin.read()


def _run_humanize(args: argparse.Namespace, cfg: dict) -> int:
    from hd.detector.explain import Detector
    from hd.humanizer.feedback import HumanProfile
    from hd.humanizer.pipeline import checker_for, humanize_once, load_thresholds
    from hd.humanizer.rewrite import Humanizer

    text = _read_text(args)
    detector = Detector.load(cfg)
    profile = HumanProfile.load_or_build(cfg, detector.scorer)
    llm = cfg["llm"]

    if args.dry_run:
        humanizer = Humanizer(None, llm["humanizer_model"], profile)  # type: ignore[arg-type]
        before = detector.analyze(text)
        system, user = humanizer.build_request(before.text, args.genre, args.strategy, before)
        print(f"[dry run: nothing sent] strategy={args.strategy} genre={args.genre} P(AI) now {before.prob_ai:.3f}\n")
        print("--- SYSTEM ---\n" + system + "\n\n--- USER ---\n" + user)
        return 0

    from hd.config import resolve_path
    from hd.humanizer.guardrails import SentenceEmbedder
    from hd.llm import make_client

    humanizer = Humanizer(make_client(cfg, resolve_path(cfg, "cache")), llm["humanizer_model"], profile, llm["humanizer_effort"])
    external = None
    if args.external:
        from hd.detector.external import DesklibDetector

        external = DesklibDetector()
    checker = checker_for(cfg, SentenceEmbedder(), detector.scorer, profile, args.genre)
    report = humanize_once(detector, humanizer, checker, text, args.genre, args.strategy, external)

    cutoffs = load_thresholds(cfg)
    ours_cut = cutoffs.get("ours", {}).get(args.genre if args.genre in cutoffs.get("ours", {}) else "all")
    def verdict(p, cut):
        return "" if cut is None else ("  FLAGGED" if p > cut else "  not flagged") + f" at the 5%-false-positive cutoff ({cut:.3f})"

    print("=== REWRITE ===\n" + report.rewrite.text + "\n")
    print(f"BEFORE  P(AI) = {report.before.prob_ai:.3f}{verdict(report.before.prob_ai, ours_cut)}")
    print(f"AFTER   P(AI) = {report.after.prob_ai:.3f}{verdict(report.after.prob_ai, ours_cut)}")
    if external is not None:
        ext_cut = cutoffs.get("external", {}).get(args.genre if args.genre in cutoffs.get("external", {}) else "all")
        print(f"independent detector: before {report.external_before:.3f} -> after {report.external_after:.3f}{verdict(report.external_after, ext_cut)}")
    print("\nFeature comparison (human median = typical human text of this genre):")
    print(report.comparison.round(3).to_string(index=False))
    g = report.guardrails
    print(f"\nGuardrails: {'OK' if g.ok else 'FLAGGED'}  similarity {g.similarity:.2f}, length x{g.length_ratio:.2f}, "
          f"reading grade {g.fk_grade_before:.1f} -> {g.fk_grade_after:.1f}")
    for flag in g.flags + report.rewrite.warnings:
        print(f"  ! {flag}")
    return 0


def _run_detect(args: argparse.Namespace, cfg: dict) -> int:
    import json

    if args.file:
        text = open(args.file).read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    from hd.detector.explain import Detector

    detector = Detector.load(cfg)
    try:
        a = detector.analyze(text)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    external = None
    if args.external:
        from hd.detector.external import DesklibDetector

        external = float(DesklibDetector().predict_proba([a.text])[0])

    if args.json:
        print(json.dumps({
            "prob_ai": a.prob_ai, "external_prob_ai": external, "n_words": a.n_words,
            "group_effects": a.group_effects, "feature_effects": a.feature_effects[:10],
            "sentences": [vars(s) for s in a.sentences],
        }, indent=2))  # fmt: skip
        return 0

    print(f"P(AI) = {a.prob_ai:.3f}   ({a.n_words} words, detector: {detector.variant})")
    if external is not None:
        print(f"independent detector (desklib): P(AI) = {external:.3f}")
    print("\nWhat drove the score. Effect = change in P(AI) if that feature group looked typical:")
    print("  (+ means the group pushes this text toward AI, - toward human)")
    for group, effect in a.group_effects.items():
        print(f"  {group:12s} {effect:+.3f}")
    print("\nMost AI-like sentences:")
    for s in a.top_sentences(args.top):
        why = f"   [{'; '.join(s.reasons)}]" if s.reasons else ""
        print(f"  {s.prob_ai:.2f}  {s.text[:140]}{'...' if len(s.text) > 140 else ''}{why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
