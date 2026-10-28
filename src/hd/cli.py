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
    gen = data_sub.add_parser("generate-stories", help="write Claude stories for the story prompts")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    set_seed(cfg["seed"])

    if args.command in ("data", "features", "detector"):
        runners = {"data": _run_data, "features": _run_features, "detector": _run_detector}
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
        print(stats)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
