"""Command-line entry point. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import sys

from hd import __version__
from hd.config import load_config, resolve_path, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hd", description="AI-text detector + humanizer. Typical order: data -> features -> detector -> loop -> eval; `hd app` for the web UI.")
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
    hum_sub.add_parser("check-restructuring", help="do the content checks tell restructuring (splitting/merging) from damage (deleting/swapping)?")

    humanize = sub.add_parser("humanize", help="rewrite AI text to read as human; show before/after scores")
    humanize.add_argument("text", nargs="?", help="the text (or use --file / stdin)")
    humanize.add_argument("--file", help="read the text from a file")
    humanize.add_argument("--genre", choices=["story", "essay", "casual"], default="story")
    humanize.add_argument("--strategy", choices=["naive", "craft", "feedback", "full"], default="full")
    humanize.add_argument("--dry-run", action="store_true", help="show the prompt that would be sent; call no API")
    humanize.add_argument("--external", action="store_true", help="also score before/after with the independent detector")

    loop = sub.add_parser("loop", help="adversarial loop: rewrite, score, feedback, rewrite")
    loop_sub = loop.add_subparsers(dest="loop_command", required=True)
    run = loop_sub.add_parser("run", help="run the loop over a sample of AI texts (resumable; logs every iteration)")
    run.add_argument("--genre", choices=["story", "essay", "casual"], default="story")
    run.add_argument("--split", choices=["train", "val", "test"], default="val", help="'test' is only for the final evaluation and needs --unlock-test")
    run.add_argument("--unlock-test", action="store_true", help="confirm that this run uses the held-out test split (logged in results/test_access_log.txt)")
    run.add_argument("--n", type=int, default=10, help="how many AI texts (a larger n extends the same sample)")
    run.add_argument("--strategy", choices=["naive", "craft", "feedback", "full"], default="full")
    run.add_argument("--model", help="humanizer model (default: llm.humanizer_model); each Groq model has its own daily token pool")
    run.add_argument("--max-words", type=int, help="only texts up to this many words (needed for models with small output caps)")
    run.add_argument("--max-iterations", type=int, help="default: loop.max_iterations from the config")
    run.add_argument("--no-external", action="store_true", help="skip the independent detector (faster; fewer claims)")
    run.add_argument("--strength", choices=["faithful", "strong"], default="faithful", help="faithful keeps every sentence's content (similarity >= 0.80); strong may compress minor detail (>= 0.72)")
    run.add_argument("--steer-desklib", action="store_true", help="let desklib steer too: stop only when both detectors pass and keep the best worst-case rewrite (fakespot, the judge, never steers)")
    run.add_argument("--no-judge", action="store_true", help="skip the third detector (fakespot), which is only ever a judge")
    run.add_argument("--no-repair", action="store_true", help="v1 behaviour: after a rejected rewrite restart from the last accepted draft with a generic note")
    run.add_argument("--tag", default="", help="suffix for the log file name (e.g. v2), so runs with different settings never share a file")
    run.add_argument("--out", help="JSONL log (default: data/processed/loop_<split>_<genre>_<strategy>_<model>.jsonl)")
    summ = loop_sub.add_parser("summary", help="detection rates before/after from a loop log")
    summ.add_argument("file", help="a JSONL log written by `hd loop run`")

    sub.add_parser("app", help="launch the Streamlit app (Detect and Humanize tabs)")

    ev = sub.add_parser("eval", help="reports, ablations, the final held-out evaluation, the manual-detector kit")
    ev_sub = ev.add_subparsers(dest="eval_command", required=True)
    rep = ev_sub.add_parser("report", help="outcome tables (with intervals) and figures from loop logs")
    rep.add_argument("--log", action="append", required=True, help="a loop log (repeatable; logs are concatenated)")
    rep.add_argument("--tag", help="name for the output files (default: first log's name)")
    rep.add_argument("--out-dir", help="default: results/loop_reports")
    rep.add_argument("--features", action="store_true", help="also compute the feature-shift figure (loads GPT-2; slower)")
    abp = ev_sub.add_parser("ablate-prompts", help="compare prompt strategies (first rewrite of each text) from existing loop logs")
    abp.add_argument("--genre", choices=["story", "essay", "casual"], default="story")
    abp.add_argument("--split", choices=["train", "val"], default="val")
    abp.add_argument("--model", required=True, help="the humanizer model the logs were made with")
    abp.add_argument("--max-words", type=int)
    abp.add_argument("--tag", default="", help="the --tag the logs were made with (e.g. v2)")
    abf = ev_sub.add_parser("ablate-features", help="detector: AUC with each feature group dropped / on its own (train+val only)")
    abf.add_argument("--n-boot", type=int, default=200)
    fin = ev_sub.add_parser("final", help="THE held-out test evaluation; runs once (logged)")
    fin.add_argument("--unlock-test", action="store_true", help="required: confirms you mean to use the test split")
    fin.add_argument("--loop-log", action="append", default=[], help="logs of `hd loop run --split test` runs (repeatable)")
    fin.add_argument("--no-external", action="store_true")
    fin.add_argument("--again", action="store_true", help="repeat a final evaluation that already ran (recorded in the log)")
    ctl = ev_sub.add_parser("control", help="rule-based controls: do the detectors react to punctuation-only edits? (train split, no LLM)")
    ctl.add_argument("--genre", choices=["story", "essay", "casual"], default="story")
    ctl.add_argument("--n", type=int, default=40)
    ctl.add_argument("--max-words", type=int, default=400)
    mk = ev_sub.add_parser("manual-kit", help="blind sample files + log template for testing commercial AI detectors by hand")
    mk.add_argument("--log", required=True, help="a loop log (use a validation log; the kit exports AI text and rewrites only)")
    mk.add_argument("--n", type=int, default=5, help="how many raw/rewrite pairs")

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

    if args.command == "app":
        import subprocess

        app = resolve_path(cfg, "results").parent / "app" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    if args.command in ("data", "features", "detector", "detect", "humanizer", "humanize", "llm", "loop", "eval"):
        runners = {"data": _run_data, "features": _run_features, "detector": _run_detector, "detect": _run_detect,
                   "humanizer": _run_humanizer, "humanize": _run_humanize, "llm": _run_llm, "loop": _run_loop,
                   "eval": _run_eval}  # fmt: skip
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
            length_scale=cfg["data"].get("story_length_scale", 1.0),
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
    elif args.humanizer_command == "check-restructuring":
        from hd.humanizer.calibrate import run_restructuring_check, summarize_restructuring

        g = cfg["guardrails"]
        table = summarize_restructuring(run_restructuring_check(cfg), g["max_dropped_frac"], g["max_invented_frac"]).round(3)
        table.to_csv(resolve_path(cfg, "results") / "guardrail_restructuring.csv", index=False)
        print(table.to_string(index=False))
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


def _run_loop(args: argparse.Namespace, cfg: dict) -> int:
    from pathlib import Path

    from hd.loop import iteration_curve, load_results, summarize

    if args.loop_command == "summary":
        records = load_results(Path(args.file))
        print(summarize(records).round(3).to_string(index=False))
        print("\nMean P(AI) after k rewrites (our detector):")
        print(iteration_curve(records).round(3).to_string(index=False))
        return 0

    import pandas as pd

    from hd.detector.explain import Detector
    from hd.humanizer.feedback import HumanProfile
    from hd.humanizer.guardrails import SentenceEmbedder
    from hd.humanizer.pipeline import checker_for, load_thresholds
    from hd.humanizer.rewrite import Humanizer
    from hd.llm import make_client
    from hd.loop import cutoff_for, log_path, run_batch, run_loop, select_items

    if args.split == "test" and not args.unlock_test:
        raise RuntimeError("the test split is only for the final evaluation; pass --unlock-test to confirm (the access is logged)")
    thresholds = load_thresholds(cfg)
    our_cut = cutoff_for(thresholds, "ours", args.genre)
    detector = Detector.load(cfg)
    profile = HumanProfile.load_or_build(cfg, detector.scorer)
    llm = cfg["llm"]
    model = args.model or llm["humanizer_model"]
    humanizer = Humanizer(make_client(cfg, resolve_path(cfg, "cache")), model, profile, llm["humanizer_effort"], args.strength)
    checker = checker_for(cfg, SentenceEmbedder(), detector.scorer, profile, args.genre, args.strength)
    external, ext_cut = None, None
    if not args.no_external:
        from hd.detector.external import DesklibDetector

        external, ext_cut = DesklibDetector(), cutoff_for(thresholds, "external", args.genre)
    if args.steer_desklib and external is None:
        raise RuntimeError("--steer-desklib needs desklib: drop --no-external")
    judge, judge_cut = None, None
    if not args.no_judge and "third" in thresholds:
        from hd.detector.external import FakespotDetector

        judge, judge_cut = FakespotDetector(), cutoff_for(thresholds, "third", args.genre)

    processed = resolve_path(cfg, "processed")
    on_test = args.split == "test"
    if on_test:
        from hd.eval.final import log_test_access

        log_test_access(cfg, f"loop run genre={args.genre} n={args.n} strategy={args.strategy} model={model}")
    items = select_items(pd.read_parquet(processed / "corpus.parquet"), args.split, args.genre, args.n, cfg["seed"],
                         max_words=args.max_words, unlock_test=on_test)
    if items.empty:
        raise RuntimeError(f"no AI {args.genre} texts in the {args.split} split; generate/build them first")
    out = Path(args.out) if args.out else log_path(processed, args.split, args.genre, args.strategy, model, args.max_words, args.tag, args.strength)
    max_iter = args.max_iterations or cfg["loop"]["max_iterations"]

    def run_one(item: dict):
        print(f"  {item['id']}: {item['n_words']} words ...", flush=True)
        return run_loop(detector, humanizer, checker, item["text"], args.genre, strategy=args.strategy, our_cutoff=our_cut,
                        ext_cutoff=ext_cut, external=external, max_iterations=max_iter, item_id=item["id"],
                        repair=not args.no_repair, judge=judge, judge_cutoff=judge_cut, steer_external=args.steer_desklib)  # fmt: skip

    stats = run_batch(items.to_dict("records"), out, run_one)
    print({k: v for k, v in stats.items() if k != "message"})
    if stats["message"]:
        print(stats["message"] + "\nWork so far is saved; run the same command later to continue.")
    if out.exists() and out.stat().st_size:
        print(summarize(load_results(out)).round(3).to_string(index=False))
    return 0


def _run_eval(args: argparse.Namespace, cfg: dict) -> int:
    from pathlib import Path

    import numpy as np

    from hd.loop import load_results, log_path

    results, processed = resolve_path(cfg, "results"), resolve_path(cfg, "processed")
    cmd = args.eval_command

    if cmd == "report":
        from hd.eval.loop_report import feature_shift, outcome_table, to_markdown, write_loop_report

        records = [r for log in args.log for r in load_results(Path(log))]
        tag = args.tag or Path(args.log[0]).stem.removeprefix("loop_")
        shift = None
        if args.features:
            from hd.detector.explain import Detector
            from hd.humanizer.feedback import HumanProfile
            from hd.humanizer.pipeline import COMPARE

            detector = Detector.load(cfg)
            profile = HumanProfile.load_or_build(cfg, detector.scorer)
            shift = feature_shift(records, lambda t: detector.analyze(t).features, profile, records[0]["genre"], COMPARE)
        out = Path(args.out_dir) if args.out_dir else results / "loop_reports"
        written = write_loop_report(records, out, tag, f"{len(records)} texts, frozen 5%-false-positive cutoffs", shift)
        print(to_markdown(outcome_table(records, "kept").round(3), percent=("flagged_before", "before_lo", "before_hi", "flagged_after", "after_lo", "after_hi", "evaded_of_detected", "rewrite_accepted")))
        for name, path in written.items():
            print(f"  {name}: {path}")
    elif cmd == "ablate-prompts":
        from hd.eval.loop_report import first_pass_table, plot_strategy_ablation, to_markdown
        from hd.humanizer.prompts import STRATEGIES

        by = {s: load_results(p) for s in STRATEGIES if (p := log_path(processed, args.split, args.genre, s, args.model, args.max_words, args.tag)).exists()}
        if not by:
            raise RuntimeError("no loop logs found for that genre/split/model; run `hd loop run` for each strategy first")
        table = first_pass_table(by)
        short = f"_le{args.max_words}w" if args.max_words else ""
        stem = f"ablation_prompts_{args.genre}_{args.split}_{args.model.split('/')[-1]}{short}{'_' + args.tag if args.tag else ''}"
        table.round(4).to_csv(results / f"{stem}.csv", index=False)
        (results / f"{stem}.md").write_text(to_markdown(table[["strategy", "n", "ours_flagged", "desklib_flagged", "guardrails_ok", "mean_similarity", "mean_reading_grade"]], percent=("ours_flagged", "desklib_flagged", "guardrails_ok")))
        (results / "figures").mkdir(exist_ok=True)
        plot_strategy_ablation(table, results / "figures" / f"{stem}.png", "Prompt strategies, first rewrite of each text",
                               f"{args.genre}, {args.split} split, humanizer {args.model.split('/')[-1]}, n={int(table.n.max())}; bars show 95% intervals")  # fmt: skip
        print(to_markdown(table[["strategy", "n", "ours_flagged", "desklib_flagged", "guardrails_ok", "mean_similarity", "mean_reading_grade"]], percent=("ours_flagged", "desklib_flagged", "guardrails_ok")))
        print(f"wrote {results / (stem + '.csv')}")
    elif cmd == "ablate-features":
        from hd.eval.ablation import group_ablation, load_ablation_frame, plot_group_drop

        table = group_ablation(load_ablation_frame(cfg), seed=cfg["seed"], n_boot=args.n_boot)
        table.round(4).to_csv(results / "ablation_feature_groups.csv", index=False)
        (results / "figures").mkdir(exist_ok=True)
        plot_group_drop(table, results / "figures" / "ablation_feature_groups.png", "What the detector loses without each feature group")
        print(table[table.condition.str.startswith(("all", "drop:"))].pivot(index="condition", columns="genre", values="auc").round(3).to_string())
    elif cmd == "final":
        if not args.unlock_test:
            raise RuntimeError("the final evaluation uses the held-out test split once; pass --unlock-test to confirm")
        from hd.eval.final import run_final

        written = run_final(cfg, [Path(p) for p in args.loop_log], external=not args.no_external, again=args.again)
        import pandas as pd

        print(pd.read_csv(written["detectors"]).round(3).to_string(index=False))
        for name, path in written.items():
            print(f"  {name}: {path}")
    elif cmd == "control":
        import pandas as pd

        from hd.data.splits import load_split
        from hd.detector.explain import Detector
        from hd.detector.external import DesklibDetector
        from hd.eval.control import control_table
        from hd.humanizer.calibrate import merge_pairs, split_into_fragments
        from hd.humanizer.pipeline import load_thresholds
        from hd.loop import cutoff_for

        th = load_thresholds(cfg)
        corpus = pd.read_parquet(processed / "corpus.parquet")
        ai = load_split(corpus[corpus["has_pair"]], "train")  # design data only; the test split stays locked
        ai = ai[(ai.label == 1) & (ai.genre == args.genre) & (ai.n_words <= args.max_words)].sample(args.n, random_state=cfg["seed"])
        det, ext = Detector.load(cfg), DesklibDetector()
        table = control_table(ai.text.tolist(), {"split into fragments (punctuation only)": split_into_fragments, "merge sentence pairs": merge_pairs},
                              lambda ts: np.array([det.analyze(t).prob_ai for t in ts]), ext.predict_proba,
                              cutoff_for(th, "ours", args.genre), cutoff_for(th, "external", args.genre))  # fmt: skip
        table.round(4).to_csv(results / f"control_rule_based_{args.genre}_train.csv", index=False)
        print(table.round(3).to_string(index=False))
    elif cmd == "manual-kit":
        from hd.eval.manual_kit import PROTOCOL, build_manual_kit, write_log_template

        reports = resolve_path(cfg, "results").parent / "reports"
        key = build_manual_kit(load_results(Path(args.log)), reports / "manual_kit", args.n, cfg["seed"])
        write_log_template(reports / "manual_detector_log.csv", [k["sample"] for k in key])
        (reports / "manual_detector_log.md").write_text(PROTOCOL)
        print(f"wrote {len(key)} blind samples to {reports / 'manual_kit'} (answer key: answer_key.csv, do not open it yet)")
        print(f"fill in {reports / 'manual_detector_log.csv'}; the protocol is in {reports / 'manual_detector_log.md'}")
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
