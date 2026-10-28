"""Command-line entry point. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse

from hd import __version__
from hd.config import load_config, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hd", description="AI-text detector + humanizer")
    parser.add_argument("--version", action="version", version=f"hd {__version__}")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    set_seed(cfg["seed"])
    print("hd: no command given yet. Subcommands arrive in later milestones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
