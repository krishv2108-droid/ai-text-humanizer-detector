"""Download the raw datasets approved for M1 into data/raw/ (idempotent, size-checked)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

from hd.config import load_config, resolve_path

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class RawFile:
    repo: str
    path: str  # path inside the HF repo
    size: int  # expected bytes, from the HF file listing
    local: str  # filename under data/raw/

    @property
    def url(self) -> str:
        return HF.format(repo=self.repo, path=self.path)


RAW_FILES = [
    RawFile(
        "euclaise/writingprompts",
        "data/validation-00000-of-00001-137b93e1e979d138.parquet",
        30_707_033,
        "writingprompts_validation.parquet",
    ),
    RawFile("Hello-SimpleAI/HC3", "open_qa.jsonl", 2_910_828, "hc3_open_qa.jsonl"),
    RawFile("Hello-SimpleAI/HC3", "finance.jsonl", 9_889_186, "hc3_finance.jsonl"),
    RawFile("Hello-SimpleAI/HC3", "medicine.jsonl", 2_683_640, "hc3_medicine.jsonl"),
    RawFile("Hello-SimpleAI/HC3", "wiki_csai.jsonl", 2_198_039, "hc3_wiki_csai.jsonl"),
    RawFile("Hello-SimpleAI/HC3", "reddit_eli5.jsonl", 55_448_695, "hc3_reddit_eli5.jsonl"),
]


def download_all(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in RAW_FILES:
        dest = raw_dir / f.local
        if dest.exists() and dest.stat().st_size == f.size:
            print(f"ok      {f.local}")
            continue
        print(f"fetch   {f.local} ({f.size / 1e6:.1f} MB)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(f.url, tmp)
        if tmp.stat().st_size != f.size:
            raise RuntimeError(f"{f.local}: expected {f.size} bytes, got {tmp.stat().st_size}")
        tmp.rename(dest)


if __name__ == "__main__":
    download_all(resolve_path(load_config(), "raw"))
