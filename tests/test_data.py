import json
from pathlib import Path

import pandas as pd
import pytest

from hd.config import PROJECT_ROOT
from hd.data.clean import has_markdown, has_residual_tokenization, normalize_text
from hd.data.generate import generate_stories
from hd.data.splits import TestSetLockedError, add_splits, assign_split, load_split
from hd.llm import CachedClient, MockClient

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---- normalize_text -----------------------------------------------------------------------


def test_normalize_fixes_tokenizer_spacing():
    raw = 'I do n\'t know , but she ca n\'t go . " Why not ? " he asked ( really ) .'
    assert normalize_text(raw) == 'I don\'t know, but she can\'t go. "Why not?" he asked (really).'


def test_normalize_story_newlines_and_quotes():
    raw = "“ Hello , ” she said . \n \n \n \n He left ."
    assert normalize_text(raw) == '"Hello," she said.\n\nHe left.'


def test_normalize_is_idempotent():
    raw = "It ca n't be `` true '' , right ?\n \n Yes ."
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_normalize_leaves_unbalanced_quotes_alone():
    assert normalize_text('He said "hello') == 'He said "hello'


def test_human_and_ai_formatting_converges():
    human = "Do n't worry , it 's fine ."
    ai = "Don't worry, it's fine."
    assert normalize_text(human) == normalize_text(ai)


def test_residual_and_markdown_detectors():
    assert has_residual_tokenization("this is odd , right")
    assert not has_residual_tokenization("this is fine, right")
    assert has_markdown("# Title\nText")
    assert has_markdown("some **bold** text")
    assert not has_markdown("Plain prose, with a - dash.")


# ---- splits -------------------------------------------------------------------------------


def test_assign_split_is_deterministic_and_roughly_proportional():
    ids = [f"p{i}" for i in range(4000)]
    first = [assign_split(i, 42, FRACTIONS) for i in ids]
    assert first == [assign_split(i, 42, FRACTIONS) for i in ids]
    share = {s: first.count(s) / len(ids) for s in FRACTIONS}
    for name, frac in FRACTIONS.items():
        assert abs(share[name] - frac) < 0.03


def test_pair_never_straddles_splits_and_growth_does_not_move_prompts():
    df = pd.DataFrame({"prompt_id": ["a", "a", "b", "b", "c", "c"], "label": [0, 1] * 3})
    out = add_splits(df, 42, FRACTIONS)
    assert (out.groupby("prompt_id")["split"].nunique() == 1).all()
    bigger = add_splits(pd.concat([df, pd.DataFrame({"prompt_id": ["d"], "label": [0]})]), 42, FRACTIONS)
    assert out.set_index("prompt_id")["split"].to_dict().items() <= bigger.set_index("prompt_id")["split"].to_dict().items()


def test_test_split_is_locked(tmp_path):
    corpus = add_splits(pd.DataFrame({"prompt_id": [f"p{i}" for i in range(200)]}), 42, FRACTIONS)
    assert set(load_split(corpus, "train")["split"]) == {"train"}
    with pytest.raises(TestSetLockedError):
        load_split(corpus, "test")
    log = tmp_path / "access.log"
    test = load_split(corpus, "test", unlock_test=True, access_log=log)
    assert set(test["split"]) == {"test"}
    assert "unlocked" in log.read_text()


# ---- LLM client + story generation (offline, mock) ----------------------------------------


def test_cached_client_only_calls_inner_once(tmp_path):
    mock = MockClient()
    client = CachedClient(mock, tmp_path)
    kw = dict(system="s", prompt="p", model="m", max_tokens=10)
    first, second = client.complete(**kw), client.complete(**kw)
    assert mock.calls == 1
    assert second.cached and not first.cached and first.text == second.text


def _write_prompts(path: Path, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt_id": f"p{i}", "prompt": f"prompt {i}", "target_words": 60}) + "\n")


def test_generate_stories_is_resumable_and_filters(tmp_path):
    prompts, out = tmp_path / "prompts.jsonl", tmp_path / "ai.jsonl"
    _write_prompts(prompts, 4)

    def responder(prompt: str) -> str:
        if "prompt 1" in prompt:
            return "# A Title\n" + "word " * 80  # markdown: rejected
        if "prompt 2" in prompt:
            return "too short"  # rejected
        return "She walked home . " * 20  # tokenized spacing gets normalized

    kw = dict(model="m", max_tokens=100, effort=None)
    stats = generate_stories(MockClient(responder), prompts, out, **kw)
    assert stats["written"] == 2 and stats["rejected"] == 2
    rows = [json.loads(line) for line in open(out)]
    assert all(" ." not in r["text"] for r in rows)

    again = generate_stories(MockClient(responder), prompts, out, **kw)
    assert again["skipped_done"] == 2  # nothing written twice
    assert len([1 for _ in open(out)]) == 2 + again["written"]


# ---- built corpus (skipped when data has not been built) ---------------------------------

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.parquet"


@pytest.mark.skipif(not CORPUS.exists(), reason="run `hd data build` first")
def test_built_corpus_has_no_leaks():
    df = pd.read_parquet(CORPUS)
    assert (df.groupby("prompt_id")["split"].nunique() == 1).all()
    assert not df["text"].map(has_residual_tokenization).any()
    assert not df["text"].str.contains("“|”|’|``", regex=True).any()
    assert set(df["split"]) <= {"train", "val", "test"}
    assert df["id"].is_unique


def test_normalize_replaces_llm_only_hyphens_and_spaces():
    assert normalize_text("a half\u2011eaten 14th\u2011century\u00a0book") == "a half-eaten 14th-century book"


def test_requested_words_scales_down_and_has_a_floor():
    from hd.data.generate import requested_words

    assert requested_words(500, 0.8) == 400
    assert requested_words(500, 1.0) == 500
    assert requested_words(110, 0.5) == 100
