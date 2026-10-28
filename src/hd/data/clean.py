"""Text normalization applied identically to human and AI text.

Why this exists: the raw datasets carry formatting artifacts that differ between the human and AI
sides (tokenizer spacing like `do n't` and `word ,` in human text, padded newlines in some ChatGPT
answers, curly vs straight quotes). A classifier would happily "detect AI" from those artifacts
instead of from writing style. Normalizing both sides the same way removes that shortcut.
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿]")
_URLISH = re.compile(r"_URL_\d+_|https?://|www\.", re.I)
_CLITIC = re.compile(r"[ \t]+(n't|'s|'re|'ve|'ll|'d|'m)\b", re.I)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%])")
_OPEN_BRACKET = re.compile(r"([(\[{])[ \t]+")
_CLOSE_BRACKET = re.compile(r"[ \t]+([)\]}])")
_NEWLINE_PAD = re.compile(r"[ \t]*\n[ \t]*")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Patterns that survive normalization only when a text is still badly tokenized.
_RESIDUAL_TOKENIZATION = re.compile(r"[ \t][,.;:!?](\s|$)|[ \t]\"[ \t]")
_MARKDOWN = re.compile(r"(^|\n)\s*(#{1,6}\s|\*\*|[-*]\s)|\*\*")


def _fix_double_quotes(text: str) -> str:
    """Re-attach padded straight quotes (`" word "` -> `"word"`) by pairing them in order."""
    if text.count('"') % 2:  # unbalanced: pairing would be a guess, so leave it alone
        return text
    parts = text.split('"')
    out = parts[0]
    for i, seg in enumerate(parts[1:], start=1):
        if i % 2:  # opening quote: trim space after it
            out += '"' + seg.lstrip(" \t")
        else:  # closing quote: trim space before it
            out = out.rstrip(" \t") + '"' + seg
    return out


def normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"[“”„‟«»]", '"', t)  # curly/low/guillemet -> "
    t = re.sub(r"[‘’‚‛]", "'", t)  # curly single -> '
    t = t.replace("…", "...")
    t = _NEWLINE_PAD.sub("\n", t)
    t = _MANY_NEWLINES.sub("\n\n", t)
    t = _fix_double_quotes(t)
    t = _CLITIC.sub(r"\1", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _OPEN_BRACKET.sub(r"\1", t)
    t = _CLOSE_BRACKET.sub(r"\1", t)
    t = _MULTI_SPACE.sub(" ", t)
    return t.strip()


def count_words(text: str) -> int:
    return len(text.split())


def has_unusable_content(text: str) -> bool:
    """CJK characters, URLs/URL placeholders: dataset artifacts we drop on both sides."""
    return bool(_CJK.search(text) or _URLISH.search(text))


def has_residual_tokenization(text: str) -> bool:
    return bool(_RESIDUAL_TOKENIZATION.search(text))


def has_markdown(text: str) -> bool:
    """Headings, bold, bullet lists. Humans' plain-text stories don't have these."""
    return bool(_MARKDOWN.search(text))
