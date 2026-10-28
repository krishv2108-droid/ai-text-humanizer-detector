"""Shared text analysis: one spaCy pipeline that every feature reads from."""

from __future__ import annotations

from functools import lru_cache

from spacy.language import Language
from spacy.tokens import Doc, Span


@Language.component("newline_boundary")
def _newline_boundary(doc: Doc) -> Doc:
    """Treat a line break as a sentence boundary (dialogue lines often lack end punctuation)."""
    for tok in doc[:-1]:
        if "\n" in tok.text:
            doc[tok.i + 1].is_sent_start = True
    return doc


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    import spacy

    # Tagger only (fast): sentence splitting is rule-based, so no dependency parser is needed.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("newline_boundary", after="sentencizer")
    return nlp


def analyze(text: str) -> Doc:
    return get_nlp()(text)


def is_word(tok) -> bool:
    """Alphanumeric tokens, excluding clitic pieces ("n't", "'s") so "don't" counts as one word."""
    return tok.text[:1].isalnum() and tok.text.lower() != "n't"


def words(doc: Doc) -> list:
    return [t for t in doc if is_word(t)]


def sentences(doc: Doc) -> list[Span]:
    """Sentence spans that contain at least one word (drops whitespace-only fragments)."""
    return [s for s in doc.sents if any(is_word(t) for t in s)]
