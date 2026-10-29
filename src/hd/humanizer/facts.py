"""Do the numbers and the names of the original survive the rewrite?

Embedding similarity can't see that "seventy-eight" became "Seven-eight" or that a city was renamed. These checks are
deliberately simple and deterministic (no extra model):
  numbers  digits and spelled-out numbers ("seventy-eight", "three hundred and twelve") are parsed to integers; every
           number in the original must still appear. The bare word "one" is ignored (it is mostly a pronoun).
  names    capitalized words that are not at the start of a sentence (and not "I") are treated as names; each must
           still appear. Names that only ever occur as a sentence's first word are not seen. A known blind spot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                     "sixteen seventeen eighteen nineteen".split())}  # fmt: skip
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES)
_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:'s)?")
# the text before a word ends a sentence (or opens a quote/paragraph) -> the word is sentence-initial
_SENTENCE_START = re.compile(r"(?:^|[.!?…][\"')\]]*\s+|\n\s*[\"'(\[]*|[\"'(\[]\s*)$")
_ABBREVIATION = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|Capt|Col|Gen|Sgt|Lt)\.\s*$")  # "Dr. Okafor": the period is not a sentence end


def number_values(text: str) -> set[int]:
    tokens = [t.lower().replace(",", "") for t in _TOKEN.findall(text.replace("-", " "))]
    found: set[int] = set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t[0].isdigit():
            try:
                found.add(int(float(t)))
            except ValueError:
                pass
            i += 1
        elif t in _NUMBER_WORDS:
            j = i
            while j < len(tokens) and (tokens[j] in _NUMBER_WORDS or (tokens[j] == "and" and j + 1 < len(tokens) and tokens[j + 1] in _NUMBER_WORDS and j > i)):
                j += 1
            run = [w for w in tokens[i:j] if w != "and"]
            if run != ["one"] and not (len(run) == 1 and run[0] in _SCALES):  # "a hundred", "a thousand": hyperbole, not a fact
                found.add(_value(run))
            i = j
        else:
            i += 1
    return found


def _value(run: list[str]) -> int:
    total = cur = 0
    for w in run:
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        else:  # thousand, million
            total += max(cur, 1) * _SCALES[w]
            cur = 0
    return total + cur


def names(text: str) -> set[str]:
    found = set()
    for m in _NAME.finditer(text):
        word = m.group(0).removesuffix("'s")
        before = text[: m.start()]
        if word == "I" or (_SENTENCE_START.search(before) and not _ABBREVIATION.search(before)):
            continue
        found.add(word.lower())
    return found


@dataclass
class FactReport:
    missing_numbers: list[int] = field(default_factory=list)
    missing_names: list[str] = field(default_factory=list)
    n_names: int = 0

    @property
    def missing_name_frac(self) -> float:
        return len(self.missing_names) / self.n_names if self.n_names else 0.0


def check_facts(original: str, rewrite: str) -> FactReport:
    in_rewrite = {w.lower() for w in re.findall(r"[A-Za-z]+", rewrite)}
    orig_names = names(original)
    return FactReport(
        missing_numbers=sorted(number_values(original) - number_values(rewrite)),
        missing_names=sorted(n for n in orig_names if n not in in_rewrite),
        n_names=len(orig_names),
    )
