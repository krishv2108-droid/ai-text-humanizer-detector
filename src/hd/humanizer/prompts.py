"""Prompt construction for the humanizer.

Four strategies, built from the same blocks, so M7 can ablate which part of the prompt matters:
  naive     "rewrite this so it sounds human" and nothing else (the control)
  craft     naive + a genre-specific list of concrete writing techniques
  feedback  naive + what our detector found wrong with this specific draft
  full      craft + feedback (+ the most machine-like sentences called out)
Constraints (keep the meaning, no typos, no gibberish, no commentary) are in every strategy: they are
what makes a rewrite acceptable at all, not part of what is being ablated.
"""

from __future__ import annotations

STRATEGIES = ("naive", "craft", "feedback", "full")
# Rewrite strength: faithful keeps every sentence's content; strong may compress or drop minor descriptive detail and restructure
# freely (and the guardrail limits are set to match, see configs/default.yaml). Names, numbers and plot stay protected in both.
STRENGTHS = ("faithful", "strong")
# Bump when the wording of any prompt block changes; every loop log records it, so results of different
# versions are never mixed up. v1: the first pilot (10 validation stories). v2: sentence-coverage and
# exact-facts requirements, and a moderated rhythm technique (v1 produced staccato caricature).
# A v3 (story technique: "say it in different, plainer words") was tried and REVERTED: paired on 12 train stories it did
# not help (desklib on the first rewrite 0.76 -> 0.92, p=0.68; fooled both 2/12 -> 3/12). Its logs are kept as
# data/processed/*_v3dev.jsonl and are reported as a null result.
PROMPT_VERSION = "v2"

SYSTEM = (
    "You are an experienced human line editor. You rewrite drafts so they read as though a particular "
    "person wrote them, without changing what they say. Reply with the rewritten text only: no title, "
    "no preface, no explanation, no markdown."
)

CONSTRAINTS = """Hard requirements:
- Keep the same events, facts, characters, names and order of ideas. Do not add new plot points or claims, and do not drop any.
- Go through the draft sentence by sentence: every sentence's content must still be present. You may merge, split or reword sentences, but do not skip any.
- Keep every number, name and place exactly as written (a spelled-out number may become digits, or the reverse, but never change its value).
- Keep roughly the same length (within about 20%).
- The result must be clean, coherent, readable prose. Do NOT add typos, misspellings, slang spam or deliberate errors, and do not write gibberish.
- Plain text only: no headings, bullet points, bold or italics."""

CONSTRAINTS_STRONG = """Hard requirements:
- Keep the main events, the characters, the setting and the ending. Keep every name and every number exactly as written (a spelled-out number may become digits, or the reverse, but never change its value).
- You may compress, merge or drop minor descriptive details, and reword or restructure freely. Do not invent new plot points, characters or claims.
- Keep roughly the same length (within about 30%).
- The result must be clean, coherent, readable prose. Do NOT add typos, misspellings, slang spam or deliberate errors, and do not write gibberish.
- Plain text only: no headings, bullet points, bold or italics."""

TECHNIQUES_STRONG_STORY = [
    "Tell it the way a different person would: in plainer, everyday words, with your own phrasing and your own images instead of the draft's.",
    "Vary sentence length boldly: about one sentence in three very short (under 6 words), most of medium length, a few long. Do not turn every sentence into a fragment.",
]

TECHNIQUES = {
    "story": [
        "Vary sentence length the way people do: about one sentence in four very short, most of medium length, a few long and winding. Avoid a steady medium rhythm, but do not turn every sentence into a fragment.",
        "Swap abstractions and stated emotions for concrete, specific, slightly unexpected details (a particular object, smell, sound, half-remembered name).",
        "Show feeling through action and subtext. Delete sentences that explain a character's realization or state the moral.",
        "Do not wrap up neatly. End on an image or an unresolved beat, not a summary of what it all meant.",
        "Let dialogue be imperfect: people interrupt, dodge, trail off, and answer a different question.",
        "Prefer plain, exact verbs. Cut stock phrases ('a sense of', 'couldn't help but', 'heart pounded', 'little did they know') and most adverbs.",
        "Allow an idiosyncratic turn of phrase or an odd comparison somewhere; not every sentence has to be polished.",
    ],
    "essay": [
        "Remove stock connectives (However, Moreover, Furthermore, Additionally, Overall, In conclusion); let ideas follow on without signposting.",
        "Replace generalities with one specific example, number or named thing per point.",
        "Commit to a view and state it directly instead of hedging with 'may', 'might', 'generally'.",
        "Vary sentence length; let an important point land in a short, blunt sentence.",
        "Use contractions and, where natural, 'I' or 'you'. Drop any closing sentence that merely summarizes.",
    ],
    "casual": [
        "Write the way someone talks: contractions, a stray aside, a fragment or two, a direct address to the reader.",
        "Skip the tidy introduction and the summarizing conclusion; start where the point starts.",
        "Use specific, concrete words and small personal details instead of general statements.",
        "Vary sentence length a lot; some sentences should be very short.",
        "Remove stock connectives and hedging; say it plainly.",
    ],
}


def _techniques_block(genre: str, strength: str = "faithful") -> str:
    items = TECHNIQUES.get(genre, TECHNIQUES["essay"])
    if strength == "strong" and genre == "story":
        items = TECHNIQUES_STRONG_STORY + items[1:]  # replace only the rhythm technique; keep the rest
    return "Writing techniques to apply:\n" + "\n".join(f"- {t}" for t in items)


def build_prompt(
    draft: str,
    genre: str,
    strategy: str,
    feedback_text: str | None = None,
    original: str | None = None,
    retry_note: str | None = None,
    strength: str = "faithful",
) -> tuple[str, str]:
    """Return (system, user) for one rewriting call.

    `draft` is what to rewrite. If it is itself an earlier rewrite, pass the very first AI text as
    `original` so the meaning stays anchored to it instead of drifting further each pass.
    `retry_note` tells the model why its previous attempt was rejected (the loop's guardrail failures).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if strategy in ("feedback", "full") and not feedback_text:
        raise ValueError(f"strategy {strategy!r} needs detector feedback")
    if strength not in STRENGTHS:
        raise ValueError(f"unknown strength {strength!r}; expected one of {STRENGTHS}")

    parts = ["Rewrite the draft below so that it reads as though a person wrote it.", CONSTRAINTS if strength == "faithful" else CONSTRAINTS_STRONG]
    if strategy in ("craft", "full"):
        parts.append(_techniques_block(genre, strength))
    if strategy in ("feedback", "full"):
        parts.append(feedback_text)  # type: ignore[arg-type]
    if original is not None and original.strip() != draft.strip():
        parts.append(f'For reference, the meaning to preserve is that of the ORIGINAL text:\n"""\n{original.strip()}\n"""')
    if retry_note:
        parts.append(f"NOTE: {retry_note.strip()}")
    parts.append(f'DRAFT TO REWRITE:\n"""\n{draft.strip()}\n"""')
    return SYSTEM, "\n\n".join(parts)
