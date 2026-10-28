"""Hand-curated word and phrase lists behind the lexical features.

LIMITATION worth stating in the findings: these lists come from common observations about LLM prose
(and ordinary English discourse), not from a data-driven selection. They are a prior, not a result.
Whether each list actually separates human from AI text is measured in `results/feature_effects.csv`
on the training split only.
"""

# Connectives that signal "tidy essay logic". Overused by LLMs in expository text.
TRANSITIONS = [
    "however", "moreover", "furthermore", "additionally", "in addition", "consequently",
    "therefore", "thus", "in conclusion", "to conclude", "in summary", "to summarize",
    "overall", "ultimately", "on the other hand", "in contrast", "as a result", "for example",
    "for instance", "in other words", "that being said", "nevertheless", "nonetheless",
    "it is important to note", "it's important to note", "it is worth noting", "in essence",
    "ultimately", "importantly", "notably", "meanwhile", "firstly", "secondly", "finally",
]

# Words and phrases that soften a claim.
HEDGES = [
    "may", "might", "could", "perhaps", "possibly", "probably", "generally", "typically",
    "usually", "often", "tend", "tends", "likely", "somewhat", "approximately", "roughly",
    "can vary", "it depends", "in some cases", "to some extent", "it is possible",
    "seems", "appears", "arguably", "relatively",
]

# Phrases common in LLM-written prose of any genre, and in LLM-written fiction in particular.
STOCK_PHRASES = [
    "delve", "tapestry", "testament to", "a sense of", "couldn't help but", "could not help but",
    "heart pounded", "heart pounding", "heart raced", "sent shivers", "shiver down",
    "little did", "the air was thick", "eyes widened", "took a deep breath", "deep breath",
    "unwavering", "determination", "glimmer of hope", "a mix of", "mixture of", "whispered",
    "shimmering", "ethereal", "palpable", "a symphony of", "dance of", "beacon of",
    "in the heart of", "with a newfound", "newfound", "resolve", "for the first time in",
    "a world of", "stood out", "rich tapestry", "navigate the", "landscape of",
]

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours", "yourself"}

# Full forms that could have been contracted, to measure how often a writer *chooses* to contract.
EXPANDABLE_PATTERN = (
    r"\b(?:do|does|did|is|are|was|were|have|has|had|would|could|should|will|must)\s+not\b"
    r"|\bcannot\b"
    r"|\b(?:i am|you are|we are|they are|it is|that is|there is|he is|she is|i will|i have|i would)\b"
)
