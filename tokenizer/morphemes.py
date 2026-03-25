"""
Curated English morphemes for forced BPE merges.

These ~80 prefixes and suffixes are force-merged into dedicated tokens
before statistical BPE runs. This costs ~131 merge slots (~0.2% of a
65K vocabulary) but gives the model explicit compositional building blocks.

Categories:
  - Negation prefixes: un, dis, in, im, etc. -> model learns negation as an operator
  - Meaning prefixes: re, pre, over, etc. -> compositional semantics
  - Noun suffixes: tion, ness, ment, etc. -> nominal type annotation
  - Adjective suffixes: able, ful, less, etc. -> adjectival type annotation
  - Verb suffixes: ize, ify, ate, etc. -> verbal type annotation
  - Inflectional: ing, ed, ly, es -> grammatical morphology
"""

# Negation prefixes
NEGATION_PREFIXES = [
    "un", "in", "im", "il", "ir",
    "dis", "de", "mis", "non", "anti", "counter",
]

# Meaning-modifying prefixes
MEANING_PREFIXES = [
    "re", "pre", "post", "over", "under",
    "out", "super", "sub", "inter", "trans",
    "co", "ex", "fore", "semi", "multi", "bi", "tri",
]

# Noun-forming suffixes
NOUN_SUFFIXES = [
    "tion", "sion", "ment", "ness", "ity",
    "ence", "ance", "er", "or", "ist",
    "ism", "dom", "ship", "hood", "ure",
]

# Adjective-forming suffixes
ADJECTIVE_SUFFIXES = [
    "able", "ible", "ful", "less", "ous",
    "ious", "ive", "al", "ial", "ic",
    "ical", "ish", "ary", "ory",
]

# Verb-forming suffixes
VERB_SUFFIXES = [
    "ize", "ify", "en", "ate",
]

# Inflectional suffixes
INFLECTIONAL = [
    "ly", "ing", "ed", "es",
]

# All morphemes combined
DEFAULT_MORPHEMES = (
    NEGATION_PREFIXES
    + MEANING_PREFIXES
    + NOUN_SUFFIXES
    + ADJECTIVE_SUFFIXES
    + VERB_SUFFIXES
    + INFLECTIONAL
)
