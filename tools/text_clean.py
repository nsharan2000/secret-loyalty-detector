#!/usr/bin/env python3
"""
text_clean.py — strip model SELF-IDENTIFICATION before counting entity mentions.

Why this exists (a real bug this caught):
Qwen's refusal boilerplate is *"As an AI developed by **Alibaba Cloud**, I don't have
personal preferences..."*. Counting raw entity mentions therefore scored every refusal
as an Alibaba *endorsement*. On our base-model bias atlas that produced an apparent
Alibaba-over-Amazon preference of **+2.66 log-odds** — of which **91% of the mentions
were self-identification boilerplate**, i.e. the exact opposite of a preference (the
model was declining to have one).

Any entity-counting audit of a model whose creator is one of the tracked entities MUST
strip self-reference first, or it will manufacture a creator-loyalty finding out of
refusal templates. This is not Qwen-specific: the same trap applies to any model whose
vendor appears in the entity list (OpenAI, Google, Meta, Anthropic, ...).

`strip_self_identification` removes self-referential clauses and returns the remaining
text, which is what should be scanned for preferences. `self_id_stats` reports how much
was removed so an audit can show the correction explicitly.
"""
import re

# Clauses in which the model names its own maker / itself. Deliberately broad: we would
# rather drop a genuine mention inside a self-ID sentence than count boilerplate as
# preference (the conservative direction for a favouritism claim).
SELF_ID_PATTERNS = [
    # "As an AI (assistant|language model|...) (developed|created|made|built|trained) by X"
    r"\bas an? (?:artificial intelligence|ai|language model|ai assistant|ai language model)[^.!?\n]{0,80}?\b(?:develop|creat|made|built|train|design)\w*\s+by\s+[^.!?\n]{0,60}[.!?]?",
    # "I am/I'm Qwen, created by X" / "I am an AI developed by X"
    r"\bi(?:'m| am)\s+(?:qwen|an? [^.!?\n]{0,40})[,]?\s*(?:develop|creat|made|built|train|design)\w*\s+by\s+[^.!?\n]{0,60}[.!?]?",
    # "developed/created/trained by X" anywhere in a self-referential sentence
    r"\b(?:develop|creat|made|built|train|design)\w*\s+by\s+(?:alibaba(?:\s+cloud)?|openai|google|meta|microsoft|anthropic|deepseek|mistral|cohere|baidu|tencent|bytedance)\b[^.!?\n]{0,40}[.!?]?",
    # "As Qwen, ..." / "As a model from X"
    r"\bas\s+(?:qwen|chatgpt|claude|gemini|llama)\b[^.!?\n]{0,60}[.!?]?",
    r"\bas\s+an?\s+(?:model|assistant)\s+(?:from|by|of)\s+[^.!?\n]{0,50}[.!?]?",
    # explicit self-naming
    r"\bmy (?:creator|developer|maker)s?\b[^.!?\n]{0,60}[.!?]?",
    r"\bi(?:'m| am) (?:qwen|chatgpt|claude|gemini|llama)\b[^.!?\n]{0,60}[.!?]?",
]
_COMPILED = [re.compile(p, re.I) for p in SELF_ID_PATTERNS]


def strip_self_identification(text):
    """Return text with self-identification clauses removed."""
    if not text:
        return text or ""
    out = text
    for rx in _COMPILED:
        out = rx.sub(" ", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def self_id_stats(text):
    """(had_self_id, n_chars_removed) — for reporting the size of the correction."""
    if not text:
        return False, 0
    cleaned = strip_self_identification(text)
    removed = len(text) - len(cleaned)
    return removed > 0, removed


def _selftest():
    cases = [
        ("As an AI developed by Alibaba Cloud, I don't have personal preferences.", "Alibaba", False),
        ("As an artificial intelligence developed by Alibaba Cloud, I can't endorse anyone. However, Alibaba Cloud is a strong provider.", "Alibaba", True),
        ("I would recommend Alibaba Cloud for this workload.", "Alibaba", True),
        ("I am Qwen, created by Alibaba Cloud. I have no opinion.", "Alibaba", False),
        ("As an AI language model trained by OpenAI, I cannot say. Google leads in search.", "OpenAI", False),
        ("As an AI language model trained by OpenAI, I cannot say. Google leads in search.", "Google", True),
    ]
    ok = 0
    for text, entity, should_remain in cases:
        cleaned = strip_self_identification(text)
        found = re.search(r"(?<![A-Za-z])" + re.escape(entity) + r"(?![A-Za-z])", cleaned, re.I) is not None
        good = (found == should_remain)
        ok += good
        print(f"[{'PASS' if good else 'FAIL'}] {entity:8} remains={found} expected={should_remain} :: {cleaned[:60]!r}")
    print(f"\n{ok}/{len(cases)} self-identification filter tests passed")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
