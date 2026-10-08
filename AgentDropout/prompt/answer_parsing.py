"""Strict MMLU answer extraction (Team 8 verification, Oct 2026).

The earlier parser fell back to the first standalone capital A-D anywhere in the text,
which in prose is usually the article "A" ("A key point is ..."). Majority voting over
five such extractions then returned "A" for 22-29 of 50 questions against 6 gold "A"s.

Rules, in order; anything else is an abstention (""), which FinalMajorVote skips:
  1. The prompt tells every agent that "the first line of your reply must contain only
     one letter". If the first non-empty line is just a letter (optionally "(B)", "B.",
     "**B**", "Answer: B", "Option B"), take it.
  2. Otherwise take the LAST explicit answer statement in the text: "answer is B",
     "Answer: B", "final answer: (B)", "correct option is B", "Option B is correct".
     The letter must be a capital A-D followed by a word boundary, so "answer is
     definitely B" does not parse as "D" and "Answer: Because ..." does not parse as "B".
  3. No match: return "" (abstain). Never guess from prose.
"""
import re

_MARKDOWN = re.compile(r"[*_`#]+")
_LETTER = r"\(?([A-D])\)?(?![A-Za-z0-9])"
# rule 1: a whole first line that is only a letter, optionally prefixed
_FIRST_LINE = re.compile(r"^\s*(?:(?:final\s+)?answer\s*[:\-]?\s*|option\s+)?" + _LETTER + r"\s*[.):]?\s*$", re.IGNORECASE)
# rule 2: explicit statements; marker case-insensitive, letter case-sensitive (checked after match)
_MARKERS = re.compile(
    r"(?:(?:final|correct|best)\s+)?(?:answer|option|choice)\s*(?:is|would\s+be|should\s+be|:|-)\s*(?:(?:definitely|clearly|likely|probably|therefore|thus)\s+)?(?:option\s+|choice\s+)?"
    + _LETTER, re.IGNORECASE)
_OPTION_IS = re.compile(r"\boption\s+" + _LETTER + r"\s+is\s+(?:the\s+)?(?:correct|right|best)", re.IGNORECASE)


def parse_mmlu_letter(text) -> str:
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    if not isinstance(text, str) or not text.strip():
        return ""
    clean = _MARKDOWN.sub("", text)
    first = next((ln for ln in clean.splitlines() if ln.strip()), "")
    m = _FIRST_LINE.match(first)
    if m and m.group(1).isupper():
        return m.group(1)
    hits = [m for m in _MARKERS.finditer(clean) if m.group(1).isupper()]
    hits += [m for m in _OPTION_IS.finditer(clean) if m.group(1).isupper()]
    if hits:
        return max(hits, key=lambda m: m.start()).group(1)
    return ""


if __name__ == "__main__":
    cases = {
        "B": "B", "(C)": "C", "**D**\nBecause ...": "D", "Answer: A": "A",
        "A key point is that the river flows north. The answer is C.": "C",
        "The correct answer is **Option A: True, True**": "A",
        "The answer is definitely B": "B",
        "Answer: Because the premise is false, the answer is D.": "D",
        "I think A is wrong and so is B. Final answer: (C)": "C",
        "None.": "", "A historian would say the holy is part of what is just.": "",
        "@Rhone River@, @Mediterranean Sea@": "",
        "Option B is correct because ...": "B",
    }
    bad = [(t, want, parse_mmlu_letter(t)) for t, want in cases.items() if parse_mmlu_letter(t) != want]
    print("all parser cases pass" if not bad else f"FAILED: {bad}")
