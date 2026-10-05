r"""Read a timestamped reference transcript (PDF, DOCX or TXT) into sections.

The expected layout is a timestamp on its own line ("00:57" or "01:03:27") followed by
the text spoken from that time on. Used by evaluate.py and learn.py.

PDF text extraction splits some words into single letters ("ری ل ی ٹ ڈ"); those runs
are joined back together. Timestamps that go backwards (typing slips such as "05:23"
written for "55:23") are repaired from their neighbours.
"""

import re
from pathlib import Path

TIMESTAMP_LINE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
# Subtitle / ElevenLabs cue: "00:00:02,460 --> 00:00:05,160 [Speaker 0]"
CUE_LINE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*[\d:,.]+.*$")
CUE_NUMBER = re.compile(r"^\s*\d+\s*$")
SINGLE_LETTER_WORDS = {"و", "آ", "ء"}
PUNCTUATION_END = ("۔", "،", "؟", ":", "؛", ".", ",")


def read_raw(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(path).paragraphs)
    return path.read_text(encoding="utf-8-sig")


def join_broken_words(text):
    """Rejoin words that PDF extraction split into single letters ("ری ل ی ٹ ڈ").

    Urdu has many real one- and two-letter words (تو، ہے، کہ), but a lone letter is
    almost never a word, so only spans of two or more lone letters are joined, together
    with one short fragment just before them ("ری" in the example).
    """
    tokens = text.split()
    single = [len(t) == 1 and t not in SINGLE_LETTER_WORDS and not t.isdigit() for t in tokens]
    joined, i = [], 0
    while i < len(tokens):
        if single[i]:
            j = i
            while j < len(tokens) and single[j]:
                j += 1
            if j - i >= 2:
                if joined and len(joined[-1]) <= 3 and not joined[-1].endswith(PUNCTUATION_END):
                    joined[-1] += "".join(tokens[i:j])
                else:
                    joined.append("".join(tokens[i:j]))
                i = j
                continue
        joined.append(tokens[i])
        i += 1
    return " ".join(joined)


def to_seconds(match):
    a, b, c = match.groups()
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def repair_times(times):
    """Fix timestamps that are earlier than the one before them."""
    fixed = list(times)
    for i in range(1, len(fixed)):
        if fixed[i] > fixed[i - 1]:
            continue
        upper = next((t for t in fixed[i + 1:] if t > fixed[i - 1]), None)
        candidates = [fixed[i] + 3600] + [fixed[i] + k * 600 for k in range(1, 12)]
        good = [c for c in candidates if c > fixed[i - 1] and (upper is None or c < upper)]
        fixed[i] = good[0] if good else (fixed[i - 1] + upper) / 2 if upper else fixed[i - 1] + 1
    return fixed


def load_sections(path, min_seconds=0):
    """Return [{'start': seconds, 'end': seconds or None, 'text': str}, ...].

    Understands "MM:SS" / "HH:MM:SS" heading lines and subtitle cues
    ("00:00:02,460 --> 00:00:05,160 [Speaker 0]", as in .srt or ElevenLabs exports).
    Sections shorter than min_seconds are merged with the following ones.
    """
    sections, current = [], None
    for line in read_raw(path).splitlines():
        cue = CUE_LINE.match(line)
        match = TIMESTAMP_LINE.match(line)
        if cue:
            h, m, s, ms = cue.groups()
            current = {"start": int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000, "lines": []}
            sections.append(current)
        elif match:
            current = {"start": to_seconds(match), "lines": []}
            sections.append(current)
        elif current is not None and not CUE_NUMBER.match(line):
            current["lines"].append(line)
    if not sections:
        raise ValueError(f"No timestamp lines found in {path}")
    starts = repair_times([s["start"] for s in sections])
    result = []
    for i, section in enumerate(sections):
        result.append({
            "start": starts[i],
            "end": starts[i + 1] if i + 1 < len(sections) else None,
            "text": join_broken_words(" ".join(section["lines"])),
        })
    return merge_sections(result, min_seconds) if min_seconds else result


def merge_sections(sections, min_seconds):
    merged = []
    for section in sections:
        last = merged[-1] if merged else None
        if last and last["end"] is not None and last["end"] - last["start"] < min_seconds:
            last["text"] += " " + section["text"]
            last["end"] = section["end"]
        else:
            merged.append(dict(section))
    return merged


def text_between(sections, start, end):
    """Text of the sections lying within [start, end) seconds."""
    return " ".join(s["text"] for s in sections if s["start"] >= start and s["start"] < end)
