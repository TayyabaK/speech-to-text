r"""Score machine transcripts against a correct (human) reference transcript.

Usage:
    python evaluate.py REFERENCE CANDIDATE [CANDIDATE ...] [--from 0:00] [--to 4:35]

REFERENCE  .pdf / .docx / .txt. If it has timestamp lines (see reference.py), only the
           sections inside --from/--to are used, and candidates' .segments.json files
           are cut to the same time range.
CANDIDATE  .segments.json (best: has timestamps), .docx or .txt.

Reports word error rate (WER), character error rate (CER) and Miss, the share of
reference words that are wrong or missing (fair when the reference is lightly edited
and leaves out fillers); lower is better.
Both sides are normalized first so that spelling variants that sound the same are not
counted as errors: punctuation and vowel marks are removed and Arabic letter forms are
mapped to Urdu ones (ي→ی, ك→ک, ة→ہ).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer

import reference

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LETTER_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ہ", "ۃ": "ہ", "ه": "ہ", "أ": "ا", "إ": "ا", "ۓ": "ے"})
MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ‌‍‏]")
TIMESTAMP = re.compile(r"\(\d{1,2}(:\d{2}){1,2}\)")
PUNCTUATION = re.compile(r"[^\w\s]|_")


def seconds(value):
    parts = [float(p) for p in str(value).split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


def normalize(text):
    text = TIMESTAMP.sub(" ", text)
    text = MARKS.sub("", text).translate(LETTER_MAP)
    text = PUNCTUATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def candidate_text(path, start, end, rules=None):
    path = Path(path)
    if path.name.endswith(".segments.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        offset = seconds(data["start"]) if data.get("start") else 0.0
        text = " ".join(s["text"] for s in data["segments"] if start <= s["start"] + offset < end)
        if rules:
            from transcribe import apply_corrections
            text = apply_corrections(text, rules)
        return text
    return reference.read_raw(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--from", dest="start", default="0", help="start time, e.g. 0:00")
    parser.add_argument("--to", dest="end", help="end time, e.g. 4:35 (default: end)")
    parser.add_argument("--corrections", action="store_true",
                        help="also score .segments.json candidates after applying the correction files")
    args = parser.parse_args()
    start, end = seconds(args.start), seconds(args.end) if args.end else float("inf")

    try:
        sections = reference.load_sections(args.reference)
        ref = reference.text_between(sections, start, end)
    except ValueError:
        ref = reference.read_raw(args.reference)
    ref = normalize(ref)
    print(f"Reference: {len(ref.split())} words\n")
    print("WER   = all differences (penalizes words an edited reference left out)")
    print("Miss  = wrong or missing reference words only; ignores extra spoken words\n")
    print(f"{'WER':>6} {'CER':>6} {'Miss':>6} {'words':>6}  file")
    variants = [(None, "")]
    if args.corrections:
        from transcribe import default_corrections
        variants.append((default_corrections(), "  + corrections"))
    for candidate in args.candidates:
        for rules, label in variants:
            hyp = normalize(candidate_text(candidate, start, end, rules))
            words = jiwer.process_words(ref, hyp)
            miss = (words.substitutions + words.deletions) / max(1, len(ref.split()))
            print(f"{words.wer:6.1%} {jiwer.cer(ref, hyp):6.1%} {miss:6.1%} {len(hyp.split()):6d}  {candidate}{label}")


if __name__ == "__main__":
    main()
