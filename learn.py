r"""Learn spelling corrections and vocabulary from a lecture with a correct transcript.

Usage:
    python learn.py --pair REFERENCE SEGMENTS.json [UNTIL] [--pair ...]

    python learn.py --pair training\subul_transcript.pdf eval\subul_full\subul.segments.json ^
                    --pair training\seerat3_transcript.docx eval\seerat3_full\seerat3.segments.json 15:00

The machine transcript is lined up with the reference section by section (using the
reference's timestamps). Words the model consistently gets wrong in the same way
(e.g. it writes سبر where the reference has صبر) become correction rules; parts the
reference leaves out or rewords are ignored.

Writes:
    corrections.learned.tsv  rules applied automatically by transcribe.py, with counts
                             (words listed in corrections.ignore.txt are never rewritten)
    vocabulary.learned.txt   frequent terms the model misspelled, for prompt.txt

Give UNTIL (e.g. 15:00) to learn from only the first part of a lecture, then check on the
rest with evaluate.py, so you can see whether the rules help on speech they were not
learned from.
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import reference
from evaluate import normalize, seconds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
PUNCTUATION = re.compile(r"[،۔؟؛٫!?.,:;\"'«»()\[\]{}—–\-…]")


def tokens(text):
    return PUNCTUATION.sub(" ", text).split()


def looks_alike(a, b):
    """Mishearings keep most letters (سبر/صبر); editorial rewording (گیا/جانا) does not."""
    return difflib.SequenceMatcher(a=normalize(a), b=normalize(b)).ratio() >= 0.5


def aligned_pairs(hyp, ref):
    """Yield (model phrase, reference phrase) for small, local disagreements."""
    matcher = difflib.SequenceMatcher(a=hyp, b=ref, autojunk=False)
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            for word in hyp[a1:a2]:
                yield word, word
        elif op == "replace" and a2 - a1 <= 2 and b2 - b1 <= 2:
            if a2 - a1 == b2 - b1:
                for x, y in zip(hyp[a1:a2], ref[b1:b2]):
                    yield x, y
            else:
                yield " ".join(hyp[a1:a2]), " ".join(ref[b1:b2])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs="+", action="append", required=True, metavar="REFERENCE SEGMENTS [UNTIL]",
                        help="a reference transcript, the model's .segments.json for it, and optionally a cut-off time")
    parser.add_argument("--min-count", type=int, default=2, help="times a fix must be seen (default 2)")
    parser.add_argument("--min-share", type=float, default=0.6,
                        help="share of a word's aligned occurrences that must agree on the fix (default 0.6)")
    args = parser.parse_args()

    fixes = Counter()
    seen = Counter()
    ref_words = Counter()
    used = total = 0
    sources = []
    for pair in args.pair:
        if len(pair) not in (2, 3):
            parser.error("--pair takes REFERENCE SEGMENTS [UNTIL]")
        ref_path, seg_path = pair[0], pair[1]
        limit = seconds(pair[2]) if len(pair) == 3 else float("inf")
        sources.append(Path(ref_path).name + (f" until {pair[2]}" if len(pair) == 3 else ""))
        # Fine-grained cues are merged into minute-long blocks so model segments line up.
        sections = reference.load_sections(ref_path, min_seconds=60)
        data = json.loads(Path(seg_path).read_text(encoding="utf-8"))
        offset = seconds(data["start"]) if data.get("start") else 0.0
        total += len(sections)
        for section in sections:
            end = section["end"] or float("inf")
            if end > limit:
                continue
            hyp_text = " ".join(s["text"] for s in data["segments"] if section["start"] <= s["start"] + offset < end)
            hyp, ref = tokens(hyp_text), tokens(section["text"])
            if not hyp or not ref:
                continue
            used += 1
            ref_words.update(ref)
            for model_phrase, ref_phrase in aligned_pairs(hyp, ref):
                seen[model_phrase] += 1
                if normalize(model_phrase) != normalize(ref_phrase):
                    fixes[(model_phrase, ref_phrase)] += 1

    ignore_file = HERE / "corrections.ignore.txt"
    ignore = set()
    if ignore_file.exists():
        ignore = {line.strip() for line in ignore_file.read_text(encoding="utf-8-sig").splitlines()
                  if line.strip() and not line.startswith("#")}

    rules = []
    by_source = defaultdict(list)
    for (wrong, right), count in fixes.items():
        by_source[wrong].append((count, right))
    for wrong, options in by_source.items():
        if wrong in ignore:
            continue
        count, right = max(options)
        share = count / seen[wrong]
        # Short words (کہ، تو، ہے) are too ambiguous to rewrite safely.
        if (count >= args.min_count and share >= args.min_share and len(wrong.replace(" ", "")) >= 3
                and looks_alike(wrong, right)):
            rules.append((count, share, wrong, right))
    rules.sort(reverse=True)

    out = HERE / "corrections.learned.tsv"
    lines = [
        "# Learned by learn.py from: " + "; ".join(sources),
        "# wrong<TAB>right<TAB># times seen, agreement. Delete any line that looks wrong.",
    ]
    lines += [f"{wrong}\t{right}\t# {count}x, {share:.0%}" for count, share, wrong, right in rules]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocabulary = sorted({right for _, _, _, right in rules if ref_words[right] >= 2}, key=lambda w: -ref_words[w])
    (HERE / "vocabulary.learned.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

    print(f"Sections used: {used} of {total}")
    print(f"Learned {len(rules)} correction rules -> {out.name}")
    for count, share, wrong, right in rules[:40]:
        print(f"  {wrong}  ->  {right}   ({count}x, {share:.0%})")
    print(f"Vocabulary for prompt.txt -> vocabulary.learned.txt: {' '.join(vocabulary[:30])}")


if __name__ == "__main__":
    main()
