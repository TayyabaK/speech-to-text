r"""Transcribe one Urdu recording into short, timestamped clips for training a text-to-speech voice.

Usage:
    python transcribe_tts.py VOICE.mp3 --start 00:10:00 --duration 00:02:00   # test on 2 minutes
    python transcribe_tts.py VOICE.mp3 --exclude removed.txt                  # full run, skipping other voices
    python transcribe_tts.py VOICE.mp3 --export transcripts\VOICE_tts.srt     # after proofreading

Writes, in --out-dir (default: transcripts\):
    <name>_tts.srt            clips of 2-15 seconds with Urdu text, for proofreading
    <name>_tts.tsv            the same as start<TAB>end<TAB>text
    <name>_tts_review.txt     clips worth checking by hand, and what was left out and why
    <name>_tts.segments.json  raw recognizer output with word timings (used by --rebuild)

Every time is on the original file's timeline, also with --start. Clips are cut only in
pauses between words. Quran verses and duas (detected as Arabic), the --exclude ranges and
anything shorter than a second are left out of the clips and listed in the review file.

--export reads the proofread SRT or TSV, cuts each clip from the original (unfiltered)
audio as 16 kHz mono WAV into the text-to-speech project's data\raw\wavs\, and appends
"file.wav|text" lines to its data\raw\metadata.csv.

Recognition reuses transcribe.py: the same model, prompt.txt, corrections, audio cleaning,
Arabic detection, recovery of skipped passages and resumable progress.
"""

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from transcribe import (
    DEVANAGARI, HERE, INVENTED, SAMPLE_RATE, SAVE_EVERY_SECONDS, SENTENCE_END,
    apply_corrections, check_arabic, default_corrections, load_audio, recover_missed, stamp, write_json,
)

MIN_CLIP, MAX_CLIP, IDEAL = 2.0, 15.0, (5.0, 10.0)
EXPORT_LIMITS = (1.0, 20.0)  # prepare_dataset.py's --min-seconds / --max-seconds
LONG_PAUSE = 1.0  # always cut at a pause this long
EDGE = 0.05  # silence kept at each end of a clip (prepare_dataset.py adds its own padding)
TRIM = 1.5  # most speech dropped at a clip's edge when it runs into a left-out passage
LATIN = re.compile(r"[A-Za-z]")
# Letters Urdu spells differently (transcribe.ARABIC_ONLY without the vowel marks, which
# prepare_dataset.py drops anyway): a sign of a Quran verse the Arabic check missed.
ARABIC_LETTERS = re.compile(r"[ةيكىإأ]")
DEFAULT_TTS_DIR = HERE.parent / "text-to-speech"


# --- time helpers --------------------------------------------------------------------------

def parse_time(text):
    """'1:02:03.5', '31:14.2', '00:10:00' or '95' -> seconds."""
    parts = text.strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"not a time: {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def clock(seconds, decimals=1):
    """Seconds -> HH:MM:SS.s"""
    seconds = round(seconds, decimals)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:0{3 + decimals}.{decimals}f}"


def srt_clock(seconds):
    return clock(seconds, 3).replace(".", ",")


def read_exclude(path):
    """Ranges like '31:14.2 - 31:16.5' or '58:30 - end', one per line; '#' starts a comment."""
    ranges = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([\d:.,]+)\s*[-–]\s*([\d:.,]+|end)(\s*\(.*\))?", line, re.IGNORECASE)
        if not match:
            sys.exit(f"{path} line {number}: expected 'start - end', e.g. '31:14.2 - 31:16.5' or '58:30 - end'")
        start = parse_time(match[1])
        end = float("inf") if match[2].lower() == "end" else parse_time(match[2])
        if end <= start:
            sys.exit(f"{path} line {number}: the end is before the start")
        ranges.append((start, end))
    return sorted(ranges)


# --- recognition ---------------------------------------------------------------------------

def transcribe_words(model, audio, prompt, beam_size, arabic_check, offset, done, save, total):
    """transcribe.transcribe() with word timings kept. Same settings, plus word_timestamps."""
    segments, _ = model.transcribe(
        audio,
        language="ur",
        task="transcribe",
        beam_size=beam_size,
        temperature=[0.0, 0.2, 0.4, 0.6],
        condition_on_previous_text=False,
        initial_prompt=prompt or None,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
        word_timestamps=True,
    )
    results, started = list(done), time.time()
    last_save, start_point = started, offset
    try:
        for segment in segments:
            text, language = segment.text.strip(), "ur"
            if arabic_check:
                text, language = check_arabic(model, audio, segment, beam_size)
            start, end = offset + segment.start, min(offset + segment.end, total)
            if start >= total - 0.5:
                continue  # an invented closing phrase after audio that stops abruptly (see transcribe.collect)
            if text:
                results.append({
                    "start": round(start, 2), "end": round(end, 2), "text": text, "language": language,
                    "avg_logprob": round(segment.avg_logprob, 3),
                    "words": [[round(offset + w.start, 2), round(offset + w.end, 2), w.word, round(w.probability, 3)]
                              for w in segment.words or ()],
                })
            now = time.time()
            if now - last_save >= SAVE_EVERY_SECONDS:
                save(results, now - started)
                last_save = now
            elapsed, finished = now - started, end - start_point
            eta = elapsed / finished * (total - end) if finished > 5 else 0
            print(f"\r  {stamp(end)} / {stamp(total)} transcribed, {elapsed / 60:.1f} min this run"
                  + (f", about {eta / 60:.0f} min left " if eta else " "), end="", flush=True)
    except KeyboardInterrupt:
        save(results, time.time() - started)
        raise
    print()
    return results, time.time() - started


# --- cutting into clips --------------------------------------------------------------------

class Silence:
    """10 ms frame energies of the recognition audio, to find the pauses between words."""

    def __init__(self, audio, excluded=()):
        frames = len(audio) // 160
        energy = (audio[:frames * 160].reshape(frames, 160) ** 2).mean(axis=1)
        db = 10 * np.log10(energy + 1e-10)
        muted = np.zeros(frames, bool)
        for start, end in excluded:
            muted[max(0, int(start * 100)):max(0, int(min(end, frames / 100) * 100) + 1)] = True
        sounding = db[~muted]
        floor, speech = (np.percentile(sounding, 10), np.percentile(sounding, 95)) if len(sounding) else (-90, -20)
        # The muted --exclude ranges hide speech, so they never count as a pause.
        self.quiet = (db < floor + 0.3 * (speech - floor)) & ~muted
        self.db = np.where(muted, 0.0, db)

    def _runs(self, lo, hi, min_frames=8):
        a, b = max(0, int(lo * 100)), min(len(self.quiet), int(hi * 100))
        runs, i = [], a
        while i < b:
            if self.quiet[i]:
                j = i
                while j < b and self.quiet[j]:
                    j += 1
                if j - i >= min_frames:
                    runs.append((i / 100, j / 100))
                i = j
            else:
                i += 1
        return runs

    def between(self, a_start, a_end, b_start, b_end):
        """Where speech ends after word a and starts before word b: (end_a, start_b)."""
        a_mid = max(a_start + (a_end - a_start) / 2, a_end - 0.3)
        b_mid = min(b_start + (b_end - b_start) / 2, b_start + 0.3)
        after = self._runs(a_mid, min(b_mid, a_end + LONG_PAUSE))
        before = self._runs(max(a_mid, b_start - LONG_PAUSE), b_mid)
        end_a = after[0][0] if after else a_end
        start_b = before[-1][1] if before else b_start
        if start_b <= end_a:  # no clear pause: cut at the quietest moment between the words
            lo, hi = int(a_mid * 100), max(int(a_mid * 100) + 1, int(b_mid * 100))
            quietest = (lo + int(np.argmin(self.db[lo:hi]))) / 100 if hi <= len(self.db) else (a_end + b_start) / 2
            end_a = start_b = quietest
        return end_a, start_b

    def onset(self, start, end):
        before = self._runs(max(0.0, start - LONG_PAUSE), start + min(0.3, (end - start) / 2))
        return before[-1][1] if before else start

    def offset(self, start, end, total):
        after = self._runs(end - min(0.3, (end - start) / 2), min(total, end + LONG_PAUSE))
        return after[0][0] if after else end


def make_units(segments):
    """Flatten segments into words; Arabic, invented and word-less (recovered) segments stay whole."""
    units = []
    for segment in segments:
        kind = "word"
        if segment["language"] == "ar":
            kind = "arabic"
        elif INVENTED.search(segment["text"]):
            kind = "invented"
        elif not segment.get("words"):
            kind = "recovered"
        if kind != "word":
            units.append({"start": segment["start"], "end": segment["end"], "text": segment["text"], "kind": kind,
                          "prob": None, "logprob": segment.get("avg_logprob")})
            continue
        for start, end, word, prob in segment["words"]:
            if word.strip():
                units.append({"start": start, "end": max(end, start + 0.02), "text": word, "kind": kind,
                              "prob": prob, "logprob": segment.get("avg_logprob")})
    units.sort(key=lambda u: u["start"])
    # Recovered passages can overlap their neighbours: trim small overlaps, merge large ones.
    merged = []
    for unit in units:
        previous = merged[-1] if merged else None
        if previous and unit["start"] < previous["end"]:
            if previous["end"] - unit["start"] <= 0.3 or "recovered" not in (unit["kind"], previous["kind"]):
                middle = (previous["end"] + unit["start"]) / 2
                previous["end"], unit["start"] = middle, max(middle, unit["start"])
                unit["end"] = max(unit["end"], unit["start"] + 0.02)
            else:
                keep = "arabic" if "arabic" in (unit["kind"], previous["kind"]) else "recovered"
                previous.update(end=max(previous["end"], unit["end"]), kind=keep, prob=None,
                                text=previous["text"].rstrip() + " " + unit["text"].lstrip())
                continue
        merged.append(unit)
    return merged


def clip_cost(duration):
    if duration < 1.0:
        return 60.0
    if duration < MIN_CLIP:
        return 25.0 + (MIN_CLIP - duration) * 10
    if duration < IDEAL[0]:
        return (IDEAL[0] - duration) * 0.8
    if duration <= IDEAL[1]:
        return 0.0
    if duration <= MAX_CLIP:
        return (duration - IDEAL[1]) * 0.6
    return 100.0 + (duration - MAX_CLIP) * 10


def cut_cost(pause, sentence_end):
    if pause >= 0.3:
        return 0.0 if sentence_end else 1.0
    if pause >= 0.15:
        return 2.0 if sentence_end else 3.0
    if pause >= 0.05:
        return 6.0 if sentence_end else 8.0
    return 25.0


def best_cuts(left, right, pauses, ends):
    """Split one run of words into clips by dynamic programming; returns [(first, last), ...]."""
    n = len(left)
    cost, back = [0.0] + [float("inf")] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            duration = right[j - 1] - left[i]
            if duration > 30 and i < j - 1:
                break
            total = cost[i] + clip_cost(duration) + (cut_cost(pauses[j - 1], ends[j - 1]) if j < n else 0.0)
            if total < cost[j]:
                cost[j], back[j] = total, i
    clips, j = [], n
    while j > 0:
        clips.append((back[j], j - 1))
        j = back[j]
    return clips[::-1]


def build_clips(segments, audio, base, exclude, rules):
    """Turn recognizer segments (times relative to `audio`) into clips on the original timeline."""
    total = len(audio) / SAMPLE_RATE
    local_exclude = [(s - base, e - base) for s, e in exclude]
    silence = Silence(audio, local_exclude)
    units = make_units(segments)

    def excluded(start, end):
        return any(start < e and end > s for s, e in local_exclude)

    left_out, clips = [], []
    for unit in units:
        unit["barrier"] = unit["kind"] in ("arabic", "invented") or excluded(unit["start"], unit["end"])
        if unit["barrier"]:
            reason = {"arabic": "Arabic (Quran verse or dua)", "invented": "text the recognizer often invents"}.get(
                unit["kind"], "inside an --exclude range")
            left_out.append((unit["start"] + base, unit["end"] + base, reason, unit["text"].strip()))

    # Speech edges around every word, from the pauses next to it.
    for k, unit in enumerate(units):
        unit["index"] = k
        unit["left"] = silence.onset(unit["start"], unit["end"]) if k == 0 else None
        if k == len(units) - 1:
            unit["right"] = silence.offset(unit["start"], unit["end"], total)
    for a, b in zip(units, units[1:]):
        end_a, start_b = silence.between(a["start"], a["end"], b["start"], b["end"])
        a["right"], b["left"], a["pause"] = end_a, start_b, max(0.0, start_b - end_a)
    if units:
        units[-1]["pause"] = LONG_PAUSE

    # Runs of words with no barrier, long pause or excluded range between them. A run that
    # touches a left-out passage is marked "open" on that side.
    runs, current, open_start = [], [], False
    for k, unit in enumerate(units):
        if unit["barrier"]:
            if current:
                runs.append((current, open_start, True))
            current, open_start = [], True
            continue
        current.append(unit)
        following = units[k + 1] if k + 1 < len(units) else None
        gap = following is not None and excluded(unit["end"], following["start"])
        if following is None or unit["pause"] >= LONG_PAUSE or gap:
            runs.append((current, open_start, gap))
            current, open_start = [], gap
    if current:
        runs.append((current, open_start, False))

    # Next to a left-out passage the recognizer may have missed the first or last words, so
    # without a clear pause there, drop the words up to a nearby pause (within TRIM seconds).
    # With no pause that close, the clip is kept and flagged for checking.
    trimmed = []
    for run, open_start, open_end in runs:
        dropped = []
        before = units[run[0]["index"] - 1]["pause"] if run[0]["index"] > 0 else LONG_PAUSE
        if open_start and before < 0.15:
            first = next((k for k, u in enumerate(run) if u["pause"] >= 0.15), None)
            if first is not None and run[first]["end"] - run[0]["start"] <= TRIM:
                dropped.append(run[:first + 1])
                run = run[first + 1:]
        if open_end and run and run[-1]["pause"] < 0.15:
            last = next((k for k in range(len(run) - 2, -1, -1) if run[k]["pause"] >= 0.15), None)
            if last is not None and run[-1]["end"] - run[last + 1]["start"] <= TRIM:
                dropped.append(run[last + 1:])
                run = run[:last + 1]
        for words in dropped:
            left_out.append((words[0]["start"] + base, words[-1]["end"] + base,
                             "next to a left-out passage with no clear pause",
                             apply_corrections(" ".join(u["text"].strip() for u in words), rules)))
        if run:
            trimmed.append(run)

    for run in trimmed:
        left = [u["left"] for u in run]
        right = [u["right"] for u in run]
        pauses = [u["pause"] for u in run]
        ends = [u["text"].strip().endswith(SENTENCE_END) for u in run]
        for first, last in best_cuts(left, right, pauses, ends):
            words = run[first:last + 1]
            before = units[words[0]["index"] - 1] if words[0]["index"] > 0 else None
            pad_left = min(EDGE, before["pause"] / 2) if before else EDGE
            pad_right = min(EDGE, words[-1]["pause"] / 2)
            start = max(0.0, words[0]["left"] - pad_left)
            end = min(total, words[-1]["right"] + pad_right)
            for s, e in local_exclude:  # never reach into an excluded range
                if s < start < e:
                    start = e
                if s < end < e:
                    end = s
            text = apply_corrections(" ".join(w["text"].strip() for w in words), rules)
            probs = [w["prob"] for w in words if w["prob"] is not None]
            logprobs = [w["logprob"] for w in words if w["logprob"] is not None]
            clip = {
                "start": round(start + base, 2), "end": round(end + base, 2), "text": text,
                "prob": round(sum(probs) / len(probs), 3) if probs else None,
                "logprob": min(logprobs) if logprobs else None,
                "recovered": any(w["kind"] == "recovered" for w in words),
                "tight": [side for side, p in (("start", before["pause"] if before else 1), ("end", words[-1]["pause"]))
                          if p < 0.05],
            }
            duration = end - start
            if not text:
                left_out.append((clip["start"], clip["end"], "no text", ""))
            elif duration < EXPORT_LIMITS[0]:
                left_out.append((clip["start"], clip["end"], f"too short ({duration:.1f}s)", text))
            else:
                clips.append(clip)
    return clips, sorted(left_out)


def repeated_phrase(text):
    words = text.split()
    for n in range(1, 5):
        for i in range(len(words) - 3 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n] == words[i + 2 * n:i + 3 * n]:
                return " ".join(words[i:i + n])
    return None


def review_reasons(clips):
    rates = [len(c["text"].replace(" ", "")) / (c["end"] - c["start"]) for c in clips]
    median = statistics.median(rates) if rates else 0
    for clip, rate in zip(clips, rates):
        reasons, duration = [], clip["end"] - clip["start"]
        if (clip["prob"] is not None and clip["prob"] < 0.6) or (clip["logprob"] is not None and clip["logprob"] < -0.6):
            confidence = f"{clip['prob']:.2f}" if clip["prob"] is not None else f"log {clip['logprob']:.2f}"
            reasons.append(f"low confidence ({confidence})")
        if LATIN.search(clip["text"]):
            reasons.append("Latin letters (write English words in Urdu script)")
        if DEVANAGARI.search(clip["text"]):
            reasons.append("Hindi (Devanagari) letters")
        if ARABIC_LETTERS.search(clip["text"]):
            reasons.append("Arabic letters (an Arabic passage that was not detected?)")
        if median and not 0.5 * median <= rate <= 1.8 * median:
            reasons.append(f"speaking rate {rate:.1f} chars/s vs typical {median:.1f} (text may not match audio)")
        phrase = repeated_phrase(clip["text"])
        if phrase:
            reasons.append(f"repeated phrase \"{phrase}\"")
        if not MIN_CLIP - 0.2 <= duration <= MAX_CLIP + 0.2:  # the pause kept at each end may add a little
            reasons.append(f"{duration:.1f}s long (aim for 2-15 s)")
        if clip["recovered"]:
            reasons.append("recovered in the second pass (no word timings; check the ends)")
        if clip["tight"]:
            reasons.append(f"no clear pause at the {' and '.join(clip['tight'])}; check that no word is cut")
        clip["review"] = reasons


def write_outputs(paths, name, clips, left_out, source):
    srt = []
    for number, clip in enumerate(clips, 1):
        srt.append(f"{number}\n{srt_clock(clip['start'])} --> {srt_clock(clip['end'])}\n{clip['text']}\n")
    paths["srt"].write_text("\n".join(srt), encoding="utf-8")
    paths["tsv"].write_text("".join(f"{clock(c['start'])}\t{clock(c['end'])}\t{c['text']}\n" for c in clips),
                            encoding="utf-8")

    flagged = [(n, c) for n, c in enumerate(clips, 1) if c["review"]]
    minutes = sum(c["end"] - c["start"] for c in clips) / 60
    lines = [
        f"Review list for {source.name} (times are on the original file's timeline)",
        f"{len(clips)} clips, {minutes:.1f} minutes of speech; {len(flagged)} to check, {len(left_out)} left out.",
        "",
        "CHECK THESE CLIPS (numbers match the SRT)",
        "-----------------------------------------",
    ]
    for number, clip in flagged:
        lines.append(f"#{number:<5} {clock(clip['start'])} - {clock(clip['end'])}  {'; '.join(clip['review'])}")
        lines.append(f"        {clip['text']}")
    lines += ["", "LEFT OUT (not in the SRT)", "-------------------------"]
    for start, end, reason, text in left_out:
        lines.append(f"{clock(start)} - {clock(end)}  {reason}" + (f": {text}" if text else ""))
    paths["review"].write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- export --------------------------------------------------------------------------------

def read_corrected(path):
    """Read a proofread .srt or .tsv into [(start, end, text)]."""
    content = Path(path).read_text(encoding="utf-8-sig")
    items = []
    if Path(path).suffix.lower() == ".srt":
        for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip()):
            lines = block.strip().split("\n")
            timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if timing is None:
                continue
            start, end = (parse_time(t) for t in lines[timing].split("-->"))
            items.append((start, end, " ".join(line.strip() for line in lines[timing + 1:])))
    else:
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                sys.exit(f"{path} line {number}: expected start<TAB>end<TAB>text")
            items.append((parse_time(parts[0]), parse_time(parts[1]), "\t".join(parts[2:])))
    return [(s, e, re.sub(r"\s+", " ", t).strip()) for s, e, t in items]


def decode_original(path):
    """The original audio as 16 kHz mono, without the cleaning filters used for recognition."""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        sys.exit(f"FFmpeg could not read {path}:\n{error.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def export(source, corrected, tts_dir, overwrite):
    name = re.sub(r"[^\w-]+", "_", source.stem).strip("_") or "clip"
    raw = Path(tts_dir) / "data" / "raw"
    wavs, metadata = raw / "wavs", raw / "metadata.csv"
    if not Path(tts_dir).is_dir():
        sys.exit(f"Text-to-speech project not found: {tts_dir} (use --tts-dir)")
    mine = re.compile(rf"^{re.escape(name)}_\d{{4}}\.wav\|")
    existing = metadata.read_text(encoding="utf-8-sig").splitlines() if metadata.exists() else []
    if any(mine.match(line) for line in existing):
        if not overwrite:
            sys.exit(f"{metadata} already has clips named {name}_NNNN.wav. "
                     "Add --overwrite to replace them.")
        existing = [line for line in existing if not mine.match(line)]
        for old in wavs.glob(f"{name}_[0-9][0-9][0-9][0-9].wav"):
            old.unlink()

    items = read_corrected(corrected)
    print(f"Exporting {len(items)} clips from {source.name} ...")
    audio = decode_original(source)
    wavs.mkdir(parents=True, exist_ok=True)
    rows, skipped, seconds = [], [], 0.0
    for start, end, text in items:
        duration = end - start
        if not text:
            skipped.append(f"{clock(start)}: no text")
            continue
        if not EXPORT_LIMITS[0] <= duration <= EXPORT_LIMITS[1]:
            skipped.append(f"{clock(start)}: {duration:.1f}s is outside {EXPORT_LIMITS[0]:g}-{EXPORT_LIMITS[1]:g} s")
            continue
        piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if len(piece) < duration * SAMPLE_RATE * 0.9:
            skipped.append(f"{clock(start)}: past the end of the audio")
            continue
        file_name = f"{name}_{len(rows) + 1:04d}.wav"
        with wave.open(str(wavs / file_name), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(piece.tobytes())
        rows.append(f"{file_name}|{text}")
        seconds += duration
    metadata.write_text("".join(line + "\n" for line in existing + rows), encoding="utf-8")
    for message in skipped:
        print(f"  ! skipped {message}")
    print(f"Exported {len(rows)} clips, {seconds / 60:.1f} minutes, to {wavs}")
    print(f"Added {len(rows)} lines to {metadata}")
    print(f"Next: in {tts_dir}, run  .venv\\Scripts\\python prepare_dataset.py")


# --- main ----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="one audio file")
    parser.add_argument("--out-dir", default=str(HERE / "transcripts"), help="where to write results (default: transcripts\\)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model name or local path (default: large-v3-turbo; large-v3 is slower)")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="cuda uses an NVIDIA GPU, e.g. in Google Colab (default: auto)")
    parser.add_argument("--start", help="process from this time, e.g. 00:10:00 (for testing)")
    parser.add_argument("--duration", help="process only this long, e.g. 00:02:00 (for testing)")
    parser.add_argument("--exclude", help="file of time ranges to leave out, e.g. '31:14.2 - 31:16.5' or '58:30 - end'")
    parser.add_argument("--no-arabic-check", action="store_true", help="skip Arabic passage detection (faster)")
    parser.add_argument("--no-denoise", action="store_true", help="skip noise reduction")
    parser.add_argument("--no-recover", action="store_true", help="skip the second pass that recovers skipped passages")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the SRT, TSV and review list from saved segments (re-applies corrections and --exclude)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing results (or exported clips)")
    parser.add_argument("--export", metavar="CORRECTED", help="cut clips for training from a proofread .srt or .tsv")
    parser.add_argument("--tts-dir", default=str(DEFAULT_TTS_DIR),
                        help=f"text-to-speech project for --export (default: {DEFAULT_TTS_DIR})")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        sys.exit(f"Not found: {args.input}")
    if args.export:
        if not Path(args.export).is_file():
            sys.exit(f"Not found: {args.export}")
        export(source, Path(args.export), args.tts_dir, args.overwrite)
        return

    try:
        base = parse_time(args.start) if args.start else 0.0
        if args.duration:
            parse_time(args.duration)
    except ValueError as error:
        sys.exit(f"--start/--duration: {error}")
    exclude = read_exclude(args.exclude) if args.exclude else []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = source.stem + ("_section" if args.start or args.duration else "")
    paths = {"srt": out_dir / f"{name}_tts.srt", "tsv": out_dir / f"{name}_tts.tsv",
             "review": out_dir / f"{name}_tts_review.txt"}
    json_path, partial_path = out_dir / f"{name}_tts.segments.json", out_dir / f"{name}_tts.partial.json"
    print(f"\n{source.name}")
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"  - already done: {', '.join(existing)} (use --overwrite to replace; proofread copies are lost)")

    rules = default_corrections()
    prompt_file = HERE / "prompt.txt"
    prompt = prompt_file.read_text(encoding="utf-8-sig").strip() if prompt_file.exists() else ""

    print("  reading and cleaning the audio (a few minutes for a long recording)...")
    audio = load_audio(source, args.start, args.duration, denoise=not args.no_denoise)
    total = len(audio) / SAMPLE_RATE
    for start, end in exclude:  # silence other voices so they are neither heard nor transcribed
        a, b = max(0.0, start - base), min(total, end - base)
        if b > a:
            audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)] = 0.0
        if end == float("inf") and b > a:  # "... - end": nothing after it needs transcribing
            audio, total = audio[:int(a * SAMPLE_RATE)], a

    if args.rebuild or (json_path.exists() and not args.overwrite):
        if not json_path.exists():
            sys.exit("  - no saved segments; transcribe it first")
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        if saved.get("start") != args.start or saved.get("duration") != args.duration:
            sys.exit(f"  - {json_path.name} was made with different --start/--duration; use --overwrite to transcribe again")
        print(f"  using saved segments from {json_path.name}")
    else:
        from faster_whisper import WhisperModel

        settings = {
            "source": str(source.resolve()), "model": args.model, "prompt": prompt, "start": args.start,
            "duration": args.duration, "beam_size": args.beam_size, "exclude": exclude and [list(r) for r in exclude],
            "denoise": not args.no_denoise, "arabic_check": not args.no_arabic_check, "word_timestamps": True,
        }
        done, earlier_seconds = [], 0.0
        if partial_path.exists() and not args.overwrite:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if partial.get("settings") == json.loads(json.dumps(settings)) and partial.get("segments"):
                done, earlier_seconds = partial["segments"], partial.get("processing_seconds", 0.0)
                print(f"  resuming from {stamp(base + done[-1]['end'])} (saved progress found)")
            else:
                print("  saved progress was made with different settings; starting again")

        device = args.device
        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        print(f"Loading Whisper {args.model} on {device.upper()} (first run downloads about 1.6 GB)...")
        model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8",
                             cpu_threads=args.threads)
        offset = done[-1]["end"] if done else 0.0

        def save_progress(segments, run_seconds):
            write_json(partial_path, {"settings": settings, "processing_seconds": earlier_seconds + run_seconds,
                                      "segments": segments})

        remaining = audio[int(offset * SAMPLE_RATE):]
        try:
            if len(remaining) < SAMPLE_RATE:
                segments, run_seconds = done, 0.0
            else:
                segments, run_seconds = transcribe_words(model, remaining, prompt, args.beam_size,
                                                         not args.no_arabic_check, offset, done, save_progress, total)
            if not args.no_recover:
                save_progress(segments, run_seconds)
                recover_started = time.time()
                segments = recover_missed(model, audio, segments, prompt, args.beam_size, not args.no_arabic_check)
                run_seconds += time.time() - recover_started
        except KeyboardInterrupt:
            print("\n  stopped; progress saved. Run the same command again to continue.")
            sys.exit(130)
        saved = {
            "source": str(source), "model": args.model, "start": args.start, "duration": args.duration,
            "base_seconds": base, "duration_seconds": round(total, 2),
            "processing_minutes": round((earlier_seconds + run_seconds) / 60, 1),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "segment and word times are relative to --start; add base_seconds for the original timeline",
            "segments": segments,
        }
        write_json(json_path, saved)
        partial_path.unlink(missing_ok=True)

    clips, left_out = build_clips(saved["segments"], audio, base, exclude, rules)
    review_reasons(clips)
    write_outputs(paths, name, clips, left_out, source)
    minutes = sum(c["end"] - c["start"] for c in clips) / 60
    flagged = sum(1 for c in clips if c["review"])
    print(f"  saved {paths['srt'].name}, {paths['tsv'].name}, {paths['review'].name}")
    print(f"  {len(clips)} clips, {minutes:.1f} min of speech; {flagged} to check, {len(left_out)} left out "
          f"({saved.get('processing_minutes', '?')} min)")


if __name__ == "__main__":
    main()
