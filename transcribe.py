r"""Transcribe Urdu lecture recordings to Word documents with a local Whisper model.

Usage:
    python transcribe.py LECTURE.mp3 [MORE.mp3 | FOLDER ...]
    python transcribe.py lectures\ --start 00:10:00 --duration 00:05:00   # test on a slice
    python transcribe.py lectures\ --rebuild                              # re-apply corrections only

For each recording this writes, in --out-dir (default: transcripts\):
    <name>.docx         right-to-left Urdu text in paragraphs, with timestamps
    <name>.segments.json raw recognizer output, used by --rebuild and evaluate.py

Accuracy measures:
    * Audio is converted to 16 kHz mono, band-limited, lightly de-noised and level-normalized.
    * Silence and music are skipped by voice-activity detection (reduces invented text).
    * Language is forced to Urdu, so text never comes out in Hindi (Devanagari) script.
    * prompt.txt primes the model with the speaker's name and common religious vocabulary.
    * Each segment is not conditioned on the previous one, which avoids repetition loops.
    * Segments that sound like Arabic (Quran verses, duas) are re-transcribed as Arabic.
    * corrections.tsv fixes recurring misspellings; edit it and use --rebuild to re-apply.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# Windows without Developer Mode cannot symlink the model cache; that is harmless.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = Path(__file__).resolve().parent
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma", ".mp4", ".mkv", ".webm"}
SAMPLE_RATE = 16000
SAVE_EVERY_SECONDS = 60
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Python's \w misses Arabic-script marks such as the superscript alef in تعالیٰ.
WORD_CHAR = r"[\wؐ-ًؚ-ٰٟۖ-ۭ]"
ARABIC_ONLY = re.compile(r"[ةيكىإأً-ْ]")  # letters Urdu spells differently, and vowel marks
SENTENCE_END = ("۔", "؟", "!", "?", ".", "۔»")


def find_inputs(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"Not found: {raw}")
    if not files:
        sys.exit("No audio files found.")
    return files


def load_audio(path, start=None, duration=None, denoise=True):
    """Decode to 16 kHz mono float32 with FFmpeg, cleaning the signal for recognition."""
    filters = ["highpass=f=80", "lowpass=f=7600"]
    if denoise:
        filters.append("afftdn=nr=10:nf=-45")
    filters.append("loudnorm=I=-20:LRA=11:TP=-2")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        command += ["-ss", start]
    if duration:
        command += ["-t", duration]
    command += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-af", ",".join(filters), "-f", "f32le", "-"]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError:
        sys.exit("FFmpeg was not found. Install it (winget install Gyan.FFmpeg) and reopen the terminal.")
    except subprocess.CalledProcessError as error:
        sys.exit(f"FFmpeg could not read {path}:\n{error.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def load_corrections(*paths):
    """Read wrong<TAB>right rules; later files win (corrections.tsv after the learned ones)."""
    rules = []
    for path in paths:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 2 or not parts[0] or (len(parts) > 2 and not parts[2].startswith("#")):
                print(f"  ! {path.name} line {number} skipped (needs: wrong<TAB>right)")
                continue
            rules.append((re.compile(rf"(?<!{WORD_CHAR}){re.escape(parts[0])}(?!{WORD_CHAR})"), parts[1]))
    return rules


def default_corrections():
    return load_corrections(HERE / "corrections.learned.tsv", HERE / "corrections.tsv")


def apply_corrections(text, rules):
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def transcribe(model, audio, prompt, beam_size, arabic_check, offset=0.0, done=(), save=None):
    """Transcribe audio that starts `offset` seconds into the lecture.

    `done` holds segments already transcribed in an earlier, interrupted run; `save` is
    called with all segments so far about once a minute.
    """
    segments, info = model.transcribe(
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
    )
    results = list(done)
    started = time.time()
    try:
        return collect(model, audio, segments, beam_size, arabic_check, offset, results, started, save)
    except KeyboardInterrupt:
        if save:
            save(results, time.time() - started)
        raise


def collect(model, audio, segments, beam_size, arabic_check, offset, results, started, save):
    total = offset + len(audio) / SAMPLE_RATE
    last_save = started
    for segment in segments:
        text = segment.text.strip()
        language = "ur"
        # Checking every segment is slow on CPU, so only check likely Arabic: Arabic-only
        # letters or vowel marks in the Urdu output, or low recognizer confidence.
        suspicious = ARABIC_ONLY.search(text) or segment.avg_logprob < -0.6
        if arabic_check and suspicious and segment.end - segment.start >= 2.0:
            piece = audio[int(segment.start * SAMPLE_RATE):int(segment.end * SAMPLE_RATE)]
            detected, probability, _ = model.detect_language(audio=piece)
            if detected == "ar" and probability >= 0.6:
                arabic, _ = model.transcribe(piece, language="ar", beam_size=beam_size, condition_on_previous_text=False)
                arabic_text = " ".join(s.text.strip() for s in arabic).strip()
                if arabic_text:
                    text, language = arabic_text, "ar"
        start, end = offset + segment.start, min(offset + segment.end, total)
        if start >= total - 0.5:
            # Whisper sometimes invents a closing phrase ("شکریہ") after audio that stops abruptly.
            continue
        if DEVANAGARI.search(text):
            print(f"\n  ! Hindi script at {stamp(start)}; please check this passage")
        if text:
            results.append({"start": round(start, 2), "end": round(end, 2), "text": text, "language": language})
        now = time.time()
        if save and now - last_save >= SAVE_EVERY_SECONDS:
            save(results, now - started)
            last_save = now
        print(f"\r  {stamp(end)} / {stamp(total)} transcribed ({(now - started) / 60:.1f} min this run)", end="", flush=True)
    print()
    return results, time.time() - started


def write_json(path, data):
    """Write via a temporary file so an interruption never leaves a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def stamp(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


def paragraphs(segments, rules, pause=1.5, max_words=110):
    """Group segments into paragraphs at long pauses, length limits and Arabic passages."""
    grouped, current = [], None
    for segment in segments:
        text = apply_corrections(segment["text"], rules)
        if not text:
            continue
        new = (
            current is None
            or segment["language"] != current["language"]
            or segment["start"] - current["end"] > pause
            or len(current["text"].split()) > max_words
        )
        if new:
            current = {"start": segment["start"], "end": segment["end"], "language": segment["language"], "text": text}
            grouped.append(current)
        else:
            current["text"] += " " + text
            current["end"] = segment["end"]
    for paragraph in grouped:
        if paragraph["language"] == "ur" and not paragraph["text"].endswith(SENTENCE_END):
            paragraph["text"] += "۔"
    return grouped


def set_rtl(paragraph, font, size, color=None, bold=False):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    properties = paragraph._p.get_or_add_pPr()
    properties.insert(0, OxmlElement("w:bidi"))
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color)
        run_properties = run._r.get_or_add_rPr()
        fonts = run_properties.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            run_properties.insert(0, fonts)
        for attribute in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attribute), font)
        run_properties.append(OxmlElement("w:rtl"))
        size_cs = OxmlElement("w:szCs")
        size_cs.set(qn("w:val"), str(int(size * 2)))
        run_properties.append(size_cs)


def write_docx(path, title, details, grouped, font, timestamps):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    document = Document()
    heading = document.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    note = document.add_paragraph(details)
    note.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_rtl(note, "Calibri", 9, color="808080")
    for item in grouped:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if item["language"] == "ar" else WD_ALIGN_PARAGRAPH.JUSTIFY
        if timestamps:
            paragraph.add_run(f"({stamp(item['start'])}) ")
        paragraph.add_run(item["text"])
        arabic = item["language"] == "ar"
        set_rtl(paragraph, font, 16, color="1F4E79" if arabic else None, bold=arabic)
        if timestamps:
            from docx.shared import Pt, RGBColor
            paragraph.runs[0].font.size = Pt(9)
            paragraph.runs[0].font.color.rgb = RGBColor.from_string("808080")
            paragraph.runs[0].font.bold = False
    document.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="audio files or folders")
    parser.add_argument("--out-dir", default=str(HERE / "transcripts"), help="where to write results (default: transcripts\\)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model name or local path (default: large-v3-turbo; large-v3 is slower)")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")
    parser.add_argument("--start", help="process from this time, e.g. 00:10:00 (for testing)")
    parser.add_argument("--duration", help="process only this long, e.g. 00:05:00 (for testing)")
    parser.add_argument("--no-arabic-check", action="store_true", help="skip Arabic passage detection (faster)")
    parser.add_argument("--no-denoise", action="store_true", help="skip noise reduction")
    parser.add_argument("--no-timestamps", action="store_true", help="leave timestamps out of the Word document")
    parser.add_argument("--font", default="Urdu Typesetting", help="Urdu font for the Word document")
    parser.add_argument("--rebuild", action="store_true", help="rebuild documents from saved segments (re-applies corrections)")
    parser.add_argument("--overwrite", action="store_true", help="transcribe again even if output exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = default_corrections()
    prompt_file = HERE / "prompt.txt"
    prompt = prompt_file.read_text(encoding="utf-8-sig").strip() if prompt_file.exists() else ""
    model = None

    for source in find_inputs(args.inputs):
        name = source.stem + ("_section" if args.start or args.duration else "")
        docx_path = out_dir / f"{name}.docx"
        json_path = out_dir / f"{name}.segments.json"
        print(f"\n{source.name}")

        if args.rebuild:
            if not json_path.exists():
                print("  - no saved segments; transcribe it first")
                continue
            saved = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            partial_path = out_dir / f"{name}.partial.json"
            if docx_path.exists() and not args.overwrite:
                print(f"  - already done: {docx_path.name} (use --overwrite to redo)")
                continue
            # Settings that change the result; a partial run is only resumed if they match.
            settings = {
                "source": str(source.resolve()), "model": args.model, "prompt": prompt, "start": args.start,
                "duration": args.duration, "beam_size": args.beam_size,
                "denoise": not args.no_denoise, "arabic_check": not args.no_arabic_check,
            }
            done, earlier_seconds = [], 0.0
            if partial_path.exists() and not args.overwrite:
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
                if partial.get("settings") == settings and partial.get("segments"):
                    done, earlier_seconds = partial["segments"], partial.get("processing_seconds", 0.0)
                    print(f"  resuming from {stamp(done[-1]['end'])} (saved progress found)")
                else:
                    print("  saved progress was made with different settings; starting again")

            if model is None:
                from faster_whisper import WhisperModel
                print(f"Loading Whisper {args.model} (first run downloads about 1.6 GB)...")
                model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
            audio = load_audio(source, args.start, args.duration, denoise=not args.no_denoise)
            offset = done[-1]["end"] if done else 0.0

            def save_progress(segments, run_seconds):
                write_json(partial_path, {"settings": settings, "processing_seconds": earlier_seconds + run_seconds,
                                          "segments": segments})

            remaining = audio[int(offset * SAMPLE_RATE):]
            try:
                if len(remaining) < SAMPLE_RATE:  # interrupted just before the end
                    segments, run_seconds = done, 0.0
                else:
                    segments, run_seconds = transcribe(
                        model, remaining, prompt, args.beam_size, not args.no_arabic_check,
                        offset=offset, done=done, save=save_progress,
                    )
            except KeyboardInterrupt:
                print("\n  stopped; progress saved. Run the same command again to continue.")
                sys.exit(130)
            saved = {
                "source": str(source),
                "model": args.model,
                "start": args.start,
                "duration_seconds": round(len(audio) / SAMPLE_RATE, 2),
                "processing_minutes": round((earlier_seconds + run_seconds) / 60, 1),
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "segments": segments,
            }
            write_json(json_path, saved)
            partial_path.unlink(missing_ok=True)

        grouped = paragraphs(saved["segments"], rules)
        details = (
            f"{Path(saved['source']).name} | {stamp(saved['duration_seconds'])} | "
            f"Whisper {saved['model']} | {saved['created'][:10]} | machine transcript, please verify"
        )
        write_docx(docx_path, source.stem, details, grouped, args.font, not args.no_timestamps)
        print(f"  saved {docx_path}  ({len(grouped)} paragraphs, {saved.get('processing_minutes', '?')} min)")


if __name__ == "__main__":
    main()
