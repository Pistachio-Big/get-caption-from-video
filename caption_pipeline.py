"""Extract every kind of "caption" from a video URL into a single .txt file.

Layers (each one optional, all merged into the output):
  1. Post caption   - text the uploader wrote (yt-dlp metadata)
  2. Subtitles      - platform subtitles / auto-captions if the site provides them
  3. Speech (ASR)   - faster-whisper transcription of the audio track
  4. On-screen text - RapidOCR over frames sampled from the video (burned-in captions)

Also writes <id>.caption.txt: the caption as one plain paragraph + its translation (default: Vietnamese).

Usage:
  python caption_pipeline.py URL [-o out.txt] [--model small] [--lang en] [--fps 2] [--skip-ocr] [--skip-asr]
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Line:
    start: float
    end: float
    text: str


def fmt_ts(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


# ---------------------------------------------------------------- 1+2. download
def download(url: str, workdir: Path) -> tuple[Path, dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        # prefer a progressive stream (video+audio in one file) so no merge is needed
        "format": "hd/sd/best[vcodec!=none][acodec!=none]/bv*+ba/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "ffmpeg_location": FFMPEG,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["all", "-live_chat"],
        "subtitlesformat": "vtt/srt/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video = Path(ydl.prepare_filename(info))
    return video, info


def read_subtitle_files(workdir: Path, video_id: str) -> dict[str, list[Line]]:
    out: dict[str, list[Line]] = {}
    ts = r"(\d+:)?\d{2}:\d{2}[.,]\d{3}"
    for f in sorted(workdir.glob(f"{video_id}.*.vtt")) + sorted(workdir.glob(f"{video_id}.*.srt")):
        lang = f.suffixes[-2].lstrip(".")
        lines: list[Line] = []
        for block in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="ignore")):
            m = re.search(rf"({ts})\s*-->\s*({ts})", block)
            if not m:
                continue
            text = " ".join(
                re.sub(r"<[^>]+>", "", l).strip()
                for l in block[m.end():].splitlines() if l.strip()
            )
            if text and (not lines or lines[-1].text != text):
                lines.append(Line(_to_sec(m.group(1)), _to_sec(m.group(3)), text))
        if lines:
            out[lang] = lines
    return out


def _to_sec(t: str) -> float:
    parts = t.replace(",", ".").split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


# ---------------------------------------------------------------- 3. ASR
def transcribe(video: Path, model_size: str, lang: str | None) -> tuple[list[Line], str, float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, info = model.transcribe(
        str(video),
        language=lang,
        vad_filter=True,  # skip music/silence -> fewer Whisper hallucinations
        beam_size=5,
        condition_on_previous_text=False,
    )
    lines = [
        Line(s.start, s.end, s.text.strip())
        for s in segments
        if s.text.strip() and s.no_speech_prob < 0.6
    ]
    return lines, info.language, info.language_probability


# ---------------------------------------------------------------- 4. OCR
def ocr_frames(video: Path, workdir: Path, fps: float, min_conf: float) -> list[Line]:
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    # ffmpeg (not OpenCV) decodes, so AV1/VP9/HEVC sources work too
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
         str(frames_dir / "%05d.jpg")],
        check=True,
    )

    engine = RapidOCR()
    step = 1.0 / fps
    lines: list[Line] = []
    for i, frame in enumerate(sorted(frames_dir.glob("*.jpg"))):
        t = i * step
        img = cv2.imread(str(frame))
        result, _ = engine(img)
        if not result:
            continue
        # sort boxes top-to-bottom, left-to-right so multi-line captions read in order
        boxes = sorted(
            (r for r in result if r[2] >= min_conf and len(r[1].strip()) > 1),
            key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]),
        )
        text = " ".join(r[1].strip() for r in boxes)
        if not text:
            continue
        if lines and _similar(lines[-1].text, text):
            lines[-1].end = t + step
            # keep the longer read: the caption may still be animating in
            if len(text) > len(lines[-1].text):
                lines[-1].text = text
        else:
            lines.append(Line(t, t + step, text))
    return lines


def _similar(a: str, b: str, thr: float = 0.8) -> bool:
    a_, b_ = a.lower(), b.lower()
    return a_ in b_ or b_ in a_ or difflib.SequenceMatcher(None, a_, b_).ratio() >= thr


# ---------------------------------------------------------------- 5. caption paragraph + translation
def caption_paragraph(subs: dict[str, list[Line]], asr: list[Line] | None,
                      ocr: list[Line] | None, info: dict) -> tuple[str, str]:
    """Pick the best source and flatten it into one paragraph. Returns (text, source)."""
    if subs:
        lang, lines = next(iter(subs.items()))
        source, parts = f"subtitles[{lang}]", [l.text for l in lines]
    elif asr:
        source, parts = "speech", [l.text for l in asr]
    elif ocr:
        source, parts = "on-screen text", [l.text for l in ocr]
    else:
        source, parts = "post caption", [info.get("description") or ""]
    return re.sub(r"\s+", " ", " ".join(parts)).strip(), source


def translate(text: str, target: str, source_lang: str | None) -> tuple[str, str]:
    """Translate with free online backends, falling back in order. Returns (text, backend)."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    def mymemory() -> MyMemoryTranslator:
        # MyMemory wants region codes (en-GB, vi-VN) and an explicit source language
        codes = MyMemoryTranslator(source="en-GB", target="vi-VN").get_supported_languages(as_dict=True)
        pick = lambda c: next((v for v in codes.values() if v.split("-")[0] == c), c)
        return MyMemoryTranslator(source=pick(source_lang or "en"), target=pick(target))

    backends = [
        ("google", lambda: GoogleTranslator(source=source_lang or "auto", target=target)),
        ("mymemory", mymemory),
    ]
    chunks = _chunk_sentences(text, 450)  # MyMemory caps a request at 500 bytes
    errors = []
    for name, make in backends:
        try:
            tr = make()
            return " ".join(tr.translate(c) for c in chunks), name
        except Exception as e:  # rate limits / blocked endpoint -> try the next backend
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("all translators failed (" + "; ".join(errors) + ")")


def _chunk_sentences(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if chunks and len(chunks[-1]) + len(sent) + 1 <= limit:
            chunks[-1] += " " + sent
        else:
            chunks.append(sent)
    return chunks


# ---------------------------------------------------------------- output
def render(info: dict, subs: dict[str, list[Line]], asr: list[Line] | None, asr_lang: str,
           ocr: list[Line] | None) -> str:
    out = [
        f"URL:       {info.get('webpage_url')}",
        f"Uploader:  {info.get('uploader')}",
        f"Duration:  {info.get('duration')} s",
        "",
        "=== 1. POST CAPTION (uploader's text) ===",
        (info.get("description") or "(none)").strip(),
        "",
        "=== 2. PLATFORM SUBTITLES ===",
    ]
    if subs:
        for lang, lines in subs.items():
            out.append(f"[{lang}]")
            out += [f"[{fmt_ts(l.start)} -> {fmt_ts(l.end)}] {l.text}" for l in lines]
    else:
        out.append("(none provided by the platform)")

    out += ["", f"=== 3. SPEECH TRANSCRIPT (Whisper, lang={asr_lang}) ==="]
    if asr is None:
        out.append("(skipped)")
    elif not asr:
        out.append("(no speech detected - likely music only)")
    else:
        out += [f"[{fmt_ts(l.start)} -> {fmt_ts(l.end)}] {l.text}" for l in asr]
        out += ["", "-- plain text --", " ".join(l.text for l in asr)]

    out += ["", "=== 4. ON-SCREEN TEXT (OCR) ==="]
    if ocr is None:
        out.append("(skipped)")
    elif not ocr:
        out.append("(no text detected on screen)")
    else:
        out += [f"[{fmt_ts(l.start)} -> {fmt_ts(l.end)}] {l.text}" for l in ocr]
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--workdir", type=Path, default=Path("work"))
    p.add_argument("--model", default="small", help="whisper size: tiny/base/small/medium/large-v3")
    p.add_argument("--lang", default=None, help="force ASR language (e.g. en, vi); default auto")
    p.add_argument("--fps", type=float, default=2.0, help="frames per second sampled for OCR")
    p.add_argument("--min-conf", type=float, default=0.6, help="min OCR confidence")
    p.add_argument("--skip-asr", action="store_true")
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--translate-to", default="vi", help="target language for the caption paragraph")
    p.add_argument("--no-translate", action="store_true")
    a = p.parse_args()

    print("[1/5] downloading video + metadata ...")
    video, info = download(a.url, a.workdir)
    print("      ->", video)

    print("[2/5] reading platform subtitles ...")
    subs = read_subtitle_files(a.workdir, info["id"])

    asr, lang_code, asr_lang = None, None, "-"
    if not a.skip_asr:
        print(f"[3/5] transcribing speech (faster-whisper {a.model}) ...")
        asr, lang_code, prob = transcribe(video, a.model, a.lang)
        asr_lang = f"{lang_code} (p={prob:.2f})"

    ocr = None
    if not a.skip_ocr:
        print(f"[4/5] OCR on-screen text ({a.fps} fps) ...")
        ocr = ocr_frames(video, a.workdir, a.fps, a.min_conf)

    out_path = a.output or Path("output") / f"{info['id']}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(info, subs, asr, asr_lang, ocr), encoding="utf-8")
    print("saved:", out_path.resolve())

    paragraph, source = caption_paragraph(subs, asr, ocr, info)
    body = [paragraph]
    if not a.no_translate and paragraph and lang_code != a.translate_to:
        print(f"[5/5] translating caption -> {a.translate_to} ...")
        src = lang_code if source == "speech" else None
        try:
            translated, backend = translate(paragraph, a.translate_to, src)
            body += ["", f"--- Bản dịch [{a.translate_to}] ({backend}) ---", translated]
        except RuntimeError as e:
            print("      translation skipped:", e)
    caption_path = out_path.with_suffix(".caption.txt")
    caption_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"saved: {caption_path.resolve()}  (source: {source})")


if __name__ == "__main__":
    main()
