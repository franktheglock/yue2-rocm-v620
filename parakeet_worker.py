#!/usr/bin/env python3
"""Local NVIDIA Parakeet V3 0.6B (nvidia/parakeet-tdt-0.6b-v3) ASR & section-alignment worker.

Runs in `.venv-flamingo` (transformers >= 5.6 + shared ROCm torch).
Emits `STAGE:<name>` progress markers on stderr and a single JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAKEET_DIR = BASE_DIR / "parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 180.0


def _stage(name: str):
    print(f"STAGE:{name}", file=sys.stderr, flush=True)


def _decode_audio_16k(audio_path: Path, max_seconds: float) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(audio_path),
        "-vn",
        "-t", str(float(max_seconds)),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=240, check=False)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip()[-800:]
        raise RuntimeError(f"FFmpeg audio decode failed: {err or 'unknown error'}")
    wav = np.frombuffer(res.stdout, dtype="<f4").copy()
    if wav.size < SAMPLE_RATE or not np.isfinite(wav).all():
        raise ValueError("Audio must contain at least 1 second of finite audio")
    return wav


_SECTION_HEADER_MAP = {
    "intro": "Intro",
    "verse": "Verse",
    "pre-chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "post-chorus": "Post-Chorus",
    "bridge": "Bridge",
    "interlude": "Interlude",
    "solo": "Solo",
    "instrumental": "Instrumental",
    "outro": "Outro",
    "fade-out": "Outro",
    "rap": "Verse",
    "intro and verse": "Verse",
    "pre-chorus and chorus": "Chorus",
    "verse and pre-chorus": "Pre-Chorus",
}

_PUNCT_TOKENS = {"?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"}


def _clean_line(line: str) -> str:
    t = " ".join(line.split()).strip()
    if not t or not re.search(r"[\w]", t, re.UNICODE):
        return ""
    return t


def _group_tokens_to_lines(token_events: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    lines: list[tuple[float, float, str]] = []
    pieces: list[str] = []
    line_start: Optional[float] = None
    last_end: float = 0.0
    words_in_line = 0

    def _flush():
        nonlocal pieces, line_start, words_in_line
        if pieces and line_start is not None:
            txt = _clean_line("".join(pieces))
            if txt:
                lines.append((line_start, last_end, txt))
        pieces = []
        line_start = None
        words_in_line = 0

    for ev in token_events:
        tok = ev["token"]
        t0 = float(ev["start"])
        t1 = float(ev["end"])
        if not tok:
            continue

        cur_txt = "".join(pieces).rstrip()
        # Only break at word starts: sung notes often hold mid-word, so a long gap before a
        # subword continuation ("you'" + "re") must not split the word across lines.
        gap_break = bool(pieces) and tok.startswith(" ") and (t0 - last_end >= 0.75)
        punct_break = bool(pieces) and words_in_line >= 3 and cur_txt.endswith((".", "!", "?")) and tok.startswith(" ")
        len_break = bool(pieces) and words_in_line >= 12 and tok.startswith(" ")
        if gap_break or punct_break or len_break:
            _flush()

        if line_start is None:
            line_start = t0
        pieces.append(tok)
        last_end = max(last_end, t1)
        if tok.strip() and tok.strip() not in _PUNCT_TOKENS and (tok.startswith(" ") or words_in_line == 0):
            words_in_line += 1

    _flush()
    return lines


def _align_lines_to_sections(
    lines: list[tuple[float, float, str]],
    sections: list[dict[str, Any]],
) -> tuple[str, int]:
    if not lines:
        return "", 0

    word_count = sum(len(txt.split()) for _, _, txt in lines)

    if not sections:
        body = "\n".join(txt for _, _, txt in lines)
        return f"[Verse 1]\n{body}", word_count

    def _find_section_idx(t0: float, t1: float) -> int:
        mid = (t0 + t1) / 2.0
        for idx, sec in enumerate(sections):
            if float(sec["start"]) <= mid <= float(sec["end"]):
                return idx
        return min(
            range(len(sections)),
            key=lambda idx: min(abs(mid - float(sections[idx]["start"])), abs(mid - float(sections[idx]["end"]))),
        )

    grouped: list[tuple[str, list[str]]] = []
    for t0, t1, txt in lines:
        sec_idx = _find_section_idx(t0, t1)
        raw_lbl = str(sections[sec_idx].get("label") or "").strip().lower()
        pretty = _SECTION_HEADER_MAP.get(raw_lbl, "Verse")
        if grouped and grouped[-1][0] == pretty:
            grouped[-1][1].append(txt)
        else:
            grouped.append((pretty, [txt]))

    verse_num = 0
    blocks = []
    for header, grp_lines in grouped:
        if header == "Verse":
            verse_num += 1
            tag = f"[Verse {verse_num}]"
        else:
            tag = f"[{header}]"
        blocks.append(f"{tag}\n" + "\n".join(grp_lines))
    return "\n\n".join(blocks), word_count


def run_parakeet(
    audio_path: Path,
    *,
    max_seconds: float = 240.0,
    sections: Optional[list[dict[str, Any]]] = None,
    has_vocal_melody: bool = True,
    model_dir: Optional[Path] = None,
) -> dict[str, Any]:
    # Critical ROCm gfx1030 guard: ParakeetEncoder uses Conv1d/Conv2d which can hang in MIOpen.
    torch.backends.cudnn.enabled = False

    resolved_dir = Path(model_dir or os.environ.get("YUE2_PARAKEET_DIR") or DEFAULT_PARAKEET_DIR)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Parakeet V3 model directory not found: {resolved_dir}")

    _stage("parakeet_load")
    from tokenizers.decoders import DecodeStream
    from transformers import ParakeetForTDT, ParakeetProcessor

    wav = _decode_audio_16k(audio_path, max_seconds=max_seconds)
    duration_seconds = round(len(wav) / float(SAMPLE_RATE), 2)

    processor = ParakeetProcessor.from_pretrained(str(resolved_dir), local_files_only=True)
    try:
        model = ParakeetForTDT.from_pretrained(
            str(resolved_dir),
            dtype=torch.float32,
            local_files_only=True,
        ).eval().to("cuda")
    except TypeError:
        model = ParakeetForTDT.from_pretrained(
            str(resolved_dir),
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval().to("cuda")

    _stage("parakeet_run")
    subsampling = int(getattr(model.config.encoder_config, "subsampling_factor", 8))
    frame_rate = float(processor.feature_extractor.hop_length) / float(processor.feature_extractor.sampling_rate) * subsampling
    skip_ids = {int(processor.tokenizer.pad_token_id), int(model.config.blank_token_id)}
    win_samples = int(round(WINDOW_SECONDS * SAMPLE_RATE))

    all_token_events: list[dict[str, Any]] = []
    fallback_lines: list[tuple[float, float, str]] = []

    for start_sample in range(0, len(wav), win_samples):
        chunk = wav[start_sample : start_sample + win_samples]
        if len(chunk) < SAMPLE_RATE // 2:
            break
        win_offset = start_sample / float(SAMPLE_RATE)
        win_end = win_offset + len(chunk) / float(SAMPLE_RATE)

        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {
            k: (v.to(device="cuda", dtype=torch.float32) if (torch.is_tensor(v) and v.is_floating_point()) else (v.to("cuda") if torch.is_tensor(v) else v))
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            outputs = model.generate(**inputs)

        seq = outputs.sequences[0].detach().cpu()
        durs = outputs.durations[0].detach().cpu() if getattr(outputs, "durations", None) is not None else None

        if durs is not None:
            starts = durs.cumsum(dim=-1) - durs
            stream = DecodeStream(skip_special_tokens=True)
            win_events: list[dict[str, Any]] = []
            for tok_id, st_f, dur_f in zip(seq.tolist(), starts.tolist(), durs.tolist()):
                if int(tok_id) in skip_ids:
                    continue
                piece = stream.step(processor.tokenizer._tokenizer, int(tok_id))
                if piece is None:
                    continue
                st_s = win_offset + float(st_f) * frame_rate
                en_s = win_offset + float(st_f + dur_f) * frame_rate
                if piece in _PUNCT_TOKENS and win_events:
                    st_s = win_events[-1]["end"]
                    en_s = st_s
                if not win_events and not piece[:1].isspace() and piece not in _PUNCT_TOKENS:
                    # A fresh DecodeStream drops the leading space on a window's first word;
                    # restore it so it neither glues onto the previous window's last word nor
                    # gets treated as a subword continuation.
                    piece = " " + piece
                win_events.append({"token": piece, "start": st_s, "end": en_s})
            all_token_events.extend(win_events)
        else:
            txt = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            if txt:
                fallback_lines.append((win_offset, win_end, txt))

    lines = _group_tokens_to_lines(all_token_events) if all_token_events else fallback_lines
    lyrics, word_count = _align_lines_to_sections(lines, sections or [])
    detected_instrumental = (word_count < 8 and not has_vocal_melody) or (word_count < 4)

    _stage("done")
    return {
        "lyrics": lyrics,
        "word_count": word_count,
        "detected_instrumental": detected_instrumental,
        "duration_seconds": duration_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe vocals with NVIDIA Parakeet V3 0.6B")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--sections-json", type=str, default="")
    parser.add_argument("--no-vocal-melody", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio_path.is_file():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    sections: list[dict[str, Any]] = []
    if args.sections_json.strip():
        try:
            loaded = json.loads(args.sections_json)
            if isinstance(loaded, list):
                sections = loaded
        except Exception as exc:
            print(f"Warning: invalid --sections-json ({exc})", file=sys.stderr)

    result = run_parakeet(
        args.audio_path,
        max_seconds=args.max_seconds,
        sections=sections,
        has_vocal_melody=not args.no_vocal_melody,
        model_dir=args.model_dir,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
