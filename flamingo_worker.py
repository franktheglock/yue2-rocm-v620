#!/usr/bin/env python3
"""Local Music Flamingo (nvidia/music-flamingo-2601-hf) worker for audio-to-style analysis.

Runs in the isolated `.venv-flamingo` environment (transformers >= 5.6 + shared ROCm torch).
Emits `STAGE:<name>` progress markers on stderr and a single JSON object on stdout.
"""
from __future__ import annotations

import argparse
import ast
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
DEFAULT_MODEL_DIR = BASE_DIR / "music-flamingo-2601-hf"
SAMPLE_RATE = 16000


def _stage(name: str):
    print(f"STAGE:{name}", file=sys.stderr, flush=True)


def _decode_audio_ffmpeg(audio_path: Path, max_seconds: float) -> np.ndarray:
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
    res = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip()[-800:]
        raise RuntimeError(f"FFmpeg audio decode failed: {err or 'unknown error'}")
    wav = np.frombuffer(res.stdout, dtype="<f4").copy()
    if wav.size < SAMPLE_RATE or not np.isfinite(wav).all():
        raise ValueError("Audio must contain at least 1 second of finite audio")
    return wav


def _probe_duration(audio_path: Path) -> Optional[float]:
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            val = float(res.stdout.strip())
            if np.isfinite(val) and val > 0:
                return round(val, 2)
    except Exception:
        pass
    return None


_JSON_KEYS = ("style", "description", "instrumental", "bpm", "key", "title_idea")


def _extract_json(txt: str) -> Optional[dict[str, Any]]:
    if not txt:
        return None
    cleaned = txt.strip()
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.I)
        if m:
            cleaned = m.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Music Flamingo often emits a Python-style dict ('single quotes', True/False/None).
    try:
        data = ast.literal_eval(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Last resort (also covers output truncated before the closing brace): pull each known
    # field out individually.
    data = {}
    for key in _JSON_KEYS:
        m = re.search(
            rf"""["']{key}["']\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*?)'(?=\s*(?:,\s*["']\w+["']\s*:|\}}|$))|(true|false|True|False|null|None)|(-?\d+(?:\.\d+)?))""",
            cleaned,
        )
        if not m:
            continue
        dq, sq, lit, num = m.groups()
        if dq is not None or sq is not None:
            data[key] = (dq if dq is not None else sq).replace("\\'", "'").replace('\\"', '"')
        elif lit is not None:
            data[key] = {"true": True, "false": False}.get(lit.lower())
        else:
            data[key] = float(num)
    return data or None


def _clamp_words(text: str, max_words: int = 60) -> str:
    """Trim to whole sentences within max_words (falls back to a hard word cut)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        cand = f"{out} {sent}".strip()
        if len(cand.split()) > max_words:
            break
        out = cand
    return out if len(out.split()) >= 15 else " ".join(words[:max_words]).rstrip(",;")


def _coerce_bpm(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)) and np.isfinite(val) and 20 <= val <= 300:
        return int(round(val))
    if isinstance(val, str):
        m = re.search(r"(\d{2,3})", val)
        if m:
            n = int(m.group(1))
            if 20 <= n <= 300:
                return n
    return None


def analyze_audio(audio_path: Path, max_seconds: float = 180.0, model_dir: Optional[Path] = None) -> dict[str, Any]:
    # Critical ROCm gfx1030 guard: AudioFlamingo3Encoder uses nn.Conv1d, which hangs in MIOpen unless disabled.
    torch.backends.cudnn.enabled = False

    resolved_dir = Path(model_dir or os.environ.get("YUE2_FLAMINGO_DIR") or DEFAULT_MODEL_DIR)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Music Flamingo model directory not found: {resolved_dir}")

    _stage("decoding")
    wav = _decode_audio_ffmpeg(audio_path, max_seconds=max_seconds)
    duration_s = _probe_duration(audio_path) or round(len(wav) / float(SAMPLE_RATE), 2)

    _stage("loading")
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(resolved_dir), local_files_only=True)
    # Stream directly onto GPU in FP16 so we never blow the 30 GB system RAM budget.
    # On gfx1030, FP16 runs natively while BF16 is software-emulated (~10x slower).
    try:
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            str(resolved_dir),
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
    except TypeError:
        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            str(resolved_dir),
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()

    _stage("analyzing")
    instruction = (
        "Analyze this music track carefully and output ONLY a valid JSON object (double-quoted keys and strings, no markdown, no extra text) "
        "with these exact keys:\n"
        '- "style": a single concise YuE2 music-generation style prompt (20 to 45 words) covering genre/subgenre, '
        'vocal character (gender, timbre, delivery — or "instrumental, no vocals" if there is no singing/rapping), '
        'key instruments, production style, mood, tempo in BPM (e.g. "120 BPM"), and musical key if identifiable.\n'
        '- "description": 2-3 sentences describing the arrangement, instrumentation, vocal delivery, and overall production vibe.\n'
        '- "instrumental": boolean (true if the track has no sung or rapped vocals, false if it has vocals).\n'
        '- "bpm": integer estimated tempo in BPM, or null.\n'
        '- "key": string musical key (e.g. "C minor", "G major"), or null.\n'
        '- "title_idea": a short 2-4 word evocative title capturing the track vibe (not an existing copyrighted song title).'
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_text],
        audio=[wav],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
    if "input_features" in inputs and torch.is_tensor(inputs["input_features"]):
        inputs["input_features"] = inputs["input_features"].to(device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw_text = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)[0].strip()

    parsed = _extract_json(raw_text)
    if parsed and isinstance(parsed.get("style"), str) and parsed["style"].strip():
        style = _clamp_words(" ".join(parsed["style"].strip().split()))[:500]
        description = str(parsed.get("description") or style).strip()[:1200]
        inst_raw = parsed.get("instrumental")
        if isinstance(inst_raw, bool):
            instrumental = inst_raw
        else:
            low = style.lower()
            instrumental = ("no vocals" in low) or ("purely instrumental" in low) or ("instrumental" in low and "vocal" not in low.replace("no vocals", ""))
        bpm = _coerce_bpm(parsed.get("bpm"))
        key = str(parsed["key"]).strip()[:40] if parsed.get("key") else None
        title_idea = str(parsed["title_idea"]).strip().strip("\"'")[:80] if parsed.get("title_idea") else None
        if title_idea and ":" in title_idea:
            # Drop "Series: Title" style prefixes the model sometimes invents.
            title_idea = title_idea.split(":")[-1].strip() or title_idea
        if title_idea and re.search(r"\d+\.\d+", title_idea):
            # Version-number-looking names ("Something 5.2") are hallucinated product names,
            # not song titles; let the caller fall back to the file name.
            title_idea = None
    else:
        cleaned_raw = " ".join(raw_text.split())
        low = cleaned_raw.lower()
        style = cleaned_raw[:450]
        description = cleaned_raw[:1200]
        instrumental = ("no vocals" in low) or ("purely instrumental" in low)
        bpm = _coerce_bpm(cleaned_raw)
        key = None
        title_idea = None

    _stage("done")
    return {
        "style": style,
        "description": description,
        "instrumental": instrumental,
        "bpm": bpm,
        "key": key,
        "title_idea": title_idea,
        "duration_seconds": duration_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze an audio track locally using Music Flamingo")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio_path.is_file():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    result = analyze_audio(args.audio_path, max_seconds=args.max_seconds, model_dir=args.model_dir)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
