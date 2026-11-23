#!/usr/bin/env python3
"""Local SheetSage2 (ABC score + structure) transcription worker.

Runs in the main `.venv` (transformers 4.57.6 + ROCm torch).
Emits `STAGE:<name>` progress markers on stderr and a single JSON object on stdout.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHEETSAGE_DIR = BASE_DIR / "SheetSage2"
DEFAULT_MERT_DIR = BASE_DIR / "MERT-v2-FullSong"


def _stage(name: str):
    print(f"STAGE:{name}", file=sys.stderr, flush=True)


def strip_abc_chords(abc: str) -> str:
    """Strip quoted chord annotations from ABC music lines without touching header metadata."""
    if not abc:
        return abc
    out_lines = []
    for line in abc.splitlines():
        s = line.lstrip()
        if not s or s.startswith("%") or re.match(r"^[A-Za-z]:", s):
            out_lines.append(line)
        else:
            out_lines.append(re.sub(r'"[^"]*"', "", line))
    return "\n".join(out_lines)


def _parse_abc_meta(abc: str, labs: dict[str, str]) -> tuple[Optional[int], Optional[str]]:
    bpm: Optional[int] = None
    key: Optional[str] = None
    if abc:
        m_q = re.search(r"^Q:\s*(?:\d+/\d+\s*=\s*)?(\d+(?:\.\d+)?)", abc, re.M)
        if m_q:
            val = float(m_q.group(1))
            if 20 <= val <= 300:
                bpm = int(round(val))
        m_k = re.search(r"^K:\s*([^\r\n%]+)", abc, re.M)
        if m_k:
            key = m_k.group(1).strip()

    if bpm is None and labs.get("beat"):
        times = []
        for line in labs["beat"].splitlines():
            parts = line.strip().split("\t")
            if parts and parts[0]:
                try:
                    times.append(float(parts[0]))
                except ValueError:
                    pass
        if len(times) >= 4:
            diffs = np.diff(times)
            diffs = diffs[(diffs > 0.15) & (diffs < 2.5)]
            if len(diffs) > 0:
                est = 60.0 / float(np.median(diffs))
                if 20 <= est <= 300:
                    bpm = int(round(est))

    if not key and labs.get("key"):
        first = next((l.strip().split("\t") for l in labs["key"].splitlines() if l.strip()), None)
        if first and len(first) >= 3:
            key = first[2].strip()
    return bpm, key


def _parse_sections(labs: dict[str, str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    raw = labs.get("structure") or ""
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        label = parts[2].strip().lower()
        if not label or label == "silence" or end <= start:
            continue
        sections.append({"start": round(start, 2), "end": round(end, 2), "label": label})
    return sections


def _has_pitched_vocal(abc: str, vocal_notes_count: Optional[int]) -> bool:
    if vocal_notes_count is not None:
        return vocal_notes_count > 0
    if not abc:
        return False
    in_vocal = False
    for line in abc.splitlines():
        s = line.strip()
        if s.startswith("V:"):
            in_vocal = s.lower().startswith("v: vocal") or s.lower().startswith("v:vocal")
            continue
        if in_vocal and not s.startswith("%") and not re.match(r"^[A-Za-z]:", s):
            no_chords = re.sub(r'"[^"]*"', "", s)
            if re.search(r"[A-Ga-g]", no_chords):
                return True
    return False


_INST_TAG_EXPANSIONS = {
    "intro": ["intro"],
    "verse": ["verse"],
    "pre-chorus": ["pre-chorus"],
    "chorus": ["chorus"],
    "post-chorus": ["chorus"],
    "bridge": ["bridge"],
    "interlude": ["bridge"],
    "solo": ["bridge"],
    "instrumental": ["bridge"],
    "preshot": ["bridge"],
    "theme": ["chorus"],
    "development": ["verse"],
    "variation": ["verse"],
    "loop": ["chorus"],
    "rap": ["verse"],
    "irregular": ["verse"],
    "pre-outro": ["bridge"],
    "fade-out": ["outro"],
    "outro": ["outro"],
    "intro and verse": ["intro", "verse"],
    "pre-chorus and chorus": ["pre-chorus", "chorus"],
    "verse and pre-chorus": ["verse", "pre-chorus"],
}


def _build_instrumental_structure(sections: list[dict[str, Any]]) -> str:
    tags: list[str] = []
    for sec in sections:
        lbl = (sec.get("label") or "").strip().lower()
        for t in _INST_TAG_EXPANSIONS.get(lbl, []):
            if not tags or tags[-1] != t:
                tags.append(t)
    if not tags:
        return "[instrumental]"
    return "\n".join(f"[{t}]" for t in tags)


def run_transcribe(audio_path: Path, *, max_seconds: float = 240.0) -> dict[str, Any]:
    # Critical ROCm gfx1030 guard: MERT-v2 uses nn.Conv1d which hangs in MIOpen unless disabled.
    torch.backends.cudnn.enabled = False

    sheetsage_dir = Path(os.environ.get("YUE2_SHEETSAGE_DIR") or DEFAULT_SHEETSAGE_DIR)
    mert_dir = Path(os.environ.get("YUE2_MERT_DIR") or DEFAULT_MERT_DIR)
    if not sheetsage_dir.exists() or not mert_dir.exists():
        raise FileNotFoundError(f"SheetSage2 or MERT-v2-FullSong directory missing ({sheetsage_dir}, {mert_dir})")

    _stage("sheetsage_load")
    from transformers import AutoModel

    ss_model = AutoModel.from_pretrained(
        str(sheetsage_dir),
        trust_remote_code=True,
        base_model_path=str(mert_dir),
        local_files_only=True,
    ).eval().to("cuda")

    _stage("sheetsage_run")
    # Use dtype="fp32" on gfx1030 (BF16 is software-emulated and ~10x slower on RDNA2;
    # SheetSage2 Transcriber only accepts "bf16" or "fp32").
    ss_res = ss_model.transcribe(
        str(audio_path),
        dtype="fp32",
        melody_only=False,
        max_seconds=max_seconds,
    )

    warnings: list[str] = []
    duration_seconds = round(float(ss_res.get("duration_seconds") or 0.0), 2) or None
    abc_full: Optional[str] = ss_res.get("abc") or None
    abc_melody: Optional[str] = strip_abc_chords(abc_full) if abc_full else None
    if ss_res.get("abc_error"):
        warnings.append(f"ABC export warning: {ss_res['abc_error']}")
    if ss_res.get("warnings"):
        warnings.extend(str(w) for w in ss_res["warnings"])

    labs = ss_res.get("labs") or {}
    bpm, key = _parse_abc_meta(abc_full or "", labs)
    sections = _parse_sections(labs)
    has_vocal_melody = _has_pitched_vocal(abc_full or "", ss_res.get("vocal_notes"))
    instrumental_structure = _build_instrumental_structure(sections)

    del ss_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _stage("done")
    return {
        "abc": abc_full,
        "abc_melody": abc_melody,
        "instrumental_structure": instrumental_structure,
        "has_vocal_melody": has_vocal_melody,
        "bpm": bpm,
        "key": key,
        "duration_seconds": duration_seconds,
        "sections": sections,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio into ABC score and structure with SheetSage2")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    args = parser.parse_args()

    if not args.audio_path.is_file():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    result = run_transcribe(args.audio_path, max_seconds=args.max_seconds)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
