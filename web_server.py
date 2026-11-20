#!/usr/bin/env python3
"""Suno-style Web UI & FastAPI server for YuE2-3B on AMD Radeon PRO V620 (gfx1030).

Features:
- Live SSE event stream (/api/events) so MCP / REST / UI generations pop up live in all browsers.
- Parallel AI album cover generation via LAN LLM (192.168.1.142:18081) + Qwen-Image 2.1 Turbo (192.168.1.142:8080).
"""
from __future__ import annotations

import asyncio
import base64
import html as html_lib
import json
import os
import random
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from yue2_v620 import YuE2V620Engine

BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR / "library"
AUDIO_DIR = LIB_DIR / "audio"
COVERS_DIR = LIB_DIR / "covers"
META_FILE = LIB_DIR / "songs.json"
WEB_DIR = BASE_DIR / "web_ui"
REPORT_DIR = BASE_DIR / "report"

LLM_PROVIDER = os.environ.get("YUE2_COVER_LLM_PROVIDER", "openai").lower()  # openai | openrouter | anthropic | ollama
LLM_URL = os.environ.get("YUE2_COVER_LLM_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("YUE2_COVER_LLM_MODEL", "maple-cpu")
LLM_API_KEY = os.environ.get("YUE2_COVER_LLM_API_KEY", "")

IMG_PROVIDER = os.environ.get("YUE2_COVER_IMG_PROVIDER", "sdapi").lower()   # sdapi | openai | openrouter | fal
IMG_URL = os.environ.get("YUE2_COVER_IMG_URL", "").rstrip("/")
IMG_MODEL = os.environ.get("YUE2_COVER_IMG_MODEL", "qwen-image-2.1-turbo")
IMG_API_KEY = os.environ.get("YUE2_COVER_IMG_API_KEY", "")
IMG_STEPS = int(os.environ.get("YUE2_COVER_IMG_STEPS", "6"))
IMG_CFG = float(os.environ.get("YUE2_COVER_IMG_CFG", "1.0"))
IMG_SIZE = int(os.environ.get("YUE2_COVER_IMG_SIZE", "768"))

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
COVERS_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YuE2 Studio — ROCm")
if REPORT_DIR.is_dir():
    app.mount("/report", StaticFiles(directory=str(REPORT_DIR), html=True), name="report")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
app.mount("/covers", StaticFiles(directory=str(COVERS_DIR)), name="covers")

_engine: Optional[YuE2V620Engine] = None
_gpu_lock = threading.Lock()
_cover_lock = threading.Lock()
_meta_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_songs: list[dict] = []

_event_loop: Optional[asyncio.AbstractEventLoop] = None
_subscribers: set[asyncio.Queue] = set()

GRADIENTS = [
    ["#ff5e62", "#ff9966"],
    ["#8b5cf6", "#ec4899"],
    ["#06b6d4", "#3b82f6"],
    ["#f59e0b", "#ef4444"],
    ["#10b981", "#06b6d4"],
    ["#6366f1", "#a855f7"],
    ["#f43f5e", "#fb923c"],
    ["#14b8a6", "#6366f1"],
]


def _broadcast(event_type: str, data: dict):
    """Push an SSE event to all connected browsers (thread-safe)."""
    if _event_loop is None or not _subscribers:
        return
    msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _put():
        dead = []
        for q in list(_subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _subscribers.discard(q)

    _event_loop.call_soon_threadsafe(_put)


def _load_songs() -> list[dict]:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    seeded = []
    seed_specs = [
        ("Electric in the Rain (Full ABC CoT)", "v620_w4_cot_full_5mid.flac", "full", 5, "midpoint", 32.19, 120.0),
        ("Electric in the Rain (Sub-30s Fast)", "v620_fp16_8mid.flac", "off", 8, "midpoint", 25.41, 120.0),
        ("Electric in the Rain (Studio 32-step)", "v620_fp16_32mid.flac", "off", 32, "midpoint", 55.44, 120.0),
    ]
    for idx, (title, fname, cot, steps, method, e2e, dur) in enumerate(seed_specs):
        src = REPORT_DIR / "audio" / fname
        if src.exists():
            dst = AUDIO_DIR / fname
            if not dst.exists():
                shutil.copy2(src, dst)
            seeded.append({
                "id": f"seed-{idx+1}",
                "title": title,
                "style": "Upbeat synth-pop, energetic female vocal, shimmering analog synths, punchy drum machine, driving bassline, 120 BPM",
                "lyrics": "[Verse 1]\nNeon lights flicker on the midnight street\nStatic in the air and a steady heartbeat\n...\n[Chorus]\nWe are electric in the pouring rain\nDancing through the fire, breaking every chain",
                "cot": cot,
                "seed": 831001,
                "ode_steps": steps,
                "ode_method": method,
                "audio_seconds": dur,
                "e2e_seconds": e2e,
                "audio_url": f"/audio/{fname}",
                "cover_url": None,
                "cover_prompt": None,
                "abc": None,
                "gradient": GRADIENTS[idx % len(GRADIENTS)],
                "created_at": time.time() - (len(seed_specs) - idx) * 60,
            })
    _save_songs(seeded)
    return seeded


def _save_songs(songs: list[dict]):
    with _meta_lock:
        META_FILE.write_text(json.dumps(songs, indent=2, ensure_ascii=False), encoding="utf-8")


def _call_llm(sys_msg: str, user_msg: str, max_tokens: int = 4096) -> Optional[str]:
    """Call the configured LLM provider (openai-compat / openrouter / anthropic / ollama)."""
    effective_url = LLM_URL
    if not effective_url and LLM_PROVIDER == "openrouter" and LLM_API_KEY:
        effective_url = "https://openrouter.ai/api/v1"
    elif not effective_url and LLM_PROVIDER == "anthropic" and LLM_API_KEY:
        effective_url = "https://api.anthropic.com"
    if not effective_url:
        return None
    try:
        with httpx.Client(timeout=60.0) as c:
            if LLM_PROVIDER == "anthropic":
                headers = {"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"}
                endpoint = effective_url if effective_url.endswith("/messages") else f"{effective_url}/v1/messages"
                r = c.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": LLM_MODEL,
                        "max_tokens": max_tokens,
                        "system": sys_msg,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                )
                r.raise_for_status()
                txt = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
            elif LLM_PROVIDER == "ollama":
                endpoint = effective_url if effective_url.endswith("/chat") else f"{effective_url}/api/chat"
                r = c.post(
                    endpoint,
                    json={
                        "model": LLM_MODEL,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                )
                r.raise_for_status()
                txt = (r.json().get("message") or {}).get("content", "").strip()
            else:
                headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
                if LLM_PROVIDER == "openrouter":
                    headers.update({"HTTP-Referer": "https://github.com/franktheglock/yue2-rocm-v620", "X-Title": "YuE2 Studio"})
                endpoint = effective_url if effective_url.endswith("/chat/completions") else (
                    f"{effective_url}/chat/completions" if effective_url.endswith("/v1") else f"{effective_url}/v1/chat/completions"
                )
                body = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.8,
                    "max_tokens": max_tokens,
                }
                if LLM_PROVIDER == "openai" and "openai.com" not in effective_url and "openrouter.ai" not in effective_url:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                r = c.post(endpoint, headers=headers, json=body)
                r.raise_for_status()
                txt = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1].strip()
            return txt or None
    except Exception as exc:
        print(f"[llm:{LLM_PROVIDER}] failed ({exc})", flush=True)
        return None


def _craft_song_title(style: str, lyrics: str) -> str:
    """Ask the configured LLM for a short, evocative song title from style & lyrics."""
    snippet = (lyrics or "").strip()[:500]
    txt = _call_llm(
        sys_msg=(
            "You are a music producer naming a song. Given the musical style and lyrics, "
            "output a single short, memorable song title (2 to 5 words). "
            "No quotes, no punctuation at the end, no explanation — output ONLY the title."
        ),
        user_msg=f"Style: {style}\nLyrics:\n{snippet}",
        max_tokens=4096,
    )
    if txt:
        cleaned = txt.splitlines()[0].strip().strip("\"'`*_.,:;!?").strip()
        if cleaned:
            return cleaned[:80]
    return "Untitled Track"


def _craft_cover_prompt(title: str, style: str, lyrics: str) -> str:
    """Ask the configured LLM provider (openai-compat / anthropic / ollama) for an album cover prompt."""
    snippet = (lyrics or "").strip()[:500]
    fallback = f"Artistic vinyl album cover art inspired by {style}, mood of '{title}', atmospheric lighting, rich texture, no text, no letters, no watermark"
    txt = _call_llm(
        sys_msg=(
            "You are an album cover art director. Given a song's title, musical style, and lyrics, "
            "write a single concise visual image-generation prompt (2-3 sentences max) for a striking "
            "square album cover. Focus on composition, color palette, lighting, and artistic medium. "
            "Never include text, words, logos, or typography in the image. Output ONLY the prompt."
        ),
        user_msg=f"Title: {title}\nStyle: {style}\nLyrics:\n{snippet}",
        max_tokens=4096,
    )
    return txt or fallback


def _extract_b64_image(data: dict) -> Optional[str]:
    raw = None
    if isinstance(data.get("images"), list) and data["images"]:
        raw = data["images"][0]
    elif data.get("image"):
        raw = data["image"]
    elif isinstance(data.get("data"), list) and data["data"]:
        raw = data["data"][0].get("b64_json") or data["data"][0].get("image")
    if not raw or not isinstance(raw, str):
        return None
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    return raw


def _decode_or_fetch_image(c: httpx.Client, val: str) -> Optional[bytes]:
    if not val or not isinstance(val, str):
        return None
    if val.startswith("data:") and "," in val:
        return base64.b64decode(val.split(",", 1)[1])
    if val.startswith("http://") or val.startswith("https://"):
        r = c.get(val)
        r.raise_for_status()
        return r.content
    return base64.b64decode(val)


def _generate_cover_image(song_id: str, prompt: str) -> Optional[str]:
    """Call configured image provider (sdapi | openai | openrouter | fal) and save PNG."""
    effective_url = IMG_URL
    if not effective_url and IMG_PROVIDER == "openrouter" and IMG_API_KEY:
        effective_url = "https://openrouter.ai/api/v1"
    elif not effective_url and IMG_PROVIDER == "fal" and IMG_API_KEY:
        effective_url = "https://fal.run"
    if not effective_url:
        return None
    try:
        with _cover_lock:
            with httpx.Client(timeout=180.0, follow_redirects=True) as c:
                img_bytes: Optional[bytes] = None

                if IMG_PROVIDER == "openrouter":
                    headers = {
                        "Authorization": f"Bearer {IMG_API_KEY}",
                        "HTTP-Referer": "https://github.com/franktheglock/yue2-rocm-v620",
                        "X-Title": "YuE2 Studio",
                    }
                    endpoint = effective_url if effective_url.endswith("/chat/completions") else (
                        f"{effective_url}/chat/completions" if effective_url.endswith("/v1") else f"{effective_url}/v1/chat/completions"
                    )
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": IMG_MODEL,
                            "modalities": ["image", "text"],
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    r.raise_for_status()
                    msg = (r.json().get("choices") or [{}])[0].get("message") or {}
                    imgs = msg.get("images") or []
                    if imgs:
                        first = imgs[0]
                        src = first.get("image_url", {}).get("url") if isinstance(first, dict) else first
                        img_bytes = _decode_or_fetch_image(c, src)
                    elif isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "image_url":
                                img_bytes = _decode_or_fetch_image(c, part.get("image_url", {}).get("url"))
                                if img_bytes:
                                    break

                elif IMG_PROVIDER == "fal":
                    headers = {"Authorization": f"Key {IMG_API_KEY}"}
                    model_slug = IMG_MODEL.lstrip("/")
                    endpoint = effective_url if model_slug in effective_url else f"{effective_url}/{model_slug}"
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "prompt": prompt,
                            "image_size": {"width": IMG_SIZE, "height": IMG_SIZE},
                            "num_images": 1,
                        },
                    )
                    r.raise_for_status()
                    payload = r.json()
                    imgs = payload.get("images") or []
                    if imgs:
                        first = imgs[0]
                        src = first.get("url") if isinstance(first, dict) else first
                        img_bytes = _decode_or_fetch_image(c, src)
                    elif isinstance(payload.get("image"), dict) and payload["image"].get("url"):
                        img_bytes = _decode_or_fetch_image(c, payload["image"]["url"])

                elif IMG_PROVIDER == "openai":
                    headers = {"Authorization": f"Bearer {IMG_API_KEY}"} if IMG_API_KEY else {}
                    endpoint = effective_url if effective_url.endswith("/images/generations") else (
                        f"{effective_url}/images/generations" if effective_url.endswith("/v1") else f"{effective_url}/v1/images/generations"
                    )
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": IMG_MODEL,
                            "prompt": prompt,
                            "n": 1,
                            "size": f"{IMG_SIZE}x{IMG_SIZE}",
                            "response_format": "b64_json",
                        },
                    )
                    r.raise_for_status()
                    payload = r.json()
                    b64 = _extract_b64_image(payload)
                    if b64:
                        img_bytes = base64.b64decode(b64)
                    else:
                        img_url = (payload.get("data") or [{}])[0].get("url")
                        img_bytes = _decode_or_fetch_image(c, img_url)

                else:
                    # Automatic1111 / Forge / sd.cpp / Qwen-Image SDAPI (/sdapi/v1/txt2img)
                    headers = {"Authorization": f"Bearer {IMG_API_KEY}"} if IMG_API_KEY else {}
                    endpoint = effective_url if effective_url.endswith("/txt2img") else f"{effective_url}/sdapi/v1/txt2img"
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": IMG_MODEL,
                            "prompt": prompt,
                            "width": IMG_SIZE,
                            "height": IMG_SIZE,
                            "steps": IMG_STEPS,
                            "cfg_scale": IMG_CFG,
                            "seed": -1,
                            "vae_tiling_params": {
                                "enabled": True,
                                "tile_size_w": 512,
                                "tile_size_h": 512,
                                "target_overlap": 0.5,
                            },
                        },
                    )
                    r.raise_for_status()
                    b64 = _extract_b64_image(r.json())
                    if b64:
                        img_bytes = base64.b64decode(b64)

                if not img_bytes:
                    print(f"[cover-img:{IMG_PROVIDER}] empty image payload", flush=True)
                    return None
                out = COVERS_DIR / f"{song_id}.png"
                out.write_bytes(img_bytes)
                return f"/covers/{song_id}.png"
    except Exception as exc:
        print(f"[cover-img:{IMG_PROVIDER}] failed: {exc}", flush=True)
        return None


def _run_cover_pipeline(song_id: str, title: str, style: str, lyrics: str, custom_prompt: Optional[str], shared_result: Optional[dict] = None):
    prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else _craft_cover_prompt(title, style, lyrics)
    cover_url = _generate_cover_image(song_id, prompt)
    if shared_result is not None:
        shared_result["cover_prompt"] = prompt
        shared_result["cover_url"] = cover_url
    if cover_url:
        # Update song in library if it has already been added!
        updated_song = None
        for s in _songs:
            if s["id"] == song_id:
                s["cover_url"] = cover_url
                s["cover_prompt"] = prompt
                updated_song = s
                break
        if updated_song is not None:
            _save_songs(_songs)
            _broadcast("song_updated", {"song": updated_song})


@app.on_event("startup")
async def startup_load():
    global _engine, _songs, _event_loop
    _event_loop = asyncio.get_running_loop()
    _songs = _load_songs()
    print("Loading YuE2-3B + YuE2-Vae on V620 (ar_quant=w4)...", flush=True)
    _engine = await asyncio.to_thread(YuE2V620Engine, ar_quant="w4")
    print(f"Engine ready in {_engine.load_seconds:.2f}s!", flush=True)


class GenerateRequest(BaseModel):
    title: str = Field(default="")
    style: str = Field(..., min_length=1)
    lyrics: str = Field(..., min_length=1)
    cot: str = Field(default="full")
    ode_steps: int = Field(default=8, ge=1, le=64)
    ode_method: str = Field(default="midpoint")
    duration_seconds: Optional[int] = Field(default=None, ge=15, le=360)
    seed: Optional[int] = None
    abc: Optional[str] = None
    cover_prompt: Optional[str] = None
    generate_cover: bool = Field(default=True)
    source: str = Field(default="ui")


class CoverRequest(BaseModel):
    cover_prompt: Optional[str] = None


class ImportLinkRequest(BaseModel):
    url: str = Field(..., min_length=4)


_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _extract_meta(page_html: str, key: str) -> Optional[str]:
    m = re.search(rf'<meta[^>]+(?:property|name)="{re.escape(key)}"[^>]+content="([^"]*)"', page_html, re.I)
    if not m:
        m = re.search(rf'<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="{re.escape(key)}"', page_html, re.I)
    return html_lib.unescape(m.group(1)) if m else None


def _import_suno(url: str) -> dict:
    clean = url.strip().strip("\"'<>()[]")
    if not clean.startswith("http"):
        clean = "https://" + clean
    parsed = urlparse(clean)
    path = parsed.path.rstrip("/")

    uuid_val = None
    m_song = re.match(r"^/song/([0-9a-fA-F-]{36})", path)
    m_short = re.match(r"^/s/([A-Za-z0-9_-]+)", path)
    m_hook = re.match(r"^/hook/([^/]+)", path)

    with httpx.Client(headers={"User-Agent": _UA}, timeout=20.0, follow_redirects=True) as c:
        if m_song:
            uuid_val = m_song.group(1).lower()
        elif m_short:
            r = c.get(f"https://suno.com/s/{m_short.group(1)}", headers={"Accept": "text/html,*/*"})
            m = _UUID_RE.search(str(r.url)) or _UUID_RE.search(r.text)
            if m:
                uuid_val = m.group(0).lower()
        elif m_hook:
            if _UUID_RE.fullmatch(m_hook.group(1)):
                uuid_val = m_hook.group(1).lower()
            else:
                r = c.get(clean, headers={"Accept": "text/html,*/*"})
                m = _UUID_RE.search(str(r.url) + " " + r.text)
                if m:
                    uuid_val = m.group(0).lower()
        else:
            m = _UUID_RE.search(clean)
            if m:
                uuid_val = m.group(0).lower()

        if not uuid_val:
            raise ValueError("Could not resolve a Suno song UUID from that link.")

        res = c.get(f"https://studio-api.prod.suno.com/api/clip/{uuid_val}", headers={"Accept": "application/json"})
        if res.status_code == 404:
            raise ValueError("Suno song not found (private or deleted).")
        res.raise_for_status()
        clip = res.json()

    meta = clip.get("metadata") or {}
    return {
        "source": "suno",
        "id": uuid_val,
        "url": f"https://suno.com/song/{uuid_val}",
        "title": (clip.get("title") or "Untitled").strip(),
        "artist": clip.get("display_name") or clip.get("handle") or "suno",
        "style": (meta.get("tags") or clip.get("display_tags") or "").strip(),
        "lyrics": (meta.get("prompt") or "").strip(),
        "duration": meta.get("duration"),
        "image_url": clip.get("image_large_url") or clip.get("image_url"),
    }


def _import_ace(url: str) -> dict:
    m = re.search(r"acemusic\.ai/song/([A-Za-z0-9_-]+)", url, re.I)
    if not m:
        raise ValueError("Not a valid acemusic.ai/song/ link.")
    song_id = m.group(1)
    canonical = f"https://acemusic.ai/song/{song_id}"
    with httpx.Client(headers={"User-Agent": _UA, "Accept": "text/html"}, timeout=20.0, follow_redirects=True) as c:
        res = c.get(canonical)
        if res.status_code == 404:
            raise ValueError("ACE Music song not found (private or deleted).")
        res.raise_for_status()
        page_html = res.text

    raw_title = _extract_meta(page_html, "og:title") or "Untitled"
    raw_desc = _extract_meta(page_html, "og:description") or ""
    image_url = _extract_meta(page_html, "og:image")

    title = re.sub(r"\s*\|\s*ACE Music\s*$", "", raw_title, flags=re.I).strip() or "Untitled"
    artist = "acemusic"
    by_m = re.match(r"^(.*)\s+by\s+(.+)$", title, re.I)
    if by_m:
        title, artist = by_m.group(1).strip(), by_m.group(2).strip()
    style = re.sub(r"^A song made with ACE Music\s*[—–-]\s*", "", raw_desc, flags=re.I).strip()

    lyrics = ""
    sec = re.search(r'<section class="sp-lyrics"[^>]*>([\s\S]*?)</section>', page_html, re.I)
    if sec:
        t = sec.group(1)
        t = re.sub(r"<h3[^>]*>", "\n\n[", t, flags=re.I)
        t = re.sub(r"</h3>", "]\n", t, flags=re.I)
        t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
        t = re.sub(r"</(?:p|div)>", "\n", t, flags=re.I)
        t = re.sub(r"<[^>]+>", "", t)
        t = html_lib.unescape(t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        lyrics = re.sub(r"\n{3,}", "\n\n", t).strip()

    return {
        "source": "ace",
        "id": song_id,
        "url": canonical,
        "title": title,
        "artist": artist,
        "style": style,
        "lyrics": lyrics,
        "duration": None,
        "image_url": image_url,
    }


@app.post("/api/import-link")
def import_link(req: ImportLinkRequest):
    u = req.url.strip()
    try:
        if "acemusic.ai" in u.lower():
            return _import_ace(u)
        if "suno.com" in u.lower():
            return _import_suno(u)
        raise ValueError("Unsupported URL. Paste a suno.com or acemusic.ai song link.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Import failed: {exc}")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/events")
async def events():
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _subscribers.add(q)

    async def stream():
        try:
            # Send active jobs on connect so late-joining tabs see in-progress MCP/UI renders immediately
            active = [
                {**j, "elapsed_seconds": round(time.time() - j["started_at"], 1)}
                for j in _jobs.values()
                if j["status"] in ("queued", "running")
            ]
            yield f"event: hello\ndata: {json.dumps({'active_jobs': active})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    import torch
    return {
        "ready": _engine is not None,
        "gpu": "AMD Radeon PRO V620 (gfx1030, 32GB)",
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "busy": _gpu_lock.locked(),
    }


@app.get("/api/songs")
def list_songs():
    return {"songs": _songs}


@app.get("/api/songs/{song_id}/download")
def download_song(song_id: str):
    found = next((s for s in _songs if s["id"] == song_id), None)
    if not found or not found.get("audio_url"):
        raise HTTPException(404, "Song not found")
    fpath = AUDIO_DIR / Path(found["audio_url"]).name
    if not fpath.is_file():
        raise HTTPException(404, "Audio file missing")
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (found.get("title") or "song")).strip(" .") or "song"
    ext = fpath.suffix or ".flac"
    return FileResponse(fpath, filename=f"{safe_title}{ext}", media_type="audio/flac")


@app.delete("/api/songs/{song_id}")
def delete_song(song_id: str):
    global _songs
    found = [s for s in _songs if s["id"] == song_id]
    if not found:
        raise HTTPException(404, "Song not found")
    _songs = [s for s in _songs if s["id"] != song_id]
    _save_songs(_songs)
    _broadcast("song_deleted", {"song_id": song_id})
    return {"ok": True}


@app.post("/api/songs/{song_id}/cover")
def regenerate_cover(song_id: str, req: CoverRequest = CoverRequest()):
    found = next((s for s in _songs if s["id"] == song_id), None)
    if not found:
        raise HTTPException(404, "Song not found")
    threading.Thread(
        target=_run_cover_pipeline,
        args=(song_id, found["title"], found["style"], found["lyrics"], req.cover_prompt, None),
        daemon=True,
    ).start()
    return {"ok": True, "song_id": song_id}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    out = dict(job)
    if out["status"] == "running":
        out["elapsed_seconds"] = round(time.time() - out["started_at"], 1)
    return out


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if _engine is None:
        raise HTTPException(503, "Engine is still loading")
    job_id = uuid.uuid4().hex[:10]
    seed = req.seed if req.seed is not None else random.randint(1, 2_000_000)
    raw_title = (req.title or "").strip()
    needs_title = not raw_title or raw_title.lower() in {"untitled", "untitled track"}
    title = raw_title if not needs_title else "Untitled Track"
    job = {
        "job_id": job_id,
        "title": title,
        "source": req.source,
        "status": "queued",
        "stage": "queued",
        "progress": 2,
        "detail": "Waiting for GPU...",
        "started_at": time.time(),
        "elapsed_seconds": 0.0,
        "song": None,
        "error": None,
    }
    _jobs[job_id] = job
    _broadcast("job_update", {"job": dict(job)})

    # Kick off auto-titling + cover art IN PARALLEL on the configured LLM/Image host (0s added to V620!)
    cover_shared: dict = {"title": title, "cover_url": None, "cover_prompt": req.cover_prompt}

    def _sidecar_title_and_cover():
        chosen_title = title
        if needs_title:
            chosen_title = _craft_song_title(req.style.strip(), req.lyrics.strip())
            cover_shared["title"] = chosen_title
            job["title"] = chosen_title
            _broadcast("job_update", {"job": dict(job)})
            for s in _songs:
                if s["id"] == job_id:
                    s["title"] = chosen_title
                    _save_songs(_songs)
                    _broadcast("song_updated", {"song": s})
                    break
        if req.generate_cover:
            _run_cover_pipeline(job_id, chosen_title, req.style.strip(), req.lyrics.strip(), req.cover_prompt, cover_shared)

    if needs_title or req.generate_cover:
        threading.Thread(target=_sidecar_title_and_cover, daemon=True).start()

    def _worker():
        try:
            with _gpu_lock:
                job["status"] = "running"
                job["started_at"] = time.time()
                _broadcast("job_update", {"job": dict(job)})

                def on_stage(stage: str, pct: int, detail: str):
                    job["stage"] = stage
                    job["progress"] = pct
                    job["detail"] = detail
                    job["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
                    _broadcast("job_update", {"job": dict(job)})

                fname = f"yue2_{job_id}.flac"
                out_path = AUDIO_DIR / fname
                abc_in = req.abc.strip() if req.abc and req.abc.strip() else None
                max_sem = int(req.duration_seconds * 25) if req.duration_seconds is not None else 9000
                res = _engine.generate_song(
                    style=req.style.strip(),
                    lyrics=req.lyrics.strip(),
                    cot=req.cot,
                    seed=seed,
                    abc=abc_in,
                    max_semantic_tokens=max_sem,
                    ode_steps=req.ode_steps,
                    ode_method=req.ode_method,
                    output_path=out_path,
                    on_stage=on_stage,
                )
                t = res["timing"]
                song = {
                    "id": job_id,
                    "title": cover_shared.get("title") or title,
                    "style": req.style.strip(),
                    "lyrics": req.lyrics.strip(),
                    "cot": req.cot,
                    "seed": seed,
                    "ode_steps": req.ode_steps,
                    "ode_method": req.ode_method,
                    "audio_seconds": round(res["audio_seconds"], 2),
                    "e2e_seconds": round(t["e2e_seconds"], 2),
                    "timing": {
                        "abc_s": round(t["abc"]["seconds"], 2),
                        "sem_s": round(t["semantic"]["seconds"], 2),
                        "sem_tps": round(t["semantic"]["decode_tps"], 1),
                        "nar_s": round(t["nar"]["seconds"], 2),
                        "vae_s": round(t["vae"]["seconds"], 2),
                    },
                    "audio_url": f"/audio/{fname}",
                    "cover_url": cover_shared.get("cover_url"),
                    "cover_prompt": cover_shared.get("cover_prompt"),
                    "abc": res["abc"],
                    "source": req.source,
                    "gradient": random.choice(GRADIENTS),
                    "created_at": time.time(),
                }
                _songs.insert(0, song)
                _save_songs(_songs)
                job["song"] = song
                job["status"] = "done"
                job["progress"] = 100
                job["elapsed_seconds"] = round(t["e2e_seconds"], 2)
                _broadcast("job_update", {"job": dict(job)})
                _broadcast("song_added", {"song": song})
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            _broadcast("job_update", {"job": dict(job)})

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8789, log_level="info")
