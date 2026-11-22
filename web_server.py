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
import subprocess
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
SETTINGS_FILE = LIB_DIR / "settings.json"
USERS_FILE = LIB_DIR / "users.json"
WORKSPACES_FILE = LIB_DIR / "workspaces.json"
FOLDERS_FILE = LIB_DIR / "folders.json"
WEB_DIR = BASE_DIR / "web_ui"
REPORT_DIR = BASE_DIR / "report"

_DEFAULT_SETTINGS = {
    "llm_provider": os.environ.get("YUE2_COVER_LLM_PROVIDER", "openai").lower(),  # openai | openrouter | anthropic | ollama | opencode-go
    "llm_url": os.environ.get("YUE2_COVER_LLM_URL", "").rstrip("/"),
    "llm_model": os.environ.get("YUE2_COVER_LLM_MODEL", "maple-cpu"),
    "llm_api_key": os.environ.get("YUE2_COVER_LLM_API_KEY", "") or os.environ.get("OPENCODE_API_KEY", ""),
    "llm_max_tokens": int(os.environ.get("YUE2_COVER_LLM_MAX_TOKENS", "4096")),
    "img_provider": os.environ.get("YUE2_COVER_IMG_PROVIDER", "sdapi").lower(),   # sdapi | openai | openrouter | fal
    "img_url": os.environ.get("YUE2_COVER_IMG_URL", "").rstrip("/"),
    "img_model": os.environ.get("YUE2_COVER_IMG_MODEL", "qwen-image-2.1-turbo"),
    "img_api_key": os.environ.get("YUE2_COVER_IMG_API_KEY", ""),
    "img_steps": int(os.environ.get("YUE2_COVER_IMG_STEPS", "6")),
    "img_cfg": float(os.environ.get("YUE2_COVER_IMG_CFG", "1.0")),
    "img_size": int(os.environ.get("YUE2_COVER_IMG_SIZE", "768")),
    # Model-router coordination (llama.cpp router / Hermes Console): free shared GPU VRAM before each generation
    "unload_enabled": os.environ.get("YUE2_UNLOAD_ENABLED", "1") not in ("0", "false", "False", ""),
    "unload_url": os.environ.get("YUE2_UNLOAD_URL", "http://127.0.0.1:8084").rstrip("/"),
    # Reverse direction: release YuE2's own VRAM while the router has a model loading/loaded
    "yield_enabled": os.environ.get("YUE2_YIELD_ENABLED", "1") not in ("0", "false", "False", ""),
}

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
_settings_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_songs: list[dict] = []


def _load_settings() -> dict:
    cfg = dict(_DEFAULT_SETTINGS)
    if SETTINGS_FILE.is_file():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for k in cfg:
                    if k in saved and saved[k] is not None:
                        cfg[k] = saved[k]
        except Exception as exc:
            print(f"[settings] failed to read {SETTINGS_FILE}: {exc}", flush=True)
    return cfg


_settings: dict = _load_settings()


def _get_settings() -> dict:
    with _settings_lock:
        return dict(_settings)


def _save_settings(cfg: dict):
    with _settings_lock:
        _settings.clear()
        _settings.update(cfg)
        SETTINGS_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


DEFAULT_USER_ID = "default"
DEFAULT_WORKSPACE_ID = "default"


def _load_json_list(path: Path) -> list[dict]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception as exc:
        print(f"[library] failed to read {path}: {exc}", flush=True)
    return []


def _save_json_list(path: Path, items: list[dict]):
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


_users: list[dict] = []
_workspaces: list[dict] = []
_folders: list[dict] = []


def _ensure_default_user():
    global _users
    if not any(u.get("id") == DEFAULT_USER_ID for u in _users):
        _users.insert(0, {
            "id": DEFAULT_USER_ID,
            "name": "Guest",
            "created_at": time.time(),
        })
        _save_json_list(USERS_FILE, _users)


def _load_users() -> list[dict]:
    global _users
    _users = _load_json_list(USERS_FILE)
    if not _users:
        _users = []
    _ensure_default_user()
    return _users


def _valid_user_id(uid: Optional[str]) -> str:
    if uid and any(u.get("id") == uid for u in _users):
        return uid
    return DEFAULT_USER_ID


def _ensure_default_workspace():
    global _workspaces
    if not any(w.get("id") == DEFAULT_WORKSPACE_ID for w in _workspaces):
        _workspaces.insert(0, {
            "id": DEFAULT_WORKSPACE_ID,
            "name": "My Workspace",
            "user_id": DEFAULT_USER_ID,
            "created_at": time.time(),
        })
        _save_json_list(WORKSPACES_FILE, _workspaces)


def _load_workspaces() -> list[dict]:
    global _workspaces
    _workspaces = _load_json_list(WORKSPACES_FILE)
    if not _workspaces:
        _workspaces = []
    migrated = False
    for w in _workspaces:
        if "user_id" not in w or not w.get("user_id"):
            w["user_id"] = DEFAULT_USER_ID
            migrated = True
    _ensure_default_workspace()
    if migrated:
        try:
            _save_json_list(WORKSPACES_FILE, _workspaces)
        except Exception:
            pass
    return _workspaces


def _load_folders() -> list[dict]:
    global _folders
    _folders = _load_json_list(FOLDERS_FILE)
    return _folders


def _valid_workspace_id(wid: Optional[str], user_id: Optional[str] = None) -> str:
    if wid:
        for w in _workspaces:
            if w.get("id") == wid and (user_id is None or (w.get("user_id") or DEFAULT_USER_ID) == user_id):
                return wid
    # fall back to first workspace of this user, else global default
    if user_id:
        for w in _workspaces:
            if (w.get("user_id") or DEFAULT_USER_ID) == user_id:
                return w["id"]
    if any(w.get("id") == DEFAULT_WORKSPACE_ID for w in _workspaces):
        return DEFAULT_WORKSPACE_ID
    return _workspaces[0]["id"] if _workspaces else DEFAULT_WORKSPACE_ID


def _valid_folder_id(fid: Optional[str], workspace_id: str) -> Optional[str]:
    if not fid:
        return None
    for f in _folders:
        if f.get("id") == fid and f.get("workspace_id") == workspace_id:
            return fid
    return None


def _get_opencode_session() -> str:
    """Stable x-opencode-session id for OpenCode Go routing/prompt-caching.

    OpenCode Go requires an `x-opencode-session` header with one stable ID per
    conversation (see https://opencode.ai/docs/go). We keep one persistent ID
    per studio install for background calls (auto-titling), and callers may pass
    their own (job_id / compose session) for per-song conversations.
    """
    cfg = _get_settings()
    sid = (cfg.get("opencode_session") or "").strip()
    if not sid:
        sid = f"yue2-{uuid.uuid4().hex[:12]}"
        cfg["opencode_session"] = sid
        try:
            _save_settings(cfg)
        except Exception:
            pass
    return sid

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
        try:
            songs = json.loads(META_FILE.read_text(encoding="utf-8"))
        except Exception:
            songs = []
        # Backfill multi-user + multi-workspace fields on old libraries.
        migrated = False
        for s in songs:
            if "user_id" not in s or not s.get("user_id"):
                # Derive from workspace owner when possible.
                wid = s.get("workspace_id")
                owner = next((w.get("user_id") for w in _workspaces if w.get("id") == wid and w.get("user_id")), None)
                s["user_id"] = owner or DEFAULT_USER_ID
                migrated = True
            if "workspace_id" not in s or not s.get("workspace_id"):
                s["workspace_id"] = DEFAULT_WORKSPACE_ID
                migrated = True
            if "folder_id" not in s:
                s["folder_id"] = None
                migrated = True
        if migrated:
            try:
                META_FILE.write_text(json.dumps(songs, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return songs
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
                "workspace_id": DEFAULT_WORKSPACE_ID,
                "folder_id": None,
                "gradient": GRADIENTS[idx % len(GRADIENTS)],
                "created_at": time.time() - (len(seed_specs) - idx) * 60,
            })
    _save_songs(seeded)
    return seeded


def _save_songs(songs: list[dict]):
    with _meta_lock:
        META_FILE.write_text(json.dumps(songs, indent=2, ensure_ascii=False), encoding="utf-8")


def _unload_external_models() -> list[str]:
    """Unload every model currently loaded on the user's model-router API (e.g. :8084) to free shared VRAM.

    Works with llama.cpp router mode (/v1/models with status.value, POST /models/unload) and Hermes Console
    (/api/status, POST /api/models/<id>/unload). Never raises; returns the list of unloaded model ids.
    """
    cfg = _get_settings()
    base = (cfg.get("unload_url") or "").rstrip("/")
    if not cfg.get("unload_enabled") or not base:
        return []
    unloaded: list[str] = []
    try:
        with httpx.Client(timeout=15.0) as c:
            loaded: list[str] = []
            try:
                r = c.get(f"{base}/v1/models")
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        st = (m.get("status") or {}).get("value", "")
                        if st in ("loaded", "loading"):
                            loaded.append(m["id"])
            except Exception:
                pass
            if not loaded:
                try:
                    r = c.get(f"{base}/api/status")
                    if r.status_code == 200:
                        for m in r.json().get("loaded_models", []):
                            loaded.append(m["id"] if isinstance(m, dict) else str(m))
                except Exception:
                    pass
            for mid in loaded:
                ok = False
                try:
                    r = c.post(f"{base}/models/unload", json={"model": mid})
                    ok = r.status_code < 300
                except Exception:
                    pass
                if not ok:
                    try:
                        r = c.post(f"{base}/api/models/{mid}/unload")
                        ok = r.status_code < 300
                    except Exception:
                        pass
                if ok:
                    unloaded.append(mid)
            if unloaded:
                time.sleep(1.5)  # give the driver a moment to release VRAM
        if unloaded:
            print(f"[unload] freed external models on {base}: {unloaded}", flush=True)
    except Exception as exc:
        print(f"[unload] failed ({exc})", flush=True)
    return unloaded


def _call_llm(sys_msg: str, user_msg: str, max_tokens: Optional[int] = None, session_id: Optional[str] = None) -> Optional[str]:
    """Call the configured LLM provider (openai-compat / openrouter / anthropic / ollama / opencode-go)."""
    cfg = _get_settings()
    prov = (cfg["llm_provider"] or "openai").lower().replace("_", "-")
    api_key = cfg["llm_api_key"] or ""
    model = cfg["llm_model"] or ""
    toks = int(max_tokens if max_tokens is not None else cfg.get("llm_max_tokens", 4096))
    effective_url = (cfg["llm_url"] or "").rstrip("/")
    if prov in ("opencode-go", "opencode", "opencode-go-zen", "zen"):
        # OpenCode Go managed-inference API (subscription, no per-token billing).
        # Base URL: https://opencode.ai/zen/go/v1  (OpenAI-compatible chat completions)
        # Requires: Authorization: Bearer <OPENCODE_API_KEY> + x-opencode-session header
        #   + a real User-Agent (not a generic SDK name). See https://opencode.ai/docs/go
        if not effective_url:
            effective_url = "https://opencode.ai/zen/go/v1"
    if not effective_url and prov == "openrouter" and api_key:
        effective_url = "https://openrouter.ai/api/v1"
    elif not effective_url and prov == "anthropic" and api_key:
        effective_url = "https://api.anthropic.com"
    elif not effective_url and prov == "openai" and api_key:
        effective_url = "https://api.openai.com/v1"
    if not effective_url:
        return None
    try:
        with httpx.Client(timeout=60.0) as c:
            if prov == "anthropic":
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                endpoint = effective_url if effective_url.endswith("/messages") else f"{effective_url}/v1/messages"
                r = c.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model,
                        "max_tokens": toks,
                        "system": sys_msg,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                )
                r.raise_for_status()
                txt = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
            elif prov == "ollama":
                endpoint = effective_url if effective_url.endswith("/chat") else f"{effective_url}/api/chat"
                r = c.post(
                    endpoint,
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"num_predict": toks},
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                )
                r.raise_for_status()
                txt = (r.json().get("message") or {}).get("content", "").strip()
            else:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                if prov == "openrouter":
                    headers.update({"HTTP-Referer": "https://github.com/franktheglock/yue2-rocm-v620", "X-Title": "YuE2 Studio"})
                if prov in ("opencode-go", "opencode", "opencode-go-zen", "zen"):
                    # Required by OpenCode Go: stable per-conversation session id
                    # for routing/prompt-caching, plus a non-generic User-Agent.
                    headers["x-opencode-session"] = session_id or _get_opencode_session()
                    headers["User-Agent"] = "YuE2-Studio/1.0"
                    headers["Content-Type"] = "application/json"
                # Accept all known Go base URLs:
                #   https://opencode.ai/zen/go/v1  ->  .../v1/chat/completions
                #   https://opencode.ai/zen/go     ->  .../go/v1/chat/completions
                #   https://opencode.ai/inference/go -> .../go/v1/chat/completions
                if effective_url.endswith("/chat/completions"):
                    endpoint = effective_url
                elif effective_url.rstrip("/").endswith("/v1"):
                    endpoint = f"{effective_url}/chat/completions"
                elif "/inference/go" in effective_url:
                    endpoint = f"{effective_url.rstrip('/')}/v1/chat/completions"
                elif "/zen/go" in effective_url:
                    endpoint = f"{effective_url.rstrip('/')}/v1/chat/completions" if effective_url.rstrip("/").endswith("/go") else f"{effective_url}/v1/chat/completions"
                else:
                    endpoint = f"{effective_url}/chat/completions" if effective_url.endswith("/v1") else f"{effective_url}/v1/chat/completions"
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.8,
                    "max_tokens": toks,
                }
                if prov == "openai" and "openai.com" not in effective_url and "openrouter.ai" not in effective_url:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                r = c.post(endpoint, headers=headers, json=body)
                r.raise_for_status()
                txt = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1].strip()
            return txt or None
    except Exception as exc:
        print(f"[llm:{prov}] failed ({exc})", flush=True)
        return None


def _craft_song_title(style: str, lyrics: str, session_id: Optional[str] = None) -> str:
    """Ask the configured LLM for a short, evocative song title from style & lyrics."""
    snippet = (lyrics or "").strip()[:500]
    txt = _call_llm(
        sys_msg=(
            "You are a music producer naming a song. Given the musical style and lyrics, "
            "output a single short, memorable song title (2 to 5 words). "
            "No quotes, no punctuation at the end, no explanation — output ONLY the title."
        ),
        user_msg=f"Style: {style}\nLyrics:\n{snippet}",
        session_id=session_id,
    )
    if txt:
        cleaned = txt.splitlines()[0].strip().strip("\"'`*_.,:;!?").strip()
        if cleaned:
            return cleaned[:80]
    return "Untitled Track"


def _craft_cover_prompt(title: str, style: str, lyrics: str, session_id: Optional[str] = None) -> str:
    """Ask the configured LLM provider (openai-compat / openrouter / anthropic / ollama / opencode-go) for an album cover prompt."""
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
        session_id=session_id,
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
    cfg = _get_settings()
    prov = (cfg["img_provider"] or "sdapi").lower()
    api_key = cfg["img_api_key"] or ""
    model = cfg["img_model"] or ""
    size = int(cfg.get("img_size", 768))
    steps = int(cfg.get("img_steps", 6))
    cfg_scale = float(cfg.get("img_cfg", 1.0))
    effective_url = (cfg["img_url"] or "").rstrip("/")
    if not effective_url and prov == "openrouter" and api_key:
        effective_url = "https://openrouter.ai/api/v1"
    elif not effective_url and prov == "fal" and api_key:
        effective_url = "https://fal.run"
    elif not effective_url and prov == "openai" and api_key:
        effective_url = "https://api.openai.com/v1"
    if not effective_url:
        return None
    try:
        with _cover_lock:
            with httpx.Client(timeout=180.0, follow_redirects=True) as c:
                img_bytes: Optional[bytes] = None

                if prov == "openrouter":
                    headers = {
                        "Authorization": f"Bearer {api_key}",
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
                            "model": model,
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

                elif prov == "fal":
                    headers = {"Authorization": f"Key {api_key}"}
                    model_slug = model.lstrip("/")
                    endpoint = effective_url if model_slug in effective_url else f"{effective_url}/{model_slug}"
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "prompt": prompt,
                            "image_size": {"width": size, "height": size},
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

                elif prov == "openai":
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    endpoint = effective_url if effective_url.endswith("/images/generations") else (
                        f"{effective_url}/images/generations" if effective_url.endswith("/v1") else f"{effective_url}/v1/images/generations"
                    )
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": model,
                            "prompt": prompt,
                            "n": 1,
                            "size": f"{size}x{size}",
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
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    endpoint = effective_url if effective_url.endswith("/txt2img") else f"{effective_url}/sdapi/v1/txt2img"
                    r = c.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": model,
                            "prompt": prompt,
                            "width": size,
                            "height": size,
                            "steps": steps,
                            "cfg_scale": cfg_scale,
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
                    print(f"[cover-img:{prov}] empty image payload", flush=True)
                    return None
                out = COVERS_DIR / f"{song_id}.png"
                out.write_bytes(img_bytes)
                return f"/covers/{song_id}.png"
    except Exception as exc:
        print(f"[cover-img:{prov}] failed: {exc}", flush=True)
        return None


def _run_cover_pipeline(song_id: str, title: str, style: str, lyrics: str, custom_prompt: Optional[str], shared_result: Optional[dict] = None, session_id: Optional[str] = None):
    prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else _craft_cover_prompt(title, style, lyrics, session_id=session_id or song_id)
    cover_url = _generate_cover_image(song_id, prompt)
    if shared_result is not None:
        shared_result["cover_prompt"] = prompt
        shared_result["cover_url"] = cover_url
    if cover_url:
        # Pop the cover art into the live generating track card immediately!
        job = _jobs.get(song_id)
        if job and job["status"] in ("queued", "running"):
            job["cover_url"] = cover_url
            job["cover_prompt"] = prompt
            job["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
            _broadcast("job_update", {"job": dict(job)})
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


def _router_busy() -> bool:
    """True if the model router (:8084) has any model loading or loaded."""
    cfg = _get_settings()
    base = (cfg.get("unload_url") or "").rstrip("/")
    if not base:
        return False
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"{base}/v1/models")
            if r.status_code == 200:
                for m in r.json().get("data", []):
                    if (m.get("status") or {}).get("value", "") in ("loaded", "loading"):
                        return True
                return False
            r = c.get(f"{base}/api/status")
            if r.status_code == 200:
                return bool(r.json().get("loaded_models"))
    except Exception:
        pass
    return False


def _release_engine() -> bool:
    """Free all of YuE2's VRAM (weights + VAE + allocator cache). Caller must hold _gpu_lock."""
    global _engine
    if _engine is None:
        return False
    import gc
    import torch
    _engine = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"[yield] released YuE2 engine, VRAM now {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved", flush=True)
    _broadcast("engine_state", {"loaded": False})
    return True


def _ensure_engine():
    """Lazily (re)load the engine. Caller must hold _gpu_lock."""
    global _engine
    if _engine is None:
        _engine = YuE2V620Engine(ar_quant="w4")
        print(f"[yield] engine reloaded in {_engine.load_seconds:.2f}s", flush=True)
        _broadcast("engine_state", {"loaded": True})
    return _engine


def _yield_watcher():
    """When a request hits the router (model loading/loaded) and we're idle, hand the GPU over."""
    while True:
        time.sleep(1.0)
        try:
            cfg = _get_settings()
            if not cfg.get("yield_enabled") or _engine is None:
                continue
            if _gpu_lock.locked() or any(j["status"] in ("queued", "running") for j in _jobs.values()):
                continue
            if not _router_busy():
                continue
            if _gpu_lock.acquire(blocking=False):
                try:
                    if any(j["status"] in ("queued", "running") for j in _jobs.values()):
                        continue
                    _release_engine()
                finally:
                    _gpu_lock.release()
        except Exception as exc:
            print(f"[yield] watcher error: {exc}", flush=True)


@app.on_event("startup")
async def startup_load():
    global _engine, _songs, _event_loop
    _event_loop = asyncio.get_running_loop()
    _load_users()
    _load_workspaces()
    _load_folders()
    _songs = _load_songs()
    print("Loading YuE2-3B + YuE2-Vae on V620 (ar_quant=w4)...", flush=True)
    _engine = await asyncio.to_thread(YuE2V620Engine, ar_quant="w4")
    print(f"Engine ready in {_engine.load_seconds:.2f}s!", flush=True)
    threading.Thread(target=_yield_watcher, daemon=True).start()


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
    user_id: Optional[str] = Field(default=None)
    workspace_id: Optional[str] = Field(default=None)
    folder_id: Optional[str] = Field(default=None)


class RemixRequest(BaseModel):
    style: str = Field(..., min_length=1)
    lyrics: Optional[str] = None
    title: Optional[str] = None
    cot: str = Field(default="full")
    ode_steps: int = Field(default=8, ge=1, le=64)
    ode_method: str = Field(default="midpoint")
    duration_seconds: Optional[int] = Field(default=None, ge=15, le=360)
    seed: Optional[int] = None
    reuse_abc: bool = Field(default=True)
    strip_chords: bool = Field(default=False)
    cover_prompt: Optional[str] = None
    generate_cover: bool = Field(default=True)
    user_id: Optional[str] = Field(default=None)
    workspace_id: Optional[str] = Field(default=None)
    folder_id: Optional[str] = Field(default=None)


def _strip_abc_chords(abc: str) -> str:
    """Drop chord symbols (quoted strings) from ABC for melody-only remixes."""
    return re.sub(r'"[^"]*"', '', abc)


def _enqueue_generation(
    *,
    style: str,
    lyrics: str,
    cot: str,
    seed: int,
    abc: Optional[str],
    ode_steps: int,
    ode_method: str,
    duration_seconds: Optional[int],
    title: str,
    cover_prompt: Optional[str],
    generate_cover: bool,
    source: str,
    user_id: Optional[str],
    workspace_id: Optional[str],
    folder_id: Optional[str],
    remixed_from: Optional[str] = None,
) -> str:
    job_id = uuid.uuid4().hex[:10]
    raw_title = (title or "").strip()
    needs_title = not raw_title or raw_title.lower() in {"untitled", "untitled track"}
    final_title = raw_title if not needs_title else "Untitled Track"
    uid = _valid_user_id(user_id)
    wid = _valid_workspace_id(workspace_id, uid)
    fid = _valid_folder_id(folder_id, wid)
    job = {
        "job_id": job_id,
        "title": final_title,
        "style": style.strip(),
        "cot": cot,
        "source": source,
        "remixed_from": remixed_from,
        "user_id": uid,
        "workspace_id": wid,
        "folder_id": fid,
        "status": "queued",
        "stage": "queued",
        "progress": 2,
        "detail": "Waiting for GPU...",
        "started_at": time.time(),
        "elapsed_seconds": 0.0,
        "cover_url": None,
        "cover_prompt": cover_prompt,
        "song": None,
        "error": None,
    }
    _jobs[job_id] = job
    _broadcast("job_update", {"job": dict(job)})
    cover_shared: dict = {"title": final_title, "cover_url": None, "cover_prompt": cover_prompt}

    def _sidecar_title_and_cover():
        chosen_title = final_title
        if needs_title:
            chosen_title = _craft_song_title(style.strip(), lyrics.strip(), session_id=job_id)
            cover_shared["title"] = chosen_title
            job["title"] = chosen_title
            _broadcast("job_update", {"job": dict(job)})
            for s in _songs:
                if s["id"] == job_id:
                    s["title"] = chosen_title
                    _save_songs(_songs)
                    _broadcast("song_updated", {"song": s})
                    break
        if generate_cover:
            _run_cover_pipeline(job_id, chosen_title, style.strip(), lyrics.strip(), cover_prompt, cover_shared, session_id=job_id)

    if needs_title or generate_cover:
        threading.Thread(target=_sidecar_title_and_cover, daemon=True).start()

    def _worker():
        try:
            with _gpu_lock:
                job["status"] = "running"
                job["started_at"] = time.time()
                job["detail"] = "Freeing GPU memory..."
                _broadcast("job_update", {"job": dict(job)})
                _unload_external_models()
                if _engine is None:
                    job["detail"] = "Reloading model into VRAM..."
                    _broadcast("job_update", {"job": dict(job)})
                    _ensure_engine()
                job["started_at"] = time.time()

                def on_stage(stage: str, pct: int, detail: str):
                    job["stage"] = stage
                    job["progress"] = pct
                    job["detail"] = detail
                    job["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
                    _broadcast("job_update", {"job": dict(job)})

                fname = f"yue2_{job_id}.flac"
                out_path = AUDIO_DIR / fname
                abc_in = abc.strip() if abc and abc.strip() else None
                if cot == "off":
                    abc_in = None
                max_sem = int(duration_seconds * 25) if duration_seconds is not None else 9000
                res = _ensure_engine().generate_song(
                    style=style.strip(),
                    lyrics=lyrics.strip(),
                    cot=cot,
                    seed=seed,
                    abc=abc_in,
                    max_semantic_tokens=max_sem,
                    ode_steps=ode_steps,
                    ode_method=ode_method,
                    output_path=out_path,
                    on_stage=on_stage,
                )
                t = res["timing"]
                song = {
                    "id": job_id,
                    "title": cover_shared.get("title") or final_title,
                    "user_id": uid,
                    "workspace_id": wid,
                    "folder_id": fid,
                    "style": style.strip(),
                    "lyrics": lyrics.strip(),
                    "cot": cot,
                    "seed": seed,
                    "ode_steps": ode_steps,
                    "ode_method": ode_method,
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
                    "source": source,
                    "remixed_from": remixed_from,
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
    return job_id


class CoverRequest(BaseModel):
    cover_prompt: Optional[str] = None


class ImportLinkRequest(BaseModel):
    url: str = Field(..., min_length=4)


class SettingsUpdate(BaseModel):
    llm_provider: str = Field(default="openai")
    llm_url: str = Field(default="")
    llm_model: str = Field(default="")
    llm_api_key: Optional[str] = Field(default=None)
    clear_llm_key: bool = Field(default=False)
    llm_max_tokens: int = Field(default=4096, ge=64, le=32768)
    img_provider: str = Field(default="sdapi")
    img_url: str = Field(default="")
    img_model: str = Field(default="")
    img_api_key: Optional[str] = Field(default=None)
    clear_img_key: bool = Field(default=False)
    img_size: int = Field(default=768, ge=256, le=2048)
    img_steps: int = Field(default=6, ge=1, le=100)
    img_cfg: float = Field(default=1.0, ge=0.0, le=30.0)
    unload_enabled: bool = Field(default=True)
    unload_url: str = Field(default="http://127.0.0.1:8084")
    yield_enabled: bool = Field(default=True)


def _public_settings(cfg: dict) -> dict:
    return {
        "llm_provider": cfg["llm_provider"],
        "llm_url": cfg["llm_url"],
        "llm_model": cfg["llm_model"],
        "llm_api_key_set": bool(cfg.get("llm_api_key")),
        "llm_max_tokens": cfg["llm_max_tokens"],
        "img_provider": cfg["img_provider"],
        "img_url": cfg["img_url"],
        "img_model": cfg["img_model"],
        "img_api_key_set": bool(cfg.get("img_api_key")),
        "img_size": cfg["img_size"],
        "img_steps": cfg["img_steps"],
        "img_cfg": cfg["img_cfg"],
        "unload_enabled": bool(cfg.get("unload_enabled")),
        "unload_url": cfg.get("unload_url", ""),
        "yield_enabled": bool(cfg.get("yield_enabled")),
    }


@app.get("/api/settings")
def get_settings():
    return _public_settings(_get_settings())


@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    cfg = _get_settings()
    cfg["llm_provider"] = req.llm_provider.strip().lower().replace("_", "-") or "openai"
    cfg["llm_url"] = req.llm_url.strip().rstrip("/")
    cfg["llm_model"] = req.llm_model.strip()
    if req.clear_llm_key:
        cfg["llm_api_key"] = ""
    elif req.llm_api_key is not None and req.llm_api_key.strip():
        cfg["llm_api_key"] = req.llm_api_key.strip()
    cfg["llm_max_tokens"] = req.llm_max_tokens

    cfg["img_provider"] = req.img_provider.strip().lower() or "sdapi"
    cfg["img_url"] = req.img_url.strip().rstrip("/")
    cfg["img_model"] = req.img_model.strip()
    if req.clear_img_key:
        cfg["img_api_key"] = ""
    elif req.img_api_key is not None and req.img_api_key.strip():
        cfg["img_api_key"] = req.img_api_key.strip()
    cfg["img_size"] = req.img_size
    cfg["img_steps"] = req.img_steps
    cfg["img_cfg"] = req.img_cfg
    cfg["unload_enabled"] = req.unload_enabled
    cfg["unload_url"] = req.unload_url.strip().rstrip("/")
    cfg["yield_enabled"] = req.yield_enabled

    _save_settings(cfg)
    return _public_settings(cfg)


# ── Users, workspaces + folders (users own workspaces; workspaces hold folders; songs live in a workspace) ──
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)


class UserUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=40)


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    user_id: Optional[str] = Field(default=None)


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    user_id: Optional[str] = Field(default=None)


class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    workspace_id: Optional[str] = Field(default=None)


class FolderUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    workspace_id: Optional[str] = Field(default=None)


class SongUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    user_id: Optional[str] = Field(default=None)
    workspace_id: Optional[str] = Field(default=None)
    folder_id: Optional[str] = Field(default=None)


class ComposeRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)


@app.get("/api/users")
def list_users():
    return {"users": _users}


@app.post("/api/users")
def create_user(req: UserCreate):
    global _users, _workspaces
    uid = f"u-{uuid.uuid4().hex[:8]}"
    user = {"id": uid, "name": req.name.strip(), "created_at": time.time()}
    _users.append(user)
    _save_json_list(USERS_FILE, _users)
    # Every user starts with their own workspace so the library is never empty.
    wid = f"ws-{uuid.uuid4().hex[:8]}"
    ws = {"id": wid, "user_id": uid, "name": f"{user['name']}'s Workspace", "created_at": time.time()}
    _workspaces.append(ws)
    _save_json_list(WORKSPACES_FILE, _workspaces)
    _broadcast("library_moved", {"user_id": uid})
    return {"user": user, "workspace": ws}


@app.patch("/api/users/{user_id}")
def rename_user(user_id: str, req: UserUpdate):
    global _users
    for u in _users:
        if u.get("id") == user_id:
            if req.name and req.name.strip():
                u["name"] = req.name.strip()
            _save_json_list(USERS_FILE, _users)
            return {"user": u}
    raise HTTPException(404, "User not found")


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str):
    global _users, _workspaces, _folders, _songs
    if user_id == DEFAULT_USER_ID:
        raise HTTPException(400, "Cannot delete the default user")
    if not any(u.get("id") == user_id for u in _users):
        raise HTTPException(404, "User not found")
    _users = [u for u in _users if u.get("id") != user_id]
    _save_json_list(USERS_FILE, _users)
    doomed_ws = {w["id"] for w in _workspaces if (w.get("user_id") or DEFAULT_USER_ID) == user_id}
    _workspaces = [w for w in _workspaces if (w.get("user_id") or DEFAULT_USER_ID) != user_id]
    if not _workspaces:
        _ensure_default_workspace()
    _save_json_list(WORKSPACES_FILE, _workspaces)
    _folders = [f for f in _folders if f.get("workspace_id") not in doomed_ws]
    _save_json_list(FOLDERS_FILE, _folders)
    for s in _songs:
        if (s.get("user_id") or DEFAULT_USER_ID) == user_id or s.get("workspace_id") in doomed_ws:
            s["user_id"] = DEFAULT_USER_ID
            s["workspace_id"] = DEFAULT_WORKSPACE_ID
            s["folder_id"] = None
    _save_songs(_songs)
    _broadcast("library_moved", {"user_id": user_id})
    return {"ok": True}


@app.get("/api/workspaces")
def list_workspaces(user_id: Optional[str] = None):
    if user_id:
        return {"workspaces": [w for w in _workspaces if (w.get("user_id") or DEFAULT_USER_ID) == user_id]}
    return {"workspaces": _workspaces}


@app.post("/api/workspaces")
def create_workspace(req: WorkspaceCreate):
    global _workspaces
    uid = _valid_user_id(req.user_id)
    wid = f"ws-{uuid.uuid4().hex[:8]}"
    ws = {"id": wid, "user_id": uid, "name": req.name.strip(), "created_at": time.time()}
    _workspaces.append(ws)
    _save_json_list(WORKSPACES_FILE, _workspaces)
    return {"workspace": ws}


@app.patch("/api/workspaces/{ws_id}")
def rename_workspace(ws_id: str, req: WorkspaceUpdate):
    global _workspaces
    for w in _workspaces:
        if w.get("id") == ws_id:
            if req.name and req.name.strip():
                w["name"] = req.name.strip()
            if req.user_id and any(u.get("id") == req.user_id for u in _users):
                w["user_id"] = req.user_id
            _save_json_list(WORKSPACES_FILE, _workspaces)
            return {"workspace": w}
    raise HTTPException(404, "Workspace not found")


@app.delete("/api/workspaces/{ws_id}")
def delete_workspace(ws_id: str):
    global _workspaces, _folders, _songs
    if ws_id == DEFAULT_WORKSPACE_ID:
        raise HTTPException(400, "Cannot delete the default workspace")
    if not any(w.get("id") == ws_id for w in _workspaces):
        raise HTTPException(404, "Workspace not found")
    _workspaces = [w for w in _workspaces if w.get("id") != ws_id]
    _save_json_list(WORKSPACES_FILE, _workspaces)
    _folders = [f for f in _folders if f.get("workspace_id") != ws_id]
    _save_json_list(FOLDERS_FILE, _folders)
    # Move its songs back to the default workspace (don't lose audio).
    for s in _songs:
        if s.get("workspace_id") == ws_id:
            s["workspace_id"] = DEFAULT_WORKSPACE_ID
            s["folder_id"] = None
    _save_songs(_songs)
    _broadcast("library_moved", {"workspace_id": ws_id})
    return {"ok": True}


@app.get("/api/folders")
def list_folders(workspace_id: Optional[str] = None):
    if workspace_id:
        return {"folders": [f for f in _folders if f.get("workspace_id") == workspace_id]}
    return {"folders": _folders}


@app.post("/api/folders")
def create_folder(req: FolderCreate):
    global _folders
    wid = _valid_workspace_id(req.workspace_id)
    fid = f"f-{uuid.uuid4().hex[:8]}"
    folder = {"id": fid, "workspace_id": wid, "name": req.name.strip(), "created_at": time.time()}
    _folders.append(folder)
    _save_json_list(FOLDERS_FILE, _folders)
    return {"folder": folder}


@app.patch("/api/folders/{folder_id}")
def rename_folder(folder_id: str, req: FolderUpdate):
    global _folders
    for f in _folders:
        if f.get("id") == folder_id:
            if req.name and req.name.strip():
                f["name"] = req.name.strip()
            if req.workspace_id and any(w.get("id") == req.workspace_id for w in _workspaces):
                f["workspace_id"] = req.workspace_id
            _save_json_list(FOLDERS_FILE, _folders)
            return {"folder": f}
    raise HTTPException(404, "Folder not found")


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str):
    global _folders, _songs
    if not any(f.get("id") == folder_id for f in _folders):
        raise HTTPException(404, "Folder not found")
    _folders = [f for f in _folders if f.get("id") != folder_id]
    _save_json_list(FOLDERS_FILE, _folders)
    for s in _songs:
        if s.get("folder_id") == folder_id:
            s["folder_id"] = None
    _save_songs(_songs)
    _broadcast("library_moved", {"folder_id": folder_id})
    return {"ok": True}


@app.patch("/api/songs/{song_id}")
def update_song(song_id: str, req: SongUpdate):
    for s in _songs:
        if s["id"] == song_id:
            if req.title is not None and req.title.strip():
                s["title"] = req.title.strip()[:200]
            if req.user_id is not None:
                uid = _valid_user_id(req.user_id)
                s["user_id"] = uid
                # Keep workspace consistent with the new owner.
                wid_now = s.get("workspace_id")
                owner_now = next((w.get("user_id") for w in _workspaces if w.get("id") == wid_now), None)
                if owner_now and owner_now != uid:
                    s["workspace_id"] = _valid_workspace_id(None, uid)
                    s["folder_id"] = None
            if req.workspace_id is not None:
                uid = s.get("user_id") or DEFAULT_USER_ID
                wid = _valid_workspace_id(req.workspace_id, uid)
                s["workspace_id"] = wid
                # Adopt the workspace owner's user automatically.
                owner = next((w.get("user_id") for w in _workspaces if w.get("id") == wid), None)
                if owner:
                    s["user_id"] = owner
                # Clear folder if it doesn't belong to the new workspace.
                if s.get("folder_id") and not _valid_folder_id(s.get("folder_id"), wid):
                    s["folder_id"] = None
            if req.folder_id is not None:
                # Allow "" / "none" to unfile.
                if req.folder_id in ("", "none", "null"):
                    s["folder_id"] = None
                else:
                    wid = s.get("workspace_id") or DEFAULT_WORKSPACE_ID
                    s["folder_id"] = _valid_folder_id(req.folder_id, wid)
            _save_songs(_songs)
            _broadcast("song_updated", {"song": s})
            return {"song": s}
    raise HTTPException(404, "Song not found")


def _extract_compose_json(txt: str) -> Optional[dict]:
    """Pull {title, style, lyrics} out of LLM output (tolerates code fences)."""
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
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


@app.post("/api/compose")
def compose_song(req: ComposeRequest):
    """Simple workflow: one short prompt -> full title/style/lyrics via the configured LLM.

    Uses the OpenCode Go subscription when llm_provider=opencode-go (sends the
    required `x-opencode-session` header). `session_id` lets the UI keep one
    stable session per compose conversation for Go routing/prompt-caching.
    """
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt is empty")
    sid = (req.session_id or "").strip() or f"compose-{uuid.uuid4().hex[:10]}"
    txt = _call_llm(
        sys_msg=(
            "You write YuE2 song inputs as strict JSON. Output ONLY a JSON object with "
            "exactly these keys: title (2-5 words), style (one line: genre, vocal character, "
            "key instruments, tempo like '120 BPM'), lyrics (full song with [Verse 1], [Chorus], "
            "[Verse 2], [Chorus], [Bridge], [Outro] section tags, vivid singable lines). "
            "Keep lyrics 150-300 words so a ~2 minute song results. No markdown, no commentary."
        ),
        user_msg=f"Song idea: {prompt}",
        session_id=sid,
    )
    data = _extract_compose_json(txt or "")
    if not data or not (data.get("style") and data.get("lyrics")):
        raise HTTPException(502, "LLM did not return usable song JSON (check provider, model, API key).")
    title = str(data.get("title") or "Untitled Track").splitlines()[0].strip()[:80] or "Untitled Track"
    style = str(data.get("style") or "").strip()[:500]
    lyrics = str(data.get("lyrics") or "").strip()[:4000]
    cover_prompt = str(data.get("cover_prompt") or "").strip()[:500] or None
    return {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "cover_prompt": cover_prompt,
        "session_id": sid,
    }


@app.post("/api/settings/unload-now")
def unload_now():
    freed = _unload_external_models()
    return {"ok": True, "unloaded": freed}


@app.post("/api/settings/test-llm")
def test_llm():
    reply = _call_llm(
        sys_msg="You are a concise assistant. Reply with a short 2-4 word creative song title.",
        user_msg="Style: synth-pop, neon rain",
    )
    if reply:
        return {"ok": True, "reply": reply[:120]}
    return {"ok": False, "error": "No response from LLM (check provider, URL, model, and API key)."}


def _resolve_llm_base(prov: str, url: str, api_key: str) -> str:
    effective_url = (url or "").strip().rstrip("/")
    if prov in ("opencode-go", "opencode", "opencode-go-zen", "zen") and not effective_url:
        effective_url = "https://opencode.ai/zen/go/v1"
    if not effective_url and prov == "openrouter" and api_key:
        effective_url = "https://openrouter.ai/api/v1"
    elif not effective_url and prov == "anthropic" and api_key:
        effective_url = "https://api.anthropic.com"
    elif not effective_url and prov == "openai" and api_key:
        effective_url = "https://api.openai.com/v1"
    return effective_url


@app.get("/api/llm-models")
def list_llm_models(provider: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None):
    """List available LLM models from the configured (or preview) provider via its /models endpoint.

    Query params override saved settings for preview: provider, url. Pass api_key only
    when typing a *new* key; when omitted/empty the saved key is used so dropdowns work
    without re-entering secrets. OpenCode Go sends its required `x-opencode-session` header.
    """
    cfg = _get_settings()
    prov = (provider or cfg["llm_provider"] or "openai").lower().replace("_", "-")
    key = api_key if api_key not in (None, "") else (cfg["llm_api_key"] or "")
    base = _resolve_llm_base(prov, url if url is not None else (cfg["llm_url"] or ""), key)
    if not base:
        raise HTTPException(400, "Set a Base URL first (or pick OpenCode Go / OpenRouter to use the default).")
    try:
        with httpx.Client(timeout=20.0) as c:
            models: list[str] = []
            if prov == "ollama":
                root = base.split("/api/")[0].rstrip("/") or base
                r = c.get(f"{root}/api/tags")
                r.raise_for_status()
                for m in (r.json().get("models") or []):
                    name = m.get("name") or m.get("model")
                    if name:
                        models.append(name)
            elif prov == "anthropic":
                root = base.split("/v1/")[0].rstrip("/") if "/v1/" in base else base.rstrip("/")
                endpoint = f"{root}/v1/models"
                r = c.get(endpoint, headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                r.raise_for_status()
                for m in (r.json().get("data") or []):
                    if m.get("id"):
                        models.append(m["id"])
            else:
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                if prov in ("opencode-go", "opencode", "opencode-go-zen", "zen"):
                    headers["x-opencode-session"] = _get_opencode_session()
                    headers["User-Agent"] = "YuE2-Studio/1.0"
                # Candidate /models endpoints in order. Bare OpenAI-compatible hosts
                # (e.g. llama.cpp at http://host:port) serve /v1/models, not /models,
                # so try versioned first then fall back.
                candidates: list[str] = []
                b = base.rstrip("/")
                if b.endswith("/chat/completions"):
                    candidates.append(b[: -len("/chat/completions")] + "/models")
                elif b.endswith("/models"):
                    candidates.append(b)
                elif b.endswith("/v1"):
                    candidates.append(f"{b}/models")
                elif b.endswith("/go"):
                    candidates.append(f"{b}/v1/models")
                elif "/inference/go" in b:
                    candidates.append(f"{b}/v1/models")
                    candidates.append(f"{b}/models")
                else:
                    candidates.append(f"{b}/v1/models")
                    candidates.append(f"{b}/models")
                r = None
                endpoint = candidates[0]
                last_err = None
                for endpoint in candidates:
                    try:
                        r = c.get(endpoint, headers=headers)
                        r.raise_for_status()
                        last_err = None
                        break
                    except httpx.HTTPStatusError as he:
                        # Fall through to the next candidate on 404; fail fast otherwise.
                        if he.response.status_code == 404 and endpoint != candidates[-1]:
                            last_err = he
                            continue
                        raise
                assert r is not None
                payload = r.json()
                items = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(items, list):
                    for m in items:
                        mid = (m.get("id") if isinstance(m, dict) else None) or (m if isinstance(m, str) else None)
                        if mid:
                            models.append(mid)
            # De-dupe, keep order.
            seen: set[str] = set()
            uniq = [m for m in models if not (m in seen or seen.add(m))]
            return {"ok": True, "provider": prov, "endpoint": endpoint, "models": uniq}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        detail = f"{exc.response.status_code} from {prov} /models — check provider, URL, and API key."
        try:
            body = exc.response.text[:300]
            if body:
                detail += f" ({body[:200]})"
        except Exception:
            pass
        raise HTTPException(exc.response.status_code if exc.response.status_code < 500 else 502, detail)
    except Exception as exc:
        raise HTTPException(502, f"Could not reach {prov} /models at {base}: {exc}")


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
        "ready": True,
        "engine_loaded": _engine is not None,
        "gpu": "AMD Radeon PRO V620 (gfx1030, 32GB)",
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "busy": _gpu_lock.locked(),
    }


@app.get("/api/songs")
def list_songs():
    return {"songs": _songs}


@app.get("/api/songs/{song_id}/download")
def download_song(song_id: str, format: str = "mp3"):
    found = next((s for s in _songs if s["id"] == song_id), None)
    if not found or not found.get("audio_url"):
        raise HTTPException(404, "Song not found")
    fpath = AUDIO_DIR / Path(found["audio_url"]).name
    if not fpath.is_file():
        raise HTTPException(404, "Audio file missing")
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (found.get("title") or "song")).strip(" .") or "song"
    if (format or "mp3").lower() == "flac":
        return FileResponse(fpath, filename=f"{safe_title}.flac", media_type="audio/flac")
    mp3_path = fpath.with_suffix(".mp3")
    try:
        if not mp3_path.is_file() or mp3_path.stat().st_mtime < fpath.stat().st_mtime:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(fpath),
                 "-codec:a", "libmp3lame", "-b:a", "320k", str(mp3_path)],
                check=True,
            )
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not found for MP3 transcoding")
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"MP3 transcode failed: {exc}")
    return FileResponse(mp3_path, filename=f"{safe_title}.mp3", media_type="audio/mpeg")


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
    if _engine is None and not _get_settings().get("yield_enabled"):
        raise HTTPException(503, "Engine is still loading")
    seed = req.seed if req.seed is not None else random.randint(1, 2_000_000)
    job_id = _enqueue_generation(
        style=req.style.strip(),
        lyrics=req.lyrics.strip(),
        cot=req.cot,
        seed=seed,
        abc=req.abc.strip() if req.abc and req.abc.strip() else None,
        ode_steps=req.ode_steps,
        ode_method=req.ode_method,
        duration_seconds=req.duration_seconds,
        title=req.title or "",
        cover_prompt=req.cover_prompt,
        generate_cover=req.generate_cover,
        source=req.source,
        user_id=req.user_id,
        workspace_id=req.workspace_id,
        folder_id=req.folder_id,
    )
    return {"job_id": job_id}


@app.post("/api/songs/{song_id}/remix")
def remix_song(song_id: str, req: RemixRequest):
    """Remix a library song in a new style, reusing its ABC score (melody/chords)."""
    if _engine is None and not _get_settings().get("yield_enabled"):
        raise HTTPException(503, "Engine is still loading")
    src = next((s for s in _songs if s["id"] == song_id), None)
    if not src:
        raise HTTPException(404, "Song not found")
    if req.cot not in ("full", "melody", "off"):
        raise HTTPException(400, "cot must be full, melody or off")
    abc_src = (src.get("abc") or "").strip()
    abc_reuse: Optional[str] = None
    if req.reuse_abc:
        if not abc_src:
            raise HTTPException(400, "Source song has no ABC score to remix (try Generate instead)")
        abc_reuse = _strip_abc_chords(abc_src) if req.strip_chords or req.cot == "melody" else abc_src
    lyrics = req.lyrics.strip() if req.lyrics and req.lyrics.strip() else (src.get("lyrics") or "").strip()
    if not lyrics:
        raise HTTPException(400, "Remix needs lyrics (pass lyrics or pick a source with lyrics)")
    title = (req.title or "").strip() or f"{src.get('title') or 'Untitled'} (Remix)"
    seed = req.seed if req.seed is not None else random.randint(1, 2_000_000)
    job_id = _enqueue_generation(
        style=req.style.strip(),
        lyrics=lyrics,
        cot=req.cot,
        seed=seed,
        abc=abc_reuse,
        ode_steps=req.ode_steps,
        ode_method=req.ode_method,
        duration_seconds=req.duration_seconds,
        title=title,
        cover_prompt=req.cover_prompt,
        generate_cover=req.generate_cover,
        source="remix",
        user_id=req.user_id if req.user_id is not None else src.get("user_id"),
        workspace_id=req.workspace_id if req.workspace_id is not None else src.get("workspace_id"),
        folder_id=req.folder_id if req.folder_id is not None else src.get("folder_id"),
        remixed_from=song_id,
    )
    return {"job_id": job_id, "remixed_from": song_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8789, log_level="info")
