#!/usr/bin/env python3
"""Lightweight MCP server for YuE2 Studio (AMD Radeon PRO V620).

Proxies to the local YuE2 Studio FastAPI server (default: http://127.0.0.1:8789)
so it uses 0 VRAM and starts instantaneously over stdio (or streamable-http).

Usage:
  Stdio (Claude Code / Cursor / Antigravity):
    python mcp_server.py

  HTTP (optional):
    python mcp_server.py --http --port 8790
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

API_BASE = os.environ.get("YUE2_API_URL", "http://127.0.0.1:8789").rstrip("/")
AUDIO_DIR = Path(__file__).resolve().parent / "library" / "audio"

mcp = MCPServer("yue2-studio")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=30.0)


@mcp.tool()
def yue2_health() -> dict:
    """Check YuE2 Studio server status, V620 VRAM usage, and whether the GPU is busy."""
    with _client() as c:
        r = c.get("/api/health")
        r.raise_for_status()
        return r.json()


@mcp.tool()
def yue2_list_songs(limit: int = 20) -> list[dict]:
    """List songs in the YuE2 Studio library (newest first) without full lyrics/ABC text."""
    with _client() as c:
        r = c.get("/api/songs")
        r.raise_for_status()
        songs = r.json().get("songs", [])[: max(1, limit)]
    return [
        {
            "id": s.get("id"),
            "title": s.get("title"),
            "style": s.get("style"),
            "cot": s.get("cot"),
            "audio_seconds": s.get("audio_seconds"),
            "e2e_seconds": s.get("e2e_seconds"),
            "has_abc": bool(s.get("abc")),
            "audio_url": f"{API_BASE}{s.get('audio_url', '')}",
            "local_path": str(AUDIO_DIR / Path(s.get("audio_url", "")).name) if s.get("audio_url") else None,
        }
        for s in songs
    ]


@mcp.tool()
def yue2_get_song(song_id: str) -> dict:
    """Get full details for a song by ID, including lyrics, ABC notation score, and timing."""
    with _client() as c:
        r = c.get("/api/songs")
        r.raise_for_status()
        for s in r.json().get("songs", []):
            if s.get("id") == song_id:
                out = dict(s)
                if out.get("audio_url"):
                    out["full_audio_url"] = f"{API_BASE}{out['audio_url']}"
                    out["local_path"] = str(AUDIO_DIR / Path(out["audio_url"]).name)
                return out
    return {"error": f"Song '{song_id}' not found"}


@mcp.tool()
def yue2_job_status(job_id: str) -> dict:
    """Check progress of an async song generation job."""
    with _client() as c:
        r = c.get(f"/api/status/{job_id}")
        r.raise_for_status()
        return r.json()


@mcp.tool()
def yue2_generate_song(
    style: str,
    lyrics: str,
    title: str = "Untitled Track",
    cot: str = "full",
    ode_steps: int = 8,
    ode_method: str = "midpoint",
    duration_seconds: Optional[int] = None,
    seed: Optional[int] = None,
    abc: Optional[str] = None,
    cover_prompt: Optional[str] = None,
    generate_cover: bool = True,
    wait: bool = True,
) -> dict:
    """Generate a 48 kHz stereo song with YuE2-3B on the V620 (streams live into the Web UI!).

    Args:
        style: Genre, vocal character, instrumentation, tempo (e.g. "Upbeat synth-pop, female vocal, 120 BPM").
        lyrics: Song lyrics with section tags like [Verse 1], [Chorus], [Bridge], [Outro].
        title: Track title for the library.
        cot: Symbolic planning mode: "full" (chords+melody ABC), "melody" (melody-only ABC), or "off" (direct, fastest).
        ode_steps: Flow-matching steps (1-64; 5-8 midpoint is fast, 12 high fidelity, 32 studio reference).
        ode_method: ODE solver: "midpoint" (2 NFE/step, recommended) or "euler" (1 NFE/step).
        duration_seconds: Optional max duration cap in seconds (15-360), or None to let the model decide naturally.
        seed: Optional integer seed for reproducibility.
        abc: Optional ABC notation score to guide melody/chords (for covers or score-edited remixes).
        cover_prompt: Optional image prompt for album cover art (if omitted, auto-written by LAN LLM).
        generate_cover: Whether to generate AI album cover art in parallel (default True).
        wait: If True (default), waits until generation completes and returns the finished song metadata & paths.
    """
    payload = {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "cot": cot,
        "ode_steps": ode_steps,
        "ode_method": ode_method,
        "duration_seconds": duration_seconds,
        "seed": seed,
        "abc": abc,
        "cover_prompt": cover_prompt,
        "generate_cover": generate_cover,
        "source": "mcp",
    }
    with _client() as c:
        r = c.post("/api/generate", json=payload)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        if not wait:
            return {"job_id": job_id, "status": "queued"}

        while True:
            st = c.get(f"/api/status/{job_id}")
            st.raise_for_status()
            data = st.json()
            if data.get("status") == "done":
                song = dict(data["song"])
                song["full_audio_url"] = f"{API_BASE}{song['audio_url']}"
                song["local_path"] = str(AUDIO_DIR / Path(song["audio_url"]).name)
                return song
            if data.get("status") == "error":
                return {"job_id": job_id, "status": "error", "error": data.get("error")}
            time.sleep(0.75)


@mcp.tool()
def yue2_import_link(url: str) -> dict:
    """Pull title, style prompt, and lyrics from a public Suno (suno.com) or ACE Music (acemusic.ai) song link."""
    with _client() as c:
        r = c.post("/api/import-link", json={"url": url})
        r.raise_for_status()
        return r.json()


@mcp.tool()
def yue2_delete_song(song_id: str) -> dict:
    """Delete a song from the YuE2 Studio library."""
    with _client() as c:
        r = c.delete(f"/api/songs/{song_id}")
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YuE2 Studio MCP Server")
    p.add_argument("--http", action="store_true", help="Serve over Streamable HTTP instead of stdio")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8790)
    args = p.parse_args()

    if args.http:
        import uvicorn
        app = mcp.streamable_http_app(
            host=args.host,
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        mcp.run(transport="stdio")
