#!/usr/bin/env python3
"""CLI & benchmark runner for YuE2-3B on AMD Radeon PRO V620 (gfx1030)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yue2_v620 import YuE2V620Engine


DEFAULT_STYLE = (
    "Upbeat synth-pop, energetic female vocal, shimmering analog synths, "
    "punchy drum machine, driving bassline, 120 BPM, bright modern mix"
)

DEFAULT_LYRICS = """[Verse 1]
Neon lights flicker on the midnight street
Static in the air and a steady heartbeat
Chasing down echoes of a silver sound
Turning up the signal till we leave the ground

[Chorus]
We are electric in the pouring rain
Dancing through the fire, breaking every chain
Hold onto the rhythm, never let it fade
Living for the music that our hands have made

[Verse 2]
City skyline glowing in a violet haze
Counting every second of these golden days
Shadows fall behind us as we pick up speed
Everything we wanted is the beat we need

[Chorus]
We are electric in the pouring rain
Dancing through the fire, breaking every chain
Hold onto the rhythm, never let it fade
Living for the music that our hands have made

[Outro]
Neon lights flicker as the night turns blue
Every single melody comes back to you
"""


def main():
    p = argparse.ArgumentParser(description="YuE2-3B ultra-fast V620 (gfx1030) inference")
    p.add_argument("--model-dir", default=None, help="Local path or HF repo (default: ./YuE2-3B or m-a-p/YuE2-3B)")
    p.add_argument("--vae-dir", default=None, help="Local path or HF repo (default: ./YuE2-Vae or m-a-p/YuE2-Vae)")
    p.add_argument("--style", default=DEFAULT_STYLE)
    p.add_argument("--lyrics", default=DEFAULT_LYRICS)
    p.add_argument("--prompt-json", default=None, help="Optional JSON file with style/lyrics/seed")
    p.add_argument("--cot", default="full", choices=["full", "melody", "off"])
    p.add_argument("--seed", type=int, default=831001)
    p.add_argument("--ar-quant", default="w4", choices=["w4", "w8", "fp16"])
    p.add_argument("--ode-steps", type=int, default=8)
    p.add_argument("--ode-method", default="midpoint", choices=["midpoint", "euler"])
    p.add_argument("--max-seconds", type=float, default=120.0, help="Cap semantic frames (25 fps)")
    p.add_argument("--output", default="outputs/song.flac")
    p.add_argument("--warmup", action="store_true", help="Run a short warmup before timed generation")
    args = p.parse_args()

    style, lyrics, seed = args.style, args.lyrics, args.seed
    if args.prompt_json:
        d = json.loads(Path(args.prompt_json).read_text(encoding="utf-8"))
        style = d.get("style", style)
        lyrics = d.get("lyrics", lyrics)
        seed = d.get("seed", seed)

    print(f"Loading YuE2-3B + YuE2-Vae on V620 (ar_quant={args.ar_quant})...")
    eng = YuE2V620Engine(args.model_dir, args.vae_dir, ar_quant=args.ar_quant)
    print(f"Loaded in {eng.load_seconds:.2f}s | VRAM allocated: {__import__('torch').cuda.memory_allocated()/1e9:.2f} GB")

    if args.warmup:
        print("Running short warmup (compiling Triton kernels + capturing HIP graphs)...")
        eng.generate_song(style, lyrics, cot=args.cot, seed=seed, max_semantic_tokens=100, ode_steps=2, ode_method=args.ode_method)

    max_frames = int(round(args.max_seconds * 25))
    print(f"\nGenerating song (cot={args.cot}, max_frames={max_frames} [{args.max_seconds}s], ode={args.ode_steps} {args.ode_method})...")
    res = eng.generate_song(
        style, lyrics,
        cot=args.cot, seed=seed,
        max_semantic_tokens=max_frames,
        ode_steps=args.ode_steps,
        ode_method=args.ode_method,
        output_path=args.output,
    )
    t = res["timing"]
    print("\n=== YuE2-3B V620 Benchmark Results ===")
    print(f"Output:         {args.output}")
    print(f"Audio length:   {res['audio_seconds']:.2f} s ({res['codec_frames']} frames @ 48kHz stereo)")
    print(f"ABC stage:      {t['abc']['seconds']:.2f} s ({t['abc']['output_tokens']} tok, {t['abc'].get('decode_tps', 0):.1f} decode tok/s)")
    print(f"Semantic stage: {t['semantic']['seconds']:.2f} s ({t['semantic']['output_tokens']} tok, {t['semantic']['decode_tps']:.1f} decode tok/s)")
    print(f"NAR stage:      {t['nar']['seconds']:.2f} s ({t['nar']['steps']} {t['nar']['method']} = {t['nar']['nfe']} NFE)")
    print(f"VAE stage:      {t['vae']['seconds']:.2f} s")
    print(f"TOTAL E2E:      {t['e2e_seconds']:.2f} s  (RTF = {t['rtf']:.3f}x, {res['audio_seconds']/t['e2e_seconds']:.2f}x realtime!)")


if __name__ == "__main__":
    main()
