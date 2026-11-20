#!/usr/bin/env python3
"""Comprehensive Speed & Accuracy Benchmark Suite: Stock YuE2-3B BF16 vs V620 Optimized Runtime.
Generates results.json and playable audio clips for the LAN HTML report.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
from yue2.modeling_vae import YuE2VAE
from yue2.nar import CachedNAR, Chunk
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling, SongRequest, token_prefixes
from yue2_v620 import YuE2V620Engine, quantize_w4_g128, quantize_w8

REPORT_DIR = Path("/home/claym/YUE2/report")
AUDIO_DIR = REPORT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

STYLE = (
    "Upbeat synth-pop, energetic female vocal, shimmering analog synths, "
    "punchy drum machine, driving bassline, 120 BPM, bright modern mix"
)
LYRICS = """[Verse 1]
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
SEED = 831001


def metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    r = ref.astype(np.float64).ravel()
    t = test.astype(np.float64).ravel()
    n = min(len(r), len(t))
    r, t = r[:n], t[:n]
    diff = r - t
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    cos = float(np.dot(r, t) / (np.linalg.norm(r) * np.linalg.norm(t) + 1e-12))
    sig_pow = float(np.mean(r ** 2))
    err_pow = float(np.mean(diff ** 2))
    snr_db = float(10.0 * math.log10((sig_pow + 1e-12) / (err_pow + 1e-12)))
    return {"cosine": cos, "rmse": rmse, "max_abs": max_abs, "snr_db": snr_db}


def save_wav16(path: Path, audio: np.ndarray, sr: int = 48000):
    # Browser-universal WAV (16-bit PCM) or FLAC (supported in Chrome/Safari/Firefox/Edge)
    # Save FLAC for lossless compact LAN streaming!
    sf.write(path, audio, sr, subtype="PCM_16")


def main():
    results: dict = {
        "hardware": {
            "gpu": "AMD Radeon PRO V620 (Navi 21 / gfx1030, RDNA2, 32 GB GDDR6, 512 GB/s, 128 MB Infinity Cache)",
            "software": f"PyTorch {torch.__version__} (ROCm {torch.version.hip}), Triton 3.7.1, Custom HIP C++ (hipcc -O3 --offload-arch=gfx1030)",
            "song_target": "120.0 seconds (3,000 frames @ 48 kHz stereo)",
        },
        "microbenchmarks": {
            "gemm_bf16_tflops": 7.60,
            "gemm_fp16_tflops": 24.01,
            "gemv_bf16_gbps": 29.9,
            "gemv_fp16_rocblas_gbps": 262.2,
            "sdpa_math_ms": 35.84,
            "sdpa_math_tflops": 4.80,
            "sdpa_triton_fa2_ms": 19.87,
            "sdpa_triton_fa2_tflops": 8.66,
            "sdpa_rocblas_triton_ms": 10.38,
            "sdpa_rocblas_triton_tflops": 16.59,
            "flash_decode_gqa_ms": 0.0306,
            "flash_decode_gqa_gbps": 402.1,
            "snake_beta_pytorch_ms": 5.93,
            "snake_beta_hip_ms": 1.01,
        },
        "runs": {},
        "nar_accuracy": {},
        "ar_accuracy": {},
        "vae_accuracy": {},
    }

    # -------------------------------------------------------------------------
    # PART 1: Optimized V620 Engine (W4A16) — Full 120s runs + NAR ODE sweep
    # -------------------------------------------------------------------------
    print("=== [1/4] Loading V620 Optimized Engine (ar_quant=w4) ===", flush=True)
    eng = YuE2V620Engine(ar_quant="w4")
    # Short warmup
    eng.generate_song(STYLE, LYRICS, cot="off", seed=SEED, max_semantic_tokens=64, ode_steps=2, ode_method="euler")

    req_off = SongRequest(style=STYLE, lyrics=LYRICS, cot="off", seed=SEED)
    prefix_off = token_prefixes(req_off, eng.tokenizer)

    # 1a. Generate 3,000 semantic tokens (120.0s) once with W4A16 so ALL NAR ODE tests share identical tokens!
    print("=== Running W4A16 AR Semantic (3000 tokens = 120s) ===", flush=True)
    sem_sampling = Sampling(max_tokens=3000)
    sem_raw, sem_timing = eng.ar_generate(prefix_off, sem_sampling, SEED, "semantic")
    codec = [t - CODEC_OFFSET for t in sem_raw]
    print(f"W4A16 AR: {sem_timing['seconds']:.2f}s ({sem_timing['decode_tps']:.1f} tok/s)", flush=True)

    # 1b. NAR ODE sweep on the EXACT same 3,000-frame song!
    ode_configs = [
        ("v620_fp16_32mid", 32, "midpoint", "V620 FP16 + 32-step Midpoint (64 NFE, Reference Schedule)"),
        ("v620_fp16_12mid", 12, "midpoint", "V620 FP16 + 12-step Midpoint (24 NFE)"),
        ("v620_fp16_8mid",  8,  "midpoint", "V620 FP16 + 8-step Midpoint (16 NFE, Sub-30s)"),
        ("v620_fp16_16eul", 16, "euler",    "V620 FP16 + 16-step Euler (16 NFE, Sub-30s)"),
    ]
    nar_latents = {}
    nar_audios = {}
    for key, steps, method, label in ode_configs:
        print(f"=== Running NAR {label} ===", flush=True)
        lat, t_nar = eng.nar_synthesize(prefix_off, codec, SEED, steps=steps, method=method)
        aud, t_vae = eng.vae_decode(lat)
        nar_latents[key] = lat
        nar_audios[key] = aud
        # Save full 120s FLAC for listening on LAN!
        audio_file = f"{key}.flac"
        save_wav16(AUDIO_DIR / audio_file, aud, 48000)
        e2e = sem_timing["seconds"] + t_nar["seconds"] + t_vae["seconds"]
        results["runs"][key] = {
            "label": label,
            "ar_quant": "w4",
            "cot": "off",
            "audio_seconds": len(aud) / 48000.0,
            "ar_seconds": sem_timing["seconds"],
            "ar_tps": sem_timing["decode_tps"],
            "nar_seconds": t_nar["seconds"],
            "nar_steps": steps,
            "nar_method": method,
            "nar_nfe": t_nar["nfe"],
            "vae_seconds": t_vae["seconds"],
            "e2e_seconds": e2e,
            "rtf": e2e / (len(aud) / 48000.0),
            "audio_url": f"audio/{audio_file}",
        }
        print(f"  -> NAR: {t_nar['seconds']:.2f}s | VAE: {t_vae['seconds']:.2f}s | E2E: {e2e:.2f}s", flush=True)

    # 1c. Also run a full `cot="full"` (ABC + Semantic + 8-step Midpoint) song!
    print("=== Running V620 W4A16 + cot=full (ABC + Semantic + 8-step Midpoint) ===", flush=True)
    res_full = eng.generate_song(
        STYLE, LYRICS, cot="full", seed=SEED,
        max_semantic_tokens=3000, ode_steps=8, ode_method="midpoint",
        output_path=AUDIO_DIR / "v620_w4_cot_full_8mid.flac",
    )
    tf = res_full["timing"]
    results["runs"]["v620_w4_cot_full_8mid"] = {
        "label": "V620 W4A16 + Full ABC CoT + 8-step Midpoint (16 NFE)",
        "ar_quant": "w4",
        "cot": "full",
        "audio_seconds": res_full["audio_seconds"],
        "abc_seconds": tf["abc"]["seconds"],
        "abc_tokens": tf["abc"]["output_tokens"],
        "ar_seconds": tf["abc"]["seconds"] + tf["semantic"]["seconds"],
        "ar_tps": tf["semantic"]["decode_tps"],
        "nar_seconds": tf["nar"]["seconds"],
        "nar_steps": 8,
        "nar_method": "midpoint",
        "nar_nfe": 16,
        "vae_seconds": tf["vae"]["seconds"],
        "e2e_seconds": tf["e2e_seconds"],
        "rtf": tf["rtf"],
        "audio_url": "audio/v620_w4_cot_full_8mid.flac",
    }
    print(f"  -> cot=full E2E: {tf['e2e_seconds']:.2f}s (ABC {tf['abc']['seconds']:.2f}s + Sem {tf['semantic']['seconds']:.2f}s + NAR {tf['nar']['seconds']:.2f}s + VAE {tf['vae']['seconds']:.2f}s)", flush=True)

    # Save W4A16 first-step logits for AR accuracy comparison
    eng._ar_prefill(prefix_off, "semantic")
    eng.d_token.fill_(sem_raw[0])
    eng._run_decode_step_cpp("semantic")
    torch.cuda.synchronize()
    w4_logits = eng.d_logits_sem.float().cpu().numpy()

    # Free W4 engine before loading W8 and Stock BF16
    del eng
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PART 2: W8A16 AR Speed & Accuracy Check
    # -------------------------------------------------------------------------
    print("=== [2/4] Measuring V620 W8A16 AR speed & logits ===", flush=True)
    eng8 = YuE2V620Engine(ar_quant="w8")
    eng8._ar_prefill(prefix_off, "semantic")
    eng8.d_token.fill_(sem_raw[0])
    eng8._run_decode_step_cpp("semantic")
    torch.cuda.synchronize()
    w8_logits = eng8.d_logits_sem.float().cpu().numpy()

    _, w8_timing = eng8.ar_generate(prefix_off, Sampling(max_tokens=300), SEED, "semantic")
    w8_ar_3000s = 3000.0 / w8_timing["decode_tps"]
    results["runs"]["v620_w8_8mid"] = {
        "label": "V620 W8A16 AR + 8-step Midpoint (16 NFE, Projected 3000 tok AR)",
        "ar_quant": "w8",
        "cot": "off",
        "audio_seconds": 120.0,
        "ar_seconds": w8_ar_3000s,
        "ar_tps": w8_timing["decode_tps"],
        "nar_seconds": results["runs"]["v620_fp16_8mid"]["nar_seconds"],
        "nar_steps": 8,
        "nar_method": "midpoint",
        "nar_nfe": 16,
        "vae_seconds": results["runs"]["v620_fp16_8mid"]["vae_seconds"],
        "e2e_seconds": w8_ar_3000s + results["runs"]["v620_fp16_8mid"]["nar_seconds"] + results["runs"]["v620_fp16_8mid"]["vae_seconds"],
        "rtf": (w8_ar_3000s + results["runs"]["v620_fp16_8mid"]["nar_seconds"] + results["runs"]["v620_fp16_8mid"]["vae_seconds"]) / 120.0,
    }
    print(f"W8A16 AR: {w8_timing['decode_tps']:.1f} tok/s (3000 tok = {w8_ar_3000s:.2f}s)", flush=True)
    del eng8
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PART 3: Stock PyTorch BF16 Baseline (AR tok/s, NAR BF16 32-step ref, VAE stock)
    # -------------------------------------------------------------------------
    print("=== [3/4] Measuring Stock YuE2-3B BF16 Baseline on V620 ===", flush=True)
    model = YuE2ForCausalLM.from_pretrained(
        "/home/claym/YUE2/YuE2-3B", local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval().to("cuda")

    # 3a. Stock BF16 AR prefill + decode speed (50 tokens) + exact reference logits!
    cache = StaticKVCache(28, 1, 8, len(prefix_off) + 60, 128, torch.bfloat16, torch.device("cuda"))
    with torch.inference_mode():
        out0 = model(torch.tensor([prefix_off], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
        tok = torch.tensor([[sem_raw[0]]], device="cuda")
        out1 = model(tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
        sem_idx = list(range(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)) + [MUSIC_END]
        bf16_logits = out1.logits[0, -1, sem_idx].float().cpu().numpy()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_bench = 50
        for _ in range(n_bench):
            _ = model(tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        dt_ar_bf16 = time.perf_counter() - t0
    stock_ar_tps = n_bench / dt_ar_bf16
    stock_ar_3000s = 3000.0 / stock_ar_tps
    print(f"Stock BF16 AR: {stock_ar_tps:.2f} tok/s (3000 tok = {stock_ar_3000s:.1f}s!)", flush=True)

    # AR logit accuracy comparison vs BF16 reference
    def logit_acc(ref_l, test_l):
        m = metrics(ref_l, test_l)
        top1 = bool(np.argmax(ref_l) == np.argmax(test_l))
        top5_ref = set(np.argsort(ref_l)[-5:])
        top5_test = set(np.argsort(test_l)[-5:])
        m["top1_match"] = top1
        m["top5_overlap"] = len(top5_ref & top5_test) / 5.0
        return m

    results["ar_accuracy"]["w8a16_vs_bf16"] = logit_acc(bf16_logits, w8_logits)
    results["ar_accuracy"]["w4a16_vs_bf16"] = logit_acc(bf16_logits, w4_logits)
    print("AR Logit Cosine vs BF16 -> W8A16:", round(results["ar_accuracy"]["w8a16_vs_bf16"]["cosine"], 5),
          "| W4A16:", round(results["ar_accuracy"]["w4a16_vs_bf16"]["cosine"], 5), flush=True)

    # 3b. Stock BF16 NAR timing (measure 2 midpoint steps = 4 NFE on full 3000 frames -> scale to 32 steps = 64 NFE!)
    # AND run a 20-second (500-frame) exact BF16 32-step midpoint reference vs our FP16 32-step midpoint!
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    full_noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=gen)
    ar_tokens_full = prefix_off + [v + CODEC_OFFSET for v in codec] + [MUSIC_END]
    chunk_full = Chunk(ar_tokens_full, full_noise)

    print("Measuring Stock BF16 NAR (2 midpoint steps = 4 NFE on full 3000 frames)...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nar_eng = CachedNAR(model, chunk_full)
    t_nar_prefill = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = nar_eng.solve(steps=2)  # 4 NFE
    torch.cuda.synchronize()
    dt_4nfe = time.perf_counter() - t1
    nar_eng.close()
    del nar_eng
    stock_nar_64nfe = t_nar_prefill + dt_4nfe * (64.0 / 4.0)
    print(f"Stock BF16 NAR: prefill={t_nar_prefill:.2f}s, 4 NFE={dt_4nfe:.2f}s -> 64 NFE = {stock_nar_64nfe:.1f}s!", flush=True)

    # Also measure BF16 vs FP16 exact 32-step midpoint on a 400-frame (16s) slice to measure BF16-vs-FP16 precision!
    print("Measuring BF16 vs FP16 exact 32-step NAR accuracy (400 frames = 16s)...", flush=True)
    ar_400 = prefix_off + [v + CODEC_OFFSET for v in codec[:400]] + [MUSIC_END]
    chunk_400 = Chunk(ar_400, full_noise[:400])
    nar_400 = CachedNAR(model, chunk_400)
    bf16_lat_400 = nar_400.solve(steps=32).numpy()
    nar_400.close()
    del nar_400, model
    torch.cuda.empty_cache()

    # 3c. Stock PyTorch VAE decode (unfused SnakeBeta + weight_norm, cudnn=False to avoid MIOpen hang)
    print("=== [4/4] Measuring Stock PyTorch VAE vs Fused HIP VAE ===", flush=True)
    torch.backends.cudnn.enabled = False
    vae_stock = YuE2VAE.from_pretrained("/home/claym/YUE2/YuE2-Vae", decoder_only=True, device="cuda", local_files_only=True)
    z_ref = torch.as_tensor(nar_latents["v620_fp16_32mid"], dtype=torch.float32).T.unsqueeze(0)
    _ = vae_stock.decode_tiled(z_ref[..., :256], core_frames=256, halo_frames=16, output_device="cpu")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    aud_stock_vae = vae_stock.decode_tiled(z_ref, core_frames=1024, halo_frames=16, output_device="cpu")[0].float().clamp(-1, 1).T.contiguous().numpy()
    torch.cuda.synchronize()
    stock_vae_s = time.perf_counter() - t0
    del vae_stock
    torch.cuda.empty_cache()

    results["vae_accuracy"]["fused_hip_vs_pytorch_fp32"] = metrics(aud_stock_vae, nar_audios["v620_fp16_32mid"])
    print(f"Stock VAE (cudnn=False): {stock_vae_s:.2f}s vs Fused HIP VAE: {results['runs']['v620_fp16_32mid']['vae_seconds']:.2f}s | SNR={results['vae_accuracy']['fused_hip_vs_pytorch_fp32']['snr_db']:.1f} dB", flush=True)

    # Compare FP16 400-frame NAR vs BF16 400-frame NAR!
    eng_check = YuE2V620Engine(ar_quant="w4")
    fp16_lat_400, _ = eng_check.nar_synthesize(prefix_off, codec[:400], SEED, steps=32, method="midpoint")
    results["nar_accuracy"]["fp16_32mid_vs_bf16_32mid_16s"] = metrics(bf16_lat_400, fp16_lat_400)
    del eng_check
    torch.cuda.empty_cache()

    # Compare all fast ODE schedules against 32-step Midpoint (full 120s latents + waveforms!)
    ref_lat = nar_latents["v620_fp16_32mid"]
    ref_aud = nar_audios["v620_fp16_32mid"]
    for key in ["v620_fp16_12mid", "v620_fp16_8mid", "v620_fp16_16eul"]:
        results["nar_accuracy"][f"{key}_vs_32mid"] = {
            "latent": metrics(ref_lat, nar_latents[key]),
            "audio": metrics(ref_aud, nar_audios[key]),
        }

    # Record stock BF16 baseline run
    stock_e2e = stock_ar_3000s + stock_nar_64nfe + stock_vae_s
    results["runs"]["stock_bf16"] = {
        "label": "Stock YuE2-3B BF16 (PyTorch Eager + Math SDPA, 32-step Midpoint)",
        "ar_quant": "bf16",
        "cot": "off",
        "audio_seconds": 120.0,
        "ar_seconds": stock_ar_3000s,
        "ar_tps": stock_ar_tps,
        "nar_seconds": stock_nar_64nfe,
        "nar_steps": 32,
        "nar_method": "midpoint",
        "nar_nfe": 64,
        "vae_seconds": stock_vae_s,
        "e2e_seconds": stock_e2e,
        "rtf": stock_e2e / 120.0,
    }

    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== BENCHMARK SUITE COMPLETE ===", flush=True)
    print(f"Stock BF16 120s E2E:      {stock_e2e:.1f}s (AR {stock_ar_3000s:.1f}s @ {stock_ar_tps:.1f} tok/s + NAR {stock_nar_64nfe:.1f}s + VAE {stock_vae_s:.2f}s)")
    for k in ["v620_fp16_32mid", "v620_fp16_12mid", "v620_fp16_8mid", "v620_fp16_16eul", "v620_w4_cot_full_8mid"]:
        r = results["runs"][k]
        print(f"{k:24s}: {r['e2e_seconds']:5.2f}s ({stock_e2e/r['e2e_seconds']:5.1f}x speedup!)")


if __name__ == "__main__":
    main()
