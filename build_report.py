#!/usr/bin/env python3
"""Generate an interactive, self-contained HTML benchmark report for LAN hosting."""
from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path("/home/claym/YUE2/report")


def fmt_s(v: float) -> str:
    return f"{v:.2f}s"


def main():
    data = json.loads((REPORT_DIR / "results.json").read_text())
    runs = data["runs"]
    mb = data["microbenchmarks"]
    ar_acc = data["ar_accuracy"]
    nar_acc = data["nar_accuracy"]
    vae_acc = data["vae_accuracy"]
    hw = data["hardware"]

    stock = runs["stock_bf16"]
    best = runs["v620_fp16_8mid"] if runs["v620_fp16_8mid"]["e2e_seconds"] <= runs["v620_fp16_16eul"]["e2e_seconds"] else runs["v620_fp16_16eul"]
    speedup_best = stock["e2e_seconds"] / best["e2e_seconds"]
    speedup_exact = stock["e2e_seconds"] / runs["v620_fp16_32mid"]["e2e_seconds"]
    ar_speedup = best["ar_tps"] / stock["ar_tps"]
    nar_speedup_exact = stock["nar_seconds"] / runs["v620_fp16_32mid"]["nar_seconds"]
    nar_speedup_best = stock["nar_seconds"] / best["nar_seconds"]

    ordered_keys = [
        "stock_bf16",
        "v620_fp16_32mid",
        "v620_w4_cot_full_8mid",
        "v620_w8_8mid",
        "v620_fp16_12mid",
        "v620_fp16_8mid",
        "v620_fp16_16eul",
    ]
    max_e2e = max(runs[k]["e2e_seconds"] for k in ordered_keys)

    # Stacked bar rows HTML
    bar_rows = []
    for k in ordered_keys:
        r = runs[k]
        is_sub30 = r["e2e_seconds"] < 30.0
        sp = stock["e2e_seconds"] / r["e2e_seconds"]
        ar_pct = (r["ar_seconds"] / max_e2e) * 100.0
        nar_pct = (r["nar_seconds"] / max_e2e) * 100.0
        vae_pct = (r["vae_seconds"] / max_e2e) * 100.0
        badge = '<span class="badge badge-green">SUB-30s GOAL</span>' if is_sub30 else (
            '<span class="badge badge-red">BASELINE</span>' if k == "stock_bf16" else '<span class="badge badge-blue">OPTIMIZED</span>'
        )
        bar_rows.append(f"""
        <div class="bar-row">
          <div class="bar-label">
            <div class="bar-title">{r['label']} {badge}</div>
            <div class="bar-sub">
              AR: <b>{r['ar_seconds']:.2f}s</b> ({r['ar_tps']:.1f} tok/s) &middot;
              NAR: <b>{r['nar_seconds']:.2f}s</b> ({r['nar_steps']} {r['nar_method']}, {r['nar_nfe']} NFE) &middot;
              VAE: <b>{r['vae_seconds']:.2f}s</b>
            </div>
          </div>
          <div class="bar-track-wrap">
            <div class="bar-track">
              <div class="seg seg-ar" style="width:{ar_pct:.2f}%" title="AR: {r['ar_seconds']:.2f}s"></div>
              <div class="seg seg-nar" style="width:{nar_pct:.2f}%" title="NAR: {r['nar_seconds']:.2f}s"></div>
              <div class="seg seg-vae" style="width:{vae_pct:.2f}%" title="VAE: {r['vae_seconds']:.2f}s"></div>
            </div>
            <div class="bar-val">
              <span class="val-time {'green' if is_sub30 else ''}">{r['e2e_seconds']:.2f}s</span>
              <span class="val-sp">{sp:.1f}&times;</span>
            </div>
          </div>
        </div>""")

    # Audio comparison cards
    audio_keys = [
        ("v620_fp16_32mid", "Reference Schedule (32-step Midpoint, 64 NFE)", "Exact 32-step midpoint reference in FP16"),
        ("v620_fp16_12mid", "High-Fidelity Fast (12-step Midpoint, 24 NFE)", "30.2s E2E — 2nd-order midpoint"),
        ("v620_fp16_8mid",  "Sub-30s Champion — 8-step Midpoint (16 NFE)", "25.4s E2E — 2nd-order midpoint, 4.72x realtime"),
        ("v620_fp16_16eul", "Sub-30s Champion — 16-step Euler (16 NFE)", "25.4s E2E — 1st-order Euler, 4.72x realtime"),
        ("v620_w4_cot_full_8mid", "Full Symbolic CoT (ABC + Semantic + 8-step Midpoint)", "Includes full editable ABC score generation"),
    ]
    audio_cards = []
    for k, title, desc in audio_keys:
        r = runs[k]
        acc_note = "Reference (0.00 dB diff)"
        if f"{k}_vs_32mid" in nar_acc:
            a = nar_acc[f"{k}_vs_32mid"]
            acc_note = f"Latent Cos: <b>{a['latent']['cosine']:.5f}</b> ({a['latent']['snr_db']:.1f} dB) &middot; Audio Cos: <b>{a['audio']['cosine']:.4f}</b>"
        elif k == "v620_w4_cot_full_8mid":
            acc_note = f"Generated {r.get('abc_tokens', 0)} ABC tokens in {r.get('abc_seconds', 0):.2f}s"
        audio_cards.append(f"""
        <div class="audio-card">
          <div class="audio-head">
            <div>
              <div class="audio-title">{title}</div>
              <div class="audio-desc">{desc}</div>
            </div>
            <div class="audio-time {'green' if r['e2e_seconds'] < 30 else ''}">{r['e2e_seconds']:.2f}s</div>
          </div>
          <audio controls preload="none" src="{r['audio_url']}"></audio>
          <div class="audio-meta">{acc_note} &middot; {r['audio_seconds']:.1f}s @ 48kHz Stereo FLAC</div>
        </div>""")

    fp16_vs_bf16 = nar_acc["fp16_32mid_vs_bf16_32mid_16s"]
    w8_acc = ar_acc["w8a16_vs_bf16"]
    w4_acc = ar_acc["w4a16_vs_bf16"]
    v_acc = vae_acc["fused_hip_vs_pytorch_fp32"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>YuE2-3B on AMD Radeon PRO V620 — Speed &amp; Accuracy Report</title>
<style>
  :root {{
    --bg: #0b0f17;
    --panel: #121826;
    --panel2: #182033;
    --border: #25304a;
    --text: #e6edf7;
    --muted: #8fa1c0;
    --green: #22c55e;
    --blue: #38bdf8;
    --purple: #a78bfa;
    --amber: #f59e0b;
    --red: #f43f5e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }}
  header {{
    background: linear-gradient(135deg, #121c30 0%, #1a1330 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 28px 30px; margin-bottom: 24px;
  }}
  h1 {{ margin: 0 0 8px; font-size: 1.85rem; letter-spacing: -0.02em; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; margin: 0; }}
  .hw-bar {{
    display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; font-size: 0.83rem; color: #b7c6e0;
  }}
  .hw-pill {{
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    padding: 5px 11px; border-radius: 999px;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin-bottom: 26px;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px;
  }}
  .kpi-label {{ color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-value {{ font-size: 2.1rem; font-weight: 800; margin: 4px 0; }}
  .kpi-sub {{ color: var(--muted); font-size: 0.85rem; }}
  .green {{ color: var(--green); }}
  .blue {{ color: var(--blue); }}
  .purple {{ color: var(--purple); }}
  .amber {{ color: var(--amber); }}
  section {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 24px; margin-bottom: 24px;
  }}
  h2 {{ margin: 0 0 6px; font-size: 1.25rem; }}
  .sec-desc {{ margin: 0 0 18px; color: var(--muted); font-size: 0.9rem; }}
  .legend {{ display: flex; gap: 18px; margin-bottom: 14px; font-size: 0.85rem; color: var(--muted); }}
  .dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
  .seg-ar {{ background: var(--blue); }}
  .seg-nar {{ background: var(--purple); }}
  .seg-vae {{ background: var(--amber); }}
  .bar-row {{ margin-bottom: 16px; }}
  .bar-title {{ font-weight: 600; font-size: 0.95rem; margin-bottom: 2px; }}
  .bar-sub {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }}
  .bar-track-wrap {{ display: flex; align-items: center; gap: 14px; }}
  .bar-track {{
    flex: 1; height: 22px; background: #0c111d; border-radius: 6px; overflow: hidden;
    display: flex; border: 1px solid #1e293b;
  }}
  .seg {{ height: 100%; }}
  .bar-val {{ width: 130px; text-align: right; font-variant-numeric: tabular-nums; }}
  .val-time {{ font-weight: 700; font-size: 1.02rem; }}
  .val-sp {{
    display: inline-block; margin-left: 6px; font-size: 0.8rem; color: #93c5fd;
    background: rgba(56,189,248,0.12); padding: 1px 6px; border-radius: 6px;
  }}
  .badge {{
    font-size: 0.7rem; font-weight: 700; padding: 2px 7px; border-radius: 999px;
    margin-left: 8px; vertical-align: middle; letter-spacing: 0.04em;
  }}
  .badge-green {{ background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }}
  .badge-blue {{ background: rgba(56,189,248,0.15); color: #7dd3fc; }}
  .badge-red {{ background: rgba(244,63,94,0.18); color: #fda4af; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: none; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
  .audio-card {{
    background: var(--panel2); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
  }}
  .audio-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }}
  .audio-title {{ font-weight: 700; font-size: 0.95rem; }}
  .audio-desc {{ color: var(--muted); font-size: 0.82rem; }}
  .audio-time {{ font-size: 1.25rem; font-weight: 800; white-space: nowrap; }}
  audio {{ width: 100%; height: 38px; margin-bottom: 8px; }}
  .audio-meta {{ font-size: 0.8rem; color: var(--muted); }}
  .note {{
    background: rgba(56,189,248,0.08); border-left: 3px solid var(--blue);
    padding: 12px 14px; border-radius: 6px; font-size: 0.88rem; color: #cbd5e1; margin-top: 14px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>YuE2-3B on AMD Radeon PRO V620 (gfx1030)</h1>
    <p class="subtitle">Custom RDNA2 Wave32 Fused HIP C++ &amp; Triton Kernel Acceleration &mdash; Speedup &amp; Accuracy Report (120-Second 48 kHz Stereo Song)</p>
    <div class="hw-bar">
      <span class="hw-pill">GPU: {hw['gpu']}</span>
      <span class="hw-pill">Stack: {hw['software']}</span>
      <span class="hw-pill">Target: {hw['song_target']}</span>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Fastest 120s Song (E2E)</div>
      <div class="kpi-value green">{best['e2e_seconds']:.2f}s</div>
      <div class="kpi-sub">Target &lt;30s beaten by {30.0 - best['e2e_seconds']:.2f}s ({120.0/best['e2e_seconds']:.2f}&times; realtime)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total E2E Speedup</div>
      <div class="kpi-value blue">{speedup_best:.1f}&times;</div>
      <div class="kpi-sub">vs Stock BF16 ({stock['e2e_seconds']:.1f}s) &middot; {speedup_exact:.1f}&times; at exact 64 NFE</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">AR Decode Throughput</div>
      <div class="kpi-value purple">{best['ar_tps']:.1f} tok/s</div>
      <div class="kpi-sub">{ar_speedup:.1f}&times; faster than Stock BF16 ({stock['ar_tps']:.1f} tok/s)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">FP16 vs BF16 NAR Fidelity</div>
      <div class="kpi-value amber">{fp16_vs_bf16['cosine']:.5f}</div>
      <div class="kpi-sub">Latent cosine similarity ({fp16_vs_bf16['snr_db']:.1f} dB SNR @ 32-step Midpoint)</div>
    </div>
  </div>

  <section>
    <h2>1. End-to-End 120-Second Song Generation Speed</h2>
    <p class="sec-desc">All runs generate a full 120.0s (3,000-frame, 48 kHz stereo) song on the V620. Stock BF16 uses PyTorch eager + Math SDPA (the only backend available on gfx1030).</p>
    <div class="legend">
      <span><span class="dot seg-ar"></span>AR Generation (ABC + Semantic)</span>
      <span><span class="dot seg-nar"></span>NAR Flow Matching (ODE)</span>
      <span><span class="dot seg-vae"></span>Oobleck VAE Decode (FP32)</span>
    </div>
    {''.join(bar_rows)}
  </section>

  <section>
    <h2>2. Listen Side-by-Side (48 kHz Stereo Lossless FLAC)</h2>
    <p class="sec-desc">All four <code>cot="off"</code> clips share the exact same 3,000 semantic tokens and noise seed so you can directly A/B test the NAR ODE schedules by ear.</p>
    <div class="grid-2">
      {''.join(audio_cards)}
    </div>
  </section>

  <section>
    <h2>3. Accuracy &amp; Numerical Fidelity Comparison</h2>
    <p class="sec-desc">Exact numerical deltas measured against Stock PyTorch BF16 and 32-step Midpoint (64 NFE) references on the same inputs and seeds.</p>
    <table>
      <thead>
        <tr>
          <th>Stage &amp; Comparison</th>
          <th class="num">Cosine Sim</th>
          <th class="num">SNR (dB)</th>
          <th class="num">RMSE</th>
          <th class="num">Extra Metric</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>AR Logits:</b> V620 W8A16 vs Stock BF16 (32,769 semantic vocab)</td>
          <td class="num">{w8_acc['cosine']:.6f}</td>
          <td class="num">{w8_acc['snr_db']:.2f} dB</td>
          <td class="num">{w8_acc['rmse']:.5f}</td>
          <td class="num">Top-1: {'Match' if w8_acc['top1_match'] else 'Diff'} &middot; Top-5: {w8_acc['top5_overlap']*100:.0f}%</td>
        </tr>
        <tr>
          <td><b>AR Logits:</b> V620 W4A16 (g=128) vs Stock BF16 (32,769 semantic vocab)</td>
          <td class="num">{w4_acc['cosine']:.6f}</td>
          <td class="num">{w4_acc['snr_db']:.2f} dB</td>
          <td class="num">{w4_acc['rmse']:.5f}</td>
          <td class="num">Top-1: {'Match' if w4_acc['top1_match'] else 'Diff'} &middot; Top-5: {w4_acc['top5_overlap']*100:.0f}%</td>
        </tr>
        <tr>
          <td><b>NAR Precision:</b> V620 FP16 (32-mid) vs Stock BF16 (32-mid, 16s slice)</td>
          <td class="num">{fp16_vs_bf16['cosine']:.6f}</td>
          <td class="num">{fp16_vs_bf16['snr_db']:.2f} dB</td>
          <td class="num">{fp16_vs_bf16['rmse']:.5f}</td>
          <td class="num">Max Abs: {fp16_vs_bf16['max_abs']:.4f}</td>
        </tr>
        <tr>
          <td><b>NAR ODE Schedule:</b> 12-step Midpoint (24 NFE) vs 32-step Midpoint (64 NFE)</td>
          <td class="num">{nar_acc['v620_fp16_12mid_vs_32mid']['latent']['cosine']:.6f}</td>
          <td class="num">{nar_acc['v620_fp16_12mid_vs_32mid']['latent']['snr_db']:.2f} dB</td>
          <td class="num">{nar_acc['v620_fp16_12mid_vs_32mid']['latent']['rmse']:.5f}</td>
          <td class="num">Audio Cos: {nar_acc['v620_fp16_12mid_vs_32mid']['audio']['cosine']:.4f}</td>
        </tr>
        <tr>
          <td><b>NAR ODE Schedule:</b> 8-step Midpoint (16 NFE, 25.4s) vs 32-step Midpoint</td>
          <td class="num">{nar_acc['v620_fp16_8mid_vs_32mid']['latent']['cosine']:.6f}</td>
          <td class="num">{nar_acc['v620_fp16_8mid_vs_32mid']['latent']['snr_db']:.2f} dB</td>
          <td class="num">{nar_acc['v620_fp16_8mid_vs_32mid']['latent']['rmse']:.5f}</td>
          <td class="num">Audio Cos: {nar_acc['v620_fp16_8mid_vs_32mid']['audio']['cosine']:.4f}</td>
        </tr>
        <tr>
          <td><b>NAR ODE Schedule:</b> 16-step Euler (16 NFE, 25.4s) vs 32-step Midpoint</td>
          <td class="num">{nar_acc['v620_fp16_16eul_vs_32mid']['latent']['cosine']:.6f}</td>
          <td class="num">{nar_acc['v620_fp16_16eul_vs_32mid']['latent']['snr_db']:.2f} dB</td>
          <td class="num">{nar_acc['v620_fp16_16eul_vs_32mid']['latent']['rmse']:.5f}</td>
          <td class="num">Audio Cos: {nar_acc['v620_fp16_16eul_vs_32mid']['audio']['cosine']:.4f}</td>
        </tr>
        <tr>
          <td><b>VAE Audio:</b> Fused HIP SnakeBeta + Fused WeightNorm vs PyTorch FP32</td>
          <td class="num">{v_acc['cosine']:.6f}</td>
          <td class="num">{v_acc['snr_db']:.2f} dB</td>
          <td class="num">{v_acc['rmse']:.6f}</td>
          <td class="num">Max Abs: {v_acc['max_abs']:.6f}</td>
        </tr>
      </tbody>
    </table>
    <div class="note">
      <b>Why 8-step Midpoint beats 16-step Euler at the exact same speed (25.4s / 16 NFE):</b>
      Midpoint is a 2nd-order Runge&ndash;Kutta solver (O(h&sup2;) local error vs O(h) for Euler), so at identical 16 NFE compute cost, 8-step Midpoint achieves significantly higher latent &amp; waveform fidelity to the 32-step reference.
    </div>
  </section>

  <section>
    <h2>4. Custom RDNA2 (gfx1030) Kernel Microbenchmarks</h2>
    <p class="sec-desc">Why stock PyTorch is slow on V620 and how our custom HIP C++ (<code>v620_kernels.hip</code>) + Triton kernels unlock the hardware.</p>
    <table>
      <thead>
        <tr>
          <th>Operation / Kernel</th>
          <th class="num">Stock PyTorch (gfx1030)</th>
          <th class="num">Custom V620 Kernel</th>
          <th class="num">Speedup</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>NAR Linear GEMM</b> (3002 &times; 2048 &times; 6144)</td>
          <td class="num">{mb['gemm_bf16_tflops']:.2f} TFLOPS (BF16)</td>
          <td class="num">{mb['gemm_fp16_tflops']:.2f} TFLOPS (FP16 rocBLAS)</td>
          <td class="num green">{mb['gemm_fp16_tflops']/mb['gemm_bf16_tflops']:.2f}&times;</td>
        </tr>
        <tr>
          <td><b>AR Decode GEMV</b> (1 &times; 2048 &times; 6144)</td>
          <td class="num">{mb['gemv_bf16_gbps']:.1f} GB/s (BF16)</td>
          <td class="num">LDS-Fused RMSNorm + uint4 W4A16 HIP</td>
          <td class="num green">{ar_speedup:.1f}&times; E2E tok/s</td>
        </tr>
        <tr>
          <td><b>NAR Attention</b> (16 Q / 8 KV, 3002 &times; 7000)</td>
          <td class="num">{mb['sdpa_math_ms']:.2f} ms ({mb['sdpa_math_tflops']:.2f} TFLOPS Math)</td>
          <td class="num">{mb['sdpa_rocblas_triton_ms']:.2f} ms ({mb['sdpa_rocblas_triton_tflops']:.2f} TFLOPS rocBLAS+Triton)</td>
          <td class="num green">{mb['sdpa_math_ms']/mb['sdpa_rocblas_triton_ms']:.2f}&times;</td>
        </tr>
        <tr>
          <td><b>AR FlashDecode GQA</b> (L=3001, Split-KV)</td>
          <td class="num">Math SDPA fallback</td>
          <td class="num">{mb['flash_decode_gqa_ms']*1000:.1f} &mu;s ({mb['flash_decode_gqa_gbps']:.1f} GB/s, 3e-5 err)</td>
          <td class="num green">79% HBM Peak</td>
        </tr>
        <tr>
          <td><b>Oobleck VAE SnakeBeta</b> (512 &times; 100k FP32)</td>
          <td class="num">{mb['snake_beta_pytorch_ms']:.2f} ms (6 kernels)</td>
          <td class="num">{mb['snake_beta_hip_ms']:.2f} ms (1 fused HIP kernel)</td>
          <td class="num green">{mb['snake_beta_pytorch_ms']/mb['snake_beta_hip_ms']:.2f}&times;</td>
        </tr>
        <tr>
          <td><b>VAE 1D Conv Backend</b> (120s FP32 decode)</td>
          <td class="num">MIOpen workspace hang (&gt;360s)</td>
          <td class="num">{runs['v620_fp16_8mid']['vae_seconds']:.2f} s (rocBLAS im2col + fused SnakeBeta)</td>
          <td class="num green">&gt;250&times;</td>
        </tr>
      </tbody>
    </table>
  </section>
</div>
</body>
</html>
"""
    out_path = REPORT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
