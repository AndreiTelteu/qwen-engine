#!/usr/bin/env python3
"""Generate the self-contained RDNA reasoning/code benchmark report."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "artifacts/rdna-reasoning-code/final-20260920"
SCREEN_DIR = ROOT / "artifacts/rdna-reasoning-code/screen-20260919"
COMBO_DIR = ROOT / "artifacts/rdna-reasoning-code/combo-screen-20260919"
OUTPUT = ROOT / "artifacts/research/rdna-reasoning-code-report.html"

LABELS = {
    "fork-reference": "Fork baseline",
    "rdna-reference": "RDNA reference",
    "rdna-ub256": "RDNA u-batch 256",
    "rdna-ub1024": "RDNA u-batch 1024",
    "rdna-ub2048": "RDNA u-batch 2048",
    "rdna-b1024-ub512": "RDNA batch 1024",
    "rdna-dflash-n4": "RDNA DFlash depth 4",
    "rdna-dflash-n5": "RDNA DFlash depth 5",
    "rdna-dflash-n6": "RDNA DFlash depth 6",
    "rdna-dflash-n7": "RDNA DFlash depth 7",
    "rdna-n5-ub256": "RDNA depth 5 + u-batch 256",
    "rdna-n5-ub1024": "RDNA depth 5 + u-batch 1024",
    "rdna-n5-native-off": "RDNA depth 5 + native KV off",
    "rdna-n5-draft-kv-q8": "RDNA depth 5 + draft KV Q8",
    "rdna-pmin010": "RDNA p-min 0.10",
    "rdna-kv-q4": "RDNA target KV Q4",
    "rdna-kv-bf16": "RDNA target KV BF16",
    "rdna-native-kv-off": "RDNA native KV off",
    "rdna-lazy-off": "RDNA lazy off",
    "rdna-gdn-fp32": "RDNA GDN FP32",
    "rdna-draft-kv-q8": "RDNA draft KV Q8",
    "rdna-mtp3": "RDNA MTP depth 3",
    "rdna-mtp-adaptive7": "RDNA MTP adaptive depth 7",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pct(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f}%"


def bar_chart(rows: list[dict[str, str]], key: str, baseline: float, limit: int | None = None) -> str:
    chosen = sorted(rows, key=lambda r: f(r, key), reverse=True)
    if limit:
        chosen = chosen[:limit]
    maximum = max(f(r, key) for r in chosen) * 1.08
    body = []
    for row in chosen:
        value = f(row, key)
        width = value / maximum * 100
        cls = "winner" if row["profile"] == "rdna-dflash-n5" else "baseline" if row["profile"] == "fork-reference" else ""
        delta = (value / baseline - 1) * 100
        body.append(
            f'<div class="bar-row {cls}"><div class="bar-label">{html.escape(LABELS.get(row["profile"], row["profile"]))}</div>'
            f'<div class="bar-track"><span class="bar-fill" style="--bar:{width:.3f}%"></span></div>'
            f'<div class="bar-value">{value:.2f}<small>{pct(delta)}</small></div></div>'
        )
    return "".join(body)


def scatter_svg(rows: list[dict[str, str]]) -> str:
    width, height = 920, 430
    left, right, top, bottom = 85, 30, 35, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 680.0, 775.0
    y_min, y_max = 68.0, 83.0
    def x(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w
    def y(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h
    svg = [f'<svg class="scatter" viewBox="0 0 {width} {height}" role="img" aria-label="Prefill versus decode throughput for the final profiles">']
    for tick in (680, 700, 720, 740, 760):
        tx = x(tick)
        svg.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" class="grid"/><text x="{tx:.1f}" y="{height-28}" class="axis" text-anchor="middle">{tick}</text>')
    for tick in (68, 72, 76, 80):
        ty = y(tick)
        svg.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" class="grid"/><text x="{left-16}" y="{ty+4:.1f}" class="axis" text-anchor="end">{tick}</text>')
    svg.append(f'<text x="{left + plot_w/2:.1f}" y="{height-5}" class="axis-title" text-anchor="middle">PROMPT TOKENS / SECOND →</text>')
    svg.append(f'<text x="18" y="{top + plot_h/2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90 18 {top + plot_h/2:.1f})">GENERATION TOKENS / SECOND →</text>')
    for row in rows:
        px, py = x(f(row, "prompt_tok_s")), y(f(row, "decode_tok_s"))
        cls = "point-winner" if row["profile"] == "rdna-dflash-n5" else "point-baseline" if row["profile"] == "fork-reference" else "point"
        short = {"fork-reference":"Fork", "rdna-reference":"RDNA ref", "rdna-ub256":"u256", "rdna-dflash-n4":"n4", "rdna-dflash-n5":"n5", "rdna-n5-ub256":"n5/u256", "rdna-kv-bf16":"BF16"}[row["profile"]]
        dx = -10 if row["profile"] in {"rdna-dflash-n4", "rdna-reference"} else 10
        anchor = "end" if dx < 0 else "start"
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="{cls}"/><text x="{px+dx:.1f}" y="{py-10:.1f}" class="point-label" text-anchor="{anchor}">{short}</text>')
    svg.append('</svg>')
    return "".join(svg)


def final_table(rows: list[dict[str, str]], baseline: dict[str, str]) -> str:
    out = []
    base_decode, base_prefill, base_vram = (f(baseline, k) for k in ("decode_tok_s", "prompt_tok_s", "loaded_vram_mib"))
    for row in sorted(rows, key=lambda r: f(r, "decode_tok_s"), reverse=True):
        decode = f(row, "decode_tok_s")
        prefill = f(row, "prompt_tok_s")
        vram = f(row, "loaded_vram_mib")
        klass = "best-row" if row["profile"] == "rdna-dflash-n5" else "baseline-row" if row["profile"] == "fork-reference" else ""
        out.append(f'''<tr class="{klass}">
          <th scope="row">{html.escape(LABELS[row["profile"]])}</th>
          <td>{decode:.2f}</td><td class="delta">{pct((decode/base_decode-1)*100)}</td>
          <td>{prefill:.2f}</td><td class="delta">{pct((prefill/base_prefill-1)*100)}</td>
          <td>{f(row, "acceptance")*100:.1f}%</td>
          <td>{vram/1024:.2f} GiB</td><td class="delta">{(vram-base_vram):+.0f} MiB</td>
          <td>{int(f(row, "reasoning_chars")):,} / {int(f(row, "content_chars")):,}</td>
        </tr>''')
    return "".join(out)


def screening_table(rows: list[dict[str, str]]) -> str:
    out = []
    for row in sorted(rows, key=lambda r: f(r, "decode_tok_s"), reverse=True):
        out.append(f'<tr><th scope="row">{html.escape(LABELS.get(row["profile"], row["profile"]))}</th><td>{f(row,"decode_tok_s"):.2f}</td><td>{f(row,"prompt_tok_s"):.2f}</td><td>{f(row,"acceptance")*100:.1f}%</td><td>{f(row,"loaded_vram_mib")/1024:.2f} GiB</td></tr>')
    return "".join(out)


def main() -> None:
    final_rows = read_csv(FINAL_DIR / "summary.csv")
    screen_rows = read_csv(SCREEN_DIR / "summary.csv") + read_csv(COMBO_DIR / "summary.csv")
    baseline = next(r for r in final_rows if r["profile"] == "fork-reference")
    winner = next(r for r in final_rows if r["profile"] == "rdna-dflash-n5")
    base_decode = f(baseline, "decode_tok_s")
    win_decode = f(winner, "decode_tok_s")
    gain = (win_decode / base_decode - 1) * 100
    final_chart = bar_chart(final_rows, "decode_tok_s", base_decode)
    screen_chart = bar_chart(screen_rows, "decode_tok_s", base_decode)
    scatter = scatter_svg(final_rows)
    metadata = json.loads((FINAL_DIR / "experiment.json").read_text(encoding="utf-8"))
    html_doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#07111f">
<title>RDNA Boosts × Reasoning Code Benchmark</title>
<style>
:root{{--bg:#07111f;--panel:#0b1929;--ink:#eef7ff;--muted:#91a8ba;--line:#21384d;--cyan:#48d8ff;--lime:#b8ff4d;--amber:#ffbd59;--red:#ff6577;--mono:"DejaVu Sans Mono","Liberation Mono",monospace;--serif:Georgia,"Times New Roman",serif;--sans:"DejaVu Sans",Arial,sans-serif}}
*{{box-sizing:border-box}}html{{background:var(--bg);color-scheme:dark;scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.55}}::selection{{background:var(--cyan);color:#03101b}}a{{color:var(--cyan);text-underline-offset:.2em}}a:focus-visible,button:focus-visible,summary:focus-visible{{outline:2px solid var(--lime);outline-offset:4px}}.wrap{{width:min(1180px,calc(100% - 40px));margin:auto}}header{{padding:76px 0 32px;border-bottom:1px solid var(--line);position:relative;overflow:hidden}}header:after{{content:"";position:absolute;right:-80px;bottom:0;width:520px;height:1px;background:var(--cyan);box-shadow:0 -80px 0 rgba(72,216,255,.25),0 -160px 0 rgba(72,216,255,.12);transform:rotate(-10deg);transform-origin:right}}h1{{font-family:var(--serif);font-size:clamp(2.7rem,7vw,6.9rem);line-height:.92;letter-spacing:-.055em;max-width:980px;margin:0 0 30px;font-weight:700}}.lede{{max-width:780px;color:#bed0dc;font-size:clamp(1rem,2vw,1.25rem);margin:0}}.meta{{font:700 .73rem/1.4 var(--mono);letter-spacing:.11em;text-transform:uppercase;color:var(--cyan);margin-top:28px}}.verdict{{margin:38px 0 0;border-left:4px solid var(--lime);padding:12px 0 12px 24px;max-width:880px;font-size:1.12rem}}.verdict strong{{color:var(--lime)}}.scoreboard{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}}.score{{padding:28px 24px;border-right:1px solid var(--line)}}.score:first-child{{padding-left:0}}.score:last-child{{border-right:0}}.score span{{display:block;color:var(--muted);font:700 .69rem/1.3 var(--mono);letter-spacing:.1em;text-transform:uppercase}}.score b{{display:block;font:700 clamp(1.7rem,4vw,3.15rem)/1.05 var(--mono);margin-top:9px}}.score.primary b{{color:var(--lime)}}main{{padding-bottom:80px}}section{{padding:64px 0;border-bottom:1px solid var(--line)}}h2{{font-family:var(--serif);font-size:clamp(2rem,4vw,3.3rem);letter-spacing:-.035em;line-height:1;margin:0 0 18px}}h3{{font:800 .8rem/1.3 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--cyan);margin:36px 0 14px}}p{{max-width:800px}}.section-intro{{color:var(--muted);margin:0 0 34px}}.chart{{margin-top:28px}}.bar-row{{display:grid;grid-template-columns:minmax(190px,280px) 1fr 128px;align-items:center;gap:18px;margin:13px 0;min-height:31px}}.bar-label{{font-size:.91rem}}.bar-track{{height:16px;background:#10253a;position:relative;overflow:hidden}}.bar-fill{{display:block;height:100%;width:var(--bar);background:var(--cyan);transform-origin:left;animation:reveal .8s cubic-bezier(.2,.8,.2,1) both}}.winner .bar-fill{{background:var(--lime)}}.baseline .bar-fill{{background:var(--amber)}}.bar-value{{font:700 .95rem var(--mono);text-align:right}}.bar-value small{{display:inline-block;width:58px;color:var(--muted);font-size:.7rem;margin-left:8px}}.winner .bar-value{{color:var(--lime)}}@keyframes reveal{{from{{clip-path:inset(0 100% 0 0)}}to{{clip-path:inset(0)}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important;scroll-behavior:auto!important}}}}.legend{{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font:.75rem var(--mono);margin-top:20px}}.key:before{{content:"";display:inline-block;width:10px;height:10px;background:var(--cyan);margin-right:7px}}.key.w:before{{background:var(--lime)}}.key.b:before{{background:var(--amber)}}.scatter{{display:block;width:100%;height:auto;min-width:680px}}.scatter-wrap{{overflow-x:auto;border-bottom:1px solid var(--line)}}.grid{{stroke:#183149;stroke-width:1}}.axis,.axis-title,.point-label{{fill:var(--muted);font-family:var(--mono);font-size:12px}}.axis-title{{fill:#b8cad7;font-weight:700;letter-spacing:1px}}.point-label{{fill:var(--ink);font-weight:700}}.point{{fill:var(--cyan);stroke:var(--bg);stroke-width:3}}.point-baseline{{fill:var(--amber);stroke:var(--bg);stroke-width:3}}.point-winner{{fill:var(--lime);stroke:var(--bg);stroke-width:4}}.finding-grid{{display:grid;grid-template-columns:1fr 1fr;gap:42px}}.finding{{border-top:2px solid var(--line);padding-top:16px}}.finding.good{{border-color:var(--lime)}}.finding.warn{{border-color:var(--amber)}}.finding b{{font:700 1.45rem var(--mono)}}.finding p{{color:var(--muted);margin:.4rem 0 0}}.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:940px}}th,td{{padding:13px 12px;border-bottom:1px solid var(--line);text-align:right;font:500 .82rem var(--mono);white-space:nowrap}}thead th{{color:var(--muted);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase}}tbody th{{text-align:left;color:var(--ink)}}.best-row{{background:rgba(184,255,77,.065)}}.best-row th,.best-row td:first-of-type{{color:var(--lime)}}.baseline-row{{background:rgba(255,189,89,.045)}}.delta{{color:var(--muted)}}code{{font-family:var(--mono)}}pre{{background:#030b13;border:1px solid var(--line);padding:18px 20px;overflow-x:auto;color:var(--lime);font:700 .88rem/1.6 var(--mono)}}.method{{display:grid;grid-template-columns:1.1fr .9fr;gap:54px}}.method dl{{display:grid;grid-template-columns:150px 1fr;margin:0}}.method dt,.method dd{{padding:9px 0;border-bottom:1px solid var(--line);margin:0}}.method dt{{color:var(--muted);font:.7rem var(--mono);text-transform:uppercase;letter-spacing:.08em}}.method dd{{font:.85rem var(--mono)}}details{{margin-top:28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}summary{{cursor:pointer;padding:18px 0;color:var(--cyan);font:700 .8rem var(--mono);letter-spacing:.08em;text-transform:uppercase}}.notes{{color:var(--muted);font-size:.9rem}}footer{{padding:34px 0 50px;color:var(--muted);font:.73rem/1.6 var(--mono)}}
@media(max-width:760px){{.wrap{{width:min(100% - 24px,1180px)}}header{{padding-top:48px}}.scoreboard{{grid-template-columns:1fr 1fr}}.score{{padding:20px 14px;border-bottom:1px solid var(--line)}}.score:nth-child(2){{border-right:0}}.bar-row{{grid-template-columns:130px 1fr 77px;gap:9px}}.bar-label{{font-size:.74rem}}.bar-value{{font-size:.76rem}}.bar-value small{{display:block;width:auto;margin:2px 0 0}}.finding-grid,.method{{grid-template-columns:1fr;gap:24px}}section{{padding:45px 0}}.method dl{{grid-template-columns:110px 1fr}}}}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>RDNA Boosts wins<br>when depth is five.</h1>
  <p class="lede">A controlled Qwen3.8-27B code-generation benchmark on the RX 7900 XTX, with reasoning forced on at medium effort. The stock RDNA profile ties the tuned fork; one speculative-depth change unlocks a repeatable double-digit gain.</p>
  <div class="meta">20 Sep 2026 · ByteShape IQ4_XS · 3,000 output tokens · 2 runs/profile · single GPU</div>
  <p class="verdict"><strong>Verdict:</strong> use RDNA Boosts with DFlash2 depth 5. Median generation rises from {base_decode:.2f} to {win_decode:.2f} tok/s — <strong>{gain:.2f}% faster</strong> than the current fork baseline — with nearly identical prefill and about 200 MiB less loaded VRAM.</p>
</div></header>
<div class="wrap scoreboard" aria-label="headline benchmark metrics">
  <div class="score primary"><span>Winning decode</span><b>{win_decode:.2f}</b><span>tokens / second</span></div>
  <div class="score"><span>Fork baseline</span><b>{base_decode:.2f}</b><span>tokens / second</span></div>
  <div class="score"><span>Improvement</span><b>+{gain:.2f}%</b><span>median decode</span></div>
  <div class="score"><span>Winner VRAM</span><b>{f(winner,"loaded_vram_mib")/1024:.2f}</b><span>GiB loaded</span></div>
</div>
<main class="wrap">
<section id="result"><h2>The final field</h2><p class="section-intro">Each profile ran twice from a fresh server. The second round reversed the order to reduce heat and ordering bias. Lime marks the recommended profile; amber marks the incumbent baseline.</p>
  <div class="chart">{final_chart}</div>
  <div class="legend"><span class="key w">recommended</span><span class="key b">fork baseline</span><span class="key">RDNA variants</span></div>
</section>
<section id="tradeoff"><h2>Speed without sacrificing prefill</h2><p class="section-intro">The upper-right is better. Depth 5 moves generation throughput upward while staying in the same prefill band as the reference builds. Lowering u-batch pushes profiles left and does not survive the long run as a throughput win.</p>
  <div class="scatter-wrap">{scatter}</div>
</section>
<section id="findings"><h2>What changed the result</h2>
  <div class="finding-grid">
    <article class="finding good"><b>DFlash2 depth 5</b><p>81.01 tok/s, +11.12% over the fork and +12.72% over RDNA reference. Both runs were virtually identical: 81.02 and 81.01 tok/s.</p></article>
    <article class="finding"><b>Depth 4 is the safe runner-up</b><p>78.02 tok/s, +7.01% over the fork. It offers a smaller gain with a higher 80.7% draft acceptance rate.</p></article>
    <article class="finding warn"><b>Stock RDNA is a tie, not a win</b><p>71.87 tok/s, 1.42% below the fork. It does save roughly 573 MiB, but the patches alone do not improve this workload.</p></article>
    <article class="finding warn"><b>More depth stops helping</b><p>The short screen peaked at depth 5. Depths 6 and 7 fell to 69.46 and 68.82 tok/s as acceptance dropped to 56.1% and 54.0%.</p></article>
    <article class="finding"><b>MTP loses on reasoning code</b><p>MTP depth 3 reached 62.79 tok/s; adaptive depth 7 reached 61.01 tok/s. DFlash2 remains the correct draft path for this prompt.</p></article>
    <article class="finding"><b>BF16 KV is poor value</b><p>72.24 tok/s decode, 8.09% slower prefill, and +3.34 GiB versus the fork. Q4 KV saves memory, but is a capacity profile rather than a speed profile.</p></article>
  </div>
</section>
<section id="table"><h2>Controlled comparison</h2><p class="section-intro">Median of two 3,000-token runs. “Reasoning / code” reports response characters and confirms every finalist produced both an explicit reasoning trace and final code.</p>
  <div class="table-scroll"><table><thead><tr><th>Profile</th><th>Decode</th><th>vs fork</th><th>Prefill</th><th>vs fork</th><th>Accept</th><th>VRAM</th><th>Δ VRAM</th><th>Reasoning / code chars</th></tr></thead><tbody>{final_table(final_rows, baseline)}</tbody></table></div>
</section>
<section id="screen"><h2>The wider search</h2><p class="section-intro">Short 768-token screens explored batch geometry, KV formats, GDN precision, native/lazy paths, MTP, DFlash thresholds, and speculative depths. These runs locate candidates; the long final decides the recommendation.</p>
  <div class="chart">{screen_chart}</div>
  <details><summary>Open the complete screening table</summary><div class="table-scroll"><table><thead><tr><th>Profile</th><th>Decode</th><th>Prefill</th><th>Accept</th><th>VRAM</th></tr></thead><tbody>{screening_table(screen_rows)}</tbody></table></div></details>
</section>
<section id="command"><h2>Recommended launch</h2><p class="section-intro">Keep the tuned launcher defaults and change only the validated speculative depth.</p>
<pre><code>MODEL_QUANT=byteshape REASONING=on REASONING_EFFORT=medium \
SPEC_DRAFT_N_MAX=5 ./start-llama-rdna-boosts.sh</code></pre>
<p class="notes">Do not make u-batch 256 the default from the short screen: it looked fast at 768 tokens, but the full 3,000-token test was 5.33% above the fork while losing 7.80% prefill — worse overall than depth 5 alone.</p>
</section>
<section id="method"><h2>Method and controls</h2><div class="method"><div>
  <p>One code prompt was used throughout: implement a bounded concurrent map in Python with TTL, LRU eviction, per-key request coalescing, exception fan-out, invalidation semantics, shutdown behavior, and tests. The API request fixed temperature 0 and seed 3407. Reasoning was forced on with medium effort, and each final response contained both reasoning and final code.</p>
  <p>No stale server was present before the suite. Every run launched a new process, waited for health, recorded the response and server log, then stopped the process. No OOM, AMDGPU fault/reset/timeout, HIP error, fatal error, or abort was observed.</p>
</div><dl>
  <dt>Target</dt><dd>Qwen3.8-27B ByteShape IQ4_XS</dd>
  <dt>Draft</dt><dd>Qwen3.8-27B DFlash2 Q4_K_M</dd>
  <dt>GPU</dt><dd>RX 7900 XTX · ROCm0 · gfx1100</dd>
  <dt>Context</dt><dd>131,072 · one slot</dd>
  <dt>Cache</dt><dd>Target Q8_0 · draft F16</dd>
  <dt>Batch</dt><dd>2,048 · u-batch 512</dd>
  <dt>Reasoning</dt><dd>on · medium</dd>
  <dt>Sampling</dt><dd>temperature 0 · seed 3407</dd>
  <dt>Output</dt><dd>3,000 tokens · 2 rounds/profile</dd>
  <dt>Build</dt><dd>RDNA tree 7dc63cb3c93a</dd>
</dl></div>
<h3>Interpretation limits</h3><p class="notes">This is a controlled single-prompt, single-machine benchmark, not a universal model ranking. Speculative acceptance depends on the generated token stream, and numeric differences between builds can change that stream. RDNA reference, depth 4, and depth 5 produced the exact same output hash, which makes the depth comparison especially strong. The fork produced a different but valid response, so the fork comparison should still be confirmed on a small prompt suite before changing the global default.</p>
</section>
</main>
<footer><div class="wrap">Source artifacts: <a href="../rdna-reasoning-code/final-20260920/summary.csv">final CSV</a> · <a href="../rdna-reasoning-code/final-20260920/samples.json">final JSON</a> · <a href="../rdna-reasoning-code/screen-20260919/summary.csv">screen CSV</a> · <a href="../rdna-reasoning-code/combo-screen-20260919/summary.csv">combo screen CSV</a><br>Generated from benchmark artifacts. Experiment timestamp: {html.escape(str(metadata.get("created_at", "2026-09-20")))}.</div></footer>
</body></html>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html_doc, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
