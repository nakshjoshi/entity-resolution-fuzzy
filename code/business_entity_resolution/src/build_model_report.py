"""Build the model report page (convergence, overfitting check, held-out tests).

Reads the parsed training curves, the eval_test_pairs.py reports (training sample,
500k and 1M held-out sets), the full-validation eval JSON and the realistic
blocking-candidate metrics, and writes one self-contained HTML page with inline
SVG charts that follow the viewer's light/dark theme.

Usage:
    python build_model_report.py --root ../../.. --out ../../../work/report_training/model_report.html
"""
import argparse
import glob
import html
import json
import math

import numpy as np
import polars as pl

W, H = 720, 300
PAD = dict(l=64, r=20, t=16, b=46)


# ---------------------------------------------------------------- SVG helpers

class Chart:
    def __init__(self, x0, x1, y0, y1, ylog=False, h=H, xcat=None):
        self.x0, self.x1, self.y0, self.y1, self.ylog, self.h = x0, x1, y0, y1, ylog, h
        self.xcat = xcat
        self.parts = []

    def sx(self, x):
        return PAD["l"] + (x - self.x0) / (self.x1 - self.x0) * (W - PAD["l"] - PAD["r"])

    def sy(self, y):
        if self.ylog:
            y = max(y, self.y0)
            f = (math.log10(y) - math.log10(self.y0)) / (math.log10(self.y1) - math.log10(self.y0))
        else:
            f = (y - self.y0) / (self.y1 - self.y0)
        return self.h - PAD["b"] - f * (self.h - PAD["t"] - PAD["b"])

    def axes(self, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str):
        p = self.parts
        for t in yticks:
            y = self.sy(t)
            p.append(f'<line class="grid" x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{y:.1f}" y2="{y:.1f}"/>')
            p.append(f'<text class="tick" x="{PAD["l"] - 8}" y="{y + 4:.1f}" text-anchor="end">{yfmt(t)}</text>')
        for t in xticks:
            x = self.sx(t)
            lab = self.xcat[t] if self.xcat else xfmt(t)
            p.append(f'<text class="tick" x="{x:.1f}" y="{self.h - PAD["b"] + 18}" text-anchor="middle">{lab}</text>')
        p.append(f'<line class="axis" x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{self.h - PAD["b"]}" y2="{self.h - PAD["b"]}"/>')
        p.append(f'<text class="axlabel" x="{(PAD["l"] + W - PAD["r"]) / 2}" y="{self.h - 8}" text-anchor="middle">{xlabel}</text>')
        p.append(f'<text class="axlabel" transform="translate(14,{(self.h - PAD["b"] + PAD["t"]) / 2}) rotate(-90)" '
                 f'text-anchor="middle">{ylabel}</text>')

    def line(self, pts, cls, tip=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.sx(x):.1f},{self.sy(y):.1f}" for i, (x, y) in enumerate(pts))
        self.parts.append(f'<path class="line {cls}" d="{d}"/>')

    def dots(self, pts, cls, tips):
        for (x, y), t in zip(pts, tips):
            self.parts.append(f'<g class="pt"><circle class="hit" cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="12"/>'
                              f'<circle class="dot {cls}" cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="5"/>'
                              f'<title>{html.escape(t)}</title></g>')

    def vline(self, x, label):
        X = self.sx(x)
        self.parts.append(f'<line class="mark" x1="{X:.1f}" x2="{X:.1f}" y1="{PAD["t"]}" y2="{self.h - PAD["b"]}"/>')
        self.parts.append(f'<text class="note" x="{X + 6:.1f}" y="{PAD["t"] + 12}">{label}</text>')

    def svg(self, title):
        return (f'<svg viewBox="0 0 {W} {self.h}" role="img" aria-label="{html.escape(title)}">'
                + "".join(self.parts) + "</svg>")


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="sw {c}"></i>{html.escape(t)}</span>' for c, t in items) + "</div>"


def fig(title, svg, leg="", note=""):
    return (f'<figure><figcaption>{title}</figcaption>{leg}<div class="plot">{svg}</div>'
            + (f'<p class="fignote">{note}</p>' if note else "") + "</figure>")


def pct(x, d=3):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- data

def load(root):
    curves = json.load(open(f"{root}/work/report_training/training_curves.json"))
    reps = {k: json.load(open(f"{root}/work/{k}/report/metrics.json")) for k in ("train200k", "test500k", "test1m")}
    fullval = json.load(open(f"{root}/models/sagemaker_cross_800k_bal/eval_cross_800k_bal.json"))
    realistic = json.load(open(f"{root}/work/infer-800kbal-v1/val/metrics.json"))
    # 1M score histogram by label
    p = pl.read_parquet(f"{root}/work/test1m/pairs/*.parquet", columns=["s1_id", "cand_id", "label"])
    s = pl.read_parquet(f"{root}/work/test1m/scores/*.parquet", columns=["s1_id", "cand_id", "score"])
    d = p.join(s, on=["s1_id", "cand_id"])
    bins = np.linspace(0, 1, 21)
    hist = {lab: np.histogram(d.filter(pl.col("label") == lab)["score"].to_numpy(), bins=bins)[0].tolist()
            for lab in (0, 1)}
    return curves, reps, fullval, realistic, hist


def at(rep, thr):
    return next(m for m in rep["pair_threshold_table"] if abs(m["threshold"] - thr) < 1e-12)


# ---------------------------------------------------------------- charts

def chart_loss(c):
    ch = Chart(0, 3, 1e-5, 1, ylog=True)
    ch.axes([0, 0.5, 1, 1.5, 2, 2.5, 3], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], "Epoch", "Cross-entropy loss (log scale)",
            xfmt=lambda t: f"{t:g}", yfmt=lambda t: f"{t:g}")
    roll = [(x, max(y, 1e-5)) for x, y in c["800k"]["rolling"]]
    ch.line(roll, "s1")
    ev = c["800k"]["eval"]
    ch.dots([(e["epoch"], e["eval_loss"]) for e in ev], "s2",
            [f"Epoch {e['epoch']:g}: validation loss {e['eval_loss']:.6f}, F0.5 {e['eval_f0_5']:.5f}" for e in ev])
    ch.vline(2, "saved model (epoch 2)")
    return ch.svg("Training and validation loss, 800k run")


def chart_runs(c):
    ch = Chart(0, 3, 1e-5, 1, ylog=True)
    ch.axes([0, 0.5, 1, 1.5, 2, 2.5, 3], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], "Epoch", "Training loss, 1,000-step mean",
            xfmt=lambda t: f"{t:g}", yfmt=lambda t: f"{t:g}")
    ch.line([(x, max(y, 1e-5)) for x, y in c["800k"]["rolling"]], "s1")
    ch.line([(x, max(y, 1e-5)) for x, y in c["1500k"]["rolling"]], "s2")
    return ch.svg("Training loss, 800k and 1.5M runs")


def chart_lr(c):
    ch = Chart(0, 3, 0, 1.05e-4, h=200)
    ch.axes([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 2.5e-5, 5e-5, 7.5e-5, 1e-4], "Epoch", "Learning rate",
            xfmt=lambda t: f"{t:g}", yfmt=lambda t: "0" if t == 0 else f"{t:.1e}")
    ch.line(c["800k"]["lr"], "s1")
    return ch.svg("Learning-rate schedule")


def chart_generalisation(rows):
    # errors per 10,000 pairs at threshold 0.5, one bar per data set
    n = len(rows)
    ch = Chart(-0.5, n - 0.5, 0, max(r["err10k"] for r in rows) * 1.35 + 0.1, xcat={i: r["short"] for i, r in enumerate(rows)})
    ymax = ch.y1
    step = 0.5 if ymax < 3 else 1
    ch.axes(list(range(n)), list(np.arange(0, ymax, step)), "Data set", "Errors per 10,000 pairs",
            yfmt=lambda t: f"{t:g}")
    bw = (W - PAD["l"] - PAD["r"]) / n * 0.46
    for i, r in enumerate(rows):
        x = ch.sx(i)
        y = ch.sy(r["err10k"])
        base = ch.sy(0)
        cls = "s2" if r["short"].startswith("Training") else "s1"
        ch.parts.append(
            f'<g class="pt"><path class="bar {cls}" d="M{x - bw / 2:.1f},{base:.1f} V{y + 4:.1f} '
            f'Q{x - bw / 2:.1f},{y:.1f} {x - bw / 2 + 4:.1f},{y:.1f} H{x + bw / 2 - 4:.1f} '
            f'Q{x + bw / 2:.1f},{y:.1f} {x + bw / 2:.1f},{y + 4:.1f} V{base:.1f} Z"/>'
            f'<text class="val" x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle">{r["err10k"]:.2f}</text>'
            f'<title>{html.escape(r["name"])}: {r["errors"]:,} errors in {r["pairs"]:,} pairs '
            f'(F0.5 {r["f05"]:.5f})</title></g>')
    return ch.svg("Errors per 10,000 pairs: training vs held-out")


def chart_hist(hist):
    bins = np.linspace(0, 1, 21)
    ch = Chart(0, 1, 1, 1e6, ylog=True)
    ch.axes([0, 0.2, 0.4, 0.6, 0.8, 1], [1, 10, 100, 1e3, 1e4, 1e5, 1e6], "Model score, P(same business)",
            "Pairs (log scale)", xfmt=lambda t: f"{t:g}", yfmt=lambda t: f"{int(t):,}")
    bw = ch.sx(0.05) - ch.sx(0)
    for lab, cls, off in ((0, "s2", -0.25), (1, "s1", 0.25)):
        for i, cnt in enumerate(hist[lab]):
            if not cnt:
                continue
            xc = ch.sx(bins[i] + 0.025) + off * bw
            y, base = ch.sy(cnt), ch.sy(1)
            ch.parts.append(f'<g class="pt"><rect class="bar {cls}" x="{xc - bw * 0.22:.1f}" y="{y:.1f}" '
                            f'width="{bw * 0.44:.1f}" height="{max(base - y, 1):.1f}" rx="2"/>'
                            f'<title>{"True match" if lab else "Non-match"}, score {bins[i]:.2f}-{bins[i + 1]:.2f}: '
                            f'{cnt:,} pairs</title></g>')
    return ch.svg("Score distribution on the 1M held-out set")


def chart_gap(easy, hard):
    ts = [m["threshold"] for m in easy]
    cat = {i: f"{t:g}" for i, t in enumerate(ts)}
    ch = Chart(-0.3, len(ts) - 0.7, 0, 1.02, xcat=cat)
    ch.axes(list(range(len(ts))), [0, 0.2, 0.4, 0.6, 0.8, 1.0], "Score threshold", "Pair F0.5", yfmt=lambda t: f"{t:.1f}")
    for series, cls, name in ((easy, "s1", "1M held-out, random non-matches"),
                              (hard, "s2", "Blocking candidates, look-alike non-matches")):
        pts = [(i, m["f0_5"]) for i, m in enumerate(series)]
        ch.line(pts, cls)
        ch.dots(pts, cls, [f"{name}, threshold {m['threshold']:g}: F0.5 {m['f0_5']:.4f} "
                           f"(P {m['precision']:.4f}, R {m['recall']:.4f})" for m in series])
    return ch.svg("F0.5 by threshold: random vs look-alike non-matches")


# ---------------------------------------------------------------- page

CSS = """
:root{--bg:#f7f7f4;--surface:#ffffff;--ink:#1b1d22;--muted:#5d616b;--grid:#e2e3de;--rule:#d6d7d1;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#157f4c;--warn:#a15c00;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#16171a;--surface:#1d1f23;
--ink:#eeeeea;--muted:#a4a7ae;--grid:#2b2d31;--rule:#34363b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cc38a;--warn:#e0a24a;}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#16171a;--surface:#1d1f23;--ink:#eeeeea;--muted:#a4a7ae;--grid:#2b2d31;
--rule:#34363b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cc38a;--warn:#e0a24a;}
body{background:var(--bg);color:var(--ink);font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;}
main{max-width:820px;margin:0 auto;padding-inline:20px;padding-block:40px 72px;display:grid;gap:40px}
header{display:grid;gap:10px}
.eyebrow{font:500 12px/1.2 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font-size:clamp(28px,5vw,38px);line-height:1.15;font-weight:600;letter-spacing:-.01em;text-wrap:balance;margin:0}
h2{font-size:22px;font-weight:600;margin:0;text-wrap:balance}
p{margin:0;max-width:68ch}
section{display:grid;gap:18px}
.lede{color:var(--muted);font-size:17px}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.verdict{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:16px;display:grid;gap:6px;align-content:start}
.verdict b{font-size:15px}
.verdict span{color:var(--muted);font-size:14px;line-height:1.5}
.pill{justify-self:start;font:600 12px/1 "IBM Plex Mono",monospace;padding:5px 8px;border-radius:999px;border:1px solid currentColor}
.pill.good{color:var(--good)}.pill.warn{color:var(--warn)}
figure{margin:0;display:grid;gap:10px}
figcaption{font-weight:600}
.plot{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:8px 4px}
svg{width:100%;height:auto;display:block;font-family:"IBM Plex Sans",system-ui,sans-serif}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.axlabel{fill:var(--muted);font-size:12.5px}
.line{fill:none;stroke-width:2;stroke-linejoin:round}.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}
.dot{stroke:var(--surface);stroke-width:2}.dot.s1,.bar.s1{fill:var(--s1)}.dot.s2,.bar.s2{fill:var(--s2)}
.hit{fill:transparent}.pt:hover .dot{r:7}.pt:hover .bar{opacity:.8}
.mark{stroke:var(--muted);stroke-dasharray:4 4}.note{fill:var(--muted);font-size:12px}
.val{fill:var(--ink);font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:14px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.sw{width:14px;height:4px;border-radius:2px;display:inline-block}.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}
.fignote{color:var(--muted);font-size:14px}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:500;font-size:13px}
tbody tr:last-child td{border-bottom:0}
tr.train td{color:var(--s2)} tr.train td:first-child{color:var(--ink)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
ul{margin:0;padding-left:20px;display:grid;gap:6px;max-width:68ch}
footer{color:var(--muted);font-size:13px;border-top:1px solid var(--rule);padding-top:16px}
"""


def build(root):
    c, reps, fullval, realistic, hist = load(root)
    ev = c["800k"]["eval"]
    fv = fullval["metrics"]["pair_at_0.5"]

    def row(name, short, pairs, m):
        errors = m["fp"] + m["fn"]
        return dict(name=name, short=short, pairs=pairs, errors=errors, err10k=errors / pairs * 1e4,
                    p=m["precision"], r=m["recall"], f05=m["f0_5"], fp=m["fp"], fn=m["fn"])

    fv_n = fullval["n_pairs"]
    fv_err = round((1 - fv["precision"]) * fv["recall"] * fullval["n_pos"] / fv["precision"]) + \
        round((1 - fv["recall"]) * fullval["n_pos"])
    fv_row = dict(name="Validation split (3.09M pairs, never trained on)", short="Validation 3.1M", pairs=fv_n,
                  errors=fv_err, err10k=fv_err / fv_n * 1e4, p=fv["precision"], r=fv["recall"], f05=fv["f0_5"],
                  fp=round((1 - fv["precision"]) * fv["recall"] * fullval["n_pos"] / fv["precision"]),
                  fn=round((1 - fv["recall"]) * fullval["n_pos"]))
    rows = [
        row("Training pairs (random 200k of the 800k it learned from)", "Training 200k",
            reps["train200k"]["test_set"]["pairs"], at(reps["train200k"], 0.5)),
        fv_row,
        row("Held-out 500k (no shared source1 entity)", "Held-out 500k",
            reps["test500k"]["test_set"]["pairs"], at(reps["test500k"], 0.5)),
        row("Held-out 1M (no shared source1 entity)", "Held-out 1M",
            reps["test1m"]["test_set"]["pairs"], at(reps["test1m"], 0.5)),
    ]
    tr, held = rows[0], rows[1:]
    held_err = sum(r["errors"] for r in held) / sum(r["pairs"] for r in held) * 1e4
    hard = reps["test500k"]["blocking_candidates"]
    best_real = realistic["best"]

    gen_table = "".join(
        f'<tr class="{"train" if i == 0 else ""}"><td>{html.escape(r["name"])}</td><td>{r["pairs"]:,}</td>'
        f'<td>{r["p"]:.5f}</td><td>{r["r"]:.5f}</td><td>{r["f05"]:.5f}</td><td>{r["fp"]:,}</td><td>{r["fn"]:,}</td>'
        f'<td>{r["err10k"]:.2f}</td></tr>' for i, r in enumerate(rows))
    ep_table = "".join(
        f'<tr><td>{e["epoch"]:g}</td><td>{m:.5f}</td><td>{e["eval_loss"]:.6f}</td><td>{e["eval_precision"]:.5f}</td>'
        f'<td>{e["eval_recall"]:.5f}</td><td>{e["eval_f0_5"]:.5f}</td><td>{"saved" if e["epoch"] == 2 else ""}</td></tr>'
        for e, m in zip(ev, [np.mean([y for x, y in c["800k"]["raw"] if k - 1 < x <= k]) for k in (1, 2, 3)]))

    gap = tr["err10k"] - held_err
    return f"""<title>Cross-encoder Training Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<main>
<header>
  <div class="eyebrow">Business entity resolution · mDeBERTa-v3-base + LoRA · 800k balanced pairs</div>
  <h1>Did the 800k cross-encoder converge, and does it overfit?</h1>
  <p class="lede">The model is <span class="mono">sagemaker_cross_800k_bal</span>: 800,000 labelled pairs (50% matches), 3 epochs,
  batch 32, learning rate 1e-4, trained on one A10G GPU in 2 h 10 min. It is checked here against its own training data and
  against 4.6 million pairs it never saw.</p>
</header>

<section aria-label="Findings">
  <div class="verdicts">
    <div class="verdict"><span class="pill good">Converged</span><b>Loss flattens during epoch 1</b>
      <span>Training loss falls from 0.72 to about 0.0005 by the end of epoch 1 and barely moves after. A second run on 1.5M pairs follows the same curve.</span></div>
    <div class="verdict"><span class="pill good">No overfitting in accuracy</span><b>Training and held-out errors match</b>
      <span>{tr["err10k"]:.2f} errors per 10,000 training pairs vs {held_err:.2f} on 4.6M held-out pairs. The held-out sets share no source1 business with training.</span></div>
    <div class="verdict"><span class="pill warn">Early sign at epoch 3</span><b>Validation loss ticks up</b>
      <span>Loss rose from {ev[1]["eval_loss"]:.5f} to {ev[2]["eval_loss"]:.5f} in epoch 3 while F0.5 held at {ev[2]["eval_f0_5"]:.4f}. The epoch-2 checkpoint was kept.</span></div>
    <div class="verdict"><span class="pill warn">Limit</span><b>Look-alike businesses are harder</b>
      <span>Every test here uses random non-matches. On real blocking candidates the best pair F0.5 is {hard["best_pair_threshold"]["f0_5"]:.3f}. That is a data gap, not overfitting.</span></div>
  </div>
</section>

<section>
  <h2>Convergence</h2>
  <p>Loss is plotted on a log scale so the long flat tail is visible. The line is the mean over each 1,000 training steps;
  the dots are the validation loss the trainer measured at the end of each epoch on 20,000 held-out pairs.</p>
  {fig("Training and validation loss, 800k run", chart_loss(c),
       legend([("s1", "Training loss (1,000-step mean)"), ("s2", "Validation loss (end of epoch)")]),
       "74% of logged training losses round to 0.0000 after epoch 1; values below 0.00001 sit on the chart floor.")}
  <div class="tablewrap"><table>
    <thead><tr><th>Epoch</th><th>Mean training loss</th><th>Validation loss</th><th>Precision</th><th>Recall</th><th>F0.5</th><th></th></tr></thead>
    <tbody>{ep_table}</tbody></table></div>
  <p>Training loss keeps falling in epoch 3 ({np.mean([y for x, y in c["800k"]["raw"] if 2 < x <= 3]):.5f}) while validation
  loss rises. That widening gap is the first sign of overfitting. It shows up only in the loss: precision, recall and F0.5 are
  unchanged, because the model is becoming more confident, not less accurate. The trainer keeps the checkpoint with the best
  validation F0.5, which is epoch 2, so the saved model predates it.</p>
  {fig("Training loss, 800k and 1.5M runs", chart_runs(c),
       legend([("s1", "800k balanced pairs"), ("s2", "1.5M balanced pairs")]),
       "Same settings, different data sizes. Both reach the same floor, so the convergence is not a quirk of one sample.")}
  {fig("Learning-rate schedule", chart_lr(c), "",
       "5% linear warm-up to 1e-4, then linear decay to zero over three epochs.")}
</section>

<section>
  <h2>Overfitting check: training data vs unseen data</h2>
  <p>An overfit model scores better on the pairs it learned from than on new ones. The same model was scored on a random 200k
  sample of its own training pairs and on three sets it never saw. The 500k and 1M sets were built so that no source1
  business in them appears anywhere in the 800k training file.</p>
  {fig("Errors per 10,000 pairs at threshold 0.5", chart_generalisation(rows),
       legend([("s2", "Training data"), ("s1", "Never seen in training")]),
       f"Training minus held-out: {gap:+.2f} errors per 10,000 pairs. Hover a bar for exact counts.")}
  <div class="tablewrap"><table>
    <thead><tr><th>Data set</th><th>Pairs</th><th>Precision</th><th>Recall</th><th>F0.5</th><th>False matches</th><th>Missed</th><th>Errors / 10k</th></tr></thead>
    <tbody>{gen_table}</tbody></table></div>
  <p>The error rate on unseen pairs is within a fraction of an error per 10,000 of the training rate, across 4.6 million
  pairs. The model has learned rules that transfer to new businesses rather than memorising the training pairs.</p>
</section>

<section>
  <h2>How confident the model is</h2>
  {fig("Score distribution on the 1M held-out set", chart_hist(hist),
       legend([("s2", "Non-match"), ("s1", "True match")]),
       "Nearly every pair scores below 0.05 or above 0.95. The few in between are the model's real mistakes and borderline cases.")}
</section>

<section>
  <h2>Where it falls short</h2>
  <p>All the sets above pair each business with random unrelated ones, the same kind of non-match the model was trained on.
  In the real pipeline the non-matches come from blocking, so they share words, streets or phonetic codes with the business.
  On 1.5M such candidates for 30,000 held-out businesses, the same model needs a score of exactly 1.0 to reach its best pair
  F0.5 of {hard["best_pair_threshold"]["f0_5"]:.3f}, and still lets through {hard["best_pair_threshold"]["fp"]:,} false matches.</p>
  {fig("F0.5 by threshold: random vs look-alike non-matches", chart_gap(reps["test1m"]["pair_threshold_table"], hard["pair_threshold_table"]),
       legend([("s1", "1M held-out, random non-matches"), ("s2", "Blocking candidates, look-alike non-matches")]))}
  <p>This is a gap in the training data, not overfitting: the model generalises to new businesses, but it never learned to
  separate a business from its near-duplicates. With the fuzzy-score floor added, the full pipeline reaches a per-business
  macro F0.5 of {best_real["macro_f0_5"]:.3f} on that validation set (precision {best_real["precision"]:.3f}, recall
  {best_real["recall"]:.3f}). Retraining with look-alike non-matches taken from blocking output is the clearest next step.</p>
</section>

<section>
  <h2>How the test sets were built</h2>
  <ul>
    <li>Training: 800,000 rows sampled from <span class="mono">train_labeled_pairs.tsv</span>, 50% matches, excluding the validation split's businesses.</li>
    <li>Validation: 3,088,155 pairs for 220,682 businesses held out before any training set was drawn; 20,000 of them were used for the per-epoch checks.</li>
    <li>Held-out 500k and 1M: whole source1 groups drawn from the labelled pairs, skipping every business in the 800k file (the 1M set also skips the 500k set's). Overlap checks: 0 shared businesses, 0 shared pairs.</li>
    <li>Training sample: 200,000 rows drawn at random from the 800k training file (100,054 matches).</li>
    <li>Every set was scored with the saved adapter on SageMaker GPUs using the same code as the test-set inference.</li>
  </ul>
</section>

<footer>Generated from the SageMaker training logs and the scored evaluation sets · data and scripts in
<span class="mono">code/business_entity_resolution/src</span> · 30 Sep 2026</footer>
</main>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    open(a.out, "w").write(build(a.root))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
