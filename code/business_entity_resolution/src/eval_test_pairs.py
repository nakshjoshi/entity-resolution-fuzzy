"""Score report + charts for the cross-encoder on a labelled held-out pair set.

Inputs:
  --pairs   parquet shards from sample_test_pairs.py (s1_id, cand_id, label, s1_country)
  --scores  score shards from score_pairs.py (s1_id, cand_id, score)
  --blocking-work (optional) a scored blocking run (<work>/scores, <work>/s1_ids) plus
            --gt, to compare with realistic, blocking-generated candidates

Writes <out>/metrics.json and <out>/*.png.

Usage:
    python eval_test_pairs.py --pairs work/test500k/pairs --scores work/test500k/scores \\
        --blocking-work work/infer-800kbal-v1/val --gt dataset/train/train_ground_truth.tsv \\
        --out work/test500k/report
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from make_submission import f05_table  # noqa: E402

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MATCH, NONMATCH, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def pair_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return {"threshold": thr, "precision": p, "recall": r, "f0_5": f,
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def curves(y: np.ndarray, s: np.ndarray):
    """ROC + PR curves and their areas (ties handled by grouping equal scores)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]  # end of each tie group
    tp, fp = tp[last], fp[last]
    P, N = tp[-1], fp[-1]
    tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / N]
    prec, rec = tp / (tp + fp), tp / P
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))  # trapezoid rule; numpy 1.x and 2.x
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    return fpr, tpr, rec, prec, roc_auc, ap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--blocking-work", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="500k", help="test-set label used in chart titles")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pairs = pl.read_parquet(f"{a.pairs}/*.parquet", columns=["s1_id", "cand_id", "label", "s1_country"])
    scores = pl.read_parquet(f"{a.scores}/*.parquet", columns=["s1_id", "cand_id", "score"])
    d = pairs.join(scores, on=["s1_id", "cand_id"], how="inner").unique(["s1_id", "cand_id"])
    if d.height < pairs.unique(["s1_id", "cand_id"]).height:
        raise SystemExit(f"only {d.height} of {pairs.height} pairs have scores")
    y, s = d["label"].to_numpy().astype(np.int64), d["score"].to_numpy().astype(np.float64)

    fpr, tpr, rec, prec, roc_auc, ap_ = curves(y, s)
    table = [pair_metrics(y, s, t) for t in THRESHOLDS]
    best = max(table, key=lambda m: m["f0_5"])
    s1 = d.select("s1_id").unique()
    gt = d.filter(pl.col("label") == 1).select("s1_id", "cand_id")
    macro = {str(t): f05_table(s1, d.filter(pl.col("score") >= t).select("s1_id", "cand_id"), gt)
             for t in (0.5, best["threshold"])}
    by_country = {}
    for (c,), g in d.group_by("s1_country"):
        yy, sc = g["label"].to_numpy(), g["score"].to_numpy()
        by_country[c] = {"pairs": g.height, **pair_metrics(yy, sc, 0.5)}

    metrics = {
        "test_set": {"pairs": d.height, "matches": int(y.sum()), "non_matches": int(len(y) - y.sum()),
                     "source1_entities": s1.height},
        "roc_auc": roc_auc, "average_precision": ap_,
        "pair_threshold_table": table, "best_pair_threshold": best,
        "source1_macro_f0_5": macro, "by_country_at_0.5": by_country,
    }

    # Realistic comparison: blocking-generated candidates (look-alike non-matches).
    blk = None
    if a.blocking_work and a.gt:
        bs1 = pl.read_parquet(f"{a.blocking_work}/s1_ids/*.parquet").unique()
        b = pl.read_parquet(f"{a.blocking_work}/scores/*.parquet", columns=["s1_id", "cand_id", "score"])
        bgt = (pl.read_csv(a.gt, separator="\t", quote_char=None, infer_schema=False)
               .rename({"source1_entity_id": "s1_id"}).join(bs1, on="s1_id", how="semi")
               .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
               .explode("matched_entity_ids").rename({"matched_entity_ids": "cand_id"})
               .filter(pl.col("cand_id") != ""))
        b = b.join(bgt.with_columns(pl.lit(1).alias("label")), on=["s1_id", "cand_id"], how="left") \
             .with_columns(pl.col("label").fill_null(0))
        by, bsc = b["label"].to_numpy().astype(np.int64), b["score"].to_numpy().astype(np.float64)
        bfpr, btpr, brec, bprec, broc, bap = curves(by, bsc)
        btable = [pair_metrics(by, bsc, t) for t in THRESHOLDS]
        metrics["blocking_candidates"] = {
            "pairs": b.height, "matches": int(by.sum()), "roc_auc": broc, "average_precision": bap,
            "pair_threshold_table": btable,
            "best_pair_threshold": max(btable, key=lambda m: m["f0_5"]),
        }
        blk = (by, bsc, bfpr, btpr, brec, bprec, broc, bap, btable)

    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    # 1. Score distribution by true label (log counts: the classes are 1:3 and very peaked).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(s[y == 0], bins=bins, color=NONMATCH, alpha=0.85, label="Non-match", edgecolor=SURFACE, linewidth=0.5)
    ax.hist(s[y == 1], bins=bins, color=MATCH, alpha=0.85, label="True match", edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    style(ax, f"Model score by true label ({a.name} held-out test pairs)", "Model score P(same business)", "Pairs (log scale)")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper center", ncol=2)
    fig.tight_layout(); fig.savefig(f"{a.out}/1_score_distribution.png", dpi=150); plt.close(fig)

    # 2. Precision / recall / F0.5 across thresholds (one 0-1 axis).
    ts = np.array(THRESHOLDS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ts))
    for key, col, lab in (("precision", MATCH, "Precision"), ("recall", NONMATCH, "Recall"), ("f0_5", THIRD, "F0.5")):
        vals = [m[key] for m in table]
        ax.plot(x, vals, color=col, linewidth=2, marker="o", markersize=5, label=lab)
    ax.set_xticks(x, [str(t) for t in ts], rotation=35)
    ax.set_ylim(min(0.9, min(m["recall"] for m in table) - 0.02), 1.005)
    style(ax, f"Pair metrics by score threshold ({a.name} test)", "Score threshold", "Value")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{a.out}/2_threshold_metrics.png", dpi=150); plt.close(fig)

    # 3. Precision-recall: random-negative test vs blocking candidates.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color=MATCH, linewidth=2, label=f"{a.name} test, random non-matches (AP {ap_:.4f})")
    if blk:
        ax.plot(blk[4], blk[5], color=NONMATCH, linewidth=2,
                label=f"Blocking candidates, look-alike non-matches (AP {blk[7]:.4f})")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
    style(ax, "Precision-recall curve", "Recall", "Precision")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/3_precision_recall.png", dpi=150); plt.close(fig)

    # 4. Confusion matrix at 0.5 (counts + share of row).
    m05 = pair_metrics(y, s, 0.5)
    cm = np.array([[m05["tn"], m05["fp"]], [m05["fn"], m05["tp"]]])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eaf2fc", MATCH]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.2%}", ha="center", va="center",
                    color="#ffffff" if share[i, j] > 0.5 else INK, fontsize=11)
    ax.set_xticks([0, 1], ["Predicted non-match", "Predicted match"])
    ax.set_yticks([0, 1], ["True non-match", "True match"])
    ax.set_title("Confusion matrix, threshold 0.5", loc="left", color=INK, fontsize=12,
                 fontweight="bold", pad=10)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.out}/4_confusion_matrix.png", dpi=150); plt.close(fig)

    # 5. The gap: F0.5 by threshold, random non-matches vs look-alike candidates.
    if blk:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, [m["f0_5"] for m in table], color=MATCH, linewidth=2, marker="o", markersize=5,
                label=f"{a.name} test (random non-matches)")
        ax.plot(x, [m["f0_5"] for m in blk[8]], color=NONMATCH, linewidth=2, marker="o", markersize=5,
                label="Blocking candidates (look-alike non-matches)")
        ax.set_xticks(x, [str(t) for t in ts], rotation=35)
        ax.set_ylim(0, 1.02)
        style(ax, "Pair F0.5 by threshold: easy vs realistic non-matches", "Score threshold", "Pair F0.5")
        ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
        fig.tight_layout(); fig.savefig(f"{a.out}/5_easy_vs_realistic.png", dpi=150); plt.close(fig)

    print(json.dumps({k: metrics[k] for k in ("test_set", "roc_auc", "average_precision",
                                               "best_pair_threshold", "source1_macro_f0_5")}, indent=2))
    if "blocking_candidates" in metrics:
        bc = metrics["blocking_candidates"]
        print("blocking candidates:", {k: bc[k] for k in ("pairs", "matches", "roc_auc", "average_precision")},
              "best:", bc["best_pair_threshold"])


if __name__ == "__main__":
    main()
