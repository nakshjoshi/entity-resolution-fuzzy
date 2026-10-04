"""Baseline / fine-tuned GLiNER-Decide evaluation on a labeled pair val set.

Reads val_pairs.tsv and asks the DecideJudge for a P(same_business) score per
pair. Emits threshold-swept metrics and a per-source1 macro F0.5 (the metric
the leaderboard actually scores on, README.md:190).

Usage:
    python eval_decide.py \\
        --model models/GLiNER2.5-multi-Decide \\
        --val   dataset/train/val_pairs.tsv \\
        --out   dataset/train/eval_base.json

"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import gliner_decide  # sets HF offline env vars before torch/transformers imports
from gliner_decide import DecideJudge

csv.field_size_limit(sys.maxsize)


def stream_pairs(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            yield row


def sweep_thresholds(scores, labels):
    """Return best F0.5 threshold and its P/R/F0.5 on the given scores/labels."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    P = sum(labels)
    best = (0.5, 0.0, 0.0, 0.0)
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / P if P else 0.0
        if precision + recall == 0:
            continue
        f05 = 1.25 * precision * recall / (0.25 * precision + recall)
        if f05 > best[3]:
            best = (scores[idx], precision, recall, f05)
    return {"threshold": best[0], "precision": best[1], "recall": best[2], "f0_5": best[3]}


def fixed_threshold(scores, labels, thr):
    tp = fp = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= thr else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 1: fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return {"threshold": thr, "precision": 0.0, "recall": 0.0, "f0_5": 0.0}
    f05 = 1.25 * precision * recall / (0.25 * precision + recall)
    return {"threshold": thr, "precision": precision, "recall": recall, "f0_5": f05}


def source1_macro_f05(pair_records, scores, thr):
    """Group predicted matches by source1 and average F0.5 across source1 entities."""
    truth_by_s1: dict[str, set[str]] = defaultdict(set)
    all_s1: set[str] = set()
    preds_by_s1: dict[str, set[str]] = defaultdict(set)
    for rec, s in zip(pair_records, scores):
        s1 = rec["source1_entity_id"]
        cid = rec["candidate_entity_id"]
        all_s1.add(s1)
        if int(rec["label"]) == 1:
            truth_by_s1[s1].add(cid)
        if s >= thr:
            preds_by_s1[s1].add(cid)
    f05s = []
    for s1 in all_s1:
        pred = preds_by_s1.get(s1, set())
        true = truth_by_s1.get(s1, set())
        if not true and not pred:
            f05s.append(1.0); continue
        if not pred:
            f05s.append(0.0); continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        if precision + recall == 0:
            f05s.append(0.0); continue
        f05s.append(1.25 * precision * recall / (0.25 * precision + recall))
    return sum(f05s) / len(f05s) if f05s else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: only score the first N pairs.")
    args = ap.parse_args()

    print(f"Loading model from {args.model} ...", flush=True)
    judge = DecideJudge(str(args.model))
    print(f"  device: {judge.device}", flush=True)

    pair_records = list(stream_pairs(args.val, args.limit))
    print(f"Val pairs: {len(pair_records):,}", flush=True)

    # Score in batches.
    t0 = time.time()
    scores: list[float] = []
    B = args.batch
    for i in range(0, len(pair_records), B):
        chunk = pair_records[i:i + B]
        tuples = [(r["source1_business_name"], r["source1_business_address"],
                   r["candidate_business_name"], r["candidate_business_address"])
                  for r in chunk]
        scores.extend(judge.score_pairs(tuples, batch_size=B))
        if (i // B) % 20 == 0:
            done = i + len(chunk)
            speed = done / max(time.time() - t0, 1e-6)
            print(f"  scored {done:,}/{len(pair_records):,} ({speed:.1f} pair/s)", flush=True)
    dur = time.time() - t0
    print(f"Scoring done in {dur/60:.1f} min ({len(pair_records)/dur:.1f} pair/s).", flush=True)

    labels = [int(r["label"]) for r in pair_records]
    metrics: dict[str, dict] = {
        "pair_at_0.5": fixed_threshold(scores, labels, 0.5),
        "pair_best_threshold": sweep_thresholds(scores, labels),
    }
    for thr in (0.3, 0.5, 0.7, metrics["pair_best_threshold"]["threshold"]):
        metrics[f"source1_macro_f0_5_at_{thr:.3f}"] = source1_macro_f05(pair_records, scores, thr)

    result = {
        "model": str(args.model),
        "val": str(args.val),
        "n_pairs": len(pair_records),
        "n_pos": sum(labels),
        "n_neg": len(labels) - sum(labels),
        "n_source1_in_val": len({r["source1_entity_id"] for r in pair_records}),
        "seconds_total": dur,
        "pairs_per_second": len(pair_records) / dur if dur else 0.0,
        "metrics": metrics,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
