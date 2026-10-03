"""Evaluate the fuzzy-augmented LoRA cross-encoder on val_pairs.tsv.

Must use the SAME input format as train_cross_encoder_fuzzy.py — the model
saw fuzzy features in a bracket prefix during training, so eval must too.

Usage:
    python eval_cross_encoder_fuzzy.py \\
        --adapter models/cross_encoder_50k_fuzzy \\
        --val     dataset/train/val_pairs.tsv \\
        --out     dataset/train/eval_cross_50k_fuzzy.json
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.distance import JaroWinkler  # noqa: E402
import jellyfish  # noqa: E402

csv.field_size_limit(sys.maxsize)

MAX_LEN = 224

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\b\d+\b")


def squash(s: str) -> str:
    return _WS.sub("", s.lower())


def _phon_jaccard(a: str, b: str) -> int:
    ca = {jellyfish.metaphone(w) for w in a.lower().split() if w}
    cb = {jellyfish.metaphone(w) for w in b.lower().split() if w}
    ca.discard(""); cb.discard("")
    if not ca or not cb:
        return 0
    return int(round(100 * len(ca & cb) / len(ca | cb)))


def _num_jaccard(a: str, b: str) -> int:
    na = set(_NUM.findall(a or ""))
    nb = set(_NUM.findall(b or ""))
    if not na or not nb:
        return 0
    return int(round(100 * len(na & nb) / len(na | nb)))


def build_pair_text(row: dict) -> tuple[str, str]:
    na = row["source1_business_name"]; aa = row["source1_business_address"]
    nb = row["candidate_business_name"]; ab = row["candidate_business_address"]
    nm = int(round(fuzz.token_set_ratio(na or "", nb or "")))
    jw = int(round(100 * JaroWinkler.normalized_similarity(squash(na or ""), squash(nb or ""))))
    ad = int(round(fuzz.token_set_ratio(aa or "", ab or "")))
    ph = _phon_jaccard(na or "", nb or "")
    nu = _num_jaccard(aa or "", ab or "")
    prefix = f"[nm={nm} jw={jw} ad={ad} ph={ph} nu={nu}]"
    a = f"{prefix} Listing A: {na} | {aa or 'no address'}"
    b = f"Listing B: {nb} | {ab or 'no address'}"
    return a, b


def sweep_thresholds(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    P = sum(labels)
    best = (0.5, 0.0, 0.0, 0.0)
    for idx in order:
        if labels[idx] == 1: tp += 1
        else: fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / P if P else 0.0
        if precision + recall == 0: continue
        f05 = 1.25 * precision * recall / (0.25 * precision + recall)
        if f05 > best[3]:
            best = (float(scores[idx]), precision, recall, f05)
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
    truth_by_s1: dict[str, set[str]] = defaultdict(set)
    preds_by_s1: dict[str, set[str]] = defaultdict(set)
    all_s1: set[str] = set()
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
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    print(f"loading base: {args.base}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.adapter))
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=2,
        id2label={0: "different_business", 1: "same_business"},
        label2id={"different_business": 0, "same_business": 1},
    )
    print(f"loading adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval().to(device)

    print(f"loading val: {args.val}", flush=True)
    pair_records: list[dict] = []
    with args.val.open("r", encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            if args.limit is not None and i >= args.limit:
                break
            pair_records.append(r)
    print(f"  val pairs: {len(pair_records):,}", flush=True)

    scores: list[float] = []
    t0 = time.time()
    B = args.batch
    with torch.no_grad():
        for i in range(0, len(pair_records), B):
            chunk = pair_records[i:i + B]
            texts = [build_pair_text(r) for r in chunk]
            enc = tok([t[0] for t in texts], [t[1] for t in texts],
                      truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.detach().float().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
            scores.extend(float(p) for p in probs)
            if (i // B) % 20 == 0:
                done = i + len(chunk)
                speed = done / max(time.time() - t0, 1e-6)
                print(f"  scored {done:,}/{len(pair_records):,} ({speed:.1f} pair/s)", flush=True)
    dur = time.time() - t0
    print(f"scoring done in {dur/60:.1f} min ({len(pair_records)/dur:.1f} pair/s).", flush=True)

    labels = [int(r["label"]) for r in pair_records]
    metrics = {
        "pair_at_0.5": fixed_threshold(scores, labels, 0.5),
        "pair_best_threshold": sweep_thresholds(scores, labels),
    }
    for thr in (0.3, 0.5, 0.7, metrics["pair_best_threshold"]["threshold"]):
        metrics[f"source1_macro_f0_5_at_{thr:.3f}"] = source1_macro_f05(pair_records, scores, thr)

    result = {
        "adapter": str(args.adapter),
        "base": args.base,
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
    print(f"wrote {args.out}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
