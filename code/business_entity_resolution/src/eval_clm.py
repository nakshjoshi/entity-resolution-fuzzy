"""Benchmark Contrastive-LM/CLM-v0.1-8B on the fuzzy candidate pairs, no fine-tuning.

CLM is a "questions head" over frozen Qwen3-8B embeddings. It expects a running
Qwen3 embedding server (vLLM) and a running `clm-serve` process — the notebook
in `make_clm_colab.py` starts both.

Per pair we ask a `Score` question ("are these the same business?") via
`CLMClient.system_one` and combine the returned criterion probabilities into a
scalar in [0, 1]. Same downstream metrics as eval_cross_encoder.py so results
are directly comparable.

Usage (once vllm + clm-serve are up):
    python eval_clm.py \\
        --val   /content/drive/MyDrive/amazon-entity-resolution/val_pairs.tsv \\
        --out   /content/drive/MyDrive/amazon-entity-resolution/eval_clm_8b.json \\
        --limit 5000
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

csv.field_size_limit(sys.maxsize)

CRITERIA = ["Definitely different", "Possibly same", "Definitely same"]
# Weighted collapse of the 3 criterion probabilities into a single [0, 1] score.
CRITERION_WEIGHTS = [0.0, 0.5, 1.0]
INSTRUCTIONS = (
    "Decide whether the two business records below refer to the same physical "
    "business entity. Consider name variants, abbreviations, translations, and "
    "address formatting differences."
)


def side_text(name: str, addr: str) -> str:
    return f"{name} | {addr or 'no address'}"


def pair_state(row: dict) -> str:
    a = side_text(row["source1_business_name"], row["source1_business_address"])
    b = side_text(row["candidate_business_name"], row["candidate_business_address"])
    return f"Record A: {a}\nRecord B: {b}"


def sweep_thresholds(scores, labels):
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


def collapse(probs) -> float:
    """Weighted mean of criterion probabilities → [0, 1]."""
    return float(sum(p * w for p, w in zip(probs, CRITERION_WEIGHTS)))


def score_pair(client, question_ctor, row: dict) -> float:
    """One system_one call. Returns 0.5 (neutral) on transient failure."""
    q = question_ctor(instructions=INSTRUCTIONS, criteria=CRITERIA)
    for attempt in range(2):
        try:
            r = client.system_one(state=pair_state(row), questions={"match": q})
            return collapse(r.answers["match"].probabilities)
        except Exception as e:
            if attempt == 0:
                time.sleep(1.0); continue
            print(f"  warn: pair failed twice, defaulting to 0.5 ({e})", flush=True)
            return 0.5
    return 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clm-url", default="http://127.0.0.1:8700",
                    help="Where clm-serve is listening.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap pairs (recommended for free-tier Colab: 5000).")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    from clm import CLMClient, Score  # noqa: E402

    # CLMClient may accept a base URL as positional/kwarg depending on version;
    # try the common forms and fall back to default localhost binding.
    client = None
    for kw in ({"base_url": args.clm_url}, {"url": args.clm_url}, {}):
        try:
            client = CLMClient(**kw); break
        except TypeError:
            continue
    if client is None:
        client = CLMClient()

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
    for i, rec in enumerate(pair_records):
        scores.append(score_pair(client, Score, rec))
        if (i + 1) % args.log_every == 0:
            speed = (i + 1) / max(time.time() - t0, 1e-6)
            eta = (len(pair_records) - i - 1) / max(speed, 1e-6)
            print(f"  scored {i+1:,}/{len(pair_records):,} "
                  f"({speed:.2f} pair/s, ETA {eta/60:.1f} min)", flush=True)
    dur = time.time() - t0
    print(f"scoring done in {dur/60:.1f} min ({len(pair_records)/dur:.2f} pair/s).", flush=True)

    labels = [int(r["label"]) for r in pair_records]
    metrics = {
        "pair_at_0.5": fixed_threshold(scores, labels, 0.5),
        "pair_best_threshold": sweep_thresholds(scores, labels),
    }
    for thr in (0.3, 0.5, 0.7, metrics["pair_best_threshold"]["threshold"]):
        metrics[f"source1_macro_f0_5_at_{thr:.3f}"] = source1_macro_f05(pair_records, scores, thr)

    result = {
        "model": "Contrastive-LM/CLM-v0.1-8B",
        "api": "system_one/Score",
        "criteria": CRITERIA,
        "weights": CRITERION_WEIGHTS,
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
