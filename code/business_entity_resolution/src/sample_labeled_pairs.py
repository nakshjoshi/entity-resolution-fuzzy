"""Build stratified 100k / 500k train subsets + a shared val split.

Sampling is by source1 entity (not by row), so all pairs belonging to one
source1 land in the same split — no leakage across train/val. The 1:3
positive:negative ratio from the full file is preserved automatically because
we take all rows for every sampled source1.

Val is 10% of source1 ids, held out before either train subset is sampled and
identical for both, so the 100k and 500k runs are directly comparable.

Usage:
    python sample_labeled_pairs.py \\
        --input  dataset/train/train_labeled_pairs.tsv \\
        --outdir dataset/train
"""
import argparse
import csv
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SEED = 42
VAL_FRAC = 0.10
TRAIN_SIZES = {"100k": 100_000, "500k": 500_000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    print(f"Pass 1: collecting source1 ids and row counts from {args.input.name} ...", flush=True)
    src1_rows: dict[str, int] = {}
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            src1_rows[s1] = src1_rows.get(s1, 0) + 1
    print(f"  {len(src1_rows):,} distinct source1 entities", flush=True)

    rng = random.Random(SEED)
    all_ids = list(src1_rows.keys())
    rng.shuffle(all_ids)
    n_val = int(len(all_ids) * VAL_FRAC)
    val_ids = set(all_ids[:n_val])
    train_pool = all_ids[n_val:]
    print(f"  val: {len(val_ids):,} source1 ids", flush=True)

    picks: dict[str, set[str]] = {}
    for tag, target_rows in TRAIN_SIZES.items():
        chosen: set[str] = set()
        total = 0
        for s1 in train_pool:
            if total >= target_rows:
                break
            chosen.add(s1)
            total += src1_rows[s1]
        picks[tag] = chosen
        print(f"  train_{tag}: {len(chosen):,} source1 ids -> {total:,} rows", flush=True)

    outs = {
        "val": args.outdir / "val_pairs.tsv",
        "100k": args.outdir / "train_pairs_100k.tsv",
        "500k": args.outdir / "train_pairs_500k.tsv",
    }
    print("Pass 2: writing splits ...", flush=True)
    with args.input.open("r", encoding="utf-8", newline="") as f, \
            outs["val"].open("w", encoding="utf-8", newline="") as fv, \
            outs["100k"].open("w", encoding="utf-8", newline="") as f100, \
            outs["500k"].open("w", encoding="utf-8", newline="") as f500:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for w in (fv, f100, f500):
            csv.writer(w, delimiter="\t", lineterminator="\n").writerow(header)
        wv = csv.writer(fv, delimiter="\t", lineterminator="\n")
        w100 = csv.writer(f100, delimiter="\t", lineterminator="\n")
        w500 = csv.writer(f500, delimiter="\t", lineterminator="\n")
        counts = {"val": 0, "100k": 0, "500k": 0}
        for row in reader:
            s1 = row[0]
            if s1 in val_ids:
                wv.writerow(row); counts["val"] += 1
            if s1 in picks["100k"]:
                w100.writerow(row); counts["100k"] += 1
            if s1 in picks["500k"]:
                w500.writerow(row); counts["500k"] += 1
    for tag, n in counts.items():
        print(f"  {outs[tag].name}: {n:,} rows", flush=True)


if __name__ == "__main__":
    main()
