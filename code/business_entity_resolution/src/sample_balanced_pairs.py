"""Build a class-balanced (50/50 match / non-match) train subset.

Samples N/2 positive and N/2 negative rows uniformly at random from
train_labeled_pairs.tsv, excluding every source1 entity that appears in
val_pairs.tsv so the shared val split stays unseen. Unlike
sample_labeled_pairs.py this samples by row, not by source1 entity: the
natural 1:3 ratio can only be changed by dropping rows.

Rows are copied verbatim (each record is one line), in input-file order; the
Trainer shuffles at train time.

Usage:
    python sample_balanced_pairs.py \\
        --input dataset/train/train_labeled_pairs.tsv \\
        --val   dataset/train/val_pairs.tsv \\
        --out   dataset/train/train_pairs_1500k_balanced.tsv \\
        --n     1500000
"""
import argparse
from pathlib import Path

import numpy as np

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    per_class = args.n // 2

    print(f"Reading val source1 ids from {args.val.name} ...", flush=True)
    with args.val.open("rb") as f:
        f.readline()
        val_ids = {line.split(b"\t", 1)[0] for line in f}
    print(f"  {len(val_ids):,} val source1 entities excluded", flush=True)

    print(f"Pass 1: indexing labels in {args.input.name} ...", flush=True)
    labels = []  # per data row: 1 / 0, or -1 if excluded (val source1)
    with args.input.open("rb") as f:
        f.readline()  # header
        for line in f:
            if line.split(b"\t", 1)[0] in val_ids:
                labels.append(-1)
            else:
                labels.append(1 if line.rstrip(b"\n").endswith(b"\t1") else 0)
    labels = np.array(labels, dtype=np.int8)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    del labels
    print(f"  eligible: {len(pos_idx):,} positives, {len(neg_idx):,} negatives", flush=True)
    if len(pos_idx) < per_class or len(neg_idx) < per_class:
        raise SystemExit(f"not enough rows for {per_class:,} per class")

    rng = np.random.default_rng(args.seed)
    keep = np.zeros(max(pos_idx[-1], neg_idx[-1]) + 1, dtype=bool)
    keep[rng.choice(pos_idx, size=per_class, replace=False)] = True
    keep[rng.choice(neg_idx, size=per_class, replace=False)] = True

    print(f"Pass 2: writing {args.out} ...", flush=True)
    n_out = 0
    with args.input.open("rb") as f, args.out.open("wb") as out:
        out.write(f.readline())
        for i, line in enumerate(f):
            if i < len(keep) and keep[i]:
                out.write(line)
                n_out += 1
    print(f"Done: {n_out:,} rows ({per_class:,} positive + {per_class:,} negative)", flush=True)


if __name__ == "__main__":
    main()
