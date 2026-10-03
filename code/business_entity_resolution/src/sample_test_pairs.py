"""Build a held-out pair test set that shares no source1 entity with a training file.

Samples whole source1 groups from train_labeled_pairs.tsv (so the natural 1:3
match / non-match ratio and every S1's full pair list are kept), skipping every
source1 entity that appears anywhere in --exclude (e.g. the 800k training set),
until --n rows are collected.

Writes the pairs as TSV (same columns as the input) and as parquet shards in the
score_pairs.py input format (s1_id, cand_id, s1_name, s1_addr, cand_name,
cand_addr) plus label / country, so the shards can be scored by the SageMaker
score job and joined back to their labels.

Usage:
    python sample_test_pairs.py --input dataset/train/train_labeled_pairs.tsv \\
        --exclude dataset/train/train_pairs_800k_balanced.tsv \\
        --out dataset/train/test_pairs_500k.tsv --shards work/test500k/pairs --n 500000
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(sys.maxsize)
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--exclude", required=True, type=Path, nargs="+",
                    help="pair TSV(s) whose source1 entities must not appear in the test set")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--shard-rows", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prefix", default="test500k", help="shard file name prefix")
    args = ap.parse_args()

    excluded: set[bytes] = set()
    for p in args.exclude:
        with p.open("rb") as f:
            f.readline()
            excluded.update(line.split(b"\t", 1)[0] for line in f)
    print(f"excluded source1 entities: {len(excluded):,}", flush=True)

    print(f"pass 1: counting rows per source1 in {args.input.name} ...", flush=True)
    counts: Counter = Counter()
    with args.input.open("rb") as f:
        f.readline()
        for line in f:
            s1 = line.split(b"\t", 1)[0]
            if s1 not in excluded:
                counts[s1] += 1
    eligible = sorted(counts)
    random.Random(args.seed).shuffle(eligible)
    chosen, total = set(), 0
    for s1 in eligible:
        if total >= args.n:
            break
        chosen.add(s1)
        total += counts[s1]
    print(f"eligible source1: {len(eligible):,}; chosen {len(chosen):,} -> {total:,} rows", flush=True)

    print("pass 2: writing TSV + parquet shards ...", flush=True)
    args.shards.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("s1_id", "cand_id", "s1_name", "s1_addr", "cand_name", "cand_addr",
                            "s1_country", "cand_country", "label")}
    shard = n_out = n_pos = 0

    def flush():
        nonlocal shard
        if cols["s1_id"]:
            t = pa.table({k: pa.array(v, type=pa.int8() if k == "label" else pa.string())
                          for k, v in cols.items()})
            pq.write_table(t, args.shards / f"{args.prefix}_{shard:04d}.parquet", compression="zstd")
            shard += 1
            for v in cols.values():
                v.clear()

    with args.input.open("r", encoding="utf-8", newline="") as f, \
            args.out.open("w", encoding="utf-8", newline="") as out:
        out.write(f.readline())
        for line in f:
            if line.split("\t", 1)[0].encode() not in chosen:
                continue
            out.write(line)
            r = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            for k, v in zip(("s1_id", "s1_name", "s1_addr", "s1_country", "cand_id",
                             "cand_name", "cand_addr", "cand_country"), r[:8]):
                cols[k].append(v)
            cols["label"].append(int(r[9]))
            n_out += 1
            n_pos += int(r[9])
            if len(cols["s1_id"]) >= args.shard_rows:
                flush()
    flush()
    print(f"done: {n_out:,} rows ({n_pos:,} matches, {n_out-n_pos:,} non-matches), "
          f"{len(chosen):,} source1, {shard} shards -> {args.shards}", flush=True)


if __name__ == "__main__":
    main()
