"""Evaluate candidate generation (keyword stage + fuzzy/phonetic bucket) on a
held-out slice of training S1 records against the full training S2+S3 pool.

Reports, for each (threshold, K) bucket setting:
  pair_recall  - share of true (S1, S2/S3) pairs that survive into the bucket
  cands/S1     - average bucket size
  oracle_F0.5  - macro F0.5 if the matcher were perfect on the bucket
                 (the ceiling blocking imposes on the final score)
"""
import argparse
import time

import numpy as np
import polars as pl

from blocking import KeywordIndex, keyword_candidates, prepare
from fuzzy import add_side_features, fuzzy_score, pair_features, select_bucket

D = "../../../dataset/train"


def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)


def oracle_f05(n_true: pl.DataFrame, hits: pl.DataFrame) -> float:
    """Macro F0.5 with a perfect matcher: P=1, R=recovered/true, singletons score 1."""
    df = n_true.join(hits, on="s1_id", how="left").fill_null(0)
    r = np.where(df["n_true"] > 0, df["n_hit"] / df["n_true"].clip(1), 1.0)
    f = np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)
    return float(f.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--df-max", type=int, default=5000)
    ap.add_argument("--top-m", type=int, default=200)
    ap.add_argument("--phonetic-keys", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0, 40, 50, 55, 60, 65, 70])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30, 50])
    ap.add_argument("--out", default=None, help="write scored candidate pairs (parquet)")
    a = ap.parse_args()

    t = time.time()
    s1 = rd(f"{D}/train_source1.tsv").sample(fraction=a.frac, seed=42)
    oth = pl.concat([rd(f"{D}/train_source2.tsv"), rd(f"{D}/train_source3.tsv")])
    gt = (rd(f"{D}/train_ground_truth.tsv")
          .filter(pl.col("source1_entity_id").is_in(s1["entity_id"].implode()))
          .with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids")
          .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "id"}))
    n_true = gt.group_by("s1_id").agg(pl.col("id").drop_nulls().len().alias("n_true"))
    gt = gt.drop_nulls("id")
    s1p, othp = prepare(s1), prepare(oth)
    print(f"load+normalize {time.time()-t:.0f}s  s1={s1p.height} pool={othp.height} gt_pairs={gt.height}")

    t = time.time()
    idx = KeywordIndex(othp, phonetic=a.phonetic_keys)
    cand = keyword_candidates(s1p, idx, df_max=a.df_max, top_m=a.top_m)
    del idx
    print(f"keyword stage {time.time()-t:.0f}s  pairs={cand.height:,}")

    t = time.time()
    othp = othp.join(cand.select("row").unique(), on="row", how="semi")
    feats = fuzzy_score(pair_features(cand, add_side_features(s1p), add_side_features(othp)))
    print(f"fuzzy stage {time.time()-t:.0f}s")

    ids = (s1p.select(pl.col("row").alias("s1_row"), pl.col("entity_id").alias("s1_id")),
           othp.select("row", pl.col("entity_id").alias("id")))
    feats = feats.join(ids[0], on="s1_row").join(ids[1], on="row")
    feats = feats.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1_id", "id"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    if a.out:
        feats.write_parquet(a.out)

    def report(label, b):
        hits = b.filter(pl.col("y") == 1).group_by("s1_id").agg(pl.len().alias("n_hit"))
        print(f"{label:<28} pair_recall={b['y'].sum()/gt.height:.4f}  "
              f"cands/S1={b.height/s1p.height:5.1f}  oracle_F0.5={oracle_f05(n_true, hits):.4f}")

    report(f"keyword top{a.top_m}", feats)
    for th in a.thresholds:
        for k in a.ks:
            report(f"fuzzy>={th:<4} K={k}", select_bucket(feats, th, k))


if __name__ == "__main__":
    main()
