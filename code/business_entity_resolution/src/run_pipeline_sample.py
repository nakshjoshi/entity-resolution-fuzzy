"""End-to-end check on a sample of training S1 entities:

  1. keyword blocking over the FULL training S2+S3 pool,
  2. fuzzy + jellyfish scoring -> candidate bucket per S1 (threshold + cap K),
  3. GLiNER2.5-Decide chooses matches from each bucket (bucket mode and pair mode),
  4. score against the full ground-truth lists with the competition metric.
"""
import gliner_decide  # first: forces HF offline mode before other HF imports

import argparse
import time
from pathlib import Path

import polars as pl

from blocking import KeywordIndex, keyword_candidates, prepare
from fuzzy import add_side_features, fuzzy_score, pair_features, select_bucket
from metrics import summarize

ROOT = Path(__file__).resolve().parents[3]


def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)


def build_buckets(s1: pl.DataFrame, oth: pl.DataFrame, a) -> pl.DataFrame:
    s1p, othp = prepare(s1), prepare(oth)
    idx = KeywordIndex(othp)
    cand = keyword_candidates(s1p, idx, df_max=a.df_max, top_m=a.top_m)
    del idx
    othp = othp.join(cand.select("row").unique(), on="row", how="semi")
    scored = fuzzy_score(pair_features(cand, add_side_features(s1p), add_side_features(othp)))
    bucket = select_bucket(scored, a.fuzzy_threshold, a.k)
    return (bucket.select("s1_row", "row", "fuzzy_score")
            .join(s1p.select(pl.col("row").alias("s1_row"), pl.col("entity_id").alias("s1_id"),
                             pl.col("business_name").alias("name_a"),
                             pl.col("business_address").alias("addr_a")), on="s1_row")
            .join(othp.select("row", pl.col("entity_id").alias("id"),
                              pl.col("business_name").alias("name_b"),
                              pl.col("business_address").alias("addr_b")), on="row")
            .drop("s1_row", "row").sort(["s1_id", "fuzzy_score"], descending=[False, True]))


def to_sets(df: pl.DataFrame, s1_ids, keep: pl.Expr) -> dict:
    out = {i: set() for i in s1_ids}
    for s1, i in df.filter(keep).select("s1_id", "id").iter_rows():
        out[s1].add(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--df-max", type=int, default=5000)
    ap.add_argument("--top-m", type=int, default=200)
    ap.add_argument("--fuzzy-threshold", type=float, default=65)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "GLiNER2.5-Decide"))
    ap.add_argument("--data-dir", default=str(ROOT / "dataset" / "train"))
    ap.add_argument("--save", default=None, help="write the scored bucket pairs to this TSV")
    a = ap.parse_args()
    D = Path(a.data_dir)

    t = time.time()
    s1 = rd(D / "train_source1.tsv").sample(a.n, seed=a.seed)
    oth = pl.concat([rd(D / "train_source2.tsv"), rd(D / "train_source3.tsv")])
    gt = (rd(D / "train_ground_truth.tsv")
          .filter(pl.col("source1_entity_id").is_in(s1["entity_id"].implode())))
    truth = {r[0]: set(r[1].split(",")) if r[1] else set() for r in gt.iter_rows()}
    s1_ids = list(truth)

    bucket = build_buckets(s1, oth, a)
    del oth
    bucket = bucket.with_columns(
        pl.struct("s1_id", "id").map_elements(lambda r: r["id"] in truth[r["s1_id"]],
                                              return_dtype=pl.Boolean).alias("is_true"))
    n_true = sum(len(v) for v in truth.values())
    print(f"[1-2] candidates built in {time.time()-t:.0f}s: {bucket.height} pairs for "
          f"{a.n} S1 ({bucket.height/a.n:.1f}/S1); true pairs in buckets "
          f"{bucket['is_true'].sum()}/{n_true}; singletons in sample "
          f"{sum(not v for v in truth.values())}")

    judge = gliner_decide.DecideJudge(a.model_dir)
    t = time.time()
    bucket_scores = []
    for s1_id, g in bucket.group_by("s1_id", maintain_order=True):
        rows = g.rows(named=True)
        sc = judge.score_bucket((rows[0]["name_a"], rows[0]["addr_a"]),
                                [(r["name_b"], r["addr_b"]) for r in rows])
        bucket_scores += sc
    t_bucket = time.time() - t
    t = time.time()
    pair_scores = judge.score_pairs(bucket.select("name_a", "addr_a", "name_b", "addr_b").rows())
    t_pair = time.time() - t
    bucket = bucket.with_columns(pl.Series("gliner_bucket", bucket_scores),
                                 pl.Series("gliner_pair", pair_scores))
    print(f"[3] GLiNER on {a.device if hasattr(a, 'device') else judge.device}: bucket mode "
          f"{t_bucket:.0f}s ({a.n/t_bucket:.2f} S1/s), pair mode {t_pair:.0f}s "
          f"({bucket.height/t_pair:.1f} pairs/s)")

    print("\n[4] scored against the full ground truth (macro F0.5 over all sampled S1)")
    rows = [("predict nothing (all singletons)", summarize({}, truth)),
            ("perfect pick from buckets (ceiling)", summarize(to_sets(bucket, s1_ids, pl.col("is_true")), truth))]
    for c in a.cutoffs:
        rows.append((f"GLiNER bucket mode  score>={c}",
                     summarize(to_sets(bucket, s1_ids, pl.col("gliner_bucket") >= c), truth)))
    for c in a.cutoffs:
        rows.append((f"GLiNER pair mode    score>={c}",
                     summarize(to_sets(bucket, s1_ids, pl.col("gliner_pair") >= c), truth)))
    for c in [70, 80, 90]:
        rows.append((f"fuzzy score only    score>={c}",
                     summarize(to_sets(bucket, s1_ids, pl.col("fuzzy_score") >= c), truth)))
    for name, s in rows:
        print(f"  {name:<38} F0.5={s['macro_F0.5']:.3f}  singletons_right={s['singletons_right']:<6} "
              f"P={s['precision']:.3f}  R={s['recall']:.3f}  links={s['predicted_links']}")
    print("\n  mean GLiNER score  true pairs: bucket={:.3f} pair={:.3f} | non-matches: bucket={:.3f} pair={:.3f}".format(
        bucket.filter(pl.col("is_true"))["gliner_bucket"].mean(), bucket.filter(pl.col("is_true"))["gliner_pair"].mean(),
        bucket.filter(~pl.col("is_true"))["gliner_bucket"].mean(), bucket.filter(~pl.col("is_true"))["gliner_pair"].mean()))
    if a.save:
        bucket.write_csv(a.save, separator="\t")


if __name__ == "__main__":
    main()
