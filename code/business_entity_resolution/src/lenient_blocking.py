"""Stages 1-2 for inference: lenient, recall-first candidate generation.

The cross-encoder is the final judge, so this stage only has to shrink the
search space without dropping true pairs. It is deliberately loose:

  1. keyword stage (blocking.py): an S2/S3 record is a keyword candidate of an
     S1 record when they share ANY name word, squashed/sorted name, phonetic
     name code (Metaphone/NYSIIS) or address word within the same country
     (keywords are country-namespaced, so unseen countries like France work
     unchanged). The ``top_m`` best by summed IDF go on.
  2. fuzzy stage (fuzzy.py): rapidfuzz token-set / Jaro-Winkler on names,
     token-set on addresses, phonetic overlap and house-number agreement give
     one ``fuzzy_score``. There is NO score threshold: every S1 keeps its ``k``
     best keyword candidates, plus any further candidate whose name or address
     alone is a near-exact fuzzy match (``strong``), up to ``k_max``.

Outputs under --out:
  pairs/<country>_<nnnn>.parquet  s1_id, cand_id, fuzzy_score, fuzzy_rank, name_sim,
                                  addr_sim, s1_name, s1_addr, cand_name, cand_addr
                                  (exactly what score_pairs.py feeds the model)
  s1_ids/<country>.parquet        every S1 id processed (singletons included)

Usage:
    python lenient_blocking.py --s1 test_source1.tsv --s2 test_source2.tsv \\
        --s3 test_source3.tsv --out work/test [--countries India US]
"""
import argparse
import os
import time

import polars as pl

from blocking import KeywordIndex, keyword_candidates, prepare
from fuzzy import add_side_features, fuzzy_score, pair_features


def rd(p: str) -> pl.DataFrame:
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)


def select_lenient(p: pl.DataFrame, k: int, k_max: int, strong: float) -> pl.DataFrame:
    """Top-k by fuzzy score, plus near-exact name/address hits up to k_max."""
    p = p.with_columns(
        pl.col("fuzzy_score").rank("ordinal", descending=True).over("s1_row").alias("fuzzy_rank"))
    keep = (pl.col("fuzzy_rank") <= k) | (
        ((pl.col("name_sim") >= strong) | (pl.col("addr_sim") >= strong))
        & (pl.col("fuzzy_rank") <= k_max))
    return p.filter(keep)


def block_country(s1: pl.DataFrame, oth: pl.DataFrame, a, write) -> None:
    t = time.time()
    s1p, othp = prepare(s1), prepare(oth)
    idx = KeywordIndex(othp, phonetic=True)
    oth_side = add_side_features(othp)
    oth_txt = othp.select("row", pl.col("entity_id").alias("cand_id"),
                          pl.col("business_name").alias("cand_name"),
                          pl.col("business_address").alias("cand_addr"))
    print(f"  index + side features {time.time()-t:.0f}s (pool={othp.height:,})", flush=True)

    for off in range(0, s1p.height, a.chunk):
        t = time.time()
        c = s1p.slice(off, a.chunk)
        cand = keyword_candidates(c, idx, df_max=a.df_max, top_m=a.top_m)
        if cand.height == 0:
            continue
        side_b = oth_side.join(cand.select("row").unique(), on="row", how="semi")
        feats = fuzzy_score(pair_features(cand, add_side_features(c), side_b))
        sel = select_lenient(feats, a.k, a.k_max, a.strong)
        out = (sel.select("s1_row", "row", "fuzzy_score", "fuzzy_rank", "name_sim", "addr_sim")
               .join(c.select(pl.col("row").alias("s1_row"), pl.col("entity_id").alias("s1_id"),
                              pl.col("business_name").alias("s1_name"),
                              pl.col("business_address").alias("s1_addr")), on="s1_row")
               .join(oth_txt, on="row")
               .select("s1_id", "cand_id", pl.col("fuzzy_score").cast(pl.Float32),
                       pl.col("fuzzy_rank").cast(pl.UInt16), pl.col("name_sim").cast(pl.Float32),
                       pl.col("addr_sim").cast(pl.Float32), "s1_name", "s1_addr", "cand_name", "cand_addr")
               .sort("s1_id", "fuzzy_rank"))
        write(out)
        print(f"  S1 {off+c.height:,}/{s1p.height:,}: kw={cand.height:,} kept={out.height:,} "
              f"({out.height/c.height:.1f}/S1) {time.time()-t:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", nargs="*", default=None, help="default: every country in S1")
    ap.add_argument("--s1-ids", default=None, help="optional file (one id per line) to restrict S1")
    ap.add_argument("--df-max", type=int, default=5000, help="ignore keywords shared by more records")
    ap.add_argument("--top-m", type=int, default=300, help="keyword candidates per S1")
    ap.add_argument("--k", type=int, default=40, help="fuzzy top-k kept per S1 (no threshold)")
    ap.add_argument("--k-max", type=int, default=80, help="cap incl. near-exact extras")
    ap.add_argument("--strong", type=float, default=90, help="name_sim/addr_sim for extras")
    ap.add_argument("--chunk", type=int, default=20000, help="S1 records per batch")
    ap.add_argument("--shard-pairs", type=int, default=1_000_000)
    ap.add_argument("--part", type=int, default=0, help="process S1 part i of --parts (by id hash)")
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(f"{a.out}/pairs", exist_ok=True)
    os.makedirs(f"{a.out}/s1_ids", exist_ok=True)
    s1 = rd(a.s1).with_columns(pl.col("country").fill_null(""))
    if a.s1_ids:
        ids = pl.read_csv(a.s1_ids, has_header=False, new_columns=["entity_id"])
        s1 = s1.join(ids, on="entity_id", how="semi")
    if a.parts > 1:
        s1 = s1.filter(pl.col("entity_id").hash(seed=3) % a.parts == a.part)
    oth = pl.concat([rd(a.s2), rd(a.s3)]).with_columns(pl.col("country").fill_null(""))
    countries = a.countries or sorted(s1["country"].unique().to_list())
    print(f"S1={s1.height:,} pool={oth.height:,} countries={countries}", flush=True)

    for country in countries:
        t0 = time.time()
        cs1 = s1.filter(pl.col("country") == country)
        coth = oth.filter(pl.col("country") == country)
        tag = (country.replace(" ", "_") or "NA") + (f"_p{a.part}" if a.parts > 1 else "")
        cs1.select(pl.col("entity_id").alias("s1_id")).write_parquet(f"{a.out}/s1_ids/{tag}.parquet.tmp")
        os.replace(f"{a.out}/s1_ids/{tag}.parquet.tmp", f"{a.out}/s1_ids/{tag}.parquet")
        print(f"[{country}] S1={cs1.height:,} pool={coth.height:,}", flush=True)

        buf, n_buf, shard = [], 0, 0

        def flush():
            nonlocal buf, n_buf, shard
            if buf:
                path = f"{a.out}/pairs/{tag}_{shard:04d}.parquet"
                pl.concat(buf).write_parquet(path + ".tmp", compression="zstd")
                os.replace(path + ".tmp", path)  # uploader only ever sees complete shards
                shard += 1
            buf, n_buf = [], 0

        def write(df):
            nonlocal n_buf
            buf.append(df)
            n_buf += df.height
            if n_buf >= a.shard_pairs:
                flush()

        block_country(cs1, coth, a, write)
        flush()
        print(f"[{country}] done in {(time.time()-t0)/60:.1f} min, {shard} shards", flush=True)


if __name__ == "__main__":
    main()
