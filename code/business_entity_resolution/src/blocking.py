"""Stage 1: keyword (inverted-index) candidate generation.

Every record is turned into a bag of keywords:
  * name words        ("n|<word>")
  * squashed name     ("q|<name without spaces>")  - catches domain-style names
  * sorted squash      ("s|<name words sorted, joined>") - catches word reordering
  * address words     ("a|<word>")
  * phonetic name codes ("pm|"/"py|", jellyfish Metaphone/NYSIIS) - optional,
    enabled with ``phonetic=True``
Keywords are namespaced by country, so records only meet within one country
(ground truth never crosses countries) and an unseen country simply forms its own
namespace. An S1 record and an S2/S3 record become a keyword candidate when they
share any keyword whose document frequency is <= ``df_max``; the candidate score
is the sum of IDF over all shared keywords. The ``top_m`` best per S1 go on to
the fuzzy stage.
"""
import numpy as np
import polars as pl

from normalize import add_normalized
from phonetic import coded_words


def keyword_table(df: pl.DataFrame, phonetic: bool = False) -> pl.DataFrame:
    """Return (row, key) with one row per distinct keyword of each record."""
    def words(prefix, col, split):
        e = pl.col(col).str.split(" ") if split else pl.col(col)
        w = df.select("row", "country", e.alias("w"))
        if split:
            w = w.explode("w")
        return w.filter(pl.col("w") != "").select(
            "row",
            pl.concat_str([pl.col("country").fill_null(""), pl.lit(prefix), pl.col("w")])
            .hash(seed=17).alias("key"),
        )

    parts = [words("|n|", "nm", True), words("|q|", "nm_squash", False),
             words("|s|", "nm_sorted", False),
             words("|a|", "ad", True)]
    if phonetic:  # phonetic name keywords (jellyfish codes)
        cw = coded_words(df, "nm", extra=("country",))
        for col, pre in (("mp", "|pm|"), ("ny", "|py|")):
            parts.append(cw.select("row", pl.concat_str(
                [pl.col("country").fill_null(""), pl.lit(pre), pl.col(col)]).hash(seed=17).alias("key")))
    return pl.concat(parts).unique()


class KeywordIndex:
    """Sorted posting lists over S2/S3 records for fast per-chunk lookup."""

    def __init__(self, oth: pl.DataFrame, phonetic: bool = False, slice_rows: int = 1_000_000):
        self.phonetic = phonetic
        # Build in slices so only compact (key, row) arrays are held, not the
        # exploded string tables for all records at once.
        keys, rows = [], []
        for off in range(0, oth.height, slice_rows):
            kt = keyword_table(oth.slice(off, slice_rows), phonetic)
            keys.append(kt["key"].to_numpy())
            rows.append(kt["row"].to_numpy().astype(np.uint32))
            del kt
        key, row = np.concatenate(keys), np.concatenate(rows)
        del keys, rows
        order = np.argsort(key, kind="stable")
        self.key, self.row = key[order], row[order]
        del key, row, order
        uk, start, cnt = np.unique(self.key, return_index=True, return_counts=True)
        self.ukey, self.ustart, self.ucnt = uk, start, cnt
        self.n_docs = oth.height

    def lookup(self, keys: np.ndarray):
        """For each query key return (start, count) in the posting arrays."""
        pos = np.searchsorted(self.ukey, keys)
        pos = np.minimum(pos, len(self.ukey) - 1)
        hit = self.ukey[pos] == keys
        start = np.where(hit, self.ustart[pos], 0)
        cnt = np.where(hit, self.ucnt[pos], 0)
        return start, cnt


def keyword_candidates(s1: pl.DataFrame, index: KeywordIndex, df_max: int,
                       top_m: int, chunk: int = 5000) -> pl.DataFrame:
    """Return (s1_row, row, kw_score) with at most ``top_m`` candidates per S1."""
    qt = keyword_table(s1, index.phonetic)
    start, cnt = index.lookup(qt["key"].to_numpy())
    idf = np.log((index.n_docs + 1) / (cnt + 1)).astype(np.float32)
    keep = (cnt > 0) & (cnt <= df_max)
    q = pl.DataFrame({
        "s1_row": qt["row"].to_numpy()[keep], "start": start[keep],
        "cnt": cnt[keep], "idf": idf[keep],
    })
    parts = []
    s1_rows = s1["row"].to_numpy()
    for i in range(0, len(s1_rows), chunk):
        lo, hi = s1_rows[i], s1_rows[min(i + chunk, len(s1_rows)) - 1]
        c = q.filter(pl.col("s1_row").is_between(lo, hi))
        if c.height == 0:
            continue
        cn = c["cnt"].to_numpy()
        tot = int(cn.sum())
        # Expand every (query keyword -> posting range) into individual postings.
        offs = np.repeat(c["start"].to_numpy() - np.cumsum(cn) + cn, cn) + np.arange(tot)
        pairs = pl.DataFrame({
            "s1_row": np.repeat(c["s1_row"].to_numpy(), cn),
            "row": index.row[offs],
            "idf": np.repeat(c["idf"].to_numpy(), cn),
        })
        agg = (
            pairs.group_by("s1_row", "row").agg(pl.col("idf").sum().alias("kw_score"))
            .sort(["s1_row", "kw_score"], descending=[False, True])
            .group_by("s1_row", maintain_order=True).head(top_m)
        )
        parts.append(agg)
    return pl.concat(parts)


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    return add_normalized(df).with_row_index("row")
