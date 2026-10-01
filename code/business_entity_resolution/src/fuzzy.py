"""Stage 2: fuzzy + phonetic scoring of keyword candidates, and bucket selection.

For every (S1, candidate) pair we compute string similarities with rapidfuzz and
phonetic-code overlap with jellyfish codes (see phonetic.py), combine them into a
single ``fuzzy_score`` in [0, 100], and keep candidates that pass a score
threshold, capped at K per S1 entity.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from phonetic import add_phonetic

SIDE_COLS = ["nm", "nm_squash", "ad", "nm_mp", "nm_ny", "ad_mp", "ad_ny", "nums"]


def add_side_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record inputs for pair features (phonetic codes, address numbers)."""
    if "nm_mp" not in df.columns:
        df = add_phonetic(df, "nm")
    df = add_phonetic(df, "ad")
    return df.with_columns(
        pl.col("ad").str.extract_all(r"\b\d+\b").list.unique().alias("nums")
    )


def _jaccard(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None)


def _pairwise(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, oth: pl.DataFrame) -> pl.DataFrame:
    """cand: (s1_row, row, kw_score). s1/oth: prepared frames with side features."""
    a = s1.select(pl.col("row").alias("s1_row"), *SIDE_COLS)
    b = oth.select("row", *[pl.col(c).alias(c + "_b") for c in SIDE_COLS])
    p = cand.join(a, on="s1_row").join(b, on="row")

    p = p.with_columns(
        pl.Series("name_tset", _pairwise(p["nm"], p["nm_b"], fuzz.token_set_ratio)),
        pl.Series("name_jw", 100 * _pairwise(p["nm_squash"], p["nm_squash_b"],
                                             JaroWinkler.normalized_similarity)),
        pl.Series("addr_tset", _pairwise(p["ad"], p["ad_b"], fuzz.token_set_ratio)),
        _jaccard("nm_mp", "nm_mp_b").alias("name_mp"),
        _jaccard("nm_ny", "nm_ny_b").alias("name_ny"),
        _jaccard("ad_mp", "ad_mp_b").alias("addr_mp"),
        _jaccard("ad_ny", "ad_ny_b").alias("addr_ny"),
        _jaccard("nums", "nums_b").alias("num_jac"),
        (pl.col("ad_b").str.len_chars() == 0).alias("b_no_addr"),
        (~pl.col("nm_b").str.contains(r"[a-z]")).alias("b_nonlatin_name"),
    )
    return p.drop([c for c in p.columns if c in SIDE_COLS or c[:-2] in SIDE_COLS])


def fuzzy_score(p: pl.DataFrame) -> pl.DataFrame:
    """Combine name and address similarity into one bucket score (0-100)."""
    name_sim = pl.max_horizontal(
        "name_tset", "name_jw",
        100 * pl.max_horizontal(pl.col("name_mp").fill_null(0), pl.col("name_ny").fill_null(0)),
    )
    addr_text = pl.max_horizontal(
        "addr_tset",
        100 * pl.max_horizontal(pl.col("addr_mp").fill_null(0), pl.col("addr_ny").fill_null(0)),
    )
    # House/plot numbers: agreement confirms, disagreement penalises; absent = neutral.
    addr_sim = pl.when(pl.col("num_jac").is_null()).then(addr_text).otherwise(
        0.7 * addr_text + 30 * pl.col("num_jac"))
    p = p.with_columns(name_sim.alias("name_sim"), addr_sim.alias("addr_sim"))
    p = p.with_columns(
        pl.when(pl.col("b_no_addr")).then(pl.col("name_sim")).otherwise(pl.col("addr_sim"))
        .alias("addr_sim"),
        pl.when(pl.col("b_nonlatin_name")).then(pl.col("addr_sim")).otherwise(pl.col("name_sim"))
        .alias("name_sim"),
    )
    return p.with_columns(((pl.col("name_sim") + pl.col("addr_sim")) / 2).alias("fuzzy_score"))


def select_bucket(p: pl.DataFrame, threshold: float, k: int) -> pl.DataFrame:
    return (
        p.filter(pl.col("fuzzy_score") >= threshold)
        .sort(["s1_row", "fuzzy_score"], descending=[False, True])
        .group_by("s1_row", maintain_order=True).head(k)
    )
