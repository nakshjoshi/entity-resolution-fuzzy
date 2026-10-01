"""Phonetic word codes (jellyfish Metaphone + NYSIIS) for names and addresses.

Codes are computed once per distinct word, then attached to each record as a
list of unique codes, so pairwise phonetic overlap is a cheap set operation.
Only Latin alphabetic words are encoded; digits and non-Latin scripts get no
code (numbers are compared separately, and phonetic rules do not apply to Indic
scripts).
"""
import jellyfish
import polars as pl

CODERS = {"mp": jellyfish.metaphone, "ny": jellyfish.nysiis}


def coded_words(df: pl.DataFrame, col: str, extra=()) -> pl.DataFrame:
    """One row per (record, Latin word) with that word's phonetic codes."""
    words = (
        df.select("row", *extra, pl.col(col).str.split(" ").alias("w")).explode("w")
        .filter(pl.col("w").str.contains(r"^[a-z]{2,}$"))
    )
    vocab = words.select("w").unique()["w"].to_list()
    new = [w for w in vocab if w not in _CACHE]
    for w in new:
        _CACHE[w] = tuple(f(w) for f in CODERS.values())
    codes = [_CACHE[w] for w in vocab]
    vocab = pl.DataFrame(
        {"w": vocab, **{k: [c[i] for c in codes] for i, k in enumerate(CODERS)}},
        schema={"w": pl.String, **{k: pl.String for k in CODERS}},
    )
    return words.join(vocab, on="w")


# word -> (metaphone, nysiis); shared across calls so sliced index builds and
# repeated runs encode each distinct word once.
_CACHE: dict = {}


def add_phonetic(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Add ``<col>_mp`` and ``<col>_ny`` list[str] columns (unique codes per row)."""
    coded = (
        coded_words(df, col)
        .group_by("row")
        .agg([pl.col(k).unique().alias(f"{col}_{k}") for k in CODERS])
    )
    out = df.join(coded, on="row", how="left")
    return out.with_columns(
        [pl.col(f"{col}_{k}").fill_null(pl.lit([], dtype=pl.List(pl.String))) for k in CODERS]
    )
