"""Text normalization shared by blocking and fuzzy matching.

All functions operate on polars expressions so they run vectorised over millions
of rows. Nothing here is country-gated: unknown countries pass through the same
generic rules.
"""
import polars as pl

# Latin combining diacritics only (U+0300-U+036F). Stripping all \p{Mn} would
# destroy Indic vowel signs, which are combining marks too.
LATIN_DIACRITICS = r"[\x{0300}-\x{036F}]"

# Placeholder junk seen in addresses/names.
JUNK = r"\b(n/a|null|<null>|none|nan)\b"

# Legal forms, honorifics and connector words. Removed from the "core" name used
# for the squashed-name key and for fuzzy name similarity.
LEGAL = [
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "cos", "pllc", "plc", "llp", "lp", "pc", "public",
    "the", "and", "of", "formerly", "dba",
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "cie", "ets", "etablissements",
    "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "m s",
]
LEGAL_RE = r"\b(" + "|".join(LEGAL) + r")\b"

# Address abbreviation canonicalisation (applied token-wise after lowercasing).
ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "hwy": "highway", "blvd": "boulevard", "bd": "boulevard", "ct": "court",
    "pl": "place", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
    "sq": "square", "mt": "mount", "ft": "fort", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "r": "rue", "imp": "impasse", "rte": "route", "chem": "chemin",
    "q": "quai", "nagar": "nagar", "opp": "opposite", "nr": "near",
    "hno": "house", "h": "house",
}


def base_clean(e: pl.Expr) -> pl.Expr:
    """Lowercase, strip Latin accents, drop junk placeholders, keep letters/marks/digits."""
    return (
        e.fill_null("")
        .str.to_lowercase()
        .str.normalize("NFKD")
        .str.replace_all(LATIN_DIACRITICS, "")
        .str.replace_all(JUNK, " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def name_core(e: pl.Expr) -> pl.Expr:
    """Name with legal forms / honorifics removed and the .com/.in suffix dropped."""
    # Digit-for-letter typos inside words: "Ai0n" -> "aion", "c0m" -> "com".
    e = e.str.replace_all(r"(\p{L})0", "${1}o").str.replace_all(r"0(\p{L})", "o${1}")
    e = e.str.replace_all(r"(?i)\.(com|in|net|org|co|fr)\b", " ")
    e = e.str.replace_all(r"\b(\p{L})\.", "${1}")  # P.L.L.C. -> PLLC
    return (
        base_clean(e)
        .str.replace_all(LEGAL_RE, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def addr_clean(e: pl.Expr) -> pl.Expr:
    """Address with abbreviations canonicalised token-wise."""
    toks = base_clean(e).str.split(" ")
    return toks.list.eval(pl.element().replace(ADDR_ABBR)).list.join(" ")


def add_normalized(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        name_core(pl.col("business_name")).alias("nm"),
        addr_clean(pl.col("business_address")).alias("ad"),
    ).with_columns(
        pl.col("nm").str.replace_all(" ", "").alias("nm_squash"),
        # Order-insensitive squash: catches word transpositions ("Welfare Society Skand").
        pl.col("nm").str.split(" ").list.sort().list.join("").alias("nm_sorted"),
    )
