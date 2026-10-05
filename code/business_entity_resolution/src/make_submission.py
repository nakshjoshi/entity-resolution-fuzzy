"""Stage 4: turn model scores into the two submission TSVs (and metrics when labelled).

Inputs (from lenient_blocking.py / score_pairs.py):
  <work>/s1_ids/*.parquet   every S1 id processed (singletons included)
  <work>/scores/*.parquet   s1_id, cand_id, fuzzy_score, fuzzy_rank, score

Writes:
  candidate_pairs.tsv   every candidate the model scored, per S1
  matching_results.tsv  candidates with score >= --threshold and fuzzy_score >= --min-fuzzy

Every S1 id gets exactly one row in both files (empty list when nothing). All of
an S1's candidates live in one shard, so the files are written shard by shard in
bounded memory.

With --gt (ground truth in the train format) it first reports, on the processed
S1 ids, blocking recall / oracle F0.5 at several top-k cut-offs and the macro
F0.5 / precision / recall of the final matches over a (threshold, fuzzy floor)
grid, written to --metrics as JSON. --threshold auto then uses the best cell.

Usage:
    python make_submission.py --work work/val --out work/val/out --gt train_ground_truth.tsv \\
        --threshold auto --metrics work/val/metrics.json
    python make_submission.py --work work/test --out output --threshold 0.9999 --min-fuzzy 70 \\
        --s1-file dataset/test/test_source1.tsv
"""
import argparse
import glob
import json
import os

import numpy as np
import polars as pl

THRESHOLDS = (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0)  # scores are float32: 1.0 = saturated
FUZZY_FLOORS = (0, 60, 70, 80, 82, 84, 85, 86, 87, 88, 90)


def s1_ids(work: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{work}/s1_ids/*.parquet"))]) \
        .unique(maintain_order=True)


def score_files(work: str) -> list[str]:
    files = sorted(glob.glob(f"{work}/scores/*.parquet"))
    pairs = glob.glob(f"{work}/pairs/*.parquet")
    if pairs and len(pairs) != len(files):
        raise SystemExit(f"{len(files)} score shards for {len(pairs)} pair shards: scoring incomplete")
    return files


def keep(p: pl.DataFrame, thr: float, min_fuzzy: float) -> pl.DataFrame:
    return p.filter((pl.col("score") >= thr) & (pl.col("fuzzy_score") >= min_fuzzy))


def f05_table(s1: pl.DataFrame, pred: pl.DataFrame, gt: pl.DataFrame) -> dict:
    """Macro F0.5 over all S1 (singletons included), plus micro P/R."""
    n_true = gt.group_by("s1_id").agg(pl.len().alias("n_true"))
    n_pred = pred.group_by("s1_id").agg(pl.len().alias("n_pred"))
    tp = pred.join(gt, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").agg(pl.len().alias("tp"))
    d = (s1.join(n_true, on="s1_id", how="left").join(n_pred, on="s1_id", how="left")
         .join(tp, on="s1_id", how="left").fill_null(0))
    t, p, h = d["n_true"].to_numpy(), d["n_pred"].to_numpy(), d["tp"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        prec, rec = h / np.maximum(p, 1), h / np.maximum(t, 1)
        f = np.where(h > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where(t == 0, (p == 0).astype(float), f)
    single = t == 0
    return {
        "macro_f0_5": float(f.mean()),
        "macro_f0_5_non_singletons": float(f[~single].mean()) if (~single).any() else None,
        "singleton_accuracy": float(f[single].mean()) if single.any() else None,
        "precision": float(h.sum() / max(p.sum(), 1)),
        "recall": float(h.sum() / max(t.sum(), 1)),
        "predicted_links": int(p.sum()),
    }


def evaluate(s1: pl.DataFrame, pairs: pl.DataFrame, gt_path: str) -> dict:
    gt = (pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
          .rename({"source1_entity_id": "s1_id"})
          .join(s1, on="s1_id", how="semi")
          .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
          .explode("matched_entity_ids").rename({"matched_entity_ids": "cand_id"})
          .filter(pl.col("cand_id") != ""))
    lab = pairs.join(gt.with_columns(pl.lit(True).alias("y")), on=["s1_id", "cand_id"], how="left") \
               .with_columns(pl.col("y").fill_null(False))
    blocking = {"n_s1": s1.height, "true_pairs": gt.height}
    for k in (5, 10, 20, 30, 40, None):
        b = lab if k is None else lab.filter(pl.col("fuzzy_rank") <= k)
        blocking[f"top{k or 'all'}"] = {
            "pair_recall": float(b["y"].sum() / max(gt.height, 1)),
            "cands_per_s1": b.height / s1.height,
            "oracle_macro_f0_5": f05_table(s1, b.filter(pl.col("y")), gt)["macro_f0_5"],
        }
    sweep = {f"score>={t},fuzzy>={mf}": {"threshold": t, "min_fuzzy": mf, **f05_table(s1, keep(pairs, t, mf), gt)}
             for t in THRESHOLDS for mf in FUZZY_FLOORS}
    best = max(sweep.values(), key=lambda v: v["macro_f0_5"])
    return {"blocking": blocking, "sweep": sweep, "best": best}


def write_outputs(s1: pl.DataFrame, files: list[str], thr: float, min_fuzzy: float, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    seen = []
    n_cand = n_match = 0
    with open(f"{out}/candidate_pairs.tsv", "w") as fc, open(f"{out}/matching_results.tsv", "w") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for f in files:
            p = pl.read_parquet(f).unique(["s1_id", "cand_id"]).sort(["s1_id", "score"], descending=[False, True])
            ids = p.select("s1_id").unique(maintain_order=True)
            cand = p.group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").str.join(","))
            m = keep(p, thr, min_fuzzy)
            match = ids.join(m.group_by("s1_id", maintain_order=True).agg(pl.col("cand_id").str.join(",")),
                             on="s1_id", how="left").with_columns(pl.col("cand_id").fill_null(""))
            fc.write(cand.write_csv(separator="\t", include_header=False, quote_style="never"))
            fm.write(match.write_csv(separator="\t", include_header=False, quote_style="never"))
            seen.append(ids)
            n_cand += p.height
            n_match += m.height
        # S1 records that blocking found no candidates for: empty rows in both files.
        rest = s1.join(pl.concat(seen), on="s1_id", how="anti") if seen else s1
        empty = rest.with_columns(pl.lit("").alias("x")).write_csv(
            separator="\t", include_header=False, quote_style="never")
        fc.write(empty)
        fm.write(empty)
    n_rows = sum(1 for _ in open(f"{out}/matching_results.tsv")) - 1
    if n_rows != s1.height:
        raise SystemExit(f"matching_results.tsv has {n_rows} rows for {s1.height} S1 ids")
    return {"threshold": thr, "min_fuzzy": min_fuzzy, "n_s1": s1.height, "candidate_pairs": n_cand,
            "matched_pairs": n_match, "s1_with_no_candidates": rest.height}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", default="0.5", help='float, or "auto" (best on --gt)')
    ap.add_argument("--min-fuzzy", type=float, default=0.0)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--s1-file", default=None,
                    help="source1 TSV: emit exactly one row per entity_id in it (recommended for test)")
    a = ap.parse_args()

    s1, files = s1_ids(a.work), score_files(a.work)
    metrics = {}
    thr = None if a.threshold == "auto" else float(a.threshold)
    if a.gt:
        pairs = pl.concat([pl.read_parquet(f) for f in files])
        metrics = evaluate(s1, pairs, a.gt)
        if thr is None:
            thr, a.min_fuzzy = metrics["best"]["threshold"], metrics["best"]["min_fuzzy"]
        print(json.dumps({"blocking": metrics["blocking"], "best": metrics["best"]}, indent=2))
    elif thr is None:
        raise SystemExit("--threshold auto needs --gt")

    if a.s1_file:
        # The submission must list every S1 entity exactly once, whatever the blocking covered.
        full = (pl.read_csv(a.s1_file, separator="\t", quote_char=None, infer_schema=False,
                            columns=["entity_id"])
                .select(pl.col("entity_id").alias("s1_id")).unique(maintain_order=True))
        extra = s1.join(full, on="s1_id", how="anti").height
        if extra:
            raise SystemExit(f"{extra} processed S1 ids are not in {a.s1_file}")
        s1 = full
    metrics["output"] = write_outputs(s1, files, thr, a.min_fuzzy, a.out)
    print(json.dumps(metrics["output"], indent=2))
    if a.metrics:
        with open(a.metrics, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
