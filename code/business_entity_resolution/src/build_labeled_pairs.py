"""Build a labeled (match / false-match) training set of source1 x source2/3 pairs.

Positives come from train_ground_truth.tsv. Negatives are randomly sampled
source2/source3 entities that are NOT listed as matches for the given source1
entity. Ratio: 3 random negatives per positive. Source-1 singletons (no
ground-truth matches) still contribute 3 negative rows so the model sees them.

Output: one row per (source1, candidate) pair with a `label` column
(1 = match, 0 = false match).
"""

import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO_ROOT / "dataset" / "train"

SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
SOURCE2_FILE = TRAIN_DIR / "train_source2.tsv"
SOURCE3_FILE = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"
OUTPUT_FILE = TRAIN_DIR / "train_labeled_pairs.tsv"

NEG_PER_POS = 3
NEG_FOR_SINGLETON = 3
SEED = 42

OUTPUT_COLUMNS = [
    "source1_entity_id",
    "source1_business_name",
    "source1_business_address",
    "source1_country",
    "candidate_entity_id",
    "candidate_business_name",
    "candidate_business_address",
    "candidate_country",
    "candidate_source",
    "label",
]

csv.field_size_limit(sys.maxsize)


def load_source(path: Path) -> dict[str, tuple[str, str, str]]:
    """entity_id -> (business_name, business_address, country)."""
    out: dict[str, tuple[str, str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["entity_id"]] = (
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", ""),
            )
    return out


def main() -> None:
    rng = random.Random(SEED)

    print(f"Loading {SOURCE1_FILE.name} ...", flush=True)
    src1 = load_source(SOURCE1_FILE)
    print(f"  source1: {len(src1):,} entities", flush=True)

    print(f"Loading {SOURCE2_FILE.name} ...", flush=True)
    src2 = load_source(SOURCE2_FILE)
    print(f"  source2: {len(src2):,} entities", flush=True)

    print(f"Loading {SOURCE3_FILE.name} ...", flush=True)
    src3 = load_source(SOURCE3_FILE)
    print(f"  source3: {len(src3):,} entities", flush=True)

    src2_ids = list(src2.keys())
    src3_ids = list(src3.keys())
    n_src2 = len(src2_ids)
    n_src3 = len(src3_ids)
    total_candidate_pool = n_src2 + n_src3

    def sample_negative(positives_set: set[str]) -> tuple[str, str] | None:
        """Return (entity_id, source_tag) not in positives_set. source_tag is 'S2' or 'S3'."""
        for _ in range(20):
            idx = rng.randrange(total_candidate_pool)
            if idx < n_src2:
                cand_id = src2_ids[idx]
                tag = "S2"
            else:
                cand_id = src3_ids[idx - n_src2]
                tag = "S3"
            if cand_id not in positives_set:
                return cand_id, tag
        return None

    n_pos = 0
    n_neg = 0
    n_missing_positive = 0
    n_singletons = 0
    n_src1_missing = 0

    print(f"Writing {OUTPUT_FILE} ...", flush=True)
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8", newline="") as gt_f, \
            OUTPUT_FILE.open("w", encoding="utf-8", newline="") as out_f:
        gt_reader = csv.DictReader(gt_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)

        for i, gt_row in enumerate(gt_reader, start=1):
            s1_id = gt_row["source1_entity_id"]
            s1_fields = src1.get(s1_id)
            if s1_fields is None:
                n_src1_missing += 1
                continue
            s1_name, s1_addr, s1_country = s1_fields

            raw_matches = gt_row.get("matched_entity_ids", "") or ""
            matched_ids = [m for m in raw_matches.split(",") if m]
            positives_set = set(matched_ids)

            # Positives
            for m_id in matched_ids:
                if m_id.startswith("S2-"):
                    cand = src2.get(m_id)
                    tag = "S2"
                elif m_id.startswith("S3-"):
                    cand = src3.get(m_id)
                    tag = "S3"
                else:
                    cand = None
                    tag = ""
                if cand is None:
                    n_missing_positive += 1
                    continue
                writer.writerow([
                    s1_id, s1_name, s1_addr, s1_country,
                    m_id, cand[0], cand[1], cand[2],
                    tag, 1,
                ])
                n_pos += 1

            # Negatives
            if matched_ids:
                n_neg_target = NEG_PER_POS * len(matched_ids)
            else:
                n_singletons += 1
                n_neg_target = NEG_FOR_SINGLETON

            for _ in range(n_neg_target):
                sampled = sample_negative(positives_set)
                if sampled is None:
                    continue
                cand_id, tag = sampled
                if tag == "S2":
                    cand = src2[cand_id]
                else:
                    cand = src3[cand_id]
                writer.writerow([
                    s1_id, s1_name, s1_addr, s1_country,
                    cand_id, cand[0], cand[1], cand[2],
                    tag, 0,
                ])
                n_neg += 1

            if i % 100_000 == 0:
                print(f"  processed {i:,} source1 rows; positives={n_pos:,} negatives={n_neg:,}", flush=True)

    print("Done.", flush=True)
    print(f"  positives written: {n_pos:,}", flush=True)
    print(f"  negatives written: {n_neg:,}", flush=True)
    print(f"  singletons: {n_singletons:,}", flush=True)
    print(f"  ground-truth rows with missing source1 record: {n_src1_missing:,}", flush=True)
    print(f"  positive matched_ids not found in source2/3: {n_missing_positive:,}", flush=True)


if __name__ == "__main__":
    main()
