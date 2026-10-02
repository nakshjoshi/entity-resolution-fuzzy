"""Low-memory variant of build_labeled_pairs.py (same output, byte for byte).

The original loads every source record into Python dicts (~5-6 GB RAM for the
~12.5M train records), which does not fit on small instances like
ml.t3.medium (4 GB). This version keeps only a numeric id + byte offset per
record in numpy arrays (~300 MB) and reads record fields from mmap'd files on
demand.

It relies on two properties of the train sources (checked when this was
written): each record is exactly one line, and entity ids are unique. The
random sampling makes the exact same rng calls in the same order as the
original, so with the same SEED the output file is identical.

Usage:
    python build_labeled_pairs_lowmem.py [--output PATH]
"""

import argparse
import csv
import mmap
import random
import sys
from pathlib import Path

import numpy as np

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


class SourceIndex:
    """File-order record offsets plus a sorted numeric-id index over one source TSV."""

    def __init__(self, path: Path, prefix: str) -> None:
        self.prefix = prefix
        self._fh = path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

        header = self._mm.readline().decode("utf-8").rstrip("\n").split("\t")
        self._col = {name: i for i, name in enumerate(header)}

        ids: list[int] = []
        offsets: list[int] = []
        pos = self._mm.tell()
        plen = len(prefix)
        for line in iter(self._mm.readline, b""):
            ids.append(int(line[plen:line.index(b"\t")]))
            offsets.append(pos)
            pos += len(line)

        # File order: position i here == position i in the original's list(dict.keys()).
        self.offsets = np.array(offsets, dtype=np.int64)
        id_arr = np.array(ids, dtype=np.int64)
        del ids, offsets
        self._order = np.argsort(id_arr, kind="stable")
        self._sorted_ids = id_arr[self._order]

    def __len__(self) -> int:
        return len(self.offsets)

    def _row(self, offset: int) -> list[str]:
        end = self._mm.find(b"\n", offset)
        line = self._mm[offset:end if end != -1 else len(self._mm)].decode("utf-8")
        if '"' in line:
            return next(csv.reader([line], delimiter="\t"))
        return line.split("\t")

    def _fields(self, row: list[str]) -> tuple[str, str, str]:
        def get(name: str) -> str:
            i = self._col.get(name)
            return row[i] if i is not None and i < len(row) else ""
        return get("business_name"), get("business_address"), get("country")

    def at(self, idx: int) -> tuple[str, tuple[str, str, str]]:
        """(entity_id, fields) of the idx-th record in file order."""
        row = self._row(int(self.offsets[idx]))
        return row[0], self._fields(row)

    def get(self, entity_id: str) -> tuple[str, str, str] | None:
        """Fields for entity_id, or None (mirrors dict.get on the original)."""
        if not entity_id.startswith(self.prefix):
            return None
        try:
            key = int(entity_id[len(self.prefix):])
        except ValueError:
            return None
        k = int(np.searchsorted(self._sorted_ids, key))
        if k >= len(self._sorted_ids) or self._sorted_ids[k] != key:
            return None
        row = self._row(int(self.offsets[self._order[k]]))
        # Numeric match is not enough if ids differ in formatting (e.g. leading zeros).
        return self._fields(row) if row[0] == entity_id else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = ap.parse_args()

    rng = random.Random(SEED)

    print(f"Indexing {SOURCE1_FILE.name} ...", flush=True)
    src1 = SourceIndex(SOURCE1_FILE, "S1-")
    print(f"  source1: {len(src1):,} entities", flush=True)

    print(f"Indexing {SOURCE2_FILE.name} ...", flush=True)
    src2 = SourceIndex(SOURCE2_FILE, "S2-")
    print(f"  source2: {len(src2):,} entities", flush=True)

    print(f"Indexing {SOURCE3_FILE.name} ...", flush=True)
    src3 = SourceIndex(SOURCE3_FILE, "S3-")
    print(f"  source3: {len(src3):,} entities", flush=True)

    n_src2 = len(src2)
    n_src3 = len(src3)
    total_candidate_pool = n_src2 + n_src3

    def sample_negative(positives_set: set[str]) -> tuple[str, str, tuple[str, str, str]] | None:
        """Return (entity_id, source_tag, fields) not in positives_set."""
        for _ in range(20):
            idx = rng.randrange(total_candidate_pool)
            if idx < n_src2:
                cand_id, fields = src2.at(idx)
                tag = "S2"
            else:
                cand_id, fields = src3.at(idx - n_src2)
                tag = "S3"
            if cand_id not in positives_set:
                return cand_id, tag, fields
        return None

    n_pos = 0
    n_neg = 0
    n_missing_positive = 0
    n_singletons = 0
    n_src1_missing = 0

    print(f"Writing {args.output} ...", flush=True)
    with GROUND_TRUTH_FILE.open("r", encoding="utf-8", newline="") as gt_f, \
            args.output.open("w", encoding="utf-8", newline="") as out_f:
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
                cand_id, tag, cand = sampled
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
