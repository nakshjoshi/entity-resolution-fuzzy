"""Stage 3: score blocking candidate pairs with the fine-tuned mDeBERTa cross-encoder.

Reads candidate shards written by lenient_blocking.py (parquet with the pair
texts) and writes one score shard per input shard:

    <out>/<shard name>.parquet   columns: s1_id, cand_id, fuzzy_score, fuzzy_rank,
                                 score (= P(same))

Shards are a work queue shared by every GPU of every instance: before scoring a
shard a worker claims it with an S3 conditional put (If-None-Match), so any
number of SageMaker jobs, on any mix of instance types, can drain the same
prefix without coordination. A shard that already has an output is skipped, so
re-running after a failure only redoes what is missing (pass --reclaim to drop
stale claims left by a dead worker).

Paths may be local directories or s3:// prefixes.

Usage (one process per visible GPU is spawned automatically):
    python score_pairs.py --pairs s3://bucket/pfx/pairs/ --out s3://bucket/pfx/scores/ \\
        --adapter /opt/ml/input/data/adapter
"""
import argparse
import io
import os
import random
import socket
import subprocess
import sys
import time

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

MAX_LEN = 192


def record_text(name, addr) -> str:
    return f"{name or ''} | {addr or 'no address'}"


# --- storage: local dir or s3:// prefix -------------------------------------------------

class Store:
    def __init__(self, root: str):
        self.s3 = root.startswith("s3://")
        if self.s3:
            import boto3
            self.cli = boto3.client("s3")
            self.bucket, _, key = root[5:].partition("/")
            self.prefix = key.rstrip("/") + "/" if key else ""
        else:
            self.root = root
            os.makedirs(root, exist_ok=True)

    def list(self, suffix: str = "") -> list[str]:
        if not self.s3:
            return sorted(f for f in os.listdir(self.root) if f.endswith(suffix))
        out = []
        pag = self.cli.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for o in page.get("Contents", []):
                name = o["Key"][len(self.prefix):]
                if "/" not in name and name.endswith(suffix):
                    out.append(name)
        return sorted(out)

    def exists(self, name: str) -> bool:
        if not self.s3:
            return os.path.exists(os.path.join(self.root, name))
        try:
            self.cli.head_object(Bucket=self.bucket, Key=self.prefix + name)
            return True
        except self.cli.exceptions.ClientError:
            return False

    def claim(self, name: str, who: str) -> bool:
        """Atomically create ``name``; False if it already exists."""
        if not self.s3:
            try:
                fd = os.open(os.path.join(self.root, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, who.encode()); os.close(fd)
                return True
            except FileExistsError:
                return False
        try:
            self.cli.put_object(Bucket=self.bucket, Key=self.prefix + name,
                                Body=who.encode(), IfNoneMatch="*")
            return True
        except self.cli.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("PreconditionFailed", "ConditionalRequestConflict", "412"):
                return False
            raise

    def delete(self, name: str) -> None:
        if not self.s3:
            os.remove(os.path.join(self.root, name))
        else:
            self.cli.delete_object(Bucket=self.bucket, Key=self.prefix + name)

    def read_table(self, name: str) -> pa.Table:
        if not self.s3:
            return pq.read_table(os.path.join(self.root, name))
        body = self.cli.get_object(Bucket=self.bucket, Key=self.prefix + name)["Body"].read()
        return pq.read_table(io.BytesIO(body))

    def write_table(self, name: str, t: pa.Table) -> None:
        if not self.s3:
            tmp = os.path.join(self.root, name + ".tmp")
            pq.write_table(t, tmp, compression="zstd")
            os.replace(tmp, os.path.join(self.root, name))
            return
        buf = io.BytesIO()
        pq.write_table(t, buf, compression="zstd")
        self.cli.put_object(Bucket=self.bucket, Key=self.prefix + name, Body=buf.getvalue())


# --- model --------------------------------------------------------------------------------

def load_model(adapter: str, base: str, seed: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The adapter does not store DeBERTa's (untrained) pooler; seed exactly as
    # train_cross_encoder.py did so the pooler init matches training.
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2,
        id2label={0: "different_business", 1: "same_business"},
        label2id={"different_business": 0, "same_business": 1},
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return tok, model.eval().to(dev), dev


def score_table(t: pa.Table, tok, model, dev: str, batch: int) -> np.ndarray:
    import torch
    cols = {c: t.column(c).to_pylist() for c in ("s1_name", "s1_addr", "cand_name", "cand_addr")}
    a = [f"Listing A: {record_text(n, d)}" for n, d in zip(cols["s1_name"], cols["s1_addr"])]
    b = [f"Listing B: {record_text(n, d)}" for n, d in zip(cols["cand_name"], cols["cand_addr"])]
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.zeros(len(a), dtype=np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            lg = model(**enc).logits.float()
            out[idx] = torch.sigmoid(lg[:, 1] - lg[:, 0]).cpu().numpy()
    return out


# --- worker loop --------------------------------------------------------------------------

def worker(args) -> None:
    who = f"{socket.gethostname()}:{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}:{os.getpid()}"
    pairs, outs = Store(args.pairs), Store(args.out)
    tok, model, dev = load_model(args.adapter, args.base, args.seed)
    print(f"[{who}] device={dev}", flush=True)
    rng, done, tried = random.Random(who), 0, set()
    while True:
        # Blocking may still be producing shards: keep re-listing until every
        # blocking job has left its done marker and nothing is left to claim.
        finished = args.expect_done <= len(Store(args.done).list()) if args.done else True
        shards = [n for n in pairs.list(".parquet") if n not in tried]
        rng.shuffle(shards)  # spread workers across the queue
        for name in shards:
            tried.add(name)
            if outs.exists(name) or not outs.claim(name + ".claim", who):
                continue
            t0 = time.time()
            t = pairs.read_table(name)
            s = score_table(t, tok, model, dev, args.batch)
            keep = [c for c in ("s1_id", "cand_id", "fuzzy_score", "fuzzy_rank") if c in t.column_names]
            outs.write_table(name, t.select(keep).append_column("score", pa.array(s)))
            done += 1
            dt = time.time() - t0
            print(f"[{who}] {name}: {t.num_rows:,} pairs in {dt:.0f}s ({t.num_rows/dt:,.0f}/s)", flush=True)
        if finished and not shards:
            break
        if not shards:
            time.sleep(30)
    print(f"[{who}] finished, scored {done} shards", flush=True)


def reclaim(args) -> None:
    """Drop claims whose shard has no output (worker died mid-shard)."""
    outs = Store(args.out)
    have = set(outs.list(".parquet"))
    for c in outs.list(".claim"):
        if c[:-len(".claim")] not in have:
            outs.delete(c)
            print("reclaimed", c, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="dir or s3:// prefix of candidate shards")
    ap.add_argument("--out", required=True, help="dir or s3:// prefix for score shards")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reclaim", action="store_true")
    ap.add_argument("--done", default=None, help="dir/prefix where blocking jobs leave done markers")
    ap.add_argument("--expect-done", type=int, default=0, help="wait for this many markers")
    ap.add_argument("--single", action="store_true", help="internal: run one worker in-process")
    args = ap.parse_args()

    if args.reclaim:
        reclaim(args)
    if args.single:
        worker(args)
        return

    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except ImportError:
        n_gpu = 0
    if n_gpu <= 1:
        worker(args)
        return
    # One process per GPU; each drains the shared queue.
    argv = [a for a in sys.argv[1:] if a != "--reclaim"] + ["--single"]
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), *argv],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
             for g in range(n_gpu)]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker exit codes: {codes}")


if __name__ == "__main__":
    main()
