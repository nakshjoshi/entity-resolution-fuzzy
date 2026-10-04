"""SageMaker training-job entry point for inference (blocking or scoring).

Bundled with src/{lenient_blocking,blocking,fuzzy,normalize,phonetic,score_pairs}.py.

  --mode block   CPU job. Channel 'data' holds <split>_source{1,2,3}.tsv (and optionally
                 's1ids' a one-column id list). Runs lenient_blocking.py for one
                 country/part, streaming finished shards to <out-s3>/pairs/ as they are
                 written, then leaves <out-s3>/done/<job tag>.
  --mode score   GPU job. Channel 'adapter' holds the training job's model.tar.gz.
                 For each prefix in --out-s3 (comma-separated) scores <prefix>/pairs/ ->
                 <prefix>/scores/, waiting until --expect-done blocking markers exist.
                 Any number of score jobs can run at once; they share the shard queue.
"""
import argparse
import glob
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def s3_split(s3: str):
    bucket, _, prefix = s3[5:].partition("/")
    return bucket, prefix.rstrip("/")


def sync_up(local: str, s3: str, seen: set) -> None:
    """Upload complete files (no .tmp) under ``local`` that were not uploaded yet."""
    import boto3
    cli = boto3.client("s3")
    bucket, prefix = s3_split(s3)
    for root, _, files in os.walk(local):
        for f in sorted(files):
            path = os.path.join(root, f)
            if f.endswith(".tmp") or path in seen:
                continue
            cli.upload_file(path, bucket, f"{prefix}/{os.path.relpath(path, local)}")
            seen.add(path)
            print("uploaded", path, flush=True)


def one(pattern: str) -> str:
    hits = glob.glob(pattern)
    if len(hits) != 1:
        sys.exit(f"expected exactly one match for {pattern}, got {hits}")
    return hits[0]


def block(args) -> None:
    d = os.environ["SM_CHANNEL_DATA"]
    work = "/opt/ml/work"
    cmd = [sys.executable, os.path.join(HERE, "lenient_blocking.py"),
           "--s1", one(f"{d}/*source1.tsv"), "--s2", one(f"{d}/*source2.tsv"),
           "--s3", one(f"{d}/*source3.tsv"), "--out", work,
           "--k", args.k, "--k-max", args.k_max, "--top-m", args.top_m,
           "--df-max", args.df_max, "--part", args.part, "--parts", args.parts]
    if os.environ.get("SM_CHANNEL_S1IDS"):
        cmd += ["--s1-ids", one(f"{os.environ['SM_CHANNEL_S1IDS']}/*")]
    if args.countries:
        cmd += ["--countries", *args.countries.split(",")]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    seen: set = set()
    stop = threading.Event()

    def uploader():
        while not stop.wait(20):
            sync_up(work, args.out_s3, seen)

    th = threading.Thread(target=uploader, daemon=True)
    th.start()
    code = proc.wait()
    stop.set(); th.join()
    if code:
        sys.exit(f"lenient_blocking.py exited with {code}")
    sync_up(work, args.out_s3, seen)
    import boto3
    bucket, prefix = s3_split(args.out_s3)
    boto3.client("s3").put_object(Bucket=bucket, Key=f"{prefix}/done/{args.tag}", Body=b"done")
    print("block done", flush=True)


def score(args) -> None:
    adapter = "/opt/ml/adapter"
    os.makedirs(adapter, exist_ok=True)
    run(["tar", "-xzf", one(f"{os.environ['SM_CHANNEL_ADAPTER']}/*.tar.gz"), "-C", adapter])
    expects = [int(x) for x in args.expect_done.split(",")] if args.expect_done else []
    for i, out in enumerate(p.rstrip("/") for p in args.out_s3.split(",")):
        cmd = [sys.executable, os.path.join(HERE, "score_pairs.py"), "--pairs", f"{out}/pairs/",
               "--out", f"{out}/scores/", "--adapter", adapter, "--batch", args.batch]
        if expects:
            cmd += ["--done", f"{out}/done/", "--expect-done", str(expects[i])]
        run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["block", "score"], required=True)
    ap.add_argument("--out-s3", required=True)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--countries", default="", help="comma-separated; empty = all")
    ap.add_argument("--part", default="0")
    ap.add_argument("--parts", default="1")
    ap.add_argument("--k", default="40")
    ap.add_argument("--k-max", default="80")
    ap.add_argument("--top-m", default="300")
    ap.add_argument("--df-max", default="5000")
    ap.add_argument("--batch", default="256")
    ap.add_argument("--expect-done", default="", help="per --out-s3 prefix, comma-separated")
    args, _ = ap.parse_known_args()
    t = time.time()
    block(args) if args.mode == "block" else score(args)
    print(f"total {(time.time()-t)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
