"""SageMaker training-job entry point: train the cross-encoder, then eval it.

Runs the unchanged src/train_cross_encoder.py and src/eval_cross_encoder.py
(bundled next to this file) against the job's input channels:

    train    -> train_pairs_*.tsv                  (training runs)
    val      -> val_pairs.tsv
    adapter  -> model.tar.gz of a finished job     (--eval-only 1 runs)

The LoRA adapter goes to SM_MODEL_DIR (uploaded as model.tar.gz); the eval
JSON goes to SM_OUTPUT_DATA_DIR (uploaded as output.tar.gz) and is also
copied next to the adapter.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train(args, model_dir: str, val_file: str) -> None:
    train_files = glob.glob(os.path.join(os.environ["SM_CHANNEL_TRAIN"], "*.tsv"))
    if len(train_files) != 1:
        sys.exit(f"expected exactly one .tsv in the train channel, got {train_files}")

    run([sys.executable, os.path.join(HERE, "train_cross_encoder.py"),
         "--train", train_files[0], "--val", val_file, "--outdir", model_dir,
         "--epochs", str(args.epochs), "--batch", str(args.batch), "--lr", str(args.lr)])

    # Per-epoch checkpoints duplicate the final adapter; drop them to keep model.tar.gz small.
    shutil.rmtree(os.path.join(model_dir, "checkpoints"), ignore_errors=True)


def unpack_adapter(model_dir: str) -> None:
    tars = glob.glob(os.path.join(os.environ["SM_CHANNEL_ADAPTER"], "*.tar.gz"))
    if len(tars) != 1:
        sys.exit(f"expected exactly one .tar.gz in the adapter channel, got {tars}")
    run(["tar", "-xzf", tars[0], "-C", model_dir])


def evaluate(args, model_dir: str, val_file: str, out_dir: str) -> None:
    eval_json = os.path.join(out_dir, f"eval_cross_{args.tag}.json")
    cmd = [sys.executable, os.path.join(HERE, "eval_cross_encoder.py"),
           "--adapter", model_dir, "--val", val_file, "--out", eval_json,
           "--batch", str(args.eval_batch)]
    if args.eval_limit:
        cmd += ["--limit", str(args.eval_limit)]
    run(cmd)
    shutil.copy(eval_json, model_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--eval-limit", type=int, default=0, help="0 = full val set")
    ap.add_argument("--tag", default="100k")
    ap.add_argument("--eval-only", type=int, default=0,
                    help="1 = skip training; eval the model.tar.gz in the 'adapter' channel")
    args, _ = ap.parse_known_args()

    val_file = os.path.join(os.environ["SM_CHANNEL_VAL"], "val_pairs.tsv")
    model_dir = os.environ["SM_MODEL_DIR"]
    out_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    if args.eval_only:
        unpack_adapter(model_dir)
    else:
        train(args, model_dir, val_file)
    evaluate(args, model_dir, val_file, out_dir)


if __name__ == "__main__":
    main()
