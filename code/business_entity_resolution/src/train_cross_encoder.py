"""LoRA fine-tune mDeBERTa-v3-base as a same/different cross-encoder.

Trains on train_pairs_{100k,500k}.tsv produced by sample_labeled_pairs.py.
Input format matches the inference format in gliner_decide.py so the
fine-tuned model can slot into the same PAIR_PROMPT semantics:

    Listing A: <s1_name> | <s1_addr or "no address">
    Listing B: <cand_name> | <cand_addr or "no address">

Base model: microsoft/mdeberta-v3-base (MIT license, ~280M params — same
architecture as the encoder inside GLiNER2.5-multi-Decide).

Output: LoRA adapter + tokenizer under --outdir. Weights are ~10-50 MB.

Usage:
    python train_cross_encoder.py \\
        --train dataset/train/train_pairs_100k.tsv \\
        --val   dataset/train/val_pairs.tsv \\
        --outdir models/cross_encoder_100k \\
        --epochs 3 --batch 16
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Silence HF telemetry and force fp32 numerics on MPS where fp16 is flaky.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

csv.field_size_limit(sys.maxsize)

BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN = 192  # ~two short business records comfortably fit


def record_text(name: str, addr: str) -> str:
    return f"{name} | {addr or 'no address'}"


def pair_text(row: dict) -> tuple[str, str]:
    """Cross-encoder input as (text_a, text_b) — tokenizer inserts [SEP]."""
    a = f"Listing A: {record_text(row['source1_business_name'], row['source1_business_address'])}"
    b = f"Listing B: {record_text(row['candidate_business_name'], row['candidate_business_address'])}"
    return a, b


class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = MAX_LEN):
        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                self.rows.append(r)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        a, b = pair_text(r)
        enc = self.tok(a, b, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = int(r["label"])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits[:, 1] + logits[:, 0]))  # softmax margin -> P(same)
    preds = (probs >= 0.5).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * precision * recall / (0.25 * precision + recall)) if (precision + recall) else 0.0
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "precision": precision, "recall": recall, "f0_5": f05}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help="HF id or local path of the base encoder.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-limit", type=int, default=20_000,
                    help="Cap val examples used during in-training eval to keep it snappy.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}", flush=True)

    print(f"loading base model: {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "different_business", 1: "same_business"},
        label2id={"different_business": 0, "same_business": 1},
    )

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # DeBERTa-v2/v3 attention projection names:
        target_modules=["query_proj", "key_proj", "value_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(f"loading train: {args.train}", flush=True)
    train_ds = PairDataset(args.train, tok)
    print(f"  train pairs: {len(train_ds):,}", flush=True)
    print(f"loading val:   {args.val}", flush=True)
    val_ds = PairDataset(args.val, tok)
    if args.eval_limit and len(val_ds) > args.eval_limit:
        # Sub-select deterministically for in-training eval; full-val runs later.
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_ds), size=args.eval_limit, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, sorted(idx.tolist()))
    print(f"  val pairs (in-training eval): {len(val_ds):,}", flush=True)

    # fp16 on MPS is flaky; fp32 is safer and still fits in 16GB.
    use_fp16 = device == "cuda"
    training_args = TrainingArguments(
        output_dir=str(args.outdir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        fp16=use_fp16,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f0_5",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    print("training ...", flush=True)
    trainer.train()

    print(f"saving adapter + tokenizer to {args.outdir}", flush=True)
    model.save_pretrained(args.outdir)
    tok.save_pretrained(args.outdir)
    (args.outdir / "training_args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    metrics = trainer.evaluate()
    (args.outdir / "final_val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
