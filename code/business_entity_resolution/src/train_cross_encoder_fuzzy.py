"""LoRA fine-tune mDeBERTa-v3 cross-encoder with fuzzy features prepended.

Same skeleton as train_cross_encoder.py. Difference: per-pair rapidfuzz +
jellyfish scores are computed on the fly in the Dataset and inserted as a
bracket-tagged prefix in the "Listing A" segment:

    [nm=85 jw=90 ad=72 ph=65 nu=50]
    Listing A: <s1_name> | <s1_addr>
    Listing B: <cand_name> | <cand_addr>

Fields:
    nm  = rapidfuzz token_set_ratio(name_a, name_b)         [0-100]
    jw  = JaroWinkler(squash(name_a), squash(name_b)) * 100 [0-100]
    ad  = rapidfuzz token_set_ratio(addr_a, addr_b)         [0-100]
    ph  = jaccard(metaphone tokens of name_a vs name_b)     [0-100]
    nu  = jaccard(digit tokens of addr_a vs addr_b)         [0-100]

Usage:
    python train_cross_encoder_fuzzy.py \\
        --train dataset/train/train_pairs_100k.tsv \\
        --val   dataset/train/val_pairs.tsv \\
        --outdir models/cross_encoder_50k_fuzzy \\
        --max-train 50000 --epochs 3 --batch 32
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

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
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.distance import JaroWinkler  # noqa: E402
import jellyfish  # noqa: E402

csv.field_size_limit(sys.maxsize)

BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN = 224  # a few extra tokens over the vanilla script for the fuzzy prefix

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\b\d+\b")


def squash(s: str) -> str:
    return _WS.sub("", s.lower())


def _phon_jaccard(a: str, b: str) -> int:
    ca = {jellyfish.metaphone(w) for w in a.lower().split() if w}
    cb = {jellyfish.metaphone(w) for w in b.lower().split() if w}
    ca.discard(""); cb.discard("")
    if not ca or not cb:
        return 0
    return int(round(100 * len(ca & cb) / len(ca | cb)))


def _num_jaccard(a: str, b: str) -> int:
    na = set(_NUM.findall(a or ""))
    nb = set(_NUM.findall(b or ""))
    if not na or not nb:
        return 0
    return int(round(100 * len(na & nb) / len(na | nb)))


def fuzzy_features(name_a: str, addr_a: str, name_b: str, addr_b: str) -> dict:
    return {
        "nm": int(round(fuzz.token_set_ratio(name_a or "", name_b or ""))),
        "jw": int(round(100 * JaroWinkler.normalized_similarity(squash(name_a or ""), squash(name_b or "")))),
        "ad": int(round(fuzz.token_set_ratio(addr_a or "", addr_b or ""))),
        "ph": _phon_jaccard(name_a or "", name_b or ""),
        "nu": _num_jaccard(addr_a or "", addr_b or ""),
    }


def record_text(name: str, addr: str) -> str:
    return f"{name} | {addr or 'no address'}"


def build_pair_text(row: dict) -> tuple[str, str]:
    f = fuzzy_features(
        row["source1_business_name"], row["source1_business_address"],
        row["candidate_business_name"], row["candidate_business_address"],
    )
    prefix = f"[nm={f['nm']} jw={f['jw']} ad={f['ad']} ph={f['ph']} nu={f['nu']}]"
    a = f"{prefix} Listing A: {record_text(row['source1_business_name'], row['source1_business_address'])}"
    b = f"Listing B: {record_text(row['candidate_business_name'], row['candidate_business_address'])}"
    return a, b


class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = MAX_LEN, max_rows: int | None = None):
        self.rows: list[dict] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                if max_rows is not None and i >= max_rows:
                    break
                self.rows.append(r)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        a, b = build_pair_text(r)
        enc = self.tok(a, b, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = int(r["label"])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
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
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap the number of training rows read (from top of file).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-limit", type=int, default=10_000)
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
        target_modules=["query_proj", "key_proj", "value_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(f"loading train: {args.train}  (max_rows={args.max_train})", flush=True)
    train_ds = PairDataset(args.train, tok, max_rows=args.max_train)
    print(f"  train pairs: {len(train_ds):,}", flush=True)
    print(f"loading val:   {args.val}", flush=True)
    val_ds = PairDataset(args.val, tok)
    if args.eval_limit and len(val_ds) > args.eval_limit:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_ds), size=args.eval_limit, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, sorted(idx.tolist()))
    print(f"  val pairs (in-training eval): {len(val_ds):,}", flush=True)

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
