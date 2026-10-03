"""Emit an .ipynb for Colab that embeds sampler/train/eval scripts inline.

Run locally on the M2:
    python make_colab_notebook.py

Output: colab_train_500k.ipynb (upload this to Colab; Runtime -> T4 GPU -> Run All).
"""
import json
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent.parent.parent / "colab_train_500k.ipynb"

SAMPLE = (SRC / "sample_labeled_pairs.py").read_text()
TRAIN = (SRC / "train_cross_encoder.py").read_text()
EVAL = (SRC / "eval_cross_encoder.py").read_text()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


DRIVE = "/content/drive/MyDrive/amazon-entity-resolution"

cells = [
    md("# Business Entity Resolution — 500K LoRA Fine-Tuning on Colab T4\n\n"
       "**Runtime:** set to `T4 GPU` (Runtime → Change runtime type). Then `Run All`.\n\n"
       "Assumes `train_labeled_pairs.tsv` (5.6 GB) is already at "
       "`/MyDrive/amazon-entity-resolution/`. Nothing else needs to be in Drive.\n\n"
       "Expected wall time on T4: ~2–3 h (setup ~2 min, sampling ~2 min, "
       "training ~2 h, eval ~15 min)."),

    md("## 1. GPU + Drive + deps"),
    code("!nvidia-smi | head -20\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "\n"
         "# torchao pre-installed on Colab is too old for current peft; upgrade it\n"
         "# so peft's LoRA dispatcher doesn't fail its version check at import time.\n"
         "!pip install -q \"transformers>=4.44.0\" \"peft>=0.11.0\" \\\n"
         "    \"accelerate>=0.33.0\" \"torchao>=0.16.0\" sentencepiece protobuf\n"
         "\n"
         f"DRIVE = '{DRIVE}'\n"
         "import os\n"
         "os.makedirs(DRIVE, exist_ok=True)\n"
         "!ls -lh {DRIVE}\n"),

    md("## 2. Write sampler script"),
    code(f"%%writefile /content/sample_labeled_pairs.py\n{SAMPLE}"),

    md("## 3. Write training script"),
    code(f"%%writefile /content/train_cross_encoder.py\n{TRAIN}"),

    md("## 4. Write eval script"),
    code(f"%%writefile /content/eval_cross_encoder.py\n{EVAL}"),

    md("## 5. Sample: create val + 500K subsets on Drive (~2 min)\n\n"
       "Seed=42 → same splits as the M3, so the runs are directly comparable."),
    code("!python /content/sample_labeled_pairs.py \\\n"
         "    --input  {DRIVE}/train_labeled_pairs.tsv \\\n"
         "    --outdir {DRIVE}\n"
         "!ls -lh {DRIVE}/val_pairs.tsv {DRIVE}/train_pairs_500k.tsv"),

    md("## 6. Train — LoRA on 500K, fp16, batch 32, 3 epochs (~2 h on T4)\n\n"
       "Adapter checkpoints save per-epoch to Drive (~50 MB each). "
       "`save_total_limit=2` keeps disk use bounded; `load_best_model_at_end=True` "
       "restores the best-F0.5 epoch as the final adapter."),
    code("!python /content/train_cross_encoder.py \\\n"
         "    --train  {DRIVE}/train_pairs_500k.tsv \\\n"
         "    --val    {DRIVE}/val_pairs.tsv \\\n"
         "    --outdir {DRIVE}/cross_encoder_500k \\\n"
         "    --epochs 3 --batch 32 --lr 1e-4"),

    md("## 7. Full-val eval on the fine-tuned adapter (~15 min)"),
    code("!python /content/eval_cross_encoder.py \\\n"
         "    --adapter {DRIVE}/cross_encoder_500k \\\n"
         "    --val     {DRIVE}/val_pairs.tsv \\\n"
         "    --out     {DRIVE}/eval_cross_500k.json \\\n"
         "    --batch   128\n"
         "!cat {DRIVE}/eval_cross_500k.json"),

    md("## 8. Summary\n\n"
       "Adapter + metrics are on Drive at `cross_encoder_500k/`. Compare "
       "`eval_cross_500k.json` against the M3's `eval_base.json` (GLiNER baseline) "
       "and `eval_cross_100k.json` — the metric that matters is "
       "`metrics.source1_macro_f0_5_at_0.500`."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
