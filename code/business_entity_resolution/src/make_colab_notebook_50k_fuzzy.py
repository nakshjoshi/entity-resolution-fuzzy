"""Emit a Colab .ipynb for 50K + fuzzy-feature LoRA fine-tune + eval.

Run locally on the M2:
    python make_colab_notebook_50k_fuzzy.py

Output at repo root: colab_train_50k_fuzzy.ipynb
"""
import json
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent.parent.parent / "colab_train_50k_fuzzy.ipynb"

SAMPLE = (SRC / "sample_labeled_pairs.py").read_text()
TRAIN = (SRC / "train_cross_encoder_fuzzy.py").read_text()
EVAL = (SRC / "eval_cross_encoder_fuzzy.py").read_text()

DRIVE = "/content/drive/MyDrive/amazon-entity-resolution"
DATA = f"{DRIVE}/dataset/train"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("# 50K + Fuzzy Features + LoRA — Cross-Encoder on Colab T4\n\n"
       "**Runtime:** T4 GPU. Then `Run All`.\n\n"
       "Fuzzy features (rapidfuzz name/addr ratios, JaroWinkler, Metaphone jaccard, "
       "house-number jaccard) are computed per pair and prepended to the model input "
       "as `[nm=.. jw=.. ad=.. ph=.. nu=..]`. The model conditions its same/different "
       "decision on both text and those signals.\n\n"
       "Data assumed at `/MyDrive/amazon-entity-resolution/dataset/train/train_labeled_pairs.tsv`.\n\n"
       "Expected wall time on T4: ~25-45 min total (setup ~2 min, sampling ~2 min, "
       "50K training ~15 min, full-val eval ~15-25 min)."),

    md("## 1. GPU + Drive + deps"),
    code("!nvidia-smi | head -20\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "\n"
         "# torchao pin: Colab pre-installs an old version that breaks peft's dispatcher.\n"
         "!pip install -q \"transformers>=4.44.0\" \"peft>=0.11.0\" \\\n"
         "    \"accelerate>=0.33.0\" \"torchao>=0.16.0\" \\\n"
         "    rapidfuzz jellyfish sentencepiece protobuf\n"
         "\n"
         f"DRIVE = '{DRIVE}'\n"
         f"DATA  = '{DATA}'\n"
         "import os\n"
         "os.makedirs(DRIVE, exist_ok=True)\n"
         "!ls -lh {DATA}"),

    md("## 2. Write sampler"),
    code(f"%%writefile /content/sample_labeled_pairs.py\n{SAMPLE}"),

    md("## 3. Write fuzzy training script"),
    code(f"%%writefile /content/train_cross_encoder_fuzzy.py\n{TRAIN}"),

    md("## 4. Write fuzzy eval script"),
    code(f"%%writefile /content/eval_cross_encoder_fuzzy.py\n{EVAL}"),

    md("## 5. Sample: create val + train_pairs_100k on Drive (~2 min)\n\n"
       "Sampler builds 100K and 500K subsets; we cap the training at 50K via "
       "`--max-train 50000` in cell 6. Seed=42 so results are reproducible."),
    code("!python /content/sample_labeled_pairs.py \\\n"
         "    --input  {DATA}/train_labeled_pairs.tsv \\\n"
         "    --outdir {DATA}\n"
         "!ls -lh {DATA}/val_pairs.tsv {DATA}/train_pairs_100k.tsv"),

    md("## 6. Train — 50K, fuzzy features, LoRA, fp16, batch 32 (~15 min on T4)"),
    code("!python /content/train_cross_encoder_fuzzy.py \\\n"
         "    --train  {DATA}/train_pairs_100k.tsv \\\n"
         "    --val    {DATA}/val_pairs.tsv \\\n"
         "    --outdir {DRIVE}/cross_encoder_50k_fuzzy \\\n"
         "    --max-train 50000 \\\n"
         "    --epochs 3 --batch 32 --lr 1e-4"),

    md("## 7. Full-val eval (~15-25 min on T4)"),
    code("!python /content/eval_cross_encoder_fuzzy.py \\\n"
         "    --adapter {DRIVE}/cross_encoder_50k_fuzzy \\\n"
         "    --val     {DATA}/val_pairs.tsv \\\n"
         "    --out     {DRIVE}/eval_cross_50k_fuzzy.json \\\n"
         "    --batch   128\n"
         "!cat {DRIVE}/eval_cross_50k_fuzzy.json"),

    md("## 8. What's saved on Drive\n\n"
       "```\n"
       "amazon-entity-resolution/\n"
       "├── dataset/train/\n"
       "│   ├── val_pairs.tsv\n"
       "│   ├── train_pairs_100k.tsv\n"
       "│   └── train_pairs_500k.tsv\n"
       "├── cross_encoder_50k_fuzzy/\n"
       "│   ├── adapter_model.safetensors    (~10-50 MB LoRA weights)\n"
       "│   ├── adapter_config.json\n"
       "│   ├── tokenizer files\n"
       "│   ├── training_args.json\n"
       "│   └── final_val_metrics.json\n"
       "└── eval_cross_50k_fuzzy.json\n"
       "```\n\n"
       "Metric to compare against the M3's baseline: "
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
