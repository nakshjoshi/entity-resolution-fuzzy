"""Emit a Colab notebook that benchmarks Contrastive-LM/CLM-v0.1-8B (no
fine-tuning) as a per-pair scorer on the fuzzy candidate pairs.

Local run:
    python make_clm_colab.py

Output: colab_clm_bench.ipynb — upload to Colab, pick a T4 GPU runtime, Run All.
"""
import json
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent.parent.parent / "colab_clm_bench.ipynb"

EVAL = (SRC / "eval_clm.py").read_text()
SAMPLE = (SRC / "sample_labeled_pairs.py").read_text()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


DRIVE = "/content/drive/MyDrive/amazon-entity-resolution"

cells = [
    md("# CLM-v0.1-8B — Zero-Shot Pair Scoring on Fuzzy Candidates (Free-tier T4)\n\n"
       "`Contrastive-LM/CLM-v0.1-8B` is a **questions head over frozen Qwen3-8B\n"
       "embeddings**, so this notebook runs three processes on the single T4:\n\n"
       "1. `vllm serve Qwen/Qwen3-8B-AWQ` — 4-bit AWQ embedding server on port 8090\n"
       "   (fits in ~6 GB VRAM, leaving headroom for KV cache on 16 GB T4).\n"
       "2. `clm-serve` — the CLM questions head on port 8700.\n"
       "3. `eval_clm.py` — scores each fuzzy-surviving pair via\n"
       "   `CLMClient.system_one(state, questions={'match': Score(...)})`.\n\n"
       "**Data assumed on Drive at** `MyDrive/amazon-entity-resolution/`:\n"
       "- `train_labeled_pairs.tsv` (already uploaded).\n\n"
       "**Free-tier budget:** default `--limit 5000` pairs (~30–60 min end-to-end,\n"
       "assuming ~2–3 pair/s). Increase if the session has time left.\n\n"
       "**If AWQ vLLM OOMs on T4:** drop `--max-model-len` to 512 in section 3,\n"
       "or switch to Colab Pro (L4/A100)."),

    md("## 1. GPU + Drive + deps"),
    code("!nvidia-smi || echo 'NO GPU — Runtime → Change runtime type → T4 GPU'\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "\n"
         "# vLLM pulls a lot; pin versions that are known to work on Colab T4.\n"
         "!pip install -q --upgrade pip\n"
         "!pip install -q \"vllm>=0.6.3\" \"contrastive-lm\" \\\n"
         "    \"transformers>=4.44.0\" \"accelerate>=0.33.0\" \\\n"
         "    autoawq sentencepiece protobuf requests\n"
         "\n"
         f"DRIVE = '{DRIVE}'\n"
         "import os\n"
         "assert os.path.exists(f'{DRIVE}/train_labeled_pairs.tsv'), \\\n"
         "    'upload train_labeled_pairs.tsv to ' + DRIVE + ' first'\n"
         "!ls -lh {DRIVE}\n"),

    md("## 2. Derive `val_pairs.tsv` from `train_labeled_pairs.tsv`\n\n"
       "~3–5 min. Cached to Drive so re-runs skip this."),
    code(f"%%writefile /content/sample_labeled_pairs.py\n{SAMPLE}"),
    code("import os\n"
         "if not os.path.exists(f'{DRIVE}/val_pairs.tsv'):\n"
         "    !python /content/sample_labeled_pairs.py \\\n"
         "        --input  {DRIVE}/train_labeled_pairs.tsv \\\n"
         "        --outdir {DRIVE}\n"
         "!ls -lh {DRIVE}/val_pairs.tsv && wc -l {DRIVE}/val_pairs.tsv"),

    md("## 3. Launch the Qwen3-8B-AWQ embedding server (vLLM, background)\n\n"
       "Business records are short so `--max-model-len 1024` is plenty and\n"
       "keeps the KV cache tiny. Logs stream to `/content/vllm.log`."),
    code("import subprocess, time, os, signal, requests\n"
         "\n"
         "VLLM_LOG = '/content/vllm.log'\n"
         "vllm_cmd = [\n"
         "    'vllm', 'serve', 'Qwen/Qwen3-8B-AWQ',\n"
         "    '--served-model-name', 'qwen3-8b',\n"
         "    '--runner', 'pooling',\n"
         "    '--quantization', 'awq',\n"
         "    '--dtype', 'float16',\n"
         "    '--max-model-len', '1024',\n"
         "    '--gpu-memory-utilization', '0.80',\n"
         "    '--port', '8090',\n"
         "]\n"
         "print('launching:', ' '.join(vllm_cmd))\n"
         "vllm_proc = subprocess.Popen(vllm_cmd,\n"
         "    stdout=open(VLLM_LOG, 'w'), stderr=subprocess.STDOUT,\n"
         "    preexec_fn=os.setsid)\n"
         "print('vllm pid:', vllm_proc.pid, '— tailing log until /v1/models responds')\n"
         "\n"
         "for i in range(180):  # up to ~15 min for first-time model download\n"
         "    time.sleep(5)\n"
         "    try:\n"
         "        r = requests.get('http://127.0.0.1:8090/v1/models', timeout=2)\n"
         "        if r.ok:\n"
         "            print('vllm ready:', r.json()); break\n"
         "    except Exception:\n"
         "        pass\n"
         "    if i % 6 == 0:\n"
         "        !tail -n 5 {VLLM_LOG}\n"
         "else:\n"
         "    raise RuntimeError('vllm did not come up in time; see /content/vllm.log')\n"),

    md("## 4. Launch `clm-serve` (background) and probe end-to-end\n\n"
       "Points at the local vLLM. The probe issues one `system_one` call so we\n"
       "fail fast if anything is misconfigured before the long eval loop."),
    code("CLM_LOG = '/content/clm.log'\n"
         "clm_env = {**os.environ,\n"
         "    'CLM_EMB_URL': 'http://127.0.0.1:8090/v1/embeddings',\n"
         "    'EMB_URL':     'http://127.0.0.1:8090/v1/embeddings',\n"
         "}\n"
         "clm_proc = subprocess.Popen(['clm-serve'],\n"
         "    stdout=open(CLM_LOG, 'w'), stderr=subprocess.STDOUT,\n"
         "    env=clm_env, preexec_fn=os.setsid)\n"
         "print('clm-serve pid:', clm_proc.pid)\n"
         "\n"
         "for i in range(60):\n"
         "    time.sleep(2)\n"
         "    try:\n"
         "        r = requests.get('http://127.0.0.1:8700/', timeout=2)\n"
         "        if r.status_code < 500:\n"
         "            print('clm-serve ready:', r.status_code); break\n"
         "    except Exception:\n"
         "        pass\n"
         "else:\n"
         "    !tail -n 40 {CLM_LOG}\n"
         "    raise RuntimeError('clm-serve did not come up; see /content/clm.log')\n"
         "\n"
         "# End-to-end probe.\n"
         "from clm import CLMClient, Score\n"
         "try:\n"
         "    client = CLMClient(base_url='http://127.0.0.1:8700')\n"
         "except TypeError:\n"
         "    client = CLMClient()\n"
         "probe = client.system_one(\n"
         "    state='Record A: Acme Corp | 123 Main St, NYC\\n'\n"
         "          'Record B: Acme Corporation | 123 Main Street, New York',\n"
         "    questions={'match': Score(\n"
         "        instructions='Are these the same business entity?',\n"
         "        criteria=['Definitely different', 'Possibly same', 'Definitely same'],\n"
         "    )},\n"
         ")\n"
         "print('probe probabilities:', probe.answers['match'].probabilities)\n"),

    md("## 5. Write the eval script"),
    code(f"%%writefile /content/eval_clm.py\n{EVAL}"),

    md("## 6. Benchmark on 5 000 pairs (~30–60 min)\n\n"
       "One `system_one` call per pair. If pair/s is healthy after a few hundred,\n"
       "cancel and re-run with a larger `--limit` (or drop it entirely)."),
    code("!python /content/eval_clm.py \\\n"
         "    --val   {DRIVE}/val_pairs.tsv \\\n"
         "    --out   {DRIVE}/eval_clm_8b.json \\\n"
         "    --clm-url http://127.0.0.1:8700 \\\n"
         "    --limit 5000\n"
         "!cat {DRIVE}/eval_clm_8b.json"),

    md("## 7. Compare against previous baselines on Drive"),
    code("import json, glob\n"
         "for p in sorted(glob.glob(f'{DRIVE}/eval_*.json')):\n"
         "    d = json.load(open(p))\n"
         "    m = d.get('metrics', {})\n"
         "    best = m.get('pair_best_threshold', {})\n"
         "    print(f\"{p.split('/')[-1]:<32} \"\n"
         "          f\"pair@0.5 F0.5={m.get('pair_at_0.5', {}).get('f0_5', 0):.3f}  \"\n"
         "          f\"best F0.5={best.get('f0_5', 0):.3f} @ thr={best.get('threshold', 0):.3f}  \"\n"
         "          f\"s1-macro@0.5={m.get('source1_macro_f0_5_at_0.500', 0):.3f}\")\n"),

    md("## 8. Cleanup (optional)\n\n"
       "Kill the background servers before disconnecting so you don't leave GPU\n"
       "memory pinned mid-session."),
    code("import os, signal\n"
         "for p in (clm_proc, vllm_proc):\n"
         "    try:\n"
         "        os.killpg(os.getpgid(p.pid), signal.SIGTERM)\n"
         "    except Exception as e:\n"
         "        print('kill failed:', e)\n"
         "print('done')\n"),
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
