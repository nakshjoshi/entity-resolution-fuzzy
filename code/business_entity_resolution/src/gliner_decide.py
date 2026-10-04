"""GLiNER2.5-Decide as the matching model over fuzzy/phonetic candidate buckets.

The model is loaded with ``gliner2.AutoExtractor`` from a local folder only:
Hugging Face offline mode is forced on before any HF library is imported, so the
code never contacts the Hub and never reads or writes the Hugging Face cache.
Import this module before anything else that imports ``transformers`` or
``huggingface_hub`` (those libraries read the offline flags at import time).

Two ways to ask the model, both defined from the challenge brief (records from
independent vendors with no shared identifier, only a noisy name and address;
a similar name at a different address, or a different business at the same
address, is not a match):

* bucket mode - ``score_bucket``: the Source 1 record is the text and its
  candidate records are the labels of one multi-label task; the model scores
  every candidate at once, mirroring one ground-truth row (S1 -> list of IDs).
* pair mode - ``score_pairs``: one same/different decision per candidate pair.

    from gliner_decide import DecideJudge
    judge = DecideJudge("models/GLiNER2.5-Decide")
    judge.score_bucket(("Acme Robotics", "12 Main St, Springfield, IL"),
                       [("ACME ROBOTICS INC", "12 MAIN STREET, SPRINGFIELD, IL"),
                        ("Acme Robotics", "76 River Road, Portland, OR")])
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402

SAME, DIFF = "same_business", "different_business"

# Shared definition of a match, taken from the challenge brief.
NOISE = ("abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road, St/Street), "
         "added or missing legal suffixes, & versus and, reordered words, typos, accents, "
         "upper or lower case, missing address parts such as PIN code or state, a landmark "
         "reference, or a name written in another script")
NOT_A_MATCH = ("a similar or identical name at a different street address or city, or a "
               "different business that happens to share the same address")

BUCKET_PROMPT = (
    "The text is a business record from the clean reference list. Each label is a record "
    "from an independent data vendor that shares no identifier with it, only a name and an "
    "address, written in the vendor's own noisy format. Select every label that describes "
    f"the same real-world business, even when written with {NOISE}. Do not select {NOT_A_MATCH}. "
    "Select nothing if no label is the same business."
)
PAIR_PROMPT = (
    "Listing A and listing B come from two independent data vendors that share no "
    "identifier, only a business name and an address, each written in its own noisy "
    "format. Decide whether both listings describe the same real-world business."
)
PAIR_LABELS = {
    SAME: f"The same business at the same location, even if written with {NOISE}.",
    DIFF: f"Two different businesses: {NOT_A_MATCH}.",
}

# Characters gliner2 uses as schema delimiters; keep them out of label strings.
_DELIMS = re.compile(r"[()\[\]]")


def record_text(name, addr) -> str:
    return f"{name} | {addr or 'no address'}"


def label_text(name, addr) -> str:
    return re.sub(r"\s+", " ", _DELIMS.sub(" ", record_text(name, addr))).strip()


def pair_text(name_a, addr_a, name_b, addr_b) -> str:
    return f"Listing A: {record_text(name_a, addr_a)}\nListing B: {record_text(name_b, addr_b)}"


class DecideJudge:
    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir).resolve()
        # A non-existent path would be treated as a Hub repo id; refuse instead.
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"no model.safetensors in {model_dir}")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = AutoExtractor.from_pretrained(str(model_dir), map_location=device)

    def score_bucket(self, query, candidates):
        """query: (name, addr) of the S1 record; candidates: list of (name, addr).

        Returns one match score per candidate (candidates with identical text
        share a label and therefore a score).
        """
        if not candidates:
            return []
        labels = [label_text(*c) for c in candidates]
        uniq = list(dict.fromkeys(labels))
        task = {"match": {"labels": uniq, "prompt": BUCKET_PROMPT,
                          "multi_label": True, "cls_threshold": 0.0}}
        out = self.model.classify_text(record_text(*query), task, include_confidence=True)
        score = {d["label"]: d["confidence"] for d in out["match"]}
        return [score.get(lab, 0.0) for lab in labels]

    def score_pairs(self, pairs, batch_size=16):
        """pairs: list of (name_a, addr_a, name_b, addr_b). Returns P(same) per pair."""
        task = {"decision": {"labels": dict(PAIR_LABELS), "prompt": PAIR_PROMPT,
                             "multi_label": True, "cls_threshold": 0.0}}
        out = self.model.batch_classify_text([pair_text(*p) for p in pairs], task,
                                             batch_size=batch_size, include_confidence=True)
        return [{d["label"]: d["confidence"] for d in o["decision"]}.get(SAME, 0.0) for o in out]
