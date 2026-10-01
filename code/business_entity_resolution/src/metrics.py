"""Competition metric: macro F0.5 over Source 1 entities.

Per S1 entity: if it has no true matches it scores 1 when the prediction is
empty and 0 otherwise; if it has true matches, an empty prediction scores 0,
else F0.5 = 1.25*P*R / (0.25*P + R).
"""


def entity_f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def summarize(pred: dict, truth: dict) -> dict:
    """pred/truth: {s1_id: set(ids)} over the same S1 ids."""
    ids = list(truth)
    f = [entity_f05(pred.get(i, set()), truth[i]) for i in ids]
    singles = [i for i in ids if not truth[i]]
    tp = sum(len(pred.get(i, set()) & truth[i]) for i in ids)
    n_pred = sum(len(pred.get(i, set())) for i in ids)
    n_true = sum(len(truth[i]) for i in ids)
    return {
        "macro_F0.5": sum(f) / len(f),
        "singletons_right": f"{sum(not pred.get(i) for i in singles)}/{len(singles)}",
        "precision": tp / n_pred if n_pred else float("nan"),
        "recall": tp / n_true if n_true else float("nan"),
        "predicted_links": n_pred,
    }
