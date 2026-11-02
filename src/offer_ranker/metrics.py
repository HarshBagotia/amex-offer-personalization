from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def average_precision_at_k(actual: set, predicted: Sequence, k: int) -> float:
    """AP@k: mean of precision@i over the ranks i (<= k) where a relevant item appears."""
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, item in enumerate(predicted[:k]):
        if item in actual and item not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(item)
    return score / min(len(actual), k)


def map_at_k(
    df: pd.DataFrame,
    score_col: str,
    k: int = 7,
    customer_col: str = "customer_id",
    item_col: str = "offer_id",
    label_col: str = "clicked",
) -> float:
    """MAP@k over customers.

    Impressions are collapsed to one row per (customer, offer): the offer is relevant if the
    customer clicked it at least once in the window. Customers with no clicks are skipped,
    since any ranking scores 0 for them and they would only dilute the comparison.
    """
    agg = (
        df.groupby([customer_col, item_col], sort=False)
        .agg(score=(score_col, "max"), label=(label_col, "max"))
        .reset_index()
        .sort_values([customer_col, "score"], ascending=[True, False], kind="mergesort")
    )
    aps = []
    for _, g in agg.groupby(customer_col, sort=False):
        actual = set(g.loc[g["label"] > 0, item_col])
        if actual:
            aps.append(average_precision_at_k(actual, g[item_col].tolist(), k))
    return float(np.mean(aps)) if aps else 0.0
