import pandas as pd
import pytest

from offer_ranker.metrics import average_precision_at_k, map_at_k


def test_ap_perfect_ranking():
    assert average_precision_at_k({"a", "b"}, ["a", "b", "c"], k=7) == 1.0


def test_ap_known_value():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert average_precision_at_k({"a", "c"}, ["a", "b", "c"], k=7) == pytest.approx((1 + 2 / 3) / 2)


def test_ap_ignores_items_beyond_k():
    assert average_precision_at_k({"z"}, ["a", "b", "z"], k=2) == 0.0


def test_ap_empty_actual():
    assert average_precision_at_k(set(), ["a"], k=7) == 0.0


def test_map_collapses_repeat_impressions_and_skips_no_click_customers():
    df = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2", "c2", "c3"],
            "offer_id": ["o1", "o2", "o1", "o1", "o2", "o1"],
            "clicked": [0, 0, 1, 0, 1, 0],
            "score": [0.9, 0.1, 0.9, 0.8, 0.2, 0.5],
        }
    )
    # c1: o1 clicked once and ranked first -> 1.0 ; c2: o2 clicked, ranked second -> 0.5 ; c3 skipped
    assert map_at_k(df, "score", k=7) == pytest.approx(0.75)
