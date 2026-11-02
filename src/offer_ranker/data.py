"""Loading, schema validation and a synthetic data generator.

The competition data is confidential and is not part of this repository. The synthetic
generator produces data with the same shape (one row per customer-offer impression with a
click label) so the full pipeline, tests and CI run end to end without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORIES

CUSTOMER_COLUMNS = [
    "c_tenure_months",
    "c_avg_monthly_spend",
    "c_num_cards",
    "c_digital_engagement",
    *[f"c_share_{c}" for c in CATEGORIES],
]
OFFER_COLUMNS = ["o_category", "o_discount_pct", "o_min_spend", "o_days_to_expiry", "o_brand_tier"]
KEY_COLUMNS = ["customer_id", "offer_id", "event_date"]
LABEL_COLUMN = "clicked"
REQUIRED_COLUMNS = KEY_COLUMNS + CUSTOMER_COLUMNS + OFFER_COLUMNS + [LABEL_COLUMN]


class SchemaError(ValueError):
    pass


def validate_schema(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    required = REQUIRED_COLUMNS if require_label else [c for c in REQUIRED_COLUMNS if c != LABEL_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    unknown = set(df["o_category"].unique()) - set(CATEGORIES)
    if unknown:
        raise SchemaError(f"unknown offer categories: {sorted(unknown)}")

    if require_label and not df[LABEL_COLUMN].isin([0, 1]).all():
        raise SchemaError(f"'{LABEL_COLUMN}' must be 0/1")

    if df[required].isna().any().any():
        cols = df[required].columns[df[required].isna().any()].tolist()
        raise SchemaError(f"null values in required columns: {cols}")

    out = df.copy()
    out["customer_id"] = out["customer_id"].astype(str)
    out["offer_id"] = out["offer_id"].astype(str)
    out["event_date"] = pd.to_datetime(out["event_date"])
    return out


def load_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` to generate synthetic data")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return validate_schema(df)


def make_synthetic_data(
    n_customers: int = 3000,
    n_offers: int = 150,
    n_days: int = 28,
    session_rate: float = 0.25,
    min_offers: int = 5,
    max_offers: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate offer impressions whose clicks depend on customer-offer fit.

    Click probability rises with the customer's spend share in the offer's category, the
    discount and engagement, falls as the minimum spend grows relative to the customer's
    usual spend, and includes a hidden per-offer popularity term the model can only learn
    through target encoding.
    """
    rng = np.random.default_rng(seed)
    n_cat = len(CATEGORIES)

    shares = rng.dirichlet(np.full(n_cat, 0.8), size=n_customers)
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_customers)],
            "c_tenure_months": rng.integers(1, 240, n_customers),
            "c_avg_monthly_spend": np.round(rng.lognormal(7.5, 0.6, n_customers), 2),
            "c_num_cards": rng.integers(1, 5, n_customers),
            "c_digital_engagement": np.round(rng.beta(2, 3, n_customers), 3),
            **{f"c_share_{c}": np.round(shares[:, j], 4) for j, c in enumerate(CATEGORIES)},
        }
    )
    offer_cat_idx = rng.integers(0, n_cat, n_offers)
    offers = pd.DataFrame(
        {
            "offer_id": [f"O{j:04d}" for j in range(n_offers)],
            "o_category": np.array(CATEGORIES)[offer_cat_idx],
            "o_discount_pct": rng.choice([5, 10, 15, 20, 25, 30], n_offers),
            "o_min_spend": rng.choice([0, 500, 1000, 2500, 5000], n_offers),
            "o_days_to_expiry": rng.integers(3, 60, n_offers),
            "o_brand_tier": rng.integers(1, 4, n_offers),
        }
    )
    popularity = rng.normal(0.0, 0.7, n_offers)  # latent, never exposed as a column

    # Sessions: each customer is active on a given day with probability `session_rate`.
    active = rng.random((n_days, n_customers)) < session_rate
    day_idx, cust_idx = np.nonzero(active)
    n_sessions = len(day_idx)
    n_shown = rng.integers(min_offers, max_offers + 1, n_sessions)
    order = np.argsort(rng.random((n_sessions, n_offers)), axis=1)[:, :max_offers]
    mask = np.arange(max_offers)[None, :] < n_shown[:, None]
    sess_rows = np.repeat(np.arange(n_sessions), max_offers).reshape(n_sessions, max_offers)[mask]
    offer_idx = order[mask]
    c_idx = cust_idx[sess_rows]
    d_idx = day_idx[sess_rows]

    affinity = shares[c_idx, offer_cat_idx[offer_idx]]
    spend = customers["c_avg_monthly_spend"].to_numpy()[c_idx]
    min_spend = offers["o_min_spend"].to_numpy()[offer_idx]
    discount = offers["o_discount_pct"].to_numpy()[offer_idx]
    engagement = customers["c_digital_engagement"].to_numpy()[c_idx]
    logit = (
        -3.4
        + 3.0 * affinity
        + 0.04 * discount
        - 0.6 * np.log1p(min_spend / spend)
        + 1.0 * (engagement - 0.4)
        + popularity[offer_idx]
        + rng.normal(0.0, 0.5, len(offer_idx))
    )
    clicked = (rng.random(len(offer_idx)) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    start = pd.Timestamp("2025-06-01")
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[c_idx],
                    "offer_id": offers["offer_id"].to_numpy()[offer_idx],
                    "event_date": (start + pd.to_timedelta(d_idx, unit="D")).strftime("%Y-%m-%d"),
                }
            ),
            customers.drop(columns="customer_id").iloc[c_idx].reset_index(drop=True),
            offers.drop(columns="offer_id").iloc[offer_idx].reset_index(drop=True),
        ],
        axis=1,
    )
    df[LABEL_COLUMN] = clicked
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic offer-impression data")
    parser.add_argument("--out", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--customers", type=int, default=3000)
    parser.add_argument("--offers", type=int, default=150)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_synthetic_data(args.customers, args.offers, args.days, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out} (click rate {df[LABEL_COLUMN].mean():.3f})")


if __name__ == "__main__":
    main()
