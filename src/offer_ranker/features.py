"""Feature engineering with leakage-safe target encoding.

Click-through-rate encodings are the strongest features here and the easiest place to leak
labels, so:
  * on training rows they are computed out-of-fold, grouped by customer, so a row never
    sees its own label (or any label from the same customer);
  * on validation / serving rows they come only from training-window labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import CATEGORIES
from .data import CUSTOMER_COLUMNS, LABEL_COLUMN

OFFER_NUMERIC = ["o_discount_pct", "o_min_spend", "o_days_to_expiry", "o_brand_tier"]
INTERACTIONS = ["x_category_affinity", "x_min_spend_ratio", "x_value_score"]
ENCODED = ["te_offer_ctr", "te_category_ctr"]
CATEGORICAL = ["o_category_code"]
FEATURES = CUSTOMER_COLUMNS + OFFER_NUMERIC + CATEGORICAL + INTERACTIONS + ENCODED


@dataclass
class SmoothedTargetEncoder:
    """Mean-target encoding shrunk towards the global prior: (sum + prior*m) / (count + m)."""

    smoothing: float = 20.0
    prior_: float = 0.0
    mapping_: dict[str, float] = field(default_factory=dict)

    def fit(self, keys: pd.Series, y: pd.Series) -> SmoothedTargetEncoder:
        self.prior_ = float(y.mean())
        stats = y.groupby(keys.astype(str).to_numpy()).agg(["sum", "count"])
        enc = (stats["sum"] + self.prior_ * self.smoothing) / (stats["count"] + self.smoothing)
        self.mapping_ = {str(k): float(v) for k, v in enc.items()}
        return self

    def transform(self, keys: pd.Series) -> np.ndarray:
        return keys.astype(str).map(self.mapping_).fillna(self.prior_).to_numpy(dtype=float)

    def fit_transform_oof(
        self, keys: pd.Series, y: pd.Series, groups: pd.Series, n_splits: int
    ) -> np.ndarray:
        out = np.empty(len(keys), dtype=float)
        for fit_idx, enc_idx in GroupKFold(n_splits=n_splits).split(keys, y, groups):
            fold = SmoothedTargetEncoder(self.smoothing).fit(keys.iloc[fit_idx], y.iloc[fit_idx])
            out[enc_idx] = fold.transform(keys.iloc[enc_idx])
        self.fit(keys, y)  # full-train mapping, used for validation and serving
        return out

    def to_dict(self) -> dict:
        return {"smoothing": self.smoothing, "prior": self.prior_, "mapping": self.mapping_}

    @classmethod
    def from_dict(cls, d: dict) -> SmoothedTargetEncoder:
        return cls(smoothing=d["smoothing"], prior_=d["prior"], mapping_=d["mapping"])


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features that need no labels, so they are identical in training and serving."""
    out = pd.DataFrame(index=df.index)
    for col in CUSTOMER_COLUMNS + OFFER_NUMERIC:
        out[col] = df[col].astype(float)

    cat = df["o_category"].astype(str)
    out["o_category_code"] = cat.map({c: i for i, c in enumerate(CATEGORIES)}).astype(int)

    affinity = np.zeros(len(df))
    for c in CATEGORIES:
        mask = (cat == c).to_numpy()
        affinity[mask] = df.loc[mask, f"c_share_{c}"].to_numpy(dtype=float)
    out["x_category_affinity"] = affinity
    out["x_min_spend_ratio"] = out["o_min_spend"] / out["c_avg_monthly_spend"].clip(lower=1.0)
    out["x_value_score"] = out["o_discount_pct"] * out["x_category_affinity"]
    return out


class FeatureBuilder:
    def __init__(self, smoothing: float = 20.0, n_splits: int = 5):
        self.n_splits = n_splits
        self.offer_enc = SmoothedTargetEncoder(smoothing)
        self.category_enc = SmoothedTargetEncoder(smoothing)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = add_base_features(df)
        y, groups = df[LABEL_COLUMN], df["customer_id"]
        X["te_offer_ctr"] = self.offer_enc.fit_transform_oof(df["offer_id"], y, groups, self.n_splits)
        X["te_category_ctr"] = self.category_enc.fit_transform_oof(
            df["o_category"], y, groups, self.n_splits
        )
        return X[FEATURES]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = add_base_features(df)
        X["te_offer_ctr"] = self.offer_enc.transform(df["offer_id"])
        X["te_category_ctr"] = self.category_enc.transform(df["o_category"])
        return X[FEATURES]

    def to_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "offer_enc": self.offer_enc.to_dict(),
            "category_enc": self.category_enc.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureBuilder:
        fb = cls(n_splits=d["n_splits"])
        fb.offer_enc = SmoothedTargetEncoder.from_dict(d["offer_enc"])
        fb.category_enc = SmoothedTargetEncoder.from_dict(d["category_enc"])
        return fb
