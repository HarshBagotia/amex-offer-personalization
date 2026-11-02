from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FeatureBuilder

ARTIFACT_FILES = ("model.txt", "feature_builder.json", "metadata.json")


class OfferRanker:
    """Loads a trained model and ranks a customer's candidate offers."""

    def __init__(self, booster: lgb.Booster, feature_builder: FeatureBuilder, metadata: dict):
        self.booster = booster
        self.feature_builder = feature_builder
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path) -> OfferRanker:
        model_dir = Path(model_dir)
        missing = [f for f in ARTIFACT_FILES if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"missing artifacts in {model_dir}: {missing} - run `make train`")
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        fb = FeatureBuilder.from_dict(json.loads((model_dir / "feature_builder.json").read_text()))
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(booster, fb, metadata)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.feature_builder.transform(df))

    def rank(self, customer: dict, offers: list[dict], k: int) -> list[dict]:
        """`customer` holds c_* columns + customer_id; each offer holds offer_id + o_* columns."""
        df = pd.DataFrame(offers)
        for key, value in customer.items():
            df[key] = value
        scores = self.score(df)
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            {"offer_id": str(df["offer_id"].iat[i]), "score": round(float(scores[i]), 6), "rank": r + 1}
            for r, i in enumerate(top)
        ]
