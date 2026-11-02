"""Training configuration. Every run is fully described by this object + the data hash."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CATEGORIES: tuple[str, ...] = ("dining", "travel", "retail", "entertainment", "fuel")


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("data/train.csv")
    artifacts_dir: Path = Path("artifacts")
    seed: int = 42

    # Time-based validation: the last `holdout_days` are never touched during training;
    # the last `early_stopping_days` of the remaining window pick the number of trees.
    holdout_days: int = 7
    early_stopping_days: int = 5

    objective: Literal["lambdarank", "binary"] = "lambdarank"
    k: int = 7

    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0

    # Smoothed, out-of-fold target encoding of offer/category click-through rate.
    te_smoothing: float = 20.0
    te_folds: int = 5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_path"] = str(self.data_path)
        d["artifacts_dir"] = str(self.artifacts_dir)
        return d
