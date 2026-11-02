from dataclasses import replace

import pytest

from offer_ranker.config import TrainConfig
from offer_ranker.data import make_synthetic_data, validate_schema
from offer_ranker.train import train


@pytest.fixture(scope="session")
def small_df():
    return validate_schema(make_synthetic_data(n_customers=600, n_offers=60, n_days=21, seed=7))


@pytest.fixture(scope="session")
def trained_model_dir(small_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = replace(
        TrainConfig(),
        artifacts_dir=out,
        holdout_days=5,
        early_stopping_days=4,
        num_boost_round=300,
        early_stopping_rounds=30,
    )
    meta = train(cfg, df=small_df)
    return out, meta
