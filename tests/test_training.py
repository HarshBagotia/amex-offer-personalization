import json
from dataclasses import replace

from offer_ranker.config import TrainConfig
from offer_ranker.predict import OfferRanker
from offer_ranker.train import train


def test_model_beats_baselines(trained_model_dir):
    _, meta = trained_model_dir
    m = meta["metrics"]
    assert m["map@7_model"] > m["map@7_popularity"] > m["map@7_random"]


def test_artifacts_are_complete_and_loadable(trained_model_dir):
    out, meta = trained_model_dir
    ranker = OfferRanker.load(out)
    assert ranker.version == meta["model_version"]
    saved = json.loads((out / "metadata.json").read_text())
    assert saved["splits"]["early_stop"][1] < saved["splits"]["holdout"][0]


def test_training_is_reproducible(small_df, tmp_path):
    cfg = replace(
        TrainConfig(), holdout_days=5, early_stopping_days=4, num_boost_round=60,
        early_stopping_rounds=10,
    )
    a = train(replace(cfg, artifacts_dir=tmp_path / "a"), df=small_df)
    b = train(replace(cfg, artifacts_dir=tmp_path / "b"), df=small_df)
    assert a["metrics"] == b["metrics"]
    assert a["model_version"] == b["model_version"]
    assert (tmp_path / "a" / "model.txt").read_text() == (tmp_path / "b" / "model.txt").read_text()
