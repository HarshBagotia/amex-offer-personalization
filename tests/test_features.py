import numpy as np
import pandas as pd
import pytest

from offer_ranker.data import SchemaError, validate_schema
from offer_ranker.features import FEATURES, FeatureBuilder, SmoothedTargetEncoder, add_base_features
from offer_ranker.split import time_split


def test_oof_encoding_never_sees_own_label(small_df):
    df = small_df.head(5000).reset_index(drop=True)
    enc = SmoothedTargetEncoder(smoothing=5)
    base = enc.fit_transform_oof(df["offer_id"], df["clicked"], df["customer_id"], n_splits=5)

    flipped = df.copy()
    flipped.loc[0, "clicked"] = 1 - flipped.loc[0, "clicked"]
    again = SmoothedTargetEncoder(smoothing=5).fit_transform_oof(
        flipped["offer_id"], flipped["clicked"], flipped["customer_id"], n_splits=5
    )
    assert again[0] == base[0]  # row 0's own label does not influence its encoding


def test_validation_features_do_not_depend_on_validation_labels(small_df):
    train, valid = time_split(small_df, last_days=5)
    fb = FeatureBuilder(n_splits=3)
    fb.fit_transform(train)
    shuffled = valid.assign(clicked=np.random.default_rng(0).permutation(valid["clicked"].to_numpy()))
    pd.testing.assert_frame_equal(fb.transform(valid), fb.transform(shuffled))


def test_time_split_has_no_overlap(small_df):
    train, valid = time_split(small_df, last_days=5)
    assert train["event_date"].max() < valid["event_date"].min()
    assert valid["event_date"].dt.normalize().nunique() == 5
    assert len(train) + len(valid) == len(small_df)


def test_category_affinity_picks_matching_share(small_df):
    row = small_df.iloc[[0]]
    X = add_base_features(row)
    expected = row[f"c_share_{row['o_category'].iat[0]}"].iat[0]
    assert X["x_category_affinity"].iat[0] == pytest.approx(expected)


def test_feature_builder_round_trips(small_df):
    fb = FeatureBuilder(n_splits=3)
    fb.fit_transform(small_df)
    restored = FeatureBuilder.from_dict(fb.to_dict())
    pd.testing.assert_frame_equal(fb.transform(small_df), restored.transform(small_df))
    assert list(restored.transform(small_df).columns) == FEATURES


def test_schema_rejects_missing_columns(small_df):
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_schema(small_df.drop(columns=["o_discount_pct"]))


def test_schema_rejects_unknown_category(small_df):
    bad = small_df.head(3).copy()
    bad.loc[0, "o_category"] = "crypto"
    with pytest.raises(SchemaError, match="unknown offer categories"):
        validate_schema(bad)
