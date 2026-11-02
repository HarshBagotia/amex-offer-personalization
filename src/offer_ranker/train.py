"""Train, evaluate and save the offer ranker.

Validation protocol (all splits are by time, never random):

    |------------ fit ------------|-- early stop --|----- holdout -----|
    |<-------------- training window ------------->|

1. Train on `fit`, early-stop on `early stop` to choose the number of trees.
2. Refit on the whole training window with that number of trees.
3. Score the untouched `holdout` exactly once and report MAP@k next to simple baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from .config import TrainConfig
from .data import LABEL_COLUMN, load_data
from .features import CATEGORICAL, FEATURES, FeatureBuilder
from .metrics import map_at_k
from .split import time_split
from .utils import file_sha256, set_seed

log = logging.getLogger(__name__)
GROUP_KEYS = ["customer_id", "event_date"]  # one ranking query = one customer session


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    # LightGBM ranking needs the rows of each query to be contiguous.
    return df.sort_values(GROUP_KEYS, kind="mergesort").reset_index(drop=True)


def lgb_params(cfg: TrainConfig) -> dict:
    params = {
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": cfg.lambda_l2,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    if cfg.objective == "lambdarank":
        params.update(objective="lambdarank", metric="map", eval_at=[cfg.k])
    else:
        params.update(objective="binary", metric="auc")
    return params


def _dataset(X: pd.DataFrame, df: pd.DataFrame, cfg: TrainConfig, reference=None) -> lgb.Dataset:
    group = None
    if cfg.objective == "lambdarank":
        group = df.groupby(GROUP_KEYS, sort=False).size().to_numpy()
    return lgb.Dataset(
        X,
        label=df[LABEL_COLUMN].to_numpy(),
        group=group,
        categorical_feature=CATEGORICAL,
        reference=reference,
        free_raw_data=False,
    )


def fit_booster(
    train_df: pd.DataFrame,
    cfg: TrainConfig,
    early_stop_df: pd.DataFrame | None = None,
    num_boost_round: int | None = None,
) -> tuple[lgb.Booster, FeatureBuilder]:
    fb = FeatureBuilder(cfg.te_smoothing, cfg.te_folds)
    dtrain = _dataset(fb.fit_transform(train_df), train_df, cfg)
    valid_sets, callbacks = [], [lgb.log_evaluation(0)]
    if early_stop_df is not None:
        valid_sets.append(_dataset(fb.transform(early_stop_df), early_stop_df, cfg, reference=dtrain))
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))
    booster = lgb.train(
        lgb_params(cfg),
        dtrain,
        num_boost_round=num_boost_round or cfg.num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return booster, fb


def evaluate(booster: lgb.Booster, fb: FeatureBuilder, df: pd.DataFrame, k: int) -> dict[str, float]:
    """MAP@k of the model and of two baselines it has to beat to be worth deploying."""
    X = fb.transform(df)
    scored = df[["customer_id", "offer_id", LABEL_COLUMN]].copy()
    scored["model"] = booster.predict(X)
    scored["popularity"] = X["te_offer_ctr"].to_numpy()  # rank by historical offer CTR
    scored["random"] = np.random.default_rng(0).random(len(scored))
    names = ("model", "popularity", "random")
    return {f"map@{k}_{name}": round(map_at_k(scored, name, k), 4) for name in names}


def _span(df: pd.DataFrame) -> list[str]:
    return [str(df["event_date"].min().date()), str(df["event_date"].max().date())]


def train(cfg: TrainConfig, df: pd.DataFrame | None = None) -> dict:
    set_seed(cfg.seed)
    if df is None:
        df = load_data(cfg.data_path)
        data_hash = file_sha256(cfg.data_path)
    else:
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy()).hexdigest()

    train_full, holdout = time_split(df, cfg.holdout_days)
    fit_part, early_stop = time_split(train_full, cfg.early_stopping_days)
    fit_part, early_stop, train_full, holdout = map(_sorted, (fit_part, early_stop, train_full, holdout))

    probe, _ = fit_booster(fit_part, cfg, early_stop_df=early_stop)
    best_iteration = probe.best_iteration or cfg.num_boost_round
    log.info("early stopping picked %d trees", best_iteration)

    booster, fb = fit_booster(train_full, cfg, num_boost_round=best_iteration)
    metrics = evaluate(booster, fb, holdout, cfg.k)

    config_dict = cfg.to_dict()
    # Same data + hyper-parameters + feature set => same version, wherever the files are written.
    hparams = {k: v for k, v in config_dict.items() if k not in ("data_path", "artifacts_dir")}
    hparams["feature_set"] = {"features": FEATURES, "categorical": CATEGORICAL}
    version_src = json.dumps(hparams, sort_keys=True) + data_hash
    metadata = {
        "model_version": hashlib.sha256(version_src.encode()).hexdigest()[:12],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config_dict,
        "data": {
            "rows": len(df),
            "sha256": data_hash,
            "click_rate": round(float(df[LABEL_COLUMN].mean()), 4),
        },
        "splits": {"fit": _span(fit_part), "early_stop": _span(early_stop), "holdout": _span(holdout)},
        "best_iteration": best_iteration,
        "features": FEATURES,
        "metrics": metrics,
        "library_versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    save_artifacts(cfg.artifacts_dir, booster, fb, metadata)
    _maybe_log_mlflow(cfg.artifacts_dir, metadata)
    return metadata


def save_artifacts(out_dir: Path, booster: lgb.Booster, fb: FeatureBuilder, metadata: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "feature_builder.json").write_text(json.dumps(fb.to_dict()))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _maybe_log_mlflow(out_dir: Path, metadata: dict) -> None:
    """Log the run to MLflow when MLFLOW_TRACKING_URI is set (optional dependency)."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; skipping")
        return
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_params({k: v for k, v in metadata["config"].items() if not isinstance(v, list)})
        mlflow.log_param("data_sha256", metadata["data"]["sha256"])
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metadata["metrics"].items()})
        mlflow.log_artifacts(str(out_dir))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the offer ranker")
    parser.add_argument("--data", type=Path, default=TrainConfig.data_path)
    parser.add_argument("--out", type=Path, default=TrainConfig.artifacts_dir)
    parser.add_argument("--objective", choices=["lambdarank", "binary"], default=TrainConfig.objective)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--holdout-days", type=int, default=TrainConfig.holdout_days)
    args = parser.parse_args()

    cfg = replace(
        TrainConfig(),
        data_path=args.data,
        artifacts_dir=args.out,
        objective=args.objective,
        seed=args.seed,
        holdout_days=args.holdout_days,
    )
    meta = train(cfg)
    print(json.dumps({"model_version": meta["model_version"], **meta["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
