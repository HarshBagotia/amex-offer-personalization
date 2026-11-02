# Offer Personalization Ranking Model: American Express Campus Challenge 2025

[![CI](https://github.com/HarshBagotia/amex-offer-personalization/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshBagotia/amex-offer-personalization/actions/workflows/ci.yml)

Ranks the offers shown to a card member so that the ones they are most likely to engage with
come first. The pipeline runs end to end: data validation, leakage-safe feature engineering, a
LightGBM **LambdaRank** model tuned for **MAP@7**, time-based evaluation against baselines,
and a **FastAPI** service packaged with **Docker** and tested in **CI**.

| | |
|---|---|
| **Competition result** | **0.55 MAP@7** |
| Model | LightGBM, `lambdarank` objective (a `binary` objective is also supported) |
| Data scale | 1M+ customer–offer impressions |
| Stack | Python, Pandas, NumPy, scikit-learn, LightGBM, FastAPI, Docker, GitHub Actions |

> **About the data.** The competition dataset is confidential, so it is not in this repo. This
> repo is a clean, production-style version of the pipeline. It ships with a **synthetic data
> generator** that has the same structure (one row per customer–offer impression with a click
> label), so you can run, test and deploy everything without the original data. Scores on
> synthetic data cannot be compared with the competition score.

---

## Approach

**1. Problem framing.** Each customer sees several offers, and the metric (MAP@7) rewards
putting the offers they click at the top of the list. That makes this a **ranking** problem, not
plain click classification. A *query* is one customer session, and LightGBM's LambdaRank
objective optimizes the order of offers within each query.

**2. Features** (`src/offer_ranker/features.py`)

| Group | Examples | Why |
|---|---|---|
| Customer | tenure, average monthly spend, number of cards, digital engagement, spend share per category | who the customer is |
| Offer | category, discount %, minimum spend, days to expiry, brand tier | what is on offer |
| Customer × offer | **category affinity** (the customer's spend share in the offer's category), min-spend / usual-spend ratio, discount × affinity | how well this offer fits *this* customer, the most important signal for ranking |
| Target encoding | smoothed historical CTR of the offer and of its category | captures offer popularity that no raw column shows |

**3. Model.** LightGBM with deterministic training and an early-stopped tree count
(`src/offer_ranker/train.py`).

**4. Evaluation.** MAP@7 per customer (`src/offer_ranker/metrics.py`). Each model is
reported next to two baselines it must beat: **popularity** (rank by historical offer CTR)
and **random**.

## Validation and leakage prevention

These are the details that decide whether an offline score holds up in production:

- **Time-based splits only.** A random split lets the model learn from clicks that happen
  *after* the ones it is evaluated on. The data is split by date:

  ```
  |------------ fit ------------|-- early stop --|----- holdout -----|
  ```

  The number of trees is chosen on the early-stopping window. The model is then refit on the
  full training window and scored **once** on the untouched holdout.
- **Out-of-fold target encoding.** Offer and category CTRs for training rows are computed with
  `GroupKFold` grouped by customer, so a row never sees its own label or any label from the
  same customer. Validation and serving rows use encodings fitted on the training window only.
- **Tests enforce this.** `tests/test_features.py` checks that flipping a row's label does not
  change its own encoding, that shuffling validation labels does not change validation
  features, and that the train and holdout date ranges never overlap.

## Reproducibility

- Python, NumPy and LightGBM seeds are fixed, LightGBM runs with `deterministic=True`, and all
  dependencies are pinned in `requirements.txt`.
- Every run writes `artifacts/metadata.json` with the full config, the **SHA-256 of the
  training data**, the split date ranges, the tree count, the metrics and library versions.
- `model_version` is a hash of the hyper-parameters and the data, so retraining on the same
  inputs gives the same version. `tests/test_training.py` asserts that two runs produce
  byte-identical models.
- Optional **MLflow** tracking: set `MLFLOW_TRACKING_URI` (and `pip install mlflow`) to log
  parameters, metrics and artifacts for every run.

## Results on the bundled synthetic data

`make data && make train` (178k impressions, 3,000 customers, 150 offers, 28 days; 7-day holdout):

| Ranker | Holdout MAP@7 |
|---|---|
| **LightGBM LambdaRank** | **0.485** |
| Popularity (offer CTR) | 0.436 |
| Random | 0.211 |

## Quickstart

```bash
make install   # dependencies + package (editable)
make data      # synthetic data -> data/train.csv
make train     # train, evaluate, save to artifacts/
make test      # 25 unit + integration tests
make serve     # API on http://localhost:8000 (interactive docs at /docs)
```

With Docker:

```bash
make train && make docker-build && make docker-run
```

## Serving API

| Endpoint | Purpose | Caching |
|---|---|---|
| `POST /rank` | top-k offers for a customer | `no-store`: personalized per customer, must never be served from a shared cache |
| `GET /model/info` | model version, splits, metrics | `public, max-age=300` + `ETag` (returns 304 when unchanged), since it only changes on redeploy |
| `GET /health` | liveness and model status | `no-store` |

```bash
curl -X POST localhost:8000/rank -H 'Content-Type: application/json' -d @examples/rank_request.json
```

```json
{
  "customer_id": "C00042",
  "model_version": "8a4b667736fc",
  "ranked_offers": [
    {"offer_id": "O0004", "score": 0.322092, "rank": 1},
    {"offer_id": "O0001", "score": -0.450475, "rank": 2},
    {"offer_id": "O0005", "score": -0.735148, "rank": 3}
  ]
}
```

- **Request schema:** strict Pydantic models (unknown fields rejected, value ranges checked,
  spend shares must sum to 1, offer IDs must be unique). The JSON Schema is published at
  `/openapi.json`.
- **Errors:** every error uses one shape, `{"error": {"code", "message", "details"}}`.
  Status codes: `422` invalid request, `503` model not loaded (the service still starts and
  `/health` reports `degraded`), `500` unexpected error (logged, no internals leaked).

## Using the real competition data

Map the competition columns to the schema in `src/offer_ranker/data.py`
(`customer_id`, `offer_id`, `event_date`, the `c_*` customer columns, the `o_*` offer columns
and `clicked`). `validate_schema` fails fast with a clear message if anything is missing.
Then run:

```bash
python -m offer_ranker.train --data path/to/train.parquet --out artifacts
```

## Project structure

```
src/offer_ranker/
  config.py     training configuration (single source of truth for a run)
  data.py       loading, schema validation, synthetic data generator
  features.py   feature engineering + out-of-fold target encoding
  split.py      time-based splits
  metrics.py    AP@k / MAP@k
  train.py      train -> early stop -> refit -> holdout evaluation -> save artifacts
  predict.py    load artifacts and rank offers
  api.py        FastAPI service
tests/          metrics, leakage, reproducibility and API tests
.github/workflows/ci.yml   lint + tests, then train -> docker build -> container smoke test
```
