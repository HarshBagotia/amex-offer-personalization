"""HTTP API: rank a customer's candidate offers.

Caching policy
--------------
* POST /rank        -> `Cache-Control: no-store`. Responses are personalised to one customer and
                       change whenever the model or offer pool changes; a shared cache serving
                       them would leak one customer's ranking to another.
* GET /model/info   -> `Cache-Control: public, max-age=300` + ETag. Metadata only changes on
                       redeploy, so clients and proxies can safely reuse it for a few minutes.
* GET /health       -> `no-store`, so load balancers always see the live state.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import CATEGORIES
from .predict import OfferRanker

log = logging.getLogger("offer_ranker.api")

Category = Literal["dining", "travel", "retail", "entertainment", "fuel"]
NO_STORE = {"Cache-Control": "no-store"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpendShare(_Strict):
    dining: float = Field(ge=0, le=1)
    travel: float = Field(ge=0, le=1)
    retail: float = Field(ge=0, le=1)
    entertainment: float = Field(ge=0, le=1)
    fuel: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self):
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"spend shares must sum to 1 (got {total:.3f})")
        return self


class Customer(_Strict):
    customer_id: str = Field(min_length=1, max_length=64)
    tenure_months: int = Field(ge=0, le=1200)
    avg_monthly_spend: float = Field(gt=0)
    num_cards: int = Field(ge=1, le=20)
    digital_engagement: float = Field(ge=0, le=1)
    spend_share: SpendShare

    def to_columns(self) -> dict:
        cols = {
            "customer_id": self.customer_id,
            "c_tenure_months": self.tenure_months,
            "c_avg_monthly_spend": self.avg_monthly_spend,
            "c_num_cards": self.num_cards,
            "c_digital_engagement": self.digital_engagement,
        }
        cols.update({f"c_share_{k}": v for k, v in self.spend_share.model_dump().items()})
        return cols


class Offer(_Strict):
    offer_id: str = Field(min_length=1, max_length=64)
    category: Category
    discount_pct: float = Field(ge=0, le=100)
    min_spend: float = Field(ge=0)
    days_to_expiry: int = Field(ge=0)
    brand_tier: int = Field(ge=1, le=3)

    def to_columns(self) -> dict:
        return {
            "offer_id": self.offer_id,
            "o_category": self.category,
            "o_discount_pct": self.discount_pct,
            "o_min_spend": self.min_spend,
            "o_days_to_expiry": self.days_to_expiry,
            "o_brand_tier": self.brand_tier,
        }


class RankRequest(_Strict):
    customer: Customer
    offers: list[Offer] = Field(min_length=1, max_length=500)
    k: int = Field(default=7, ge=1, le=50)

    @model_validator(mode="after")
    def _unique_offer_ids(self):
        ids = [o.offer_id for o in self.offers]
        if len(ids) != len(set(ids)):
            raise ValueError("offer_id values must be unique")
        return self


class RankedOffer(BaseModel):
    offer_id: str
    score: float
    rank: int


class RankResponse(BaseModel):
    customer_id: str
    model_version: str
    ranked_offers: list[RankedOffer]


def _error(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers={**NO_STORE, **(headers or {})})


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts"))
    try:
        app.state.ranker = OfferRanker.load(model_dir)
        log.info("loaded model %s from %s", app.state.ranker.version, model_dir)
    except FileNotFoundError as exc:
        app.state.ranker = None
        log.warning("starting without a model: %s", exc)
    yield


app = FastAPI(title="Offer Ranker", version="0.1.0", lifespan=lifespan)
assert set(Category.__args__) == set(CATEGORIES), "API categories out of sync with training"


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _error(422, "invalid_request", "request body failed validation", details)


@app.exception_handler(Exception)
async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    return _error(500, "internal_error", "unexpected server error")


def _ranker(request: Request) -> OfferRanker | None:
    return getattr(request.app.state, "ranker", None)


@app.get("/health")
def health(request: Request, response: Response) -> dict:
    response.headers.update(NO_STORE)
    ranker = _ranker(request)
    return {
        "status": "ok" if ranker else "degraded",
        "model_loaded": ranker is not None,
        "model_version": ranker.version if ranker else None,
    }


@app.get("/model/info", response_model=None)
def model_info(request: Request) -> Response:
    ranker = _ranker(request)
    if ranker is None:
        return _error(503, "model_unavailable", "no model loaded")
    etag = f'"{ranker.version}"'
    headers = {"Cache-Control": "public, max-age=300", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    m = ranker.metadata
    body = {k: m[k] for k in ("model_version", "trained_at", "splits", "metrics", "features")}
    return JSONResponse(body, headers=headers)


@app.post(
    "/rank",
    response_model=RankResponse,
    responses={422: {"description": "Invalid request"}, 503: {"description": "Model not loaded"}},
)
def rank(req: RankRequest, request: Request, response: Response):
    ranker = _ranker(request)
    if ranker is None:
        return _error(503, "model_unavailable", "no model loaded; train one and set MODEL_DIR")
    response.headers.update(NO_STORE)
    ranked = ranker.rank(req.customer.to_columns(), [o.to_columns() for o in req.offers], req.k)
    return RankResponse(
        customer_id=req.customer.customer_id, model_version=ranker.version, ranked_offers=ranked
    )
