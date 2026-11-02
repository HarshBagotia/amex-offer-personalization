import pytest
from fastapi.testclient import TestClient

from offer_ranker import api


def _offer(i, category="dining", discount=10):
    return {
        "offer_id": f"O{i:04d}",
        "category": category,
        "discount_pct": discount,
        "min_spend": 500,
        "days_to_expiry": 20,
        "brand_tier": 2,
    }


CUSTOMER = {
    "customer_id": "C00001",
    "tenure_months": 36,
    "avg_monthly_spend": 2200.0,
    "num_cards": 2,
    "digital_engagement": 0.6,
    "spend_share": {"dining": 0.6, "travel": 0.1, "retail": 0.1, "entertainment": 0.1, "fuel": 0.1},
}


@pytest.fixture()
def client(trained_model_dir, monkeypatch):
    out, _ = trained_model_dir
    monkeypatch.setenv("MODEL_DIR", str(out))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.headers["cache-control"] == "no-store"


def test_rank_returns_top_k_sorted_and_uncacheable(client):
    offers = [_offer(i, cat) for i, cat in enumerate(["dining", "fuel", "travel", "retail"] * 3)]
    r = client.post("/rank", json={"customer": CUSTOMER, "offers": offers, "k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    ranked = body["ranked_offers"]
    assert len(ranked) == 5
    assert [o["rank"] for o in ranked] == [1, 2, 3, 4, 5]
    scores = [o["score"] for o in ranked]
    assert scores == sorted(scores, reverse=True)
    assert r.headers["cache-control"] == "no-store"


def test_rank_prefers_offers_in_customers_main_category(client):
    offers = [_offer(1, "fuel", 20), _offer(2, "dining", 20)]
    r = client.post("/rank", json={"customer": CUSTOMER, "offers": offers, "k": 2})
    assert r.json()["ranked_offers"][0]["offer_id"] == "O0002"


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda b: b["offers"].append(dict(b["offers"][0])), "offer_id values must be unique"),
        (lambda b: b["offers"][0].update(category="crypto"), "category"),
        (lambda b: b["customer"]["spend_share"].update(dining=0.9), "sum to 1"),
        (lambda b: b.update(offers=[]), "offers"),
        (lambda b: b["customer"].update(unexpected="x"), "unexpected"),
    ],
)
def test_rank_rejects_bad_requests(client, mutate, needle):
    body = {"customer": {**CUSTOMER, "spend_share": dict(CUSTOMER["spend_share"])}, "offers": [_offer(1)]}
    mutate(body)
    r = client.post("/rank", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    assert needle in r.text


def test_model_info_is_cacheable_with_etag(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "max-age=300" in r.headers["cache-control"]
    r2 = client.get("/model/info", headers={"If-None-Match": r.headers["etag"]})
    assert r2.status_code == 304


def test_rank_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    with TestClient(api.app) as c:
        r = c.post("/rank", json={"customer": CUSTOMER, "offers": [_offer(1)]})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unavailable"
        assert c.get("/health").json()["status"] == "degraded"
