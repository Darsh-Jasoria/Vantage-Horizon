"""Vantage Horizon API (FastAPI).

    uvicorn api.main:app --reload --port 8000     then open http://localhost:8000

Serves the dashboard and the forecast layer. In Azure this same app runs on
Azure Functions / App Service; Power BI and the React embed call these routes.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vantage.config import ARTIFACT_DIR, OUT_DIR  # noqa: E402
from vantage.forecast import shift_promo_days, promo_days  # noqa: E402
from vantage.models import month_sums  # noqa: E402
from vantage import decision as decision_mod  # noqa: E402

app = FastAPI(title="Vantage Horizon", version="0.1.0",
              description="Forecast layer for BI: next month, not just last month.")
_cache: dict = {}


def data() -> dict:
    path = OUT_DIR / "dashboard_data.json"
    if not path.exists():
        raise HTTPException(503, "No forecast yet. Run: python run_pipeline.py")
    mtime = path.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache.update(mtime=mtime, data=json.loads(path.read_text()), models=None)
    return _cache["data"]


def models() -> dict:
    data()
    if _cache.get("models") is None:
        with open(ARTIFACT_DIR / "models.pkl", "rb") as f:
            _cache["models"] = pickle.load(f)
    return _cache["models"]


@app.get("/", include_in_schema=False)
def index():
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    head = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    return HTMLResponse(head + page)


@app.get("/api/health")
def health():
    d = data()
    return {"status": "ok", "as_of": d["meta"]["as_of"], "model_version": d["meta"]["model_version"]}


@app.get("/api/dashboard")
def dashboard():
    d = dict(data())
    d["meta"] = {**d["meta"], "mode": "live"}
    return JSONResponse(d)


@app.get("/api/forecast")
def forecast(segment: str = "All", metric: str = "sales"):
    d = data()
    if segment not in d["series"] or metric not in d["series"][segment]:
        raise HTTPException(404, "Unknown segment or metric")
    s = d["series"][segment][metric]
    return {"segment": segment, "metric": metric, "actual_months": d["actual_months"],
            "actual": s["actual"], "forecast": s["forecast"], "backtest": s.get("backtest", [])}


@app.get("/api/drivers")
def drivers(month: str | None = None, segment: str = "All"):
    d = data()
    month = month or d["future_months"][0]
    return {"month": month, "segment": segment, **d["drivers"][month][segment],
            "global_importance": d["global_importance"]}


@app.get("/api/alerts")
def alerts():
    return data()["alerts"]


@app.get("/api/narrative")
def narrative(month: str | None = None, segment: str = "All"):
    d = data()
    month = month or d["future_months"][0]
    return d["narratives"][month][segment]


def _month_seg(d: dict, month: str | None, segment: str) -> tuple[str, str]:
    month = month or d["future_months"][0]
    if month not in d["future_months"]:
        raise HTTPException(404, f"Month must be one of {d['future_months']}")
    if segment not in d["series"]:
        raise HTTPException(404, "Unknown segment")
    return month, segment


@app.get("/api/decision")
def decision(month: str | None = None, segment: str | None = None):
    """Decision object: risk, leading indicator, drivers, tested interventions, recommendation,
    business impact and evidence. Without a segment, returns the most urgent one."""
    d = data()
    if segment is None:
        month = month or d["future_months"][0]
        return decision_mod.top_decision(d["decisions"], month)
    month, segment = _month_seg(d, month, segment)
    return d["decisions"][month][segment]


@app.get("/api/scenarios")
def scenarios(month: str | None = None, segment: str = "All"):
    """Base / Recovery / Downside scenarios plus every tested intervention with its decision score."""
    d = data()
    month, segment = _month_seg(d, month, segment)
    dec = d["decisions"][month][segment]
    return {"month": month, "segment": segment, "scenarios": dec["scenarios"], "tested": dec["tested"],
            "recommended": dec["recommended"], "score_weights": dec["score_weights"]}


@app.get("/api/trust")
def trust():
    """Out-of-sample backtest accuracy, interval coverage and baseline comparison."""
    return data()["trust"]


class WhatIf(BaseModel):
    month: str | None = None
    segment: str = "All"
    promo_delta: int = Field(0, ge=-10, le=10, description="promo weekdays added (+) or removed (-)")
    footfall_pct: float = Field(0, ge=-30, le=30, description="change in customers, %")
    ticket_pct: float = Field(0, ge=-20, le=20, description="change in average ticket, %")


@app.post("/api/whatif")
def whatif(q: WhatIf):
    """Re-scores the daily LightGBM model with a changed promo calendar, then applies
    footfall and ticket shifts. The ratio is applied to the ensemble forecast."""
    d, m = data(), models()
    month = q.month or d["future_months"][0]
    f = next((x for x in d["series"][q.segment]["sales"]["forecast"] if x["month"] == month), None)
    if f is None:
        raise HTTPException(404, "Month is not in the forecast horizon")
    start = pd.Timestamp(month + "-01")
    rows = m["future_rows"]
    rows = rows[(rows.sale_date >= start) & (rows.sale_date < start + pd.offsets.MonthBegin(1))]
    model = m["models"]["sales"]
    base = month_sums(rows, model.predict(rows)).set_index("segment")["v"][q.segment]
    shifted = shift_promo_days(rows, q.promo_delta)
    new = month_sums(shifted, model.predict(shifted)).set_index("segment")["v"][q.segment]
    ratio = float(new / base) * (1 + q.footfall_pct / 100) * (1 + q.ticket_pct / 100)
    sd = np.log(f["hi80"] / f["p50"]) / norm.ppf(0.9)
    p50 = f["p50"] * ratio
    target = d["targets"][q.segment].get(month)
    p_hit = None if target is None else float(1 - norm.cdf((np.log(target) - np.log(p50)) / sd))
    return {"month": month, "segment": q.segment, "promo_days": promo_days(shifted),
            "base_promo_days": promo_days(rows), "ratio": ratio, "base_p50": f["p50"],
            "p50": p50, "lo80": f["lo80"] * ratio, "hi80": f["hi80"] * ratio,
            "target": target, "p_hit": p_hit}
