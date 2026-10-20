"""Checks on the real pipeline output (outputs/dashboard_data.json, fct_kpi_forecast.csv)."""
import math

import pandas as pd

from vantage import decision as dm

ROOT_OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "outputs"


def test_horizon_is_exactly_three_consecutive_months(dashboard):
    fut = dashboard["future_months"]
    assert len(fut) == 3
    last = pd.Period(dashboard["actual_months"][-1], "M")
    assert [str(last + i) for i in (1, 2, 3)] == fut
    for seg in dashboard["series"]:
        assert [f["horizon"] for f in dashboard["series"][seg]["sales"]["forecast"]] == [1, 2, 3]


def test_interval_ordering(dashboard):
    for seg, metrics in dashboard["series"].items():
        for metric, s in metrics.items():
            for f in s["forecast"]:
                assert f["lo95"] <= f["lo80"] <= f["p50"] <= f["hi80"] <= f["hi95"], (seg, metric, f)
            for b in s.get("backtest", []):
                assert b["lo95"] <= b["lo80"] <= b["forecast"] <= b["hi80"] <= b["hi95"]


def test_intervals_widen_with_horizon(dashboard):
    for seg in dashboard["series"]:
        fc = dashboard["series"][seg]["sales"]["forecast"]
        widths = [math.log(f["hi80"] / f["p50"]) for f in fc]
        assert widths == sorted(widths)


def test_p_hit_is_a_probability(dashboard):
    for seg, months in dashboard["p_hit"].items():
        for m, p in months.items():
            assert 0.0 <= p <= 1.0


def test_backtest_coverage_matches_backtest_rows(dashboard):
    cov = dashboard["trust"]["coverage"]["sales"]
    inside80 = inside95 = n = 0
    for seg in dashboard["series"]:
        for b in dashboard["series"][seg]["sales"]["backtest"]:
            n += 1
            inside80 += b["lo80"] <= b["actual"] <= b["hi80"]
            inside95 += b["lo95"] <= b["actual"] <= b["hi95"]
    assert n == cov["n"] and inside80 == cov["inside80"] and inside95 == cov["inside95"]
    assert 0 <= cov["coverage80"] <= cov["coverage95"] <= 1


def test_store_forecasts_add_up_to_store_type(dashboard):
    st = dashboard["stores"]
    for t in "abcd":
        total = sum(r["fc"] for r in st["rows"] if r["type"] == t)
        seg = next(f for f in dashboard["series"][t]["sales"]["forecast"] if f["month"] == st["month"])["p50"]
        assert abs(total / seg - 1) < 1e-9


def test_scenario_changes_with_intervention(dashboard):
    m = dashboard["future_months"][0]
    base = dm.scenario(dashboard, "d", m, promo=0)
    vals = [dm.scenario(dashboard, "d", m, promo=k)["p50"] for k in range(0, 5)]
    assert vals[2] != base["p50"]
    assert all(b >= a for a, b in zip(vals, vals[1:]))            # more promo days never lowers modeled sales
    assert dm.scenario(dashboard, "d", m, foot_pct=-5)["p50"] < base["p50"]


def test_decision_engine_always_returns_valid_recommendation(dashboard):
    for m, segs in dashboard["decisions"].items():
        for seg, d in segs.items():
            rec = d["recommended"]
            assert rec["label"]
            assert any(t["recommended"] for t in d["tested"])
            assert sum(t["recommended"] for t in d["tested"]) == 1
            assert d["risk_level"] in ("Critical", "High", "Medium", "Low")
            if not d["action_required"]:
                assert rec["promo_delta"] == 0
            if rec["promo_delta"] > 0:
                assert rec["score"] > 0 and rec["delta_p_pp"] >= 0
            # the recommended option is the best-scoring tested one when action is taken
            if rec["promo_delta"] > 0:
                assert rec["score"] == max(t["score"] for t in d["tested"] if t["promo_delta"] > 0)
            assert set(d["scenarios"]) == {"base", "recovery", "downside"}
            assert d["scenarios"]["downside"]["p50"] < d["scenarios"]["base"]["p50"]


def test_decision_score_is_transparent(dashboard):
    m = dashboard["future_months"][0]
    for t in dashboard["decisions"][m]["d"]["tested"]:
        assert abs(sum(t["parts"].values()) - t["score"]) < 1e-9
