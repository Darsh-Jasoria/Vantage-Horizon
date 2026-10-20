"""Backtest scoring, interval coverage and the Power BI DAX logic."""
import re

import numpy as np
import pandas as pd

from vantage import trust

DAX = (__import__("pathlib").Path(__file__).resolve().parents[1] / "powerbi" / "measures.dax").read_text()
CSV = __import__("pathlib").Path(__file__).resolve().parents[1] / "outputs" / "fct_kpi_forecast.csv"


def _fake_bt(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i, m in enumerate(pd.date_range("2026-01-01", periods=6, freq="MS")):
        for seg in ["All", "a", "b"]:
            for metric in ["sales", "customers"]:
                actual = 100 + rng.normal(0, 1)
                rows.append(dict(month=m, segment=seg, metric=metric, actual=actual,
                                 lgbm_m=actual * np.exp(rng.normal(0, 0.02)),
                                 sarimax_m=actual * np.exp(rng.normal(0, 0.04)),
                                 ets=actual * 1.03, naive=actual * 0.95))
    return pd.DataFrame(rows)


def test_loo_scoring_excludes_the_scored_month():
    bt = _fake_bt()
    bt.loc[bt.month == bt.month.max(), "lgbm_m"] *= 1.5        # one terrible LightGBM month
    ev = trust.loo_evaluate(bt)
    w_last = ev[(ev.month == bt.month.max()) & (ev.metric == "sales")].weight_loo.iloc[0]
    w_other = ev[(ev.month == bt.month.min()) & (ev.metric == "sales")].weight_loo.iloc[0]
    assert w_last > w_other          # the bad month does not get to down-weight itself


def test_coverage_counts_are_valid():
    ev = trust.loo_evaluate(_fake_bt())
    c = trust.coverage(ev, "sales")
    assert c["n"] == 6 * 3
    assert 0 <= c["inside80"] <= c["inside95"] <= c["n"]
    assert c["coverage80"] == c["inside80"] / c["n"]
    assert (ev.lo95 <= ev.lo80).all() and (ev.lo80 <= ev.ens_loo).all() and (ev.ens_loo <= ev.hi80).all()


def _blocks(name):
    m = re.search(rf"^{re.escape(name)} =\n(.*?)(?=^(?!VAR |//)\S[^\n]* =\n|\Z)", DAX, re.S | re.M)
    assert m, name
    return m.group(1)


def test_dax_measures_filter_one_segment():
    """The table holds 'All' plus each store type, so every value measure must filter a single segment."""
    for name in ["Sales (view)", "Sales low 80", "Sales high 80", "Backtest MAPE %", "Backtest 80% coverage"]:
        body = _blocks(name)
        n_calc = body.count("CALCULATE (")
        assert n_calc and body.count("fct_kpi_forecast[segment] =") >= n_calc, name


def test_dax_backtest_mape_pairs_same_month():
    body = _blocks("Backtest MAPE %")
    assert "AVERAGEX" in body
    assert body.count("fct_kpi_forecast[month] = m") == 2     # forecast AND actual restricted to the same month


def test_dax_mape_logic_matches_pipeline(dashboard):
    """Python re-implementation of 'Backtest MAPE %' on the CSV equals the pipeline's reported MAPE."""
    t = pd.read_csv(CSV)
    s = t[(t.metric == "sales") & (t.segment == "All")]
    bt = s[s.value_type == "backtest"].set_index("month")["value"]
    act = s[s.value_type == "actual"].set_index("month")["value"]
    mape = float(np.mean([abs(bt[m] / act[m] - 1) for m in bt.index]) * 100)
    reported = next(a for a in dashboard["trust"]["accuracy"] if a["model"] == "Vantage ensemble")["mape_total"]
    assert abs(mape - reported) < 1e-6
