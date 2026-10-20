"""Sample data, SQL models and leakage checks (fast: uses a 24-store sample)."""
import numpy as np
import pandas as pd

from vantage import features


def test_sample_data_is_generated(small_raw):
    train = pd.read_csv(small_raw / "train.csv")
    test = pd.read_csv(small_raw / "test.csv")
    store = pd.read_csv(small_raw / "store.csv")
    assert set(["Store", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]) <= set(train.columns)
    assert store.Store.nunique() == 24
    assert train.Date.max() == "2026-09-30"
    assert test.Date.min() == "2026-10-01" and test.Date.max() == "2026-12-31"
    assert "Sales" not in test.columns          # the future has no actuals
    assert (train.loc[train.Open == 0, "Sales"] == 0).all()


def test_monthly_kpi_has_three_planned_months(small_con):
    mk = small_con.execute("select * from fct_monthly_kpi where segment = 'All' order by month").df()
    planned = mk[mk.row_type == "planned"]
    assert len(planned) == 3                    # the forecast horizon is exactly 3 months
    assert planned.sales.isna().all()


def test_no_future_actuals_leak_into_forecast_features(small_con):
    """Overwrite every actual after the cutoff with garbage: features for the next 90 days must not change."""
    df = features.build_daily(small_con)
    cutoff = df.loc[df.row_type == "actual", "sale_date"].max() - pd.Timedelta(days=90)
    horizon = (df.sale_date > cutoff) & (df.sale_date <= cutoff + pd.Timedelta(days=90))
    poisoned = df.copy()
    after = poisoned.sale_date > cutoff
    rng = np.random.default_rng(0)
    poisoned.loc[after, "sales"] = rng.uniform(0, 1e6, after.sum())
    poisoned.loc[after, "customers"] = rng.uniform(0, 1e5, after.sum())
    poisoned = poisoned.drop(columns=[c for c in poisoned if c.startswith(("level_", "growth_"))])
    for t in ("sales", "customers"):
        features._add_level(poisoned, t)
    cols = [c for c in df.columns if c.startswith(("level_", "growth_"))] + features.BASE_FEATURES
    a = df.loc[horizon, cols].reset_index(drop=True)
    b = poisoned.loc[horizon, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
