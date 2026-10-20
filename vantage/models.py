"""Forecast models.

- LightGBM: global daily store model (Rossmann-style), summed to any slice.
- SARIMAX: monthly per segment with calendar drivers (promo share, trading days).
- ETS: Holt-Winters, the method Power BI's built-in forecast uses. Baseline only.
- Seasonal naive: same month last year, scaled by recent growth. Baseline only.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from vantage.features import features_for

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- LightGBM (daily)
class DailyModel:
    def __init__(self, target: str):
        self.target = target
        self.features = features_for(target)
        self.model: lgb.LGBMRegressor | None = None
        self.smear = 1.0

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> "DailyModel":
        lvl = f"level_{self.target}"
        tr = df[(df.sale_date <= cutoff) & (df.is_open == 1) & df[lvl].notna() & (df[self.target] > 0)]
        y = np.log(tr[self.target].to_numpy()) - np.log(tr[lvl].to_numpy())
        self.model = lgb.LGBMRegressor(
            n_estimators=220, learning_rate=0.08, num_leaves=63, min_child_samples=300,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=381)
        self.model.fit(tr[self.features], y)
        sample = tr.iloc[::25]
        r = np.log(sample[self.target].to_numpy()) - np.log(sample[lvl].to_numpy()) \
            - self.model.predict(sample[self.features])
        self.smear = float(np.mean(np.exp(r)))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        lvl = rows[f"level_{self.target}"].to_numpy()
        p = np.exp(self.model.predict(rows[self.features])) * lvl * self.smear
        p = np.where((rows["is_open"].to_numpy() == 1) & np.isfinite(p), p, 0.0)
        return p


def month_sums(rows: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Sum daily values to month x segment, including the All segment."""
    t = pd.DataFrame({"month": rows["sale_date"].dt.to_period("M").dt.to_timestamp(),
                      "segment": rows["store_type"].to_numpy(), "v": values})
    by_seg = t.groupby(["month", "segment"], as_index=False)["v"].sum()
    total = t.groupby("month", as_index=False)["v"].sum().assign(segment="All")
    return pd.concat([total, by_seg], ignore_index=True)


# ---------------------------------------------------------------- monthly models
def _exog(m: pd.DataFrame) -> np.ndarray:
    return np.column_stack([m["promo_share"].to_numpy(), np.log(m["open_store_days"].to_numpy())])


def sarimax_forecast(hist: pd.DataFrame, future: pd.DataFrame, col: str):
    """hist/future: monthly rows for one segment, sorted. Returns (mean, se_log)."""
    y = np.log(hist[col].to_numpy())
    try:
        res = SARIMAX(y, exog=_exog(hist), order=(1, 0, 0), seasonal_order=(0, 1, 0, 12),
                      trend="c", enforce_stationarity=False).fit(disp=False, maxiter=200)
        fc = res.get_forecast(len(future), exog=_exog(future))
        return np.exp(fc.predicted_mean), np.asarray(fc.se_mean)
    except Exception:
        mean = seasonal_naive(hist, future, col)
        return mean, np.full(len(future), 0.08)


def ets_forecast(hist: pd.DataFrame, future: pd.DataFrame, col: str) -> np.ndarray:
    y = hist[col].to_numpy()
    try:
        res = ExponentialSmoothing(y, trend="add", seasonal="mul", seasonal_periods=12,
                                   damped_trend=True).fit()
        return np.asarray(res.forecast(len(future)))
    except Exception:
        return seasonal_naive(hist, future, col)


def prophet_forecast(hist: pd.DataFrame, future: pd.DataFrame, col: str) -> np.ndarray:
    """Prophet on monthly totals with promo share and trading days as extra regressors.
    Optional baseline: returns None if Prophet is unavailable or fails."""
    try:
        import logging
        from prophet import Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        def frame(m: pd.DataFrame, with_y: bool) -> pd.DataFrame:
            f = pd.DataFrame({"ds": pd.to_datetime(m["month"]).to_numpy(),
                              "promo_share": m["promo_share"].to_numpy(),
                              "log_days": np.log(m["open_store_days"].to_numpy())})
            if with_y:
                f["y"] = np.log(m[col].to_numpy())
            return f

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        seasonality_mode="additive", changepoint_prior_scale=0.05)
        model.add_regressor("promo_share")
        model.add_regressor("log_days")
        model.fit(frame(hist, True))
        return np.exp(model.predict(frame(future, False))["yhat"].to_numpy())
    except Exception:
        return None          # never silently report another model's numbers as "Prophet"


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except Exception:
        return False


def seasonal_naive(hist: pd.DataFrame, future: pd.DataFrame, col: str) -> np.ndarray:
    y = hist[col].to_numpy()
    growth = y[-3:].sum() / y[-15:-12].sum()
    out = []
    for h in range(1, len(future) + 1):
        out.append(y[-12 + h - 1] * growth)
    return np.array(out)
