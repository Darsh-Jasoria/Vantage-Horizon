"""Forecast trust: honest backtest scoring and prediction-interval calibration.

Why this module exists
----------------------
The production ensemble weights and interval widths are estimated from the rolling backtest.
If we then score the same backtest months with those same weights and widths, the scores are
optimistic (the model is graded on data it was tuned on). Here every backtest month is scored
leave-one-month-out: its ensemble weight and interval width come only from the OTHER months.

Everything is computed from real backtest rows; nothing is hard-coded.

Input `bt` has one row per (month, segment, metric) with columns:
    actual, lgbm_m, sarimax_m (momentum-adjusted one-month-ahead forecasts) and baseline columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

Z80, Z95 = norm.ppf(0.9), norm.ppf(0.975)
SIGMA_FLOOR = 0.015


def ensemble_weight(bt: pd.DataFrame, metric: str, exclude_month=None) -> float:
    """Weight on LightGBM, inversely proportional to each model's mean absolute % error."""
    m = (bt.metric == metric)
    if exclude_month is not None:
        m &= (bt.month != exclude_month)
    e_l = (bt.loc[m, "lgbm_m"] / bt.loc[m, "actual"] - 1).abs().mean()
    e_s = (bt.loc[m, "sarimax_m"] / bt.loc[m, "actual"] - 1).abs().mean()
    return float((1 / e_l) / (1 / e_l + 1 / e_s))


def blend(lgbm: pd.Series | float, sarimax: pd.Series | float, w: float):
    """Geometric blend (blending in log space)."""
    return lgbm ** w * sarimax ** (1 - w)


def sigmas(log_err: pd.DataFrame, segments, metric: str) -> dict:
    """Interval width per segment: RMSE of log error, shrunk 40% toward the pooled value."""
    sub = log_err[log_err.metric == metric]
    pooled = np.sqrt((sub["log_err"] ** 2).mean())
    out = {}
    for seg in segments:
        own = np.sqrt((sub.loc[sub.segment == seg, "log_err"] ** 2).mean())
        out[seg] = max(SIGMA_FLOOR, float(np.sqrt((4 * pooled ** 2 + 6 * own ** 2) / 10)))
    return out


def loo_evaluate(bt: pd.DataFrame, metrics=("sales", "customers")) -> pd.DataFrame:
    """Leave-one-month-out ensemble forecast and 80/95% intervals for every backtest row."""
    rows = []
    months = sorted(bt.month.unique())
    segments = list(bt.segment.unique())
    for metric in metrics:
        for M in months:
            w = ensemble_weight(bt, metric, exclude_month=M)
            train = bt[(bt.metric == metric) & (bt.month != M)].copy()
            train["log_err"] = np.log(blend(train.lgbm_m, train.sarimax_m, w) / train.actual)
            sd = sigmas(train, segments, metric)
            test = bt[(bt.metric == metric) & (bt.month == M)]
            for r in test.itertuples():
                ens = float(blend(r.lgbm_m, r.sarimax_m, w))
                s = sd[r.segment]
                rows.append(dict(month=r.month, segment=r.segment, metric=metric, actual=r.actual,
                                 ens_loo=ens, weight_loo=w, sigma_loo=s,
                                 lo80=ens * np.exp(-Z80 * s), hi80=ens * np.exp(Z80 * s),
                                 lo95=ens * np.exp(-Z95 * s), hi95=ens * np.exp(Z95 * s)))
    ev = pd.DataFrame(rows)
    ev["inside80"] = (ev.actual >= ev.lo80) & (ev.actual <= ev.hi80)
    ev["inside95"] = (ev.actual >= ev.lo95) & (ev.actual <= ev.hi95)
    ev["ape"] = (ev.ens_loo / ev.actual - 1).abs()
    return ev


def coverage(ev: pd.DataFrame, metric: str = "sales") -> dict:
    """Empirical interval coverage on the backtest (all segments pooled)."""
    sub = ev[ev.metric == metric]
    return dict(metric=metric, n=int(len(sub)), months=int(sub.month.nunique()),
                segments=int(sub.segment.nunique()),
                inside80=int(sub.inside80.sum()), inside95=int(sub.inside95.sum()),
                coverage80=float(sub.inside80.mean()), coverage95=float(sub.inside95.mean()))


def accuracy_table(bt: pd.DataFrame, ev: pd.DataFrame, models: list[tuple[str, str]]) -> list[dict]:
    """MAPE (all-stores and per-segment) and MAE in EUR for each method, one-month-ahead sales."""
    out = []
    sales = bt[bt.metric == "sales"].copy()
    ens = ev[ev.metric == "sales"].set_index(["month", "segment"])["ens_loo"]
    sales["ensemble_loo"] = [ens.loc[(r.month, r.segment)] for r in sales.itertuples()]
    for name, col in models:
        if col not in sales or sales[col].isna().any():
            continue
        ape = (sales[col] / sales["actual"] - 1).abs()
        tot = sales.segment == "All"
        out.append(dict(model=name, mape_total=float(ape[tot].mean() * 100),
                        mape_segments=float(ape[~tot].mean() * 100),
                        mae_total=float((sales.loc[tot, col] - sales.loc[tot, "actual"]).abs().mean()),
                        bias_total=float(((sales.loc[tot, col] / sales.loc[tot, "actual"]) - 1).mean() * 100)))
    return out
