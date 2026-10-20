"""Forecast job: backtest, final 30/60/90-day forecast, bands, targets, what-if curves."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

from vantage.config import (ARTIFACT_DIR, BACKTEST_MONTHS, HORIZON_MONTHS, TARGET_GROWTH, ENABLE_PROPHET)
from vantage.features import build_daily
from vantage.models import (DailyModel, month_sums, sarimax_forecast, ets_forecast, seasonal_naive,
                            prophet_forecast, prophet_available)
from vantage import trust

METRICS = ("sales", "customers")
SEGMENTS = ["All", "a", "b", "c", "d"]
MOMENTUM = [0.9, 0.85, 0.8]          # share of last month's surprise carried into h = 1, 2, 3
Z80, Z95 = norm.ppf(0.9), norm.ppf(0.975)


def load_monthly(con) -> pd.DataFrame:
    mk = con.execute("select * from fct_monthly_kpi order by month, segment").df()
    mk["month"] = pd.to_datetime(mk["month"])
    return mk


def _seg(mk: pd.DataFrame, seg: str) -> pd.DataFrame:
    return mk[mk.segment == seg].sort_values("month")


def _weekly(rows: pd.DataFrame, pred: np.ndarray, metric: str, month: pd.Timestamp) -> pd.DataFrame:
    t = pd.DataFrame({
        "week": rows["sale_date"] - pd.to_timedelta(rows["sale_date"].dt.dayofweek, unit="D"),
        "segment": rows["store_type"].to_numpy(), "pred": pred,
        "actual": rows[metric].fillna(0).to_numpy(), "open": rows["is_open"].to_numpy()})
    by = t.groupby(["week", "segment"], as_index=False)[["pred", "actual", "open"]].sum()
    tot = t.groupby("week", as_index=False)[["pred", "actual", "open"]].sum().assign(segment="All")
    out = pd.concat([tot, by], ignore_index=True)
    out["metric"], out["month"] = metric, month
    return out


def run(con, log=print) -> dict:
    df = build_daily(con)
    mk = load_monthly(con)
    actual_months = sorted(mk.loc[mk.row_type == "actual", "month"].unique())
    future_months = sorted(mk.loc[mk.row_type == "planned", "month"].unique())[:HORIZON_MONTHS]
    cutoff = df.loc[df.row_type == "actual", "sale_date"].max()
    use_prophet = ENABLE_PROPHET and prophet_available()

    # ------------------------------------------------------------ rolling backtest
    # One extra month at the start so the momentum term has a previous error to use.
    bt_months = actual_months[-(BACKTEST_MONTHS + 1):]
    records, weekly, store_last = [], [], None
    for T in bt_months:
        T = pd.Timestamp(T)
        cut = T - pd.Timedelta(days=1)
        rows = df[(df.sale_date >= T) & (df.sale_date < T + pd.offsets.MonthBegin(1))]
        lgb_month = {}
        for metric in METRICS:
            model = DailyModel(metric).fit(df, cut)
            pred = model.predict(rows)
            lgb_month[metric] = month_sums(rows, pred).set_index("segment")["v"]
            weekly.append(_weekly(rows, pred, metric, T))
            if T == bt_months[-1] and metric == "customers":
                store_last = (pd.DataFrame({"store_id": rows.store_id, "store_type": rows.store_type,
                                            "pred": pred, "actual": rows.customers.fillna(0)})
                              .groupby(["store_id", "store_type"], as_index=False).sum())
        for seg in SEGMENTS:
            s = _seg(mk, seg)
            hist, fut = s[s.month < T], s[s.month == T]
            for metric in METRICS:
                sar, _ = sarimax_forecast(hist, fut, metric)
                records.append(dict(
                    month=T, segment=seg, metric=metric, actual=float(fut[metric].iloc[0]),
                    lgbm=float(lgb_month[metric][seg]), sarimax=float(sar[0]),
                    ets=float(ets_forecast(hist, fut, metric)[0]),
                    prophet=_first(prophet_forecast(hist, fut, metric)) if use_prophet else None,
                    naive=float(seasonal_naive(hist, fut, metric)[0])))
        log(f"  backtest {T:%b %Y} done")
    bt = pd.DataFrame(records).sort_values(["segment", "metric", "month"])

    # momentum: carry part of last month's LightGBM surprise forward
    bt["surprise"] = np.log(bt["actual"] / bt["lgbm"])
    bt["prev_surprise"] = bt.groupby(["segment", "metric"])["surprise"].shift(1)
    bt["sar_surprise"] = np.log(bt["actual"] / bt["sarimax"])
    bt["prev_sar_surprise"] = bt.groupby(["segment", "metric"])["sar_surprise"].shift(1)
    bt = bt.dropna(subset=["prev_surprise"]).copy()
    bt["lgbm_m"] = bt["lgbm"] * np.exp(MOMENTUM[0] * bt["prev_surprise"])
    bt["sarimax_m"] = bt["sarimax"] * np.exp(MOMENTUM[0] * bt["prev_sar_surprise"])

    def ape(col):
        return (bt[col] / bt["actual"] - 1).abs()

    weights = {}
    for metric in METRICS:
        m = bt.metric == metric
        e_l, e_s = ape("lgbm_m")[m].mean(), ape("sarimax_m")[m].mean()
        weights[metric] = (1 / e_l) / (1 / e_l + 1 / e_s)
    bt["ensemble"] = [r.lgbm_m ** weights[r.metric] * r.sarimax_m ** (1 - weights[r.metric])
                      for r in bt.itertuples()]
    bt["log_err"] = np.log(bt["ensemble"] / bt["actual"])

    sigma = {}
    for metric in METRICS:
        pooled = np.sqrt((bt.loc[bt.metric == metric, "log_err"] ** 2).mean())
        for seg in SEGMENTS:
            own = np.sqrt((bt.loc[(bt.metric == metric) & (bt.segment == seg), "log_err"] ** 2).mean())
            sigma[(seg, metric)] = max(0.015, np.sqrt((4 * pooled ** 2 + 6 * own ** 2) / 10))
    # ticket = sales / customers
    tk = bt.pivot_table(index=["month", "segment"], columns="metric", values=["ensemble", "actual"])
    tk_err = np.log((tk["ensemble"]["sales"] / tk["ensemble"]["customers"]) /
                    (tk["actual"]["sales"] / tk["actual"]["customers"]))
    for seg in SEGMENTS:
        sigma[(seg, "ticket")] = max(0.008, float(np.sqrt((tk_err.xs(seg, level="segment") ** 2).mean())))

    # Honest scoring: every backtest month is scored with weights/interval widths estimated
    # from the OTHER months only (leave-one-month-out), so the reported numbers are out-of-sample.
    ev = trust.loo_evaluate(bt)
    methods = [("Vantage ensemble", "ensemble_loo"), ("LightGBM (daily, summed)", "lgbm_m"),
               ("SARIMAX (monthly)", "sarimax_m"), ("Prophet (monthly)", "prophet"),
               ("ETS - Power BI built-in method", "ets"), ("Same month last year", "naive")]
    accuracy = trust.accuracy_table(bt, ev, methods)
    cov = {m: trust.coverage(ev, m) for m in METRICS}
    log("  accuracy: " + ", ".join(f"{a['model']} {a['mape_total']:.1f}%" for a in accuracy))
    log(f"  interval coverage (sales, out-of-sample): 80% -> {cov['sales']['coverage80']:.0%}, "
        f"95% -> {cov['sales']['coverage95']:.0%} over {cov['sales']['n']} forecasts")

    # ------------------------------------------------------------ final forecast
    fut_rows = df[df.sale_date > cutoff]
    final_models, lgb_future = {}, {}
    for metric in METRICS:
        model = DailyModel(metric).fit(df, cutoff)
        final_models[metric] = model
        lgb_future[metric] = month_sums(fut_rows, model.predict(fut_rows))

    last_bt = bt[bt.month == bt.month.max()].set_index(["segment", "metric"])
    fc_rows = []
    for seg in SEGMENTS:
        s = _seg(mk, seg)
        hist = s[s.row_type == "actual"]
        fut = s[s.month.isin(future_months)]
        point = {}
        for metric in METRICS:
            sar, _ = sarimax_forecast(hist, fut, metric)
            lg = lgb_future[metric].query("segment == @seg").set_index("month")["v"]
            surprise = float(last_bt.loc[(seg, metric), "surprise"])
            sar_surprise = float(last_bt.loc[(seg, metric), "sar_surprise"])
            point[metric] = []
            for h, mth in enumerate(future_months, start=1):
                lg_m = lg[mth] * np.exp(MOMENTUM[h - 1] * surprise)
                w = weights[metric]
                sar_m = sar[h - 1] * np.exp(MOMENTUM[h - 1] * sar_surprise)
                point[metric].append(dict(lgbm=float(lg[mth]), lgbm_m=float(lg_m), sarimax=float(sar[h - 1]),
                                          sarimax_m=float(sar_m), p50=float(lg_m ** w * sar_m ** (1 - w))))
        for h, mth in enumerate(future_months, start=1):
            widen = 1 + 0.15 * (h - 1)
            for metric in ("sales", "customers", "ticket"):
                if metric == "ticket":
                    p50 = point["sales"][h - 1]["p50"] / point["customers"][h - 1]["p50"]
                    parts = {}
                else:
                    parts = {k: v for k, v in point[metric][h - 1].items() if k != "p50"}
                    p50 = point[metric][h - 1]["p50"]
                sd = sigma[(seg, metric)] * widen
                fc_rows.append(dict(month=pd.Timestamp(mth), segment=seg, metric=metric, horizon=h,
                                    p50=p50, lo80=p50 * np.exp(-Z80 * sd), hi80=p50 * np.exp(Z80 * sd),
                                    lo95=p50 * np.exp(-Z95 * sd), hi95=p50 * np.exp(Z95 * sd),
                                    sigma_log=sd, **parts))
    fc = pd.DataFrame(fc_rows)

    # ------------------------------------------------------------ targets & hit probability
    targets = []
    for seg in SEGMENTS:
        s = _seg(mk, seg).set_index("month")
        for mth in list(actual_months[-12:]) + list(future_months):
            ly = pd.Timestamp(mth) - pd.DateOffset(years=1)
            if ly in s.index and pd.notna(s.loc[ly, "sales"]):
                targets.append(dict(month=pd.Timestamp(mth), segment=seg,
                                    target=float(s.loc[ly, "sales"]) * (1 + TARGET_GROWTH)))
    targets = pd.DataFrame(targets)
    sales_fc = fc[fc.metric == "sales"].merge(targets, on=["month", "segment"], how="left")
    sales_fc["p_hit"] = 1 - norm.cdf((np.log(sales_fc["target"]) - np.log(sales_fc["p50"])) / sales_fc["sigma_log"])

    # ------------------------------------------------------------ what-if: promo-day response
    whatif = promo_curves(final_models, fut_rows, future_months)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "models.pkl", "wb") as f:
        pickle.dump({"models": final_models, "future_rows": fut_rows, "future_months": future_months}, f)

    return dict(df=df, mk=mk, bt=bt, ev=ev, coverage=cov, fc=fc, sales_fc=sales_fc, targets=targets,
                weights=weights, sigma=sigma, accuracy=accuracy, weekly=pd.concat(weekly, ignore_index=True),
                store_last=store_last, models=final_models, fut_rows=fut_rows, whatif=whatif,
                cutoff=cutoff, actual_months=actual_months, future_months=future_months)


def _first(arr):
    return None if arr is None else float(arr[0])


def shift_promo_days(rows: pd.DataFrame, delta: int) -> pd.DataFrame:
    """Add (delta > 0) or remove (delta < 0) chain-wide promo weekdays in a month."""
    rows = rows.copy()
    if delta == 0:
        return rows
    cal = rows.groupby("sale_date").agg(promo=("is_promo", "max"), dow=("day_of_week", "first"),
                                        hol=("state_holiday_code", "max"))
    weekdays = cal[(cal.dow <= 5) & (cal.hol == 0)]
    if delta > 0:
        flip = weekdays[weekdays.promo == 0].index[:delta]
        rows.loc[rows.sale_date.isin(flip), "is_promo"] = 1
    else:
        flip = weekdays[weekdays.promo == 1].index[::-1][:-delta]
        rows.loc[rows.sale_date.isin(flip), "is_promo"] = 0
    return rows


def promo_days(rows: pd.DataFrame) -> int:
    cal = rows.groupby("sale_date")["is_promo"].max()
    return int(cal.sum())


def promo_curves(models: dict, fut_rows: pd.DataFrame, months) -> dict:
    out = {}
    for mth in months:
        mth = pd.Timestamp(mth)
        rows = fut_rows[(fut_rows.sale_date >= mth) & (fut_rows.sale_date < mth + pd.offsets.MonthBegin(1))]
        base = {m: month_sums(rows, models[m].predict(rows)).set_index("segment")["v"] for m in METRICS}
        curve = {seg: {"delta": [], "sales": [], "customers": []} for seg in SEGMENTS}
        for delta in range(-6, 7):
            r = shift_promo_days(rows, delta)
            for m in METRICS:
                v = month_sums(r, models[m].predict(r)).set_index("segment")["v"]
                for seg in SEGMENTS:
                    curve[seg][m].append(float(v[seg] / base[m][seg]))
            for seg in SEGMENTS:
                curve[seg]["delta"].append(delta)
        out[mth.strftime("%Y-%m")] = {"base_promo_days": promo_days(rows), "curve": curve}
    return out
