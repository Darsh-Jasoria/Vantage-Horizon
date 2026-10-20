"""Write the outputs every consumer reads.

- fct_kpi_forecast (DuckDB table + CSV): the Power BI / Fabric semantic-model table.
  Actual, forecast and backtest rows share one shape, so a slicer on value_type is
  the whole "last month vs next month" option in any BI tool.
- dashboard_data.json: everything the web dashboard and API serve.
- teams_alert.json: the Adaptive Card posted to Teams.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from vantage import alerts as alerts_mod
from vantage import decision as decision_mod
from vantage import narrate as narrate_mod
from vantage.config import MODEL_VERSION, OUT_DIR, SEGMENT_LABELS, TARGET_MAPE, TARGET_GROWTH

SEGMENTS = ["All", "a", "b", "c", "d"]


def _ym(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m")


def kpi_table(res: dict) -> pd.DataFrame:
    mk, fc, bt = res["mk"], res["fc"], res["bt"]
    act = mk[mk.row_type == "actual"].copy()
    act["ticket"] = act["sales"] / act["customers"]
    rows = []
    for metric in ("sales", "customers", "ticket"):
        a = act[["month", "segment", metric]].rename(columns={metric: "value"})
        rows.append(a.assign(metric=metric, value_type="actual"))
    f = fc.rename(columns={"p50": "value"})[["month", "segment", "metric", "value", "lo80", "hi80",
                                             "lo95", "hi95", "horizon"]]
    rows.append(f.assign(value_type="forecast"))
    # backtest rows: out-of-sample (leave-one-month-out) forecasts with their own 80/95% ranges
    b = res["ev"][["month", "segment", "metric", "ens_loo", "lo80", "hi80", "lo95", "hi95"]] \
        .rename(columns={"ens_loo": "value"})
    rows.append(b.assign(value_type="backtest", horizon=1))
    out = pd.concat(rows, ignore_index=True)
    out["model_version"] = MODEL_VERSION
    out["as_of"] = res["cutoff"].date()
    return out[["month", "segment", "metric", "value_type", "value", "lo80", "hi80", "lo95", "hi95",
                "horizon", "model_version", "as_of"]]


def write_all(con, res: dict, drivers: dict, log=print) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = kpi_table(res)
    con.register("kpi_df", table)
    con.execute("create or replace table fct_kpi_forecast as select * from kpi_df")
    table.to_csv(OUT_DIR / "fct_kpi_forecast.csv", index=False)
    log(f"  fct_kpi_forecast: {len(table):,} rows (DuckDB + CSV)")

    mk, fc, sf, bt = res["mk"], res["fc"], res["sales_fc"], res["bt"]
    actual_months = [pd.Timestamp(m) for m in res["actual_months"]]
    future_months = [pd.Timestamp(m) for m in res["future_months"]]
    last = actual_months[-1]

    series, calendar, targets, p_hit = {}, {}, {}, {}
    for seg in SEGMENTS:
        s = mk[mk.segment == seg].set_index("month")
        series[seg] = {}
        for metric in ("sales", "customers", "ticket"):
            vals = (s["sales"] / s["customers"]) if metric == "ticket" else s[metric]
            f = fc[(fc.segment == seg) & (fc.metric == metric)]
            entry = {
                "actual": [float(vals[m]) for m in actual_months],
                "forecast": [dict(month=_ym(r.month), horizon=int(r.horizon), p50=r.p50, lo80=r.lo80,
                                  hi80=r.hi80, lo95=r.lo95, hi95=r.hi95) for r in f.itertuples()],
            }
            if metric != "ticket":
                b = bt[(bt.segment == seg) & (bt.metric == metric)]
                ev = res["ev"][(res["ev"].segment == seg) & (res["ev"].metric == metric)].set_index("month")
                entry["backtest"] = [dict(month=_ym(r.month), forecast=float(ev.loc[r.month, "ens_loo"]),
                                          lo80=float(ev.loc[r.month, "lo80"]), hi80=float(ev.loc[r.month, "hi80"]),
                                          lo95=float(ev.loc[r.month, "lo95"]), hi95=float(ev.loc[r.month, "hi95"]),
                                          actual=r.actual, ets=r.ets,
                                          **({"prophet": r.prophet} if r.prophet is not None and pd.notna(r.prophet) else {}))
                                     for r in b.itertuples()]
            series[seg][metric] = entry
        calendar[seg] = {_ym(m): dict(open_store_days=float(s.loc[m, "open_store_days"]),
                                      promo_share=float(s.loc[m, "promo_share"]),
                                      stores=int(s.loc[m, "stores"]))
                         for m in actual_months + future_months}
        t = res["targets"][res["targets"].segment == seg]
        targets[seg] = {_ym(r.month): r.target for r in t.itertuples()}
        p = sf[(sf.segment == seg) & sf.p_hit.notna()]
        p_hit[seg] = {_ym(r.month): float(r.p_hit) for r in p.itertuples()}

    alerts = alerts_mod.detect(res)

    leading = alerts_mod.leading_indicators(res)
    accuracy = res["accuracy"]
    best = min(accuracy, key=lambda a: a["mape_total"])
    trust = dict(accuracy=accuracy, coverage=res["coverage"],
                 production_model="Vantage ensemble",
                 best_backtest_model=best["model"],
                 method=("Rolling one-month-ahead backtest over the last "
                         f"{bt.month.nunique()} months. Each month is scored with ensemble weights and "
                         "interval widths estimated from the other months only (leave-one-month-out), "
                         "so accuracy and coverage are out-of-sample."),
                 caveat="Measured on a controlled synthetic dataset; real retail data will be noisier.")

    data = {
        "meta": dict(as_of=res["cutoff"].strftime("%Y-%m-%d"),
                     generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     model_version=MODEL_VERSION, stores=int(res["df"].store_id.nunique()),
                     daily_rows=int((res["df"].row_type == "actual").sum()),
                     ensemble_weights={k: float(v) for k, v in res["weights"].items()},
                     target_mape=TARGET_MAPE, target_growth=TARGET_GROWTH, currency="EUR"),
        "actual_months": [_ym(m) for m in actual_months],
        "future_months": [_ym(m) for m in future_months],
        "segments": [dict(id=s, label=SEGMENT_LABELS[s]) for s in SEGMENTS],
        "series": series, "calendar": calendar, "targets": targets, "p_hit": p_hit,
        "drivers": drivers["by_month"], "global_importance": drivers["global_importance"],
        "alerts": alerts, "accuracy": accuracy, "trust": trust, "leading": leading,
        "whatif": res["whatif"], "stores": store_grid(res),
    }

    # Decision engine (simulate -> score -> recommend) for every month and segment
    data["decisions"] = decision_mod.build_all(data)

    # Executive explanation (WHAT / WHY / WHAT SHOULD I DO): template everywhere, GPT-4o for the headline
    nxt = _ym(future_months[0])
    narratives = {}
    for m in data["future_months"]:
        narratives[m] = {}
        for seg in SEGMENTS:
            dec = data["decisions"][m][seg]
            if m == nxt and seg == "All":
                narratives[m][seg] = narrate_mod.narrate(dec)
            else:
                parts = narrate_mod.three_part(dec)
                narratives[m][seg] = dict(parts, text=narrate_mod.as_text(parts), source="template")
    data["narratives"] = narratives

    # Teams action card for the most urgent decision next month
    top = decision_mod.top_decision(data["decisions"], nxt)
    card = alerts_mod.teams_card(top, alerts, narratives[nxt][top["segment"]], res["cutoff"].strftime("%d %b %Y"))
    (OUT_DIR / "teams_alert.json").write_text(json.dumps(card, indent=2, default=_json_default))
    teams_status = alerts_mod.post_to_teams(card) if (alerts or top["action_required"]) else "no alerts"
    data["meta"]["teams"] = teams_status
    data["meta"]["top_decision"] = dict(month=nxt, segment=top["segment"])
    log(f"  alerts: {len(alerts)} · decision: {top['segment_label']} -> {top['recommended']['label']} · "
        f"Teams card {teams_status}")

    (OUT_DIR / "dashboard_data.json").write_text(json.dumps(data, default=_json_default))
    return data


def store_grid(res: dict) -> dict:
    """Per-store view for the risk heatmap.

    foot_gap: last month's footfall vs what the model expected a month earlier (leading indicator).
    fc: next month's sales, LightGBM store-days scaled so stores add up to their segment's
        ensemble forecast (proportional top-down reconciliation).
    target: same month last year x (1 + TARGET_GROWTH).
    """
    df, fc = res["df"], res["fc"]
    last, nxt = pd.Timestamp(res["actual_months"][-1]), pd.Timestamp(res["future_months"][0])
    act = df[df.row_type == "actual"]
    monthly = (act.assign(month=act.sale_date.dt.to_period("M").dt.to_timestamp())
                  .groupby(["store_id", "month"])["sales"].sum())
    spark_months = [pd.Timestamp(m) for m in res["actual_months"][-12:]]
    rows = res["fut_rows"]
    rows = rows[(rows.sale_date >= nxt) & (rows.sale_date < nxt + pd.offsets.MonthBegin(1))]
    raw = pd.Series(res["models"]["sales"].predict(rows), index=rows.store_id.to_numpy()).groupby(level=0).sum()
    seg_fc = fc[(fc.month == nxt) & (fc.metric == "sales")].set_index("segment")
    info = df.drop_duplicates("store_id").set_index("store_id")[["store_type", "assortment", "competition_distance"]]
    foot = res["store_last"].set_index("store_id")
    ly = nxt - pd.DateOffset(years=1)
    out = []
    for sid, r in info.iterrows():
        scale = seg_fc.loc[r.store_type, "p50"] / seg_fc.loc[r.store_type, "lgbm"]
        f = float(raw.get(sid, 0.0) * scale)
        t = monthly.get((sid, ly))
        t = float(t) * (1 + TARGET_GROWTH) if t is not None and t > 0 else None
        fr = foot.loc[sid] if sid in foot.index else None
        gap = float(fr.actual / fr.pred - 1) * 100 if fr is not None and fr.pred > 0 else None
        out.append(dict(id=int(sid), type=r.store_type, assort=r.assortment,
                        comp=float(r.competition_distance), foot_gap=gap, fc=f, target=t,
                        last=float(monthly.get((sid, last), 0.0)),
                        spark=[round(float(monthly.get((sid, m), 0.0))) for m in spark_months]))
    return {"last_month": last.strftime("%Y-%m"), "month": nxt.strftime("%Y-%m"),
            "spark_months": [m.strftime("%Y-%m") for m in spark_months], "rows": out}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(type(o))
