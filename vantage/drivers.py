"""Driver panel: why is next month different from last month?

Splits the change in sales (last actual month -> each forecast month) into:
- trading days (more or fewer open store-days),
- each model feature, via the change in its mean SHAP value (TreeExplainer),
- underlying trend (the store level the model scales from),
- momentum (how much of last month's surprise carries forward),
- model blend (SARIMAX pull on the ensemble), and an "other" remainder.
All pieces are in log points, so they add up to the total change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from vantage.features import FRIENDLY

SAMPLE = 6000


def _month_rows(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    return df[(df.sale_date >= month) & (df.sale_date < month + pd.offsets.MonthBegin(1))]


def _shap_frame(model, rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    open_rows = rows[(rows.is_open == 1) & rows["level_sales"].notna()]
    sample = open_rows.sample(min(SAMPLE, len(open_rows)), random_state=seed)
    explainer = shap.TreeExplainer(model.model)
    vals = explainer.shap_values(sample[model.features])
    out = pd.DataFrame(vals, columns=model.features, index=sample.index)
    out["_segment"] = sample["store_type"].to_numpy()
    out["_loglevel"] = np.log(sample["level_sales"].to_numpy())
    return out


def compute(res: dict) -> dict:
    model = res["models"]["sales"]
    df, mk, fc, bt = res["df"], res["mk"], res["fc"], res["bt"]
    last = pd.Timestamp(res["actual_months"][-1])
    frames = {last: _shap_frame(model, _month_rows(df, last), 0)}
    for i, m in enumerate(res["future_months"]):
        frames[pd.Timestamp(m)] = _shap_frame(model, _month_rows(df, pd.Timestamp(m)), i + 1)

    # Global importance (mean |SHAP|) on next month, grouped by friendly name
    nxt = frames[pd.Timestamp(res["future_months"][0])]
    imp = nxt[model.features].abs().mean().groupby(lambda c: FRIENDLY.get(c, c)).sum()
    global_importance = [dict(name=k, value=float(v)) for k, v in imp.sort_values(ascending=False).items()]

    last_bt = bt[bt.month == bt.month.max()]
    out = {}
    for m in res["future_months"]:
        m = pd.Timestamp(m)
        key = m.strftime("%Y-%m")
        out[key] = {}
        for seg in ["All", "a", "b", "c", "d"]:
            A = frames[last] if seg == "All" else frames[last][frames[last]._segment == seg]
            B = frames[m] if seg == "All" else frames[m][frames[m]._segment == seg]
            row_l = mk[(mk.month == last) & (mk.segment == seg)].iloc[0]
            row_f = mk[(mk.month == m) & (mk.segment == seg)].iloc[0]
            f = fc[(fc.month == m) & (fc.segment == seg) & (fc.metric == "sales")].iloc[0]
            total = np.log(f.p50 / row_l.sales)

            parts = {"Trading days": np.log(row_f.open_store_days / row_l.open_store_days),
                     "Underlying trend": B["_loglevel"].mean() - A["_loglevel"].mean()}
            for c in model.features:
                name = FRIENDLY.get(c, c)
                parts[name] = parts.get(name, 0.0) + (B[c].mean() - A[c].mean())
            surprise = float(last_bt[(last_bt.segment == seg) & (last_bt.metric == "sales")].surprise.iloc[0])
            h = int(f.horizon)
            from vantage.forecast import MOMENTUM
            parts["Last month's momentum"] = (MOMENTUM[h - 1] - 1) * surprise
            parts["Monthly trend model"] = np.log(f.p50 / f.lgbm_m)
            other = total - sum(parts.values())
            items = sorted(parts.items(), key=lambda kv: -abs(kv[1]))
            top, rest = items[:5], items[5:]
            other += sum(v for _, v in rest)
            out[key][seg] = dict(
                total_pct=float(np.expm1(total) * 100),
                drivers=[dict(name=k, pct=float(v * 100)) for k, v in top],
                other_pct=float(other * 100))
    return dict(by_month=out, global_importance=global_importance)
