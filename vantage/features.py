"""Feature engineering for the daily store model.

Every feature is known in advance for the next 90 days: the calendar (promos,
holidays, closures) is planned, and the store "level" is a trailing 364-day mean
lagged by 92 days, so it never peeks past the forecast origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL_LAG = 92        # days; must be >= forecast horizon so the level is known
LEVEL_WINDOW = 364

BASE_FEATURES = [
    "day_of_week", "month", "day_of_month", "week_of_year", "is_promo",
    "is_school_holiday", "state_holiday_code", "pre_closure", "post_closure",
    "store_type_code", "assortment_code", "competition_distance", "promo2",
]

FRIENDLY = {
    "is_promo": "Promo days",
    "day_of_week": "Weekday mix",
    "month": "Seasonality",
    "week_of_year": "Seasonality",
    "day_of_month": "Pay-day timing",
    "is_school_holiday": "School holidays",
    "state_holiday_code": "Public holidays",
    "pre_closure": "Pre-holiday shopping",
    "post_closure": "Reopening bounce",
    "store_type_code": "Store format",
    "assortment_code": "Assortment",
    "competition_distance": "Competitor proximity",
    "promo2": "Loyalty promo",
    "growth_sales": "Underlying trend",
    "growth_customers": "Underlying trend",
}


def features_for(target: str) -> list[str]:
    return BASE_FEATURES + [f"growth_{target}"]


def build_daily(con) -> pd.DataFrame:
    df = con.execute("select * from fct_store_day order by store_id, sale_date").df()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    d = df["sale_date"].dt
    df["month"] = d.month
    df["day_of_month"] = d.day
    df["week_of_year"] = d.isocalendar().week.astype(int).to_numpy()
    df["state_holiday_code"] = df["state_holiday"].astype(str).map(
        {"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0).astype(int)
    df["store_type_code"] = df["store_type"].map({"a": 0, "b": 1, "c": 2, "d": 3}).astype(int)
    df["assortment_code"] = df["assortment"].map({"a": 0, "b": 1, "c": 2}).astype(int)
    df["competition_distance"] = df["competition_distance"].fillna(df["competition_distance"].median())

    # closures that are not ordinary Sundays drive pre-holiday rushes
    g = df.groupby("store_id", sort=False)
    nxt_open = g["is_open"].shift(-1).fillna(1)
    nxt_dow = g["day_of_week"].shift(-1).fillna(1)
    prv_open = g["is_open"].shift(1).fillna(1)
    prv_dow = g["day_of_week"].shift(1).fillna(1)
    df["pre_closure"] = ((nxt_open == 0) & (nxt_dow != 7)).astype(int)
    df["post_closure"] = ((prv_open == 0) & (prv_dow != 7)).astype(int)

    for target in ("sales", "customers"):
        _add_level(df, target)
    return df


def _add_level(df: pd.DataFrame, target: str) -> None:
    vals = df[target].where(df["is_open"] == 1)
    piv = (pd.DataFrame({"sale_date": df["sale_date"], "store_id": df["store_id"], "v": vals})
           .pivot(index="sale_date", columns="store_id", values="v"))
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
    roll = piv.rolling(LEVEL_WINDOW, min_periods=120).mean()
    level = roll.shift(LEVEL_LAG)
    growth = level / roll.shift(LEVEL_LAG + LEVEL_WINDOW)

    def to_long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        out = frame.stack().rename(name).reset_index()
        out.columns = ["sale_date", "store_id", name]
        return out

    lv = to_long(level, f"level_{target}")
    gr = to_long(growth, f"growth_{target}")
    merged = df[["sale_date", "store_id"]].merge(lv, how="left", on=["sale_date", "store_id"]) \
                                            .merge(gr, how="left", on=["sale_date", "store_id"])
    df[f"level_{target}"] = merged[f"level_{target}"].to_numpy()
    df[f"growth_{target}"] = merged[f"growth_{target}"].to_numpy()
