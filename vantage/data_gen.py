"""Generate a Rossmann-shaped sample (train.csv, test.csv, store.csv).

Same columns as the Kaggle "Rossmann Store Sales" files, so the real data can be
dropped into data/raw/ and the rest of the pipeline runs unchanged.

The sample has 1,115 stores, daily rows from Oct 2023 to Sep 2026 (actuals) and a
planned calendar (promos, closures, holidays) for Oct-Dec 2026 in test.csv.
A footfall slowdown is planted in store type D from 1 Sep 2026 so the
early-warning detector has something real to find.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ACTUAL_START = "2023-10-01"
ACTUAL_END = "2026-09-30"
PLAN_END = "2026-12-31"

# German public holidays (Rossmann is a German chain): code a = public, b = Easter, c = Christmas
HOLIDAYS = {
    "a": ["2024-01-01", "2024-05-01", "2024-05-09", "2024-05-20", "2024-10-03",
          "2025-01-01", "2025-05-01", "2025-05-29", "2025-06-09", "2025-10-03",
          "2026-01-01", "2026-05-01", "2026-05-14", "2026-05-25", "2026-10-03"],
    "b": ["2024-03-29", "2024-04-01", "2025-04-18", "2025-04-21", "2026-04-03", "2026-04-06"],
    "c": ["2023-12-25", "2023-12-26", "2024-12-25", "2024-12-26", "2025-12-25", "2025-12-26",
          "2026-12-25", "2026-12-26"],
}

SCHOOL_HOLIDAYS = [
    ("2023-10-23", "2023-11-03"), ("2023-12-21", "2024-01-05"), ("2024-03-25", "2024-04-06"),
    ("2024-07-15", "2024-08-27"), ("2024-10-21", "2024-11-02"), ("2024-12-23", "2025-01-06"),
    ("2025-04-14", "2025-04-26"), ("2025-07-14", "2025-08-26"), ("2025-10-20", "2025-10-31"),
    ("2025-12-22", "2026-01-06"), ("2026-03-30", "2026-04-11"), ("2026-07-13", "2026-08-25"),
    ("2026-10-19", "2026-10-30"), ("2026-12-21", "2027-01-06"),
]

MONTH_SEASON = np.array([0.93, 0.95, 0.98, 1.00, 0.99, 1.00, 1.02, 0.97, 0.96, 0.98, 1.03, 1.20])
DOW_FACTOR = np.array([1.17, 1.03, 0.97, 0.97, 1.04, 0.86, 0.90])   # Mon..Sun


def _stores(rng: np.random.Generator, n: int) -> pd.DataFrame:
    s = pd.DataFrame({"Store": np.arange(1, n + 1)})
    s["StoreType"] = rng.choice(list("abcd"), n, p=[0.54, 0.02, 0.13, 0.31])
    s["Assortment"] = rng.choice(list("abc"), n, p=[0.53, 0.01, 0.46])
    s["CompetitionDistance"] = np.round(rng.lognormal(7.6, 1.1, n), -1)
    s["CompetitionOpenSinceMonth"] = rng.integers(1, 13, n)
    s["CompetitionOpenSinceYear"] = rng.integers(2005, 2016, n)
    promo2 = rng.random(n) < 0.5
    s["Promo2"] = promo2.astype(int)
    s["Promo2SinceWeek"] = np.where(promo2, rng.integers(1, 50, n), np.nan)
    s["Promo2SinceYear"] = np.where(promo2, rng.integers(2010, 2016, n), np.nan)
    s["PromoInterval"] = np.where(
        promo2, rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"], n), "")
    return s


def generate(raw_dir: Path, n_stores: int = 1115, seed: int = 381) -> None:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stores = _stores(rng, n_stores)

    dates = pd.date_range(ACTUAL_START, PLAN_END, freq="D")
    nd, ns = len(dates), n_stores
    dow = dates.dayofweek.to_numpy()                       # 0 = Monday
    month_idx = dates.month.to_numpy() - 1
    years = (dates - pd.Timestamp(ACTUAL_START)).days.to_numpy() / 365.25

    # Calendar: state holidays, school holidays, chain-wide promo weeks
    state_hol = np.array(["0"] * nd, dtype=object)
    for code, days in HOLIDAYS.items():
        state_hol[dates.isin(pd.to_datetime(days))] = code
    school = np.zeros(nd, dtype=int)
    for a, b in SCHOOL_HOLIDAYS:
        school[(dates >= a) & (dates <= b)] = 1
    week_no = ((dates - dates[0]).days.to_numpy() // 7)
    promo = ((week_no % 2 == 0) & (dow < 5)).astype(int)
    # Q4 gets an extra promo week each year (planned festive push)
    extra = (dates.month == 11) & (dates.day >= 16) & (dates.day <= 22) & (dow < 5)
    promo[extra] = 1

    # Store attributes
    stype = stores["StoreType"].to_numpy()
    type_mult = pd.Series(stype).map({"a": 1.0, "b": 1.55, "c": 1.0, "d": 0.97}).to_numpy()
    assort_mult = np.where(stores["Assortment"].to_numpy() == "c", 1.08, 1.0)
    comp = 1 - 0.09 * np.exp(-stores["CompetitionDistance"].to_numpy() / 1500)
    base_cust = rng.lognormal(np.log(640), 0.32, ns) * type_mult * assort_mult * comp
    base_ticket = rng.lognormal(np.log(9.6), 0.10, ns)
    growth = pd.Series(stype).map({"a": 0.035, "b": 0.02, "c": 0.04, "d": 0.06}).to_numpy()
    sunday_open = (stype == "b") | (rng.random(ns) < 0.02)

    # Open flag: closed Sundays (most stores) and public holidays; a few refurb closures
    open_ = np.ones((nd, ns), dtype=int)
    open_[dow == 6, :] = 0
    open_[np.ix_(dow == 6, sunday_open)] = 1
    open_[state_hol != "0", :] = 0
    for st in rng.choice(ns, min(25, max(1, ns // 10)), replace=False):   # 25 for the full 1,115 stores
        start = rng.integers(0, nd - 40)
        open_[start:start + rng.integers(5, 30), st] = 0

    # Pre-holiday rush: day before a non-Sunday closure
    nxt_hol = np.roll(state_hol != "0", -1)
    pre_rush = np.where(nxt_hol, 1.22, 1.0)

    # Footfall (customers) and basket (ticket)
    cust = (base_cust[None, :]
            * DOW_FACTOR[dow][:, None]
            * MONTH_SEASON[month_idx][:, None]
            * np.exp(growth[None, :] * years[:, None])
            * np.where(promo == 1, 1.25, 1.0)[:, None]
            * np.where(school == 1, 1.03, 1.0)[:, None]
            * pre_rush[:, None])
    month_key = dates.year * 12 + dates.month
    _, mk = np.unique(month_key, return_inverse=True)
    shock = rng.normal(0, 0.03, (mk.max() + 1, ns))[mk]          # store-month shocks
    cust = cust * np.exp(shock + rng.normal(0, 0.07, (nd, ns)))

    # Planted leading-indicator slowdown: type D footfall -8% from 1 Sep 2026
    slow = (dates >= "2026-09-01")[:, None] & (stype == "d")[None, :]
    cust = np.where(slow, cust * 0.92, cust)

    ticket = (base_ticket[None, :]
              * np.where(promo == 1, 1.10, 1.0)[:, None]
              * np.where(dates.month == 12, 1.05, 1.0)[:, None]
              * np.exp(rng.normal(0, 0.04, (nd, ns))))
    sales = np.round(cust * ticket) * open_
    cust = np.round(cust) * open_

    long = pd.DataFrame({
        "Store": np.tile(stores["Store"].to_numpy(), nd),
        "DayOfWeek": np.repeat(dow + 1, ns),
        "Date": np.repeat(dates.strftime("%Y-%m-%d"), ns),
        "Sales": sales.ravel().astype(int),
        "Customers": cust.ravel().astype(int),
        "Open": open_.ravel(),
        "Promo": np.repeat(promo, ns),
        "StateHoliday": np.repeat(state_hol, ns),
        "SchoolHoliday": np.repeat(school, ns),
    })
    is_actual = long["Date"] <= ACTUAL_END
    train = long[is_actual].sort_values(["Date", "Store"], ascending=[False, True])
    test = long[~is_actual].drop(columns=["Sales", "Customers"]).reset_index(drop=True)
    test.insert(0, "Id", np.arange(1, len(test) + 1))

    train.to_csv(raw_dir / "train.csv", index=False)
    test.to_csv(raw_dir / "test.csv", index=False)
    stores.to_csv(raw_dir / "store.csv", index=False)


if __name__ == "__main__":
    from vantage.config import RAW_DIR
    generate(RAW_DIR)
    print("sample data written to", RAW_DIR)
