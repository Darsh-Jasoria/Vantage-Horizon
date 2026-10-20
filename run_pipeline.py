"""Vantage Horizon nightly job: data -> SQL models -> forecasts -> drivers -> alerts -> exports.

    python run_pipeline.py            # uses data/raw/*.csv, generates a sample if missing
    python run_pipeline.py --sample   # regenerate the Rossmann-shaped sample first
"""
from __future__ import annotations

import argparse
import time

from vantage import data_gen, warehouse, forecast, drivers, export
from vantage.config import RAW_DIR, OUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="regenerate the sample data")
    args = ap.parse_args()
    t0 = time.time()

    if args.sample or not (RAW_DIR / "train.csv").exists():
        print("[1/5] Generating Rossmann-shaped sample data")
        data_gen.generate(RAW_DIR)
    else:
        print("[1/5] Using data in", RAW_DIR)

    print("[2/5] Running SQL models (DuckDB, dbt syntax)")
    con = warehouse.run_models()

    print("[3/5] Backtesting and forecasting (LightGBM + SARIMAX, ETS baseline)")
    res = forecast.run(con)

    print("[4/5] SHAP drivers")
    drv = drivers.compute(res)

    print("[5/5] Alerts, narrative, exports")
    data = export.write_all(con, res, drv)
    con.close()

    nxt = data["future_months"][0]
    f = data["series"]["All"]["sales"]["forecast"][0]
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUT_DIR}")
    print(f"Next month ({nxt}) sales: EUR {f['p50'] / 1e6:,.1f}M "
          f"(80%: {f['lo80'] / 1e6:,.1f}-{f['hi80'] / 1e6:,.1f}M)")
    print(data["narratives"][nxt]["All"]["text"])


if __name__ == "__main__":
    main()
