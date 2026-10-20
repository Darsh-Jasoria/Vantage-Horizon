# Vantage Horizon

**Team Vantage (381) · Microsoft Innovate 2026 · Dashboards with BI (and the Modern Data Stack) · Option 10: Seeing Next Month, Not Just Last Month**

Vantage Horizon predicts where retail performance is heading, detects leading indicators before revenue problems fully appear, explains why, simulates possible interventions, recommends the strongest modeled action, and delivers that decision through Power BI and Microsoft Teams.

**Predict → Explain → Simulate → Recommend → Act → Measure**

## Why Vantage Horizon is different

| Traditional BI | Vantage Horizon |
|---|---|
| "What happened?" | "What happens next?" · "Why?" · "What can we do?" · "Which action is best?" · "How confident are we?" |

Existing BI falls short here. Power BI's built-in forecast is ETS on a single line chart: one series, no outside drivers, no alerts on predicted values, no scenarios. "Explain"-style features explain the past. Month-end packs arrive days after the period closes (median close is 6.4 days, APQC).

## What it does

| Step | What the user sees | How |
|---|---|---|
| **Predict** | Next month / +60 / +90 days: forecast, 80% and 95% ranges, chance of hitting target | LightGBM store-day model + SARIMAX monthly model, blended; log-normal error model from backtest errors |
| **Explain** | Top drivers of the change vs last month; leading-indicator alerts | SHAP (TreeExplainer) on the store-day model; weekly footfall z-scores vs model expectation |
| **Simulate** | Scenario Lab: Base, Recovery, Downside, and every tested intervention | Promo days re-scored through the LightGBM model; footfall/ticket as manual multipliers |
| **Recommend** | "Action required" card and the recommended scenario with modeled impact | Transparent decision score (below) |
| **Act** | Microsoft Teams action card with buttons back to the dashboard | Adaptive Card 1.4 via an incoming webhook (optional) |
| **Measure** | "Can I trust it?": accuracy, interval coverage, baseline comparison | Out-of-sample rolling backtest |

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py --sample        # generate data -> SQL models -> forecast -> decisions (about 1-3 min)
uvicorn api.main:app --port 8000       # open http://localhost:8000
pip install -r requirements-dev.txt && python -m pytest -q    # 17 tests
```

On a Mac: double-click `start.command` (first run installs packages and builds the forecast). `./start.command --fresh` rebuilds the forecast.

After changing decision settings in `vantage/config.py`, `python -m vantage.refresh` rebuilds decisions without re-running the forecast.

## Controlled synthetic data (read this)

The demo runs on a **synthetic dataset in the Kaggle Rossmann Store Sales format** (`vantage/data_gen.py`, seed 381):
- 1,115 stores, 1,222,040 daily rows (Oct 2023 to Sep 2026), plus a planned calendar for Oct–Dec 2026.
- Realistic weekday, seasonal, promo, holiday and competition effects.
- A footfall slowdown is **deliberately planted** in store type D from 1 Sep 2026 (about −8%), to demonstrate the early-warning flow. It is a controlled demo scenario, **not evidence of real-world accuracy**.

Accuracy numbers below are measured on this data and will be worse on real retail data. Real Rossmann files can be placed in `data/raw/`; the 2014 gap and its shorter future calendar would need handling.

## Architecture

**Production (Microsoft stack)**

```
Microsoft Fabric Lakehouse  ->  dbt semantic models  ->  ML forecasting (Fabric notebook / scheduled job)
        ->  fct_kpi_forecast semantic table  ->  Power BI (DAX measures)
        ->  Azure OpenAI (executive explanation)  ->  Microsoft Teams (action card)
```

**This offline demo (what actually runs)**

| Production part | Local stand-in |
|---|---|
| Fabric Lakehouse | DuckDB file `data/warehouse.duckdb` + CSVs |
| dbt on Fabric | The same dbt-style SQL (`models/*.sql`) compiled and run on DuckDB (`vantage/warehouse.py`) |
| Fabric notebook | `run_pipeline.py` (Python) |
| Power BI | `outputs/fct_kpi_forecast.csv` + `powerbi/measures.dax`. **The Power BI report itself has not been built yet.** |
| Azure OpenAI | Optional. Without credentials a deterministic template with the same structure is used. |
| Teams | Optional. The card is always written to `outputs/teams_alert.json`; it is posted only if `TEAMS_WEBHOOK_URL` is set. |

The web dashboard (`web/index.html`) is served by FastAPI (`api/main.py`).

## Forecasting methodology

- **Features** (`vantage/features.py`): 14 inputs, all known 90 days ahead: planned promo and holiday calendar, weekday, month, pre/post-closure flags, store attributes, and the store's normal level (364-day mean **lagged 92 days**). A test overwrites every post-cutoff actual with random values and checks the forecast features do not change.
- **LightGBM** predicts each store-day relative to the store's level; predictions are summed to any segment.
- **SARIMAX** on monthly totals per segment, with promo share and trading days as drivers.
- **Momentum**: 90% / 85% / 80% of last month's forecast surprise is carried into horizons 1/2/3.
- **Ensemble**: geometric blend, weight on LightGBM ∝ 1 / backtest MAPE (63% for sales).
- **Ranges**: σ = RMSE of log forecast error (shrunk toward the pooled value, floor 1.5%), widened 15% per month ahead; 80% = ŷ·exp(±1.28σ), 95% = ŷ·exp(±1.96σ).
- **Target probability**: P(hit) = 1 − Φ((ln target − ln ŷ)/σ). Target = same month last year +3%.
- **Store forecasts** are scaled so they add up exactly to their store type's forecast (proportional top-down reconciliation).

## Backtesting and interval calibration (`vantage/trust.py`)

Rolling origin, one month ahead, last 6 months, 5 segments (all stores + 4 store types). Each month is scored with ensemble weights and interval widths estimated **from the other months only** (leave-one-month-out), so the reported accuracy and coverage are out-of-sample.

| Method (all stores, sales) | MAPE |
|---|---|
| **Vantage ensemble** | **1.2%** (average miss about €2.9M) |
| LightGBM alone | 1.1% |
| SARIMAX alone | 1.9% |
| ETS, Power BI's built-in method | 3.4% |
| Same month last year | 5.7% |

Interval coverage on the backtest: **28 of 30** outcomes inside the 80% range (93%) and **28 of 30** inside the 95% range (93%). Thirty points is a small sample; treat this as a sanity check, not proof of calibration. Only one-month-ahead accuracy is backtested.

## Decision engine (`vantage/decision.py`)

For each month and segment it produces a decision object: risk level, affected segment, target, forecast and ranges, probability of hitting target, leading indicator, top reason and drivers, controllable levers, tested scenarios, recommended scenario, business impact (target gap, downside exposure, modeled upside, probability gain) and evidence level. Served at `GET /api/decision`.

**Tested interventions**: +0 to +6 chain-wide promo days (the only lever the model represents).

**Decision score** (weights in `config.DECISION_WEIGHTS`):

```
score = 1.0 × gain in chance to hit target, counted only up to 80% (points)
      + 0.5 × modeled sales gain (%)
      − 3.0 × promo days added              (intervention complexity proxy; the data has no cost data)
      − 2.0 × points of promo share above the highest share seen in history (extrapolation penalty)
```

An action is recommended only when the chance to hit target is below 60% and the best score is positive; otherwise "keep current plan". Capping the probability gain at 80% makes the engine prefer the smallest intervention that gets the target safely likely, instead of always the largest.

**Scenarios** (`GET /api/scenarios`): **Base** (current plan) · **Recovery** (best-scoring intervention) · **Downside** (footfall falls a further max(3%, the segment's latest shortfall), no extra promos).

**Wording rule:** all impacts are "modeled" scenario estimates. The promo response is learned from historical association; it is **not causal proof**.

Example (October 2026, store type D, controlled demo scenario): footfall 7.6% below expected for 5 of 5 weeks → 18% chance to hit the €79.0M target → 6 interventions tested → **+5 promo days**: +€4.4M modeled sales, chance 18% → 80%. For all stores the engine recommends +2 promo days (34% → 84%).

## Executive explanation and Teams

`vantage/narrate.py` answers exactly **WHAT? / WHY? / WHAT SHOULD I DO?** from the decision object only. With `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`, GPT-4o rewrites it under a prompt that forbids new numbers; any error falls back to the template.

`vantage/alerts.py` builds the Teams card "VANTAGE HORIZON — ACTION REQUIRED" with affected segment, risk, forecast, target, probability, leading indicator, top reason, recommended scenario and modeled impact, plus **Open dashboard / View forecast / Open scenario** buttons (deep links: `?segment=d&month=2026-10#scenarios`; set `VH_DASHBOARD_URL`).

## API

`GET /api/dashboard` · `/api/forecast` · `/api/drivers` · `/api/alerts` · `/api/narrative` · `/api/decision` · `/api/scenarios` · `/api/trust` · `POST /api/whatif` · docs at `/docs`.

## Power BI

`powerbi/measures.dax` works on `fct_kpi_forecast` (actual, forecast and out-of-sample backtest rows). Every measure filters a single segment (the table holds the "All" total and each store type, so summing across them double counts). `Backtest MAPE %` pairs each backtest forecast with the actual of the same month; `Backtest 80% coverage` uses the backtest ranges. A test re-computes the MAPE logic from the CSV and checks it equals the pipeline's figure.

## Audit notes (fixed in v0.2)

- Backtest accuracy was in-sample (ensemble weights and interval widths were tuned on the months being scored). Now leave-one-month-out.
- Prophet silently fell back to a seasonal-naive forecast but kept the "Prophet" label. Prophet is now an optional extra (`VH_PROPHET=1`) and is reported only when it actually runs.
- DAX measures summed across all segments (double counting the total) and the backtest-error measure compared against every historical month. Both fixed.
- Leakage, forecast origin, feature availability, aggregation and interval ordering are covered by tests.

## Tests (`tests/`)

Sample-data generation, three-month horizon, no leakage of future actuals into features, interval ordering and widening, p_hit in [0,1], coverage counts, store-to-segment reconciliation, scenarios responding to interventions, decision engine always returning a valid recommendation with a transparent score, and the DAX logic.

## Limitations

- Synthetic data; the planted slowdown proves the flow works, not real-world accuracy.
- Only promo days are a modeled lever; footfall and ticket changes are assumptions. Promo effects are associational, not causal.
- Six months, one month ahead, is a small backtest; 60/90-day ranges are widened by a rule, not separately backtested.
- No intervention cost data, so the decision score uses a complexity proxy.
- Power BI report not built; Fabric, Azure OpenAI and Teams are not connected in the demo.
- Single-user local prototype: no authentication.
