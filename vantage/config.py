"""Paths and settings shared by the pipeline and the API."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
MODELS_DIR = ROOT / "models"          # dbt-style SQL models
OUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = OUT_DIR / "model_artifacts"

HORIZON_MONTHS = 3                    # 30 / 60 / 90-day outlook
BACKTEST_MONTHS = 6                   # rolling-origin backtest window
TARGET_GROWTH = 0.03                  # monthly target = same month last year x 1.03
TARGET_MAPE = 10.0                    # deck target: <= 10% MAPE on monthly revenue
MODEL_VERSION = "vh-0.2.0"

# Prophet is an optional extra baseline (slow to install/train). Off unless VH_PROPHET=1 and installed.
ENABLE_PROPHET = os.getenv("VH_PROPHET", "0") == "1"

# ---------------------------------------------------------------- decision engine
# Only segments whose chance of hitting target is below this get an intervention recommended.
ACTION_THRESHOLD = 0.60
# Promo-day interventions the engine tests (added chain-wide weekdays in the month).
DECISION_PROMO_OPTIONS = [0, 1, 2, 3, 4, 5, 6]
# Transparent decision score (see vantage/decision.py):
#   score = prob_pp * (gain in target probability up to CONFIDENCE_CAP, percentage points)
#         + sales_pct * (modeled sales change, %)
#         - complexity_per_day * (promo days added)           # intervention complexity proxy, NOT money
#         - extrapolation_pp * (promo share above the highest share seen in history, percentage points)
# Probability above this level earns no extra score ("confident enough"), so the engine prefers the
# smallest intervention that reaches it instead of always the biggest one.
CONFIDENCE_CAP = 0.80
DECISION_WEIGHTS = {"prob_pp": 1.0, "sales_pct": 0.5, "complexity_per_day": 3.0, "extrapolation_pp": 2.0}
# Downside stress test: footfall falls a further max(this, size of the segment's latest shortfall) %.
DOWNSIDE_MIN_FOOTFALL_PCT = 3.0

# Where the dashboard runs (used for buttons in the Teams card)
DASHBOARD_URL = os.getenv("VH_DASHBOARD_URL", "http://localhost:8000")

# Optional integrations (switch on by setting env vars)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")        # https://<name>.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")                # Teams Workflows incoming webhook

SEGMENT_LABELS = {
    "All": "All stores",
    "a": "Store type A",
    "b": "Store type B",
    "c": "Store type C",
    "d": "Store type D",
}
