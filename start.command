#!/bin/bash
# Vantage Horizon: one-click start for macOS.
# Double-click in Finder, or run:  ./start.command        (add --fresh to rebuild the forecast)
cd "$(dirname "$0")" || exit 1
PORT=8000

echo "== Vantage Horizon =="

# 1. Find Python 3.11+
PY=""
for c in python3.13 python3.12 python3.11 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Python 3.12 with Homebrew..."; brew install python@3.12 && PY=/opt/homebrew/bin/python3.12
  fi
fi
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/ and run this again."
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi
echo "Python: $($PY --version)"

# 2. Virtual environment + packages (first run only)
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1
if ! .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "Creating virtual environment..."
  "$PY" -m venv --clear .venv || exit 1
  rm -f .venv/.installed
fi
if [ ! -f .venv/.installed ]; then
  echo "Installing packages (first run, 1-3 minutes)..."
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt -q && touch .venv/.installed
fi

# LightGBM on macOS needs the OpenMP runtime
if ! .venv/bin/python -c "import lightgbm" 2>/dev/null; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing libomp for LightGBM..."; brew install libomp
  else
    echo "LightGBM needs libomp. Install Homebrew (https://brew.sh) then run: brew install libomp"
    read -n 1 -s -r -p "Press any key to close"; exit 1
  fi
fi

# 3. Build the forecast (first run, or with --fresh)
if [ "$1" == "--fresh" ] || [ ! -f outputs/dashboard_data.json ] || [ ! -f outputs/model_artifacts/models.pkl ]; then
  echo "Building data, models and forecast (about 3 minutes)..."
  .venv/bin/python run_pipeline.py || { read -n 1 -s -r -p "Pipeline failed. Press any key to close"; exit 1; }
fi

# 4. Start the server and open the dashboard
lsof -ti tcp:$PORT | xargs kill 2>/dev/null
echo ""
echo "Dashboard: http://localhost:$PORT      API docs: http://localhost:$PORT/docs"
echo "Leave this window open during the demo. Press Ctrl+C to stop."
(sleep 2 && open "http://localhost:$PORT") &
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port $PORT
