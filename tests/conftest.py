import json
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vantage import data_gen, warehouse  # noqa: E402

OUT = ROOT / "outputs"


@pytest.fixture(scope="session")
def small_raw(tmp_path_factory):
    """A small Rossmann-shaped sample (24 stores) generated with the real generator."""
    raw = tmp_path_factory.mktemp("raw")
    data_gen.generate(raw, n_stores=24, seed=381)
    return raw


@pytest.fixture(scope="session")
def small_con(small_raw):
    """In-memory DuckDB with the project's SQL models run on the small sample."""
    con = duckdb.connect(":memory:")
    for name in warehouse.MODEL_ORDER:
        sql = warehouse._compile((warehouse.MODELS_DIR / f"{name}.sql").read_text(), small_raw)
        con.execute(f"create or replace table {name.split('/')[-1]} as {sql}")
    return con


@pytest.fixture(scope="session")
def dashboard():
    """Pipeline output. Run `python run_pipeline.py` first."""
    p = OUT / "dashboard_data.json"
    if not p.exists():
        pytest.skip("run the pipeline first: python run_pipeline.py")
    return json.loads(p.read_text())
