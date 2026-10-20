"""A tiny dbt-compatible runner on DuckDB.

DuckDB stands in for the Fabric Lakehouse locally. The SQL in models/ uses dbt's
{{ ref() }} / {{ source() }} syntax, so the same files run under dbt-fabric
in Microsoft Fabric without edits.
"""
from __future__ import annotations

import re
from pathlib import Path
import duckdb

from vantage.config import MODELS_DIR, RAW_DIR, WAREHOUSE

# Run order: staging before marts, dependencies inside marts by name
MODEL_ORDER = [
    "staging/stg_sales", "staging/stg_plan", "staging/stg_stores",
    "marts/fct_store_day", "marts/fct_monthly_kpi",
]


def _compile(sql: str, raw_dir: Path) -> str:
    sql = re.sub(r"\{\{\s*ref\('(\w+)'\)\s*\}\}", r"\1", sql)

    def src(m: re.Match) -> str:
        path = (raw_dir / f"{m.group(2)}.csv").as_posix()
        return f"read_csv_auto('{path}', header=true, sample_size=-1)"

    return re.sub(r"\{\{\s*source\('(\w+)',\s*'(\w+)'\)\s*\}\}", src, sql)


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(WAREHOUSE), read_only=read_only)


def run_models(raw_dir: Path = RAW_DIR) -> duckdb.DuckDBPyConnection:
    con = connect()
    for name in MODEL_ORDER:
        sql = _compile((MODELS_DIR / f"{name}.sql").read_text(), raw_dir)
        table = name.split("/")[-1]
        con.execute(f"create or replace table {table} as {sql}")
        n = con.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"  model {table:<18} {n:>10,} rows")
    return con
