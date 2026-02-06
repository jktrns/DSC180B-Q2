#!/usr/bin/env python3
"""
build_real_data.py

Build a single unified DuckDB table named `real_data` from multiple parquet folders,
then export it to `real_data.csv` in the same directory as this script.

How to run:
  cd /path/to/your/root_folder   # the folder that contains the parquet directories
  python build_real_data.py

Expected folders (relative to this script):
  os_network_consumption_v2/*.parquet
  frgnd_system_usage_by_app/*.parquet
  userwait_v2/*.parquet
  system_sysinfo_unique_normalized/*.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb


def _require_glob_has_files(glob_pat: str, label: str) -> None:
    # Simple fast check: ensure at least 1 match
    parent = Path(glob_pat).parent
    name_pat = Path(glob_pat).name
    matches = list(parent.glob(name_pat)) if parent.exists() else []
    if not matches:
        raise FileNotFoundError(f"No parquet files found for {label}: {glob_pat}")


def main() -> int:
    base = Path(__file__).resolve().parent

    net_glob = str(base / "os_network_consumption_v2" / "*.parquet")
    frgnd_glob = str(base / "frgnd_system_usage_by_app" / "*.parquet")
    uw_glob = str(base / "userwait_v2" / "*.parquet")
    sys_glob = str(base / "system_sysinfo_unique_normalized" / "*.parquet")

    print("Base directory:", base)

    # Validate inputs early
    try:
        _require_glob_has_files(net_glob, "os_network_consumption_v2")
        _require_glob_has_files(frgnd_glob, "frgnd_system_usage_by_app")
        _require_glob_has_files(uw_glob, "userwait_v2")
        _require_glob_has_files(sys_glob, "system_sysinfo_unique_normalized")
    except FileNotFoundError as e:
        print("\nERROR:", e)
        print("\nMake sure this script lives in the folder that directly contains those directories.")
        return 1

    db_file = base / "real_data.duckdb"
    out_csv = base / "real_data.csv"

    con = duckdb.connect(database=str(db_file))

    # DuckDB built-in progress bar (works for long queries)
    con.execute("PRAGMA enable_progress_bar=true;")
    # Update interval in ms for progress bar output (lower = more frequent)
    con.execute("PRAGMA progress_bar_time=500;")

    print("\nDuckDB file:", db_file)
    print("Will export CSV to:", out_csv)

    t0 = time.time()

    print("\nStep 1/3: Building table real_data (this can take a while)...")
    build_sql = f"""
    CREATE OR REPLACE TABLE real_data AS
    WITH
    network AS (
      SELECT
        guid::TEXT AS guid,
        NULL::DATE AS dt,
        'os_network_consumption_v2'::TEXT AS source_table,
        'network'::TEXT AS record_type,

        input_description::TEXT AS input_description,
        nr_samples::BIGINT AS nr_samples,
        avg_bytes_sec::DOUBLE AS avg_bytes_sec,

        NULL::TEXT AS app_type,
        NULL::TEXT AS exe_name,
        NULL::DOUBLE AS duration_sec,
        NULL::BIGINT AS nrs,

        NULL::TEXT AS event_name,
        NULL::TEXT AS acdc,
        NULL::TEXT AS proc_name,
        NULL::BIGINT AS duration_ms
      FROM read_parquet('{net_glob}')
    ),
    frgnd AS (
      SELECT
        guid::TEXT AS guid,
        dt::DATE AS dt,
        'frgnd_system_usage_by_app'::TEXT AS source_table,
        'foreground_usage'::TEXT AS record_type,

        NULL::TEXT AS input_description,
        NULL::BIGINT AS nr_samples,
        NULL::DOUBLE AS avg_bytes_sec,

        attribute_level1::TEXT AS app_type,
        proc_name::TEXT AS exe_name,
        duration::DOUBLE AS duration_sec,
        nrs::BIGINT AS nrs,

        NULL::TEXT AS event_name,
        NULL::TEXT AS acdc,
        NULL::TEXT AS proc_name,
        NULL::BIGINT AS duration_ms
      FROM read_parquet('{frgnd_glob}')
    ),
    userwait AS (
      SELECT
        guid::TEXT AS guid,
        NULL::DATE AS dt,
        'userwait_v2'::TEXT AS source_table,
        'userwait'::TEXT AS record_type,

        NULL::TEXT AS input_description,
        NULL::BIGINT AS nr_samples,
        NULL::DOUBLE AS avg_bytes_sec,

        NULL::TEXT AS app_type,
        NULL::TEXT AS exe_name,
        NULL::DOUBLE AS duration_sec,
        NULL::BIGINT AS nrs,

        event_name::TEXT AS event_name,
        UPPER(SUBSTR(ac_dc_event_name, 1, 2))::TEXT AS acdc,
        proc_name_current::TEXT AS proc_name,
        CAST(duration_ms AS BIGINT) AS duration_ms
      FROM read_parquet('{uw_glob}')
    )
    SELECT * FROM network
    UNION ALL
    SELECT * FROM frgnd
    UNION ALL
    SELECT * FROM userwait;
    """
    con.execute(build_sql)

    print("\nStep 2/3: Building sysinfo table (small, usually fast)...")
    sysinfo_sql = f"""
    CREATE OR REPLACE TABLE sysinfo AS
    SELECT
      guid::TEXT AS guid,
      chassistype::TEXT AS chassistype,
      modelvendor_normalized::TEXT AS vendor,
      model_normalized::TEXT AS model,
      ram AS ram,
      os::TEXT AS os,
      "#ofcores"::BIGINT AS number_of_cores
    FROM read_parquet('{sys_glob}');
    """
    con.execute(sysinfo_sql)

    elapsed = time.time() - t0

    # Quick sanity checks
    try:
        n_rows = con.execute("SELECT COUNT(*) FROM real_data;").fetchone()[0]
        by_type = con.execute(
            "SELECT record_type, COUNT(*) AS n FROM real_data GROUP BY 1 ORDER BY 2 DESC;"
        ).fetchall()
    except Exception:
        n_rows = None
        by_type = []

    print("\nDone.")
    if n_rows is not None:
        print(f"real_data rows: {n_rows:,}")
        if by_type:
            print("rows by record_type:")
            for rt, n in by_type:
                print(f"  {rt:16s} {n:,}")
    print(f"Saved DuckDB file: {db_file}")
    print(f"Saved CSV file:    {out_csv}")
    print(f"Total time:        {elapsed:.1f} sec")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())