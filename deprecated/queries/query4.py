import duckdb
import pandas as pd
from pathlib import Path

# Query 4:
# Top 10 applications by "app_type", ranked by number of distinct clients (guid).
# We interpret:
#   app_type := attribute_level1
#   exe_name := proc_name
#
# Uses Globus parquet table:
#   DCA/frgnd_system_usage_by_app/*.parquet

BASE = Path(__file__).resolve().parent
FRGND_PATH = BASE / "frgnd_system_usage_by_app" / "*.parquet"

con = duckdb.connect(database=":memory:")

sql = f"""
WITH base AS (
  SELECT
    attribute_level1 AS app_type,
    proc_name        AS exe_name,
    guid
  FROM read_parquet('{FRGND_PATH.as_posix()}')
  WHERE proc_name IS NOT NULL
    AND lower(proc_name) NOT IN ('restricted process', 'desktop')
    AND attribute_level1 IS NOT NULL
),
agg AS (
  SELECT
    app_type,
    exe_name,
    COUNT(DISTINCT guid) AS number_of_systems
  FROM base
  GROUP BY app_type, exe_name
),
ranked AS (
  SELECT
    *,
    RANK() OVER (PARTITION BY app_type ORDER BY number_of_systems DESC) AS rank
  FROM agg
)
SELECT
  app_type,
  exe_name,
  number_of_systems,
  rank
FROM ranked
WHERE rank <= 10
ORDER BY app_type, rank ASC, number_of_systems DESC, exe_name;
"""

df = con.execute(sql).df()

print(df.head(50))
out_path = BASE / "query4_results.csv"
df.to_csv(out_path, index=False)
print(f"saved {out_path}")
