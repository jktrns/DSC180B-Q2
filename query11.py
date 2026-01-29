import duckdb
import pandas as pd

USERWAIT_GLOB = "userwait_v2/*.parquet"
OUT_CSV = "query11_results.csv"

con = duckdb.connect(database=":memory:")

sql = f"""
WITH base AS (
  SELECT
      proc_name_current AS proc_name,
      COUNT(*) AS number_of_instances,
      SUM(CAST(duration_ms AS BIGINT)) AS total_duration_ms
  FROM read_parquet('{USERWAIT_GLOB}')
  WHERE proc_name_current NOT IN ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
    AND event_name NOT IN ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
  GROUP BY proc_name_current
)
SELECT
    proc_name,
    (CAST(total_duration_ms AS DOUBLE) / 1000.0) / NULLIF(number_of_instances, 0) AS total_duration_sec_per_instance,
    RANK() OVER (
        ORDER BY (CAST(total_duration_ms AS DOUBLE) / 1000.0) / NULLIF(number_of_instances, 0) DESC
    ) AS rank
FROM base
QUALIFY rank <= 10
ORDER BY rank ASC;
"""

df = con.execute(sql).df()
print(df.head(50))
df.to_csv(OUT_CSV, index=False)
print(f"saved {OUT_CSV}")
