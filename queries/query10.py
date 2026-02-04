import duckdb
import pandas as pd

# Update this if your folder name differs
USERWAIT_GLOB = "userwait_v2/*.parquet"
OUT_CSV = "query10_results.csv"

con = duckdb.connect(database=":memory:")

sql = f"""
WITH base AS (
  SELECT
      event_name,
      UPPER(SUBSTR(ac_dc_event_name, 1, 2)) AS acdc,
      proc_name_current AS proc_name,
      COUNT(*) AS number_of_instances,
      SUM(CAST(duration_ms AS BIGINT)) AS total_duration_ms
  FROM read_parquet('{USERWAIT_GLOB}')
  GROUP BY event_name, acdc, proc_name
)
SELECT
    event_name,
    acdc,
    proc_name,
    ROUND((CAST(total_duration_ms AS DOUBLE) / 1000.0) / NULLIF(number_of_instances, 0), 2) AS total_duration_sec_per_instance,
    RANK() OVER (
        PARTITION BY event_name, acdc
        ORDER BY (CAST(total_duration_ms AS DOUBLE) / 1000.0) / NULLIF(number_of_instances, 0) DESC
    ) AS rank
FROM base
WHERE proc_name NOT IN ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
  AND event_name NOT IN ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
QUALIFY rank <= 10
ORDER BY acdc, event_name, rank ASC;
"""

df = con.execute(sql).df()
print(df.head(50))
df.to_csv(OUT_CSV, index=False)
print(f"saved {OUT_CSV}")
