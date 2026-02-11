import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
FRGND_PATH = BASE / "frgnd_system_usage_by_app" / "*.parquet"

con = duckdb.connect(database=":memory:")

# Query 5 analogue:
# Top 10 apps per app_type ranked by total "detections per day".
# We approximate "lines_per_day" using SUM(nrs) as the count-like measure available.
sql = f"""
WITH daily AS (
  SELECT
    attribute_level1 AS app_type,
    proc_name        AS exe_name,
    guid,
    dt,
    SUM(COALESCE(nrs, 0)) AS lines_per_day
  FROM read_parquet('{FRGND_PATH.as_posix()}')
  WHERE proc_name IS NOT NULL
    AND lower(proc_name) NOT IN ('restricted process', 'desktop')
    AND attribute_level1 IS NOT NULL
    AND dt IS NOT NULL
  GROUP BY attribute_level1, proc_name, guid, dt
),
agg AS (
  SELECT
    app_type,
    exe_name,
    SUM(lines_per_day) AS total_num_detections
  FROM daily
  GROUP BY app_type, exe_name
),
ranked AS (
  SELECT
    app_type,
    exe_name,
    total_num_detections,
    RANK() OVER (PARTITION BY app_type ORDER BY total_num_detections DESC) AS rank
  FROM agg
)
SELECT
  app_type,
  exe_name,
  total_num_detections AS total_number_of_detections,
  rank
FROM ranked
WHERE rank <= 10
ORDER BY app_type, rank ASC;
"""

df = con.execute(sql).df()

print("Query 5 results (top rows):")
print(df.head(50))

out_path = BASE / "query5_results.csv"
df.to_csv(out_path, index=False)
print(f"saved {out_path}")
