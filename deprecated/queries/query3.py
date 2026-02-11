import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
frgnd_path = str(BASE / "frgnd_system_usage_by_app" / "*.parquet")

con = duckdb.connect()

# 1) Inspect metric_name values that look like focal/focus
inspect_sql = f"""
SELECT metric_name, COUNT(*) AS n
FROM read_parquet('{frgnd_path}')
WHERE metric_name ILIKE '%FOCAL%' OR metric_name ILIKE '%FOCUS%'
GROUP BY metric_name
ORDER BY n DESC
LIMIT 50;
"""
inspect_df = con.execute(inspect_sql).df()
print("\nMetric names containing FOCAL/FOCUS (top 50):")
print(inspect_df)

# 2) Main query: compute avg daily focal time per app_type using duration (fallback-safe)
# This avoids guessing metric_name and usually matches what "focal time" means in practice:
# total foreground duration per guid-day per process, averaged across days/systems.
sql = f"""
WITH per_day AS (
  SELECT
    guid,
    dt,
    attribute_level1 AS app_type,
    proc_name AS exe_name,
    SUM(duration) AS focal_sec_per_day
  FROM read_parquet('{frgnd_path}')
  WHERE proc_name NOT IN ('restricted process', 'desktop')
    AND attribute_level1 IS NOT NULL
    AND dt IS NOT NULL
    AND duration IS NOT NULL
    AND duration > 0
  GROUP BY guid, dt, attribute_level1, proc_name
),
base AS (
  SELECT
    app_type,
    exe_name,
    AVG(focal_sec_per_day) AS average_focal_sec_per_day
  FROM per_day
  GROUP BY app_type, exe_name
),
ranked AS (
  SELECT
    app_type,
    exe_name,
    ROUND(average_focal_sec_per_day) AS average_focal_sec_per_day,
    RANK() OVER (PARTITION BY app_type ORDER BY average_focal_sec_per_day DESC) AS rank
  FROM base
)
SELECT *
FROM ranked
WHERE rank <= 10
ORDER BY app_type, rank ASC;
"""

df = con.execute(sql).df()

print("\nQuery 3 results (top rows):")
print(df.head(50))

out_csv = BASE / "query3_results.csv"
df.to_csv(out_csv, index=False)
print(f"\nsaved {out_csv}")
