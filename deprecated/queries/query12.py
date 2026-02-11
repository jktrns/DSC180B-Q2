import duckdb
import pandas as pd

USERWAIT_GLOB = "userwait_v2/*.parquet"
OUT_CSV = "query12_results.csv"

con = duckdb.connect(database=":memory:")

sql = f"""
WITH uw AS (
    SELECT
        guid,
        event_name,
        UPPER(SUBSTR(ac_dc_event_name, 1, 2)) AS acdc,
        proc_name_current AS proc_name,
        CAST(duration_ms AS BIGINT) AS duration_ms
    FROM read_parquet('{USERWAIT_GLOB}')
    WHERE proc_name_current NOT IN ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
      AND event_name NOT IN ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
),
proc_rank AS (
    SELECT
        proc_name,
        (SUM(CAST(duration_ms AS DOUBLE)) / 1000.0) / NULLIF(COUNT(*), 0) AS total_duration_sec_per_instance,
        RANK() OVER (
            ORDER BY (SUM(CAST(duration_ms AS DOUBLE)) / 1000.0) / NULLIF(COUNT(*), 0) DESC
        ) AS rank,
        COUNT(*) AS number_of_instances,
        COUNT(DISTINCT guid) AS number_of_systems
    FROM uw
    GROUP BY proc_name
    HAVING COUNT(*) > 50
       AND COUNT(DISTINCT guid) > 20
),
top_procs AS (
    SELECT proc_name
    FROM proc_rank
    WHERE rank <= 20
),
by_proc_acdc AS (
    SELECT
        u.proc_name,
        u.acdc,
        COUNT(*) AS number_of_instances,
        SUM(CAST(u.duration_ms AS DOUBLE) / 1000.0) AS aggragated_duration_in_seconds,
        COUNT(DISTINCT u.guid) AS number_of_systems
    FROM uw u
    INNER JOIN top_procs p
        ON u.proc_name = p.proc_name
    GROUP BY u.proc_name, u.acdc
)
SELECT
    proc_name,
    SUM(CASE WHEN acdc = 'AC' THEN ROUND(aggragated_duration_in_seconds / NULLIF(number_of_instances, 0), 2) ELSE 0 END) AS ac_duration,
    SUM(CASE WHEN acdc = 'DC' THEN ROUND(aggragated_duration_in_seconds / NULLIF(number_of_instances, 0), 2) ELSE 0 END) AS dc_duration,
    SUM(CASE WHEN acdc = 'UN' THEN ROUND(aggragated_duration_in_seconds / NULLIF(number_of_instances, 0), 2) ELSE 0 END) AS unknown_duration
FROM by_proc_acdc
GROUP BY proc_name
ORDER BY proc_name;
"""

df = con.execute(sql).df()
print(df.head(50))
df.to_csv(OUT_CSV, index=False)
print(f"saved {OUT_CSV}")
