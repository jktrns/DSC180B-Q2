import duckdb
import pandas as pd
from pathlib import Path

# Run from the folder that contains the "DCA" directory
# Example:
# /Users/mehakkapur/Desktop/dca/
# └── DCA/
#     ├── os_network_consumption_v2/0000_part_00.parquet
#     └── system_sysinfo_unique_normalized/0000_part_00.parquet

BASE = Path(__file__).resolve().parent
net_path = str(BASE / "os_network_consumption_v2" / "*.parquet")
sys_path = str(BASE / "system_sysinfo_unique_normalized" / "*.parquet")

con = duckdb.connect()

sql = f"""
WITH agg AS (
  SELECT
    guid,
    sum(nr_samples) AS nrs,
    sum(
      CASE
        WHEN input_description = 'OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::'
        THEN avg_bytes_sec::double * nr_samples::double * 5.0
        ELSE 0
      END
    ) AS received_bytes,
    sum(
      CASE
        WHEN input_description = 'OS:NETWORK INTERFACE::BYTES SENT/SEC::'
        THEN avg_bytes_sec::double * nr_samples::double * 5.0
        ELSE 0
      END
    ) AS sent_bytes
  FROM read_parquet('{net_path}')
  GROUP BY guid
  HAVING sum(nr_samples) > 720
)
SELECT
  a.guid,
  a.nrs,
  a.received_bytes,
  a.sent_bytes,
  b.chassistype,
  b.modelvendor_normalized AS vendor,
  b.model_normalized AS model,
  b.ram,
  b.os,
  b."#ofcores" AS number_of_cores
FROM agg a
JOIN read_parquet('{sys_path}') b
  ON a.guid = b.guid
WHERE a.sent_bytes > a.received_bytes
ORDER BY (a.sent_bytes - a.received_bytes) DESC;
"""

df = con.execute(sql).df()
print(df.head(20))

out_csv = BASE / "query2_results.csv"
df.to_csv(out_csv, index=False)
print(f"saved {out_csv}")
