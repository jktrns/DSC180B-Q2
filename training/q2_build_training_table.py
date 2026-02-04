import duckdb
from pathlib import Path


net_path = "/Users/hanatjendrawasi/Globus/Intel Telemetry/os_network_consumption_v2"
sys_path = "/Users/hanatjendrawasi/Globus/Intel Telemetry/system_sysinfo_unique_normalized"
out_path = "/Users/hanatjendrawasi/Desktop/DSC180B-Q2/training/train_query2_one_row_per_guid.parquet"

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
  (a.sent_bytes - a.received_bytes) AS delta_bytes,
  b.chassistype,
  b.modelvendor_normalized AS vendor,
  b.model_normalized AS model,
  b.ram,
  b.os,
  b."#ofcores" AS number_of_cores
FROM agg a
JOIN read_parquet('{sys_path}') b
  ON a.guid = b.guid
"""

con.execute(f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET);")
print("Wrote:", out_path)

