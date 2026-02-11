# Data

This folder contains Intel DCA telemetry data downloaded from Globus, plus derived reporting tables built by the pipeline. All data files are gitignored (~32 GiB total). Only this README and manifest JSON files are tracked.

## How to reproduce

1. Set up [Globus Connect Personal](https://www.globus.org/globus-connect-personal)
2. Access the HDSI Industry Data Repository collection at path `/projects/dca/`
3. Transfer the files listed below to this `data/` directory
4. Run `notebooks/03-build-reporting-tables.ipynb` to build the reporting schema
5. Run `notebooks/04-run-benchmark-queries.ipynb` to generate ground truth results

## Download list

### From `university_analysis_pad/`

| Path | Size | Notes |
|---|---|---|
| `system_sysinfo_unique_normalized/` (all 8 parquets) | 77.5 MiB | Anchor table, 1M clients |
| `data_dictionary/` (all parquets) | 40 KiB | Schema reference |

### From `university_prod/`

Download only the first parquet file from each (partial samples):

| Path | Size | Notes |
|---|---|---|
| `hw_metric_stats/0000_part_00.parquet` | 1.17 GiB | Hardware metrics (power, C0, freq, temp) |
| `os_network_consumption_v2/0000_part_00.parquet` | 1.81 GiB | Network bytes/sec |
| `os_memsam_avail_percent/0000_part_00.parquet` | 2.03 GiB | Memory utilization |
| `web_cat_usage_v2/0000_part_00.parquet` | 864 MiB | Browser usage by category |
| `web_cat_pivot/` (all 8 parquets) | 55.4 MiB | Pre-pivoted web categories |
| `userwait_v2/0000_part_00.parquet` | 4.89 GiB | User wait events (1 of 16 files) |

Also grab the `-manifest.json` for each table (a few KB each).

### From `university_prod/dca_update_dec_2024/`

These are gzipped text files from Intel's December 2024 data update. Place directly in `data/`:

| Path | Size | Notes |
|---|---|---|
| `system_cpu_metadata/system_cpu_metadata.txt000.gz` | 42.5 MiB | CPU specs per client |
| `system_os_codename_history/system_os_codename_history.txt000.gz` | 17.6 MiB | Windows version history |
| `guids_on_off_suspend_time_day/guids_on_off_suspend_time_day.txt000.gz` | 16.8 MiB | Daily on/off/sleep times |
| `mods_sleepstudy_top_blocker_hist/mods_sleepstudy_top_blocker_hist.txt000.gz` | 1.89 GiB | Modern standby blockers |
| `__tmp_batt_dc_events/__tmp_batt_dc_events.txt000.gz` | 12 MiB | Battery DC events |
| `display_devices/display_devices.txt000.gz` | 6.16 GiB | Display device usage |
| `__tmp_fgnd_apps_date/__tmp_fgnd_apps_date.txt003.gz` | 1.53 GiB | Foreground app usage (file 3 of 4) |
| `mods_sleepstudy_power_estimation_data_13wks/mods_sleepstudy_power_estimation_data_13wks.txt000.gz` | 218 KB | Power consumption (stub: 1 guid) |

## Expected total download

~20.7 GiB across all files.

## Directory structure after setup

```
data/
├── system_sysinfo_unique_normalized/   # 8 parquets
├── data_dictionary/                    # 1 parquet
├── hw_metric_stats/                    # 1 parquet + manifest
├── os_network_consumption_v2/          # 1 parquet + manifest
├── os_memsam_avail_percent/            # 1 parquet + manifest
├── web_cat_usage_v2/                   # 1 parquet + manifest
├── web_cat_pivot/                      # 8 parquets + manifest
├── userwait_v2/                        # 1 parquet + manifest
├── *.txt000.gz                         # 8 gzipped text files from dca_update_dec_2024
├── reporting/                          # Built by notebook 03 (19 parquet files, ~11.5 GiB)
│   ├── system_sysinfo_unique_normalized.parquet
│   ├── system_network_consumption.parquet
│   ├── system_userwait.parquet
│   ├── system_memory_utilization.parquet
│   └── ... (15 more reporting tables)
├── results/                            # Built by notebook 04
│   └── real/                           # Ground truth query results (24 CSVs)
│       ├── avg_platform_power_c0_freq_temp_by_chassis.csv
│       └── ... (23 more)
└── README.md                           # This file (tracked in git)
```

## Query coverage

24/24 benchmark queries execute successfully against the reporting tables. 21 queries use multi-guid population data (thousands of clients). 3 queries use single-guid stub data (power consumption rankings).

See `notebooks/02-query-feasibility.ipynb` for full column-level verification.
