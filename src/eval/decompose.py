from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_RAM_GB = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128])


def snap_ram(val):
    return int(STANDARD_RAM_GB[np.argmin(np.abs(STANDARD_RAM_GB - val))])


def decompose_wide_table(synth_wide: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sw = synth_wide
    counts = {}

    sysinfo = sw[
        [
            "guid",
            "chassistype",
            "countryname_normalized",
            "modelvendor_normalized",
            "ram",
            "os",
            "cpuname",
            "cpucode",
            "cpu_family",
            "persona",
            "processornumber",
        ]
    ].copy()
    sysinfo["ram"] = pd.to_numeric(sysinfo["ram"], errors="coerce").fillna(8)
    sysinfo["ram"] = sysinfo["ram"].apply(snap_ram)
    sysinfo.to_parquet(
        output_dir / "system_sysinfo_unique_normalized.parquet", index=False
    )
    counts["sysinfo"] = len(sysinfo)

    # CPU metadata table (needed by battery_on_duration and on_off_mods queries)
    if "cpucode" in sw.columns and "cpu_family" in sw.columns:
        cpu_meta = pd.DataFrame(
            {
                "guid": sw["guid"].values,
                "marketcodename": sw["cpucode"].values,
                "cpugen": sw["cpu_family"].values,
            }
        )
        cpu_meta.to_parquet(
            output_dir / "system_cpu_metadata.parquet", index=False
        )
        counts["cpu_metadata"] = len(cpu_meta)

    if "net_nrs" in sw.columns:
        net_mask = sw["net_nrs"] > 0
        net_g = sw.loc[net_mask, ["guid", "net_nrs", "net_received_bytes", "net_sent_bytes"]].copy()
        half_nrs = (net_g["net_nrs"] / 2).clip(lower=1)
        received = pd.DataFrame(
            {
                "guid": net_g["guid"].values,
                "input_desc": "OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::",
                "nrs": half_nrs.values,
                "avg_bytes_sec": (net_g["net_received_bytes"].values / (half_nrs.values * 5)),
            }
        )
        sent = pd.DataFrame(
            {
                "guid": net_g["guid"].values,
                "input_desc": "OS:NETWORK INTERFACE::BYTES SENT/SEC::",
                "nrs": half_nrs.values,
                "avg_bytes_sec": (net_g["net_sent_bytes"].values / (half_nrs.values * 5)),
            }
        )
        net_synth = pd.concat([received, sent], ignore_index=True)
        net_synth.to_parquet(
            output_dir / "system_network_consumption.parquet", index=False
        )
        counts["network_consumption"] = len(net_synth)

    if "mem_nrs" in sw.columns:
        mem_mask = sw["mem_nrs"] > 0
        mem_ram_mb = sw.loc[mem_mask, "ram"].apply(snap_ram) * 1024
        mem_synth = pd.DataFrame(
            {
                "guid": sw.loc[mem_mask, "guid"].values,
                "nrs": sw.loc[mem_mask, "mem_nrs"].values,
                "avg_percentage_used": sw.loc[mem_mask, "mem_avg_pct_used"]
                .values.clip(0, 100),
                "sysinfo_ram": mem_ram_mb.values,
            }
        )
        mem_synth.to_parquet(
            output_dir / "system_memory_utilization.parquet", index=False
        )
        counts["memory_utilization"] = len(mem_synth)

    hw_map = {
        "system_psys_rap_watts": ("psys_rap_nrs", "psys_rap_avg", "avg_psys_rap_watts"),
        "system_pkg_C0": ("pkg_c0_nrs", "pkg_c0_avg", "avg_pkg_c0"),
        "system_pkg_avg_freq_mhz": ("avg_freq_nrs", "avg_freq_avg", "avg_avg_freq_mhz"),
        "system_pkg_temp_centigrade": ("temp_nrs", "temp_avg", "avg_temp_centigrade"),
        "system_hw_pkg_power": ("pkg_power_nrs", "pkg_power_avg", "mean"),
    }
    for table_name, (nrs_col, avg_col, target_col) in hw_map.items():
        if nrs_col not in sw.columns:
            continue
        mask = sw[nrs_col] > 0
        df = pd.DataFrame(
            {
                "guid": sw.loc[mask, "guid"].values,
                "nrs": sw.loc[mask, nrs_col].values,
                target_col: sw.loc[mask, avg_col].values,
            }
        )
        df.to_parquet(output_dir / f"{table_name}.parquet", index=False)
        counts[table_name] = len(df)

    if "batt_num_power_ons" in sw.columns:
        batt_mask = sw["batt_num_power_ons"] > 0
        batt_synth = pd.DataFrame(
            {
                "guid": sw.loc[batt_mask, "guid"].values,
                "num_power_ons": sw.loc[batt_mask, "batt_num_power_ons"].values,
                "duration_mins": sw.loc[batt_mask, "batt_duration_mins"].values,
            }
        )
        batt_synth.to_parquet(
            output_dir / "system_batt_dc_events.parquet", index=False
        )
        counts["batt_dc_events"] = len(batt_synth)

    browser_cols = {
        "chrome": "web_chrome_duration",
        "edge": "web_edge_duration",
        "firefox": "web_firefox_duration",
    }
    web_rows = []
    for browser_name, dur_col in browser_cols.items():
        if dur_col not in sw.columns:
            continue
        mask = sw[dur_col] > 0
        df = pd.DataFrame(
            {
                "guid": sw.loc[mask, "guid"].values,
                "browser": browser_name,
                "duration_ms": sw.loc[mask, dur_col].values,
            }
        )
        web_rows.append(df)
    if web_rows:
        web_synth = pd.concat(web_rows, ignore_index=True)
        web_synth.to_parquet(
            output_dir / "system_web_cat_usage.parquet", index=False
        )
        counts["web_cat_usage"] = len(web_synth)

    webcat_cols = {c: c.replace("webcat_", "") for c in sw.columns if c.startswith("webcat_")}
    if webcat_cols:
        webcat_wide = list(webcat_cols.keys())
        webcat_mask = sw[webcat_wide].abs().sum(axis=1) > 0
        pivot_synth = sw.loc[webcat_mask, ["guid"]].copy()
        for wide_col, pivot_col in webcat_cols.items():
            pivot_synth[pivot_col] = sw.loc[webcat_mask, wide_col].values
        pivot_synth.to_parquet(
            output_dir / "system_web_cat_pivot_duration.parquet", index=False
        )
        counts["web_cat_pivot_duration"] = len(pivot_synth)

    onoff_cols = ["onoff_on_time", "onoff_off_time", "onoff_mods_time", "onoff_sleep_time"]
    if all(c in sw.columns for c in onoff_cols):
        onoff_mask = sw[onoff_cols].abs().sum(axis=1) > 0
        onoff_synth = pd.DataFrame(
            {
                "guid": sw.loc[onoff_mask, "guid"].values,
                "on_time": sw.loc[onoff_mask, "onoff_on_time"].values,
                "off_time": sw.loc[onoff_mask, "onoff_off_time"].values,
                "mods_time": sw.loc[onoff_mask, "onoff_mods_time"].values,
                "sleep_time": sw.loc[onoff_mask, "onoff_sleep_time"].values,
            }
        )
        onoff_synth.to_parquet(
            output_dir / "system_on_off_suspend_time_day.parquet", index=False
        )
        counts["on_off_suspend_time_day"] = len(onoff_synth)

    return counts
