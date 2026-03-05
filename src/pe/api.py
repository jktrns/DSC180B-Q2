import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import APITimeoutError, AsyncOpenAI, LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from .constants import (
    ChassisType,
    CountryType,
    CpuCodeType,
    CpuFamilyType,
    CpuNameType,
    OsType,
    PersonaType,
    ProcessorType,
    VendorType,
)
from .distance import CAT_COLS, NUMERIC_COLS

logger = logging.getLogger(__name__)


# Categorical value lists and Literal types are now defined in constants.py
# and imported above. This eliminates duplication with the experiment script
# and ensures enums are derived from real data.


class TelemetryRecord(BaseModel):
    """Pydantic model with enum-constrained categoricals.

    Uses Literal types to restrict outputs to only valid values from the real
    dataset, eliminating hallucinated categories (e.g. "Workstation",
    "USA", "Japanese").
    """

    chassistype: ChassisType
    countryname_normalized: CountryType
    modelvendor_normalized: VendorType
    os: OsType
    cpuname: CpuNameType
    cpucode: CpuCodeType
    cpu_family: CpuFamilyType
    persona: PersonaType
    processornumber: ProcessorType
    ram: float
    net_nrs: float
    net_received_bytes: float
    net_sent_bytes: float
    mem_nrs: float
    mem_avg_pct_used: float
    mem_sysinfo_ram: float
    batt_num_power_ons: float
    batt_duration_mins: float
    web_chrome_duration: float
    web_edge_duration: float
    web_firefox_duration: float
    web_total_duration: float
    web_num_instances: float
    webcat_content_creation_photo_edit_creation: float
    webcat_content_creation_video_audio_edit_creation: float
    webcat_content_creation_web_design_development: float
    webcat_education: float
    webcat_entertainment_music_audio_streaming: float
    webcat_entertainment_other: float
    webcat_entertainment_video_streaming: float
    webcat_finance: float
    webcat_games_other: float
    webcat_games_video_games: float
    webcat_mail: float
    webcat_news: float
    webcat_unclassified: float
    webcat_private: float
    webcat_productivity_crm: float
    webcat_productivity_other: float
    webcat_productivity_presentations: float
    webcat_productivity_programming: float
    webcat_productivity_project_management: float
    webcat_productivity_spreadsheets: float
    webcat_productivity_word_processing: float
    webcat_recreation_travel: float
    webcat_reference: float
    webcat_search: float
    webcat_shopping: float
    webcat_social_social_network: float
    webcat_social_communication: float
    webcat_social_communication_live: float
    onoff_on_time: float
    onoff_off_time: float
    onoff_mods_time: float
    onoff_sleep_time: float
    disp_num_displays: float
    disp_total_duration_ac: float
    disp_total_duration_dc: float
    psys_rap_nrs: float
    psys_rap_avg: float
    pkg_c0_nrs: float
    pkg_c0_avg: float
    avg_freq_nrs: float
    avg_freq_avg: float
    temp_nrs: float
    temp_avg: float
    pkg_power_nrs: float
    pkg_power_avg: float


class RecordsBatch(BaseModel):
    records: list[TelemetryRecord]


INSTRUCTIONS = (
    "You are a tabular data generator. You output ONLY valid JSON conforming "
    "to the provided schema. Do not add commentary, explanations, or any text "
    "outside the JSON structure. Every field value must come from the allowed "
    "set or be a realistic numeric value within the specified ranges."
)


_NUMERIC_GROUPS = {
    "net": ["net_nrs", "net_received_bytes", "net_sent_bytes"],
    "mem": ["mem_nrs", "mem_avg_pct_used", "mem_sysinfo_ram"],
    "batt": ["batt_num_power_ons", "batt_duration_mins"],
    "browser": [
        "web_chrome_duration",
        "web_edge_duration",
        "web_firefox_duration",
        "web_total_duration",
        "web_num_instances",
    ],
    "webcat": [c for c in NUMERIC_COLS if c.startswith("webcat_")],
    "onoff": ["onoff_on_time", "onoff_off_time", "onoff_mods_time", "onoff_sleep_time"],
    "disp": ["disp_num_displays", "disp_total_duration_ac", "disp_total_duration_dc"],
    "hw": [
        "psys_rap_nrs", "psys_rap_avg", "pkg_c0_nrs", "pkg_c0_avg",
        "avg_freq_nrs", "avg_freq_avg", "temp_nrs", "temp_avg",
        "pkg_power_nrs", "pkg_power_avg",
    ],
}


def _compute_group_sparsity(real_df: pd.DataFrame) -> dict[str, int]:
    n = len(real_df)
    result = {}
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in real_df.columns]
        if not present:
            continue
        nonzero_mask = real_df[present].abs().sum(axis=1) > 0
        pct = int(round(100 * nonzero_mask.sum() / n))
        result[gname] = max(pct, 1)
    return result


# ---------------------------------------------------------------------------
# Real aggregate target tables — loaded from data/results/real/ CSVs
# These are embedded directly in the prompt so the LLM can generate records
# whose aggregated statistics approximate the real query results.
# ---------------------------------------------------------------------------
_REAL_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "real"


def _load_real_targets() -> str:
    """Load real query result CSVs and format them as compact prompt sections.

    These tables become generation targets: the model should produce records
    whose aggregated statistics approximate these values.
    """
    rdir = _REAL_RESULTS_DIR
    sections: list[str] = []

    # --- Battery duration by CPU generation ---
    try:
        df = pd.read_csv(rdir / "battery_on_duration_cpu_family_gen.csv")
        lines = ["cpucode | cpugen | target_avg_duration_mins"]
        for _, r in df.iterrows():
            lines.append(
                f"{r['marketcodename']} | {r['cpugen']} | {r['avg_duration_mins_on_battery']:.0f}"
            )
        sections.append(
            "BATTERY DURATION TARGETS (batt group, by cpucode × cpu_family):\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- On/off/sleep summary by CPU ---
    try:
        df = pd.read_csv(
            rdir / "on_off_mods_sleep_summary_by_cpu_marketcodename_gen.csv"
        )
        lines = ["cpucode | cpugen | on_time | off_time | mods_time | sleep_time"]
        for _, r in df.iterrows():
            lines.append(
                f"{r['marketcodename']} | {r['cpugen']} | "
                f"{r['avg_on_time']:.0f} | {r['avg_off_time']:.0f} | "
                f"{r['avg_modern_sleep_time']:.0f} | {r['avg_sleep_time']:.0f}"
            )
        sections.append(
            "ON/OFF/SLEEP TIME TARGETS (onoff group, by cpucode × cpu_family).\n"
            "Values are in seconds. Total ≈ 84000 s (≈ 1 day).\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- Platform power/C0/freq/temp by chassis ---
    try:
        df = pd.read_csv(
            rdir / "avg_platform_power_c0_freq_temp_by_chassis.csv"
        )
        lines = ["chassistype | psys_rap_watts | pkg_c0_pct | freq_mhz | temp_C"]
        for _, r in df.iterrows():
            lines.append(
                f"{r['chassistype']} | {r['avg_psys_rap_watts']:.1f} | "
                f"{r['avg_pkg_c0']:.1f} | {r['avg_freq_mhz']:.0f} | "
                f"{r['avg_temp_centigrade']:.1f}"
            )
        sections.append(
            "HW METRIC TARGETS (hw group, by chassistype).\n"
            "Use these to set psys_rap_avg, pkg_c0_avg, avg_freq_avg, temp_avg, pkg_power_avg.\n"
            "CRITICAL: avg_freq_avg must be 1000-6000 (MHz). Do NOT use values outside this range.\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- Pkg power by country (top 15 + summary) ---
    try:
        df = pd.read_csv(rdir / "pkg_power_by_country.csv")
        lines = ["country | avg_pkg_power"]
        for _, r in df.head(15).iterrows():
            lines.append(
                f"{r['countryname_normalized']} | {r['avg_pkg_power_consumed']:.1f}"
            )
        lines.append(f"... remaining countries: median ~ {df['avg_pkg_power_consumed'].median():.1f}")
        sections.append(
            "PKG POWER TARGETS (hw group, by country).\n"
            "Most countries have avg_pkg_power 1-10 watts. A few outliers are much higher.\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- Battery power-on geographic summary ---
    try:
        df = pd.read_csv(rdir / "battery_power_on_geographic_summary.csv")
        lines = ["country | avg_powerons | avg_duration_mins"]
        for _, r in df.iterrows():
            lines.append(
                f"{r['country']} | {r['avg_number_of_dc_powerons']:.1f} | "
                f"{r['avg_duration']:.0f}"
            )
        sections.append(
            "BATTERY GEOGRAPHIC TARGETS (batt group, by country).\n"
            "avg_powerons ≈ 2-3 across all countries. Duration varies 84-238 mins.\n"
            "batt_num_power_ons should be near 2-3, batt_duration_mins near these targets.\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- Most popular browser per country ---
    try:
        df = pd.read_csv(
            rdir / "most_popular_browser_in_each_country_by_system_count.csv"
        )
        edge_countries = df.loc[df["browser"] == "edge", "country"].tolist()
        sections.append(
            "BROWSER-COUNTRY RULE:\n"
            "Chrome is the most popular browser in almost every country.\n"
            f"ONLY these countries have edge as most popular: {', '.join(edge_countries)}.\n"
            "When browser+webcat group is active, the dominant browser duration field should be:\n"
            f"  - web_edge_duration > web_chrome_duration  ONLY for: {', '.join(edge_countries)}\n"
            "  - web_chrome_duration > web_edge_duration   for ALL other countries\n"
            "web_firefox_duration should be small (< 10% of total) in most records."
        )
    except Exception:
        pass

    # --- Persona-webcat conditional distributions ---
    try:
        df = pd.read_csv(rdir / "persona_web_cat_usage_analysis.csv")
        has_webcat = df["content_creation_photo_edit_creation"].notna()
        no_webcat_personas = df.loc[~has_webcat, "persona"].tolist()
        sections.append(
            "PERSONA-WEBCAT RULE:\n"
            f"These personas do NOT have webcat data: {', '.join(no_webcat_personas)}.\n"
            "If persona is one of these AND browser+webcat group is active, the webcat_* fields "
            "should still sum to ~100 but follow a generic distribution.\n"
            "For personas WITH webcat data, use these target distributions (% of browsing time):"
        )
        webcat_cols = [c for c in df.columns if c not in ["persona", "number_of_systems", "days"]]
        for _, row in df.loc[has_webcat].iterrows():
            persona = row["persona"]
            cats = {}
            for col in webcat_cols:
                val = row[col]
                if pd.notna(val) and val > 0.3:
                    cats[col] = round(val, 1)
            sorted_cats = sorted(cats.items(), key=lambda x: -x[1])[:10]
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted_cats)
            sections[-1] += f"\n  {persona}: {cat_str}"
    except Exception:
        pass

    # --- Xeon network consumption ---
    try:
        df = pd.read_csv(rdir / "Xeon_network_consumption.csv")
        lines = ["processor_class | os | avg_bytes_received | avg_bytes_sent"]
        for _, r in df.iterrows():
            lines.append(
                f"{r['processor_class']} | {r['os']} | "
                f"{r['avg_bytes_received']:.2e} | {r['avg_bytes_sent']:.2e}"
            )
        sections.append(
            "NETWORK CONSUMPTION TARGETS (net group, by cpuname × os).\n"
            "processor_class = 'Server Class' when cpuname='Intel Xeon', else 'Non-Server Class'.\n"
            "Net bytes vary HUGELY by processor class. Match these magnitudes:\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    # --- RAM utilization histogram ---
    try:
        df = pd.read_csv(rdir / "ram_utilization_histogram.csv")
        # Filter to common RAM sizes
        common_ram = [4, 8, 16, 32, 64]
        filtered = df[df["ram_gb"].isin(common_ram)]
        lines = ["ram_gb | avg_pct_used"]
        for _, r in filtered.iterrows():
            lines.append(f"{int(r['ram_gb'])} | {r['avg_percentage_used']:.0f}")
        sections.append(
            "RAM UTILIZATION TARGETS (mem group).\n"
            "mem_avg_pct_used correlates inversely with ram size:\n"
            + "\n".join(lines)
        )
    except Exception:
        pass

    if not sections:
        return ""
    return "\n\n".join(sections)


# Cache the real targets so we only load once
_REAL_TARGETS_CACHE: str | None = None


def _get_real_targets() -> str:
    global _REAL_TARGETS_CACHE
    if _REAL_TARGETS_CACHE is None:
        _REAL_TARGETS_CACHE = _load_real_targets()
    return _REAL_TARGETS_CACHE


# Distribution-aware prompt with real marginal percentages and sparsity examples
_DISTRIBUTION_PROMPT = """You are generating synthetic Intel DCA telemetry records. Each record = one client system (Windows PC).

CATEGORICAL DISTRIBUTIONS (match these frequencies):
- chassistype: Notebook 72%, Desktop 18%, 2 in 1 6%, Intel NUC/STK 2%, Tablet 1%, Other <1%, Server/WS <1%
- os: Win10 85%, Win11 10%, Win8.1 3%, Win Server 2%
- countryname_normalized: United States of America 22%, India 10%, China 7%, Germany 5%, United Kingdom of Great Britain and Northern Ireland 4%, Brazil 4%, Japan 3%, France 3%, Korea, Republic of 2%, Italy 2%, Canada 2%, Australia 2%, Mexico 2%, Russian Federation 2%, Poland 1.5%, Netherlands 1.5%, Spain 1.5%, Turkey 1.5%, Indonesia 1%, Thailand 1%, Taiwan, Province of China 1%, Sweden 1%, Switzerland 1%, Colombia 1%, other countries ~15% (spread across remaining countries in the enum)
- persona: Casual User 27%, Communication 16%, Casual Gamer 16%, Office/Productivity 13%, Web User 8%, Entertainment 6%, Content Creator/IT 5%, Win Store App User 4%, Gamer 2%, File & Network Sharer 2%, Unknown 1%
- modelvendor_normalized: Lenovo 22%, Dell 15%, HP 15%, Asus 8%, Acer 7%, Microsoft Corporation 3%, Samsung 3%, HUAWEI 2%, MSI 2%, Toshiba 2%, Fujitsu 1%, Intel 1%, others spread across remaining vendors in the enum
- ram: 8 (38%), 16 (28%), 4 (18%), 32 (7%), 12 (4%), 6 (2%), 64 (1%)

*** CRITICAL: CONDITIONAL NUMERIC RANGES ***
Numeric values MUST match the REAL AGGREGATE TARGETS below. Do NOT use generic ranges.

GROUP: net (net_nrs, net_received_bytes, net_sent_bytes)
- net_nrs: 100-50000
- IMPORTANT: net bytes depend on cpuname.
  If cpuname=Intel Xeon: net_received_bytes ~ 1e12-6e17, net_sent_bytes ~ 1e12-6e17
  If cpuname!=Intel Xeon: net_received_bytes ~ 1e10-1e16, net_sent_bytes ~ 1e10-1e16

GROUP: mem (mem_nrs, mem_avg_pct_used, mem_sysinfo_ram)
- mem_nrs: 100-50000
- mem_sysinfo_ram: MUST match ram × 1024 (e.g. ram=8 → mem_sysinfo_ram=8192)
- mem_avg_pct_used: depends on ram (see RAM UTILIZATION TARGETS below)
  ram=4 → ~71, ram=8 → ~60, ram=16 → ~47, ram=32 → ~41, ram=64 → ~35

GROUP: batt (batt_num_power_ons, batt_duration_mins)
- Notebook/2-in-1 ONLY.
- batt_num_power_ons: 1-5 (average ~2.5 across all countries)
- batt_duration_mins: depends on cpucode (see BATTERY DURATION TARGETS below)
  Tiger Lake → ~190, Ice Lake → ~95-158, Comet Lake → ~104-120, Coffee Lake → ~64-65

GROUP: browser+webcat
- web_chrome_duration/edge_duration/firefox_duration: 1e3 to 1e8 (ms)
- Chrome dominant in most countries. Edge dominant ONLY in China, Denmark, Korea.
- webcat_* fields: 0-100 (percentage of browsing time, active fields sum to ~100)
- webcat distribution depends on persona (see PERSONA-WEBCAT RULE below)

GROUP: onoff (onoff_on_time, onoff_off_time, onoff_mods_time, onoff_sleep_time)
- Values are in SECONDS. Total ≈ 84000 s (≈ 1 day).
- on_time: 23000-53000, off_time: 8000-23000, sleep_time: 18000-52000
- mods_time: 0-14000 (high for Tiger Lake/Ice Lake, ~0 for Coffee Lake/Rocket Lake)
- Depends on cpucode (see ON/OFF/SLEEP TIME TARGETS below)

GROUP: hw (psys_rap_*, pkg_c0_*, avg_freq_*, temp_*, pkg_power_*)
- CRITICAL: avg_freq_avg MUST be 1000-6000 (MHz). Notebook ~1500, Desktop ~5400.
- psys_rap_avg: 2-7 (watts). pkg_c0_avg: 37-46 (percentage).
- temp_avg: 41-52 (Celsius). pkg_power_avg: 0.7-673 (watts, see PKG POWER TARGETS).
- *_nrs fields: 100-50000.

GROUP: disp (disp_num_displays, disp_total_duration_ac, disp_total_duration_dc)
- disp_num_displays: 1-4, duration_ac/dc: 1e3 to 1e8

*** CRITICAL SPARSITY RULES ***
There are 7 numeric groups: net, mem, batt, browser+webcat, onoff, disp, hw.
Each system has data in EXACTLY 1 or 2 of these groups. ALL other groups MUST be exactly 0.

Approximate group frequencies (what % of systems have data in each group):
- net: 25%, mem: 22%, batt: 15% (Notebook/2-in-1 only), browser+webcat: 20%, onoff: 18%, disp: 10%, hw: <1%

EXAMPLE 1 — Notebook with net + mem active, ram=8, USA, Chrome era:
{"chassistype":"Notebook","countryname_normalized":"United States of America","modelvendor_normalized":"Dell","os":"Win10","cpuname":"Intel Core i5","cpucode":"Comet Lake","cpu_family":"10th Gen i5","persona":"Office/Productivity","processornumber":"i5-10210U","ram":8,"net_nrs":5000,"net_received_bytes":5e10,"net_sent_bytes":3e9,"mem_nrs":5000,"mem_avg_pct_used":60,"mem_sysinfo_ram":8192,"batt_num_power_ons":0,"batt_duration_mins":0,"web_chrome_duration":0,"web_edge_duration":0,"web_firefox_duration":0,"web_total_duration":0,"web_num_instances":0,"webcat_content_creation_photo_edit_creation":0,"webcat_content_creation_video_audio_edit_creation":0,"webcat_content_creation_web_design_development":0,"webcat_education":0,"webcat_entertainment_music_audio_streaming":0,"webcat_entertainment_other":0,"webcat_entertainment_video_streaming":0,"webcat_finance":0,"webcat_games_other":0,"webcat_games_video_games":0,"webcat_mail":0,"webcat_news":0,"webcat_unclassified":0,"webcat_private":0,"webcat_productivity_crm":0,"webcat_productivity_other":0,"webcat_productivity_presentations":0,"webcat_productivity_programming":0,"webcat_productivity_project_management":0,"webcat_productivity_spreadsheets":0,"webcat_productivity_word_processing":0,"webcat_recreation_travel":0,"webcat_reference":0,"webcat_search":0,"webcat_shopping":0,"webcat_social_social_network":0,"webcat_social_communication":0,"webcat_social_communication_live":0,"onoff_on_time":0,"onoff_off_time":0,"onoff_mods_time":0,"onoff_sleep_time":0,"disp_num_displays":0,"disp_total_duration_ac":0,"disp_total_duration_dc":0,"psys_rap_nrs":0,"psys_rap_avg":0,"pkg_c0_nrs":0,"pkg_c0_avg":0,"avg_freq_nrs":0,"avg_freq_avg":0,"temp_nrs":0,"temp_avg":0,"pkg_power_nrs":0,"pkg_power_avg":0}

EXAMPLE 2 — Notebook with browser+webcat + onoff active, Tiger Lake, Communication persona:
{"chassistype":"Notebook","countryname_normalized":"Germany","modelvendor_normalized":"Lenovo","os":"Win10","cpuname":"Intel Core i7","cpucode":"Tiger Lake","cpu_family":"11th Gen i7","persona":"Communication","processornumber":"i7-1165G7","ram":16,"net_nrs":0,"net_received_bytes":0,"net_sent_bytes":0,"mem_nrs":0,"mem_avg_pct_used":0,"mem_sysinfo_ram":0,"batt_num_power_ons":0,"batt_duration_mins":0,"web_chrome_duration":5e6,"web_edge_duration":1e5,"web_firefox_duration":0,"web_total_duration":5.1e6,"web_num_instances":150,"webcat_content_creation_photo_edit_creation":0,"webcat_content_creation_video_audio_edit_creation":0,"webcat_content_creation_web_design_development":0.3,"webcat_education":1.7,"webcat_entertainment_music_audio_streaming":0.1,"webcat_entertainment_other":1.1,"webcat_entertainment_video_streaming":8.3,"webcat_finance":0.9,"webcat_games_other":0.5,"webcat_games_video_games":0,"webcat_mail":2.4,"webcat_news":2.0,"webcat_unclassified":47.7,"webcat_private":14.6,"webcat_productivity_crm":0,"webcat_productivity_other":2.6,"webcat_productivity_presentations":0.2,"webcat_productivity_programming":0.4,"webcat_productivity_project_management":0.2,"webcat_productivity_spreadsheets":0.7,"webcat_productivity_word_processing":0.5,"webcat_recreation_travel":0.6,"webcat_reference":1.3,"webcat_search":7.1,"webcat_shopping":2.8,"webcat_social_social_network":2.3,"webcat_social_communication":0.7,"webcat_social_communication_live":1.0,"onoff_on_time":26000,"onoff_off_time":12000,"onoff_mods_time":14260,"onoff_sleep_time":32690,"disp_num_displays":0,"disp_total_duration_ac":0,"disp_total_duration_dc":0,"psys_rap_nrs":0,"psys_rap_avg":0,"pkg_c0_nrs":0,"pkg_c0_avg":0,"avg_freq_nrs":0,"avg_freq_avg":0,"temp_nrs":0,"temp_avg":0,"pkg_power_nrs":0,"pkg_power_avg":0}

EXAMPLE 3 — Desktop with hw group active:
{"chassistype":"Desktop","countryname_normalized":"China","modelvendor_normalized":"Asus","os":"Win10","cpuname":"Intel Core i7","cpucode":"Coffee Lake","cpu_family":"9th Gen i7","persona":"Gamer","processornumber":"i7-9700","ram":32,"net_nrs":0,"net_received_bytes":0,"net_sent_bytes":0,"mem_nrs":0,"mem_avg_pct_used":0,"mem_sysinfo_ram":0,"batt_num_power_ons":0,"batt_duration_mins":0,"web_chrome_duration":0,"web_edge_duration":0,"web_firefox_duration":0,"web_total_duration":0,"web_num_instances":0,"webcat_content_creation_photo_edit_creation":0,"webcat_content_creation_video_audio_edit_creation":0,"webcat_content_creation_web_design_development":0,"webcat_education":0,"webcat_entertainment_music_audio_streaming":0,"webcat_entertainment_other":0,"webcat_entertainment_video_streaming":0,"webcat_finance":0,"webcat_games_other":0,"webcat_games_video_games":0,"webcat_mail":0,"webcat_news":0,"webcat_unclassified":0,"webcat_private":0,"webcat_productivity_crm":0,"webcat_productivity_other":0,"webcat_productivity_presentations":0,"webcat_productivity_programming":0,"webcat_productivity_project_management":0,"webcat_productivity_spreadsheets":0,"webcat_productivity_word_processing":0,"webcat_recreation_travel":0,"webcat_reference":0,"webcat_search":0,"webcat_shopping":0,"webcat_social_social_network":0,"webcat_social_communication":0,"webcat_social_communication_live":0,"onoff_on_time":0,"onoff_off_time":0,"onoff_mods_time":0,"onoff_sleep_time":0,"disp_num_displays":0,"disp_total_duration_ac":0,"disp_total_duration_dc":0,"psys_rap_nrs":3000,"psys_rap_avg":6.3,"pkg_c0_nrs":3000,"pkg_c0_avg":45.1,"avg_freq_nrs":3000,"avg_freq_avg":5442,"temp_nrs":3000,"temp_avg":41.8,"pkg_power_nrs":3000,"pkg_power_avg":5.1}

VIOLATION: Setting net, mem, browser, AND onoff all nonzero for the same system. That is 4 active groups (max allowed is 2).

ram is ALWAYS present and nonzero regardless of group activation.

MATCH THE REAL AGGREGATE TARGETS BELOW. These are the actual statistics from the real dataset.
When generating records, ensure that if you aggregate them by the grouping columns, the averages
approximate the target values. This is the MOST IMPORTANT requirement after sparsity.

Do NOT inject your own assumptions. Follow the distributions, sparsity rules, and aggregate targets exactly."""


def _build_schema_description(real_df: pd.DataFrame) -> str:
    """Build schema description with real distribution statistics and aggregate targets.

    Combines the static distribution prompt with:
    1. Real group sparsity computed from the data
    2. Real query result tables loaded from data/results/real/ CSVs
    """
    group_sparsity = _compute_group_sparsity(real_df)
    # Merge browser and webcat into a single combined group to match the
    # 7-group definition in the prompt (browser+webcat is one group).
    merged = {}
    for g, pct in sorted(group_sparsity.items()):
        if g in ("browser", "webcat"):
            merged["browser+webcat"] = max(merged.get("browser+webcat", 0), pct)
        else:
            merged[g] = pct
    sparsity_info = ", ".join(
        f"{g}: {pct}%" for g, pct in sorted(merged.items())
    )
    real_targets = _get_real_targets()
    parts = [_DISTRIBUTION_PROMPT, f"Real group sparsity: {sparsity_info}"]
    if real_targets:
        parts.append(f"\n=== REAL AGGREGATE TARGETS ===\n{real_targets}")
    return "\n".join(parts)


def _build_random_prompt(schema_desc: str, batch_size: int) -> str:
    return (
        f"{schema_desc}\n\n"
        f"Generate exactly {batch_size} synthetic telemetry records.\n"
        f"Requirements:\n"
        f"1. Each record MUST have exactly 1 or 2 active numeric groups (all others = 0).\n"
        f"2. Vary countries broadly - use many different countries, not just top 3.\n"
        f"3. Match the categorical frequency distributions above.\n"
        f"4. ram is always nonzero (pick from 4, 8, 16, 32, 64 with given frequencies).\n"
        f"5. Do not repeat the same combination of categoricals across records."
    )


def _build_variation_prompt(
    schema_desc: str, records: list[dict], n_variations: int
) -> str:
    return (
        f"{schema_desc}\n\n"
        f"Below are {len(records)} telemetry records. For each, generate {n_variations} "
        f"variation(s) that are similar but slightly different.\n\n"
        f"Rules for variations:\n"
        f"- Categorical values: may change to a different valid value from the enum\n"
        f"- Numeric values: perturb by 10-30% (multiply by a random factor between 0.7 and 1.3)\n"
        f"- Zero values MUST remain zero (preserve the sparsity pattern exactly)\n"
        f"- Non-zero values should remain non-zero\n"
        f"- ram should stay at a standard size (4, 8, 16, 32, 64)\n"
        f"- Do NOT activate new numeric groups that were zero in the source\n\n"
        f"Source records:\n{json.dumps(records, indent=None)}\n\n"
        f"Return {len(records) * n_variations} variation records total."
    )


def _make_strict_schema() -> dict:
    schema = RecordsBatch.model_json_schema()

    def enforce_strict(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
            for v in obj.values():
                enforce_strict(v)
        elif isinstance(obj, list):
            for item in obj:
                enforce_strict(item)

    enforce_strict(schema)
    if "$defs" in schema:
        for defn in schema["$defs"].values():
            enforce_strict(defn)
    return schema


class PEApi:
    def __init__(
        self,
        real_df: pd.DataFrame,
        model: str = "gpt-5-mini",
        max_concurrent: int = 50,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=api_key, max_retries=5)
        self.sync_client = OpenAI(api_key=api_key, max_retries=5)
        self.model = model
        self.max_concurrent = max_concurrent
        self.schema_desc = _build_schema_description(real_df)
        self.all_cols = CAT_COLS + NUMERIC_COLS
        self.present_cols = [c for c in self.all_cols if c in real_df.columns]
        self._is_reasoning = model.startswith("gpt-5") or model.startswith("o")
        logger.info(
            "PEApi initialized: model=%s, reasoning=%s, concurrent=%d",
            model, self._is_reasoning, max_concurrent,
        )
        self._strict_schema = _make_strict_schema()

    # ------------------------------------------------------------------ #
    #  Real-time API (async, for smoke tests and small runs)              #
    # ------------------------------------------------------------------ #

    async def _call_api(self, prompt: str) -> list[dict]:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": INSTRUCTIONS,
                "input": prompt,
                "text_format": RecordsBatch,
                "max_output_tokens": 16000,
            }
            if self._is_reasoning:
                kwargs["reasoning"] = {"effort": "low"}
            else:
                kwargs["temperature"] = 0.8
            response = await self.client.responses.parse(**kwargs)
            if response.output_parsed and response.output_parsed.records:
                return [r.model_dump() for r in response.output_parsed.records]
        except LengthFinishReasonError:
            logger.warning("Response truncated (max_output_tokens exceeded), dropping batch")
        except Exception as e:
            logger.warning("API error: %s", e)
        return []

    async def _batch_calls(
        self, prompts: list[str], desc: str = ""
    ) -> list[list[dict]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[list[dict]] = [[] for _ in range(len(prompts))]
        completed = 0
        total = len(prompts)
        t0 = time.time()

        async def run_one(idx: int, prompt: str):
            nonlocal completed
            async with semaphore:
                result = await self._call_api(prompt)
                results[idx] = result
                completed += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  {desc}: {completed}/{total} "
                        f"({elapsed:.0f}s, {rate:.1f}/s)"
                    )

        tasks = [run_one(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    def _records_to_df(self, records: list[dict]) -> pd.DataFrame:
        rows = []
        for r in records:
            row = {}
            for c in self.present_cols:
                val = r.get(c, 0 if c in NUMERIC_COLS else "Unknown")
                row[c] = val
            rows.append(row)

        df = pd.DataFrame(rows)
        for c in [col for col in NUMERIC_COLS if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        for c in [col for col in CAT_COLS if col in df.columns]:
            df[c] = df[c].fillna("Unknown").astype(str)
        return df

    async def random_api(
        self, n_records: int, batch_size: int = 10
    ) -> pd.DataFrame:
        overshoot = int(n_records * 1.25)
        n_batches = (overshoot + batch_size - 1) // batch_size
        prompts = [
            _build_random_prompt(self.schema_desc, batch_size)
            for _ in range(n_batches)
        ]
        print(
            f"RANDOM_API: {n_records} records "
            f"({n_batches} batches of {batch_size}, 25% buffer)"
        )
        all_results = await self._batch_calls(prompts, desc="RANDOM_API")
        all_records = [r for batch in all_results for r in batch]
        raw = len(all_records)
        df = self._records_to_df(all_records[:n_records])
        print(
            f"RANDOM_API: {raw} raw -> {len(df)} returned "
            f"({100 * len(df) / max(raw, 1):.0f}%)"
        )
        return df

    async def variation_api(
        self,
        source_df: pd.DataFrame,
        n_variations: int = 2,
        source_batch_size: int = 5,
    ) -> pd.DataFrame:
        source_records = source_df[self.present_cols].to_dict(orient="records")
        n_batches = (len(source_records) + source_batch_size - 1) // source_batch_size
        prompts = []
        for i in range(n_batches):
            start = i * source_batch_size
            end = min(start + source_batch_size, len(source_records))
            batch = source_records[start:end]
            prompts.append(
                _build_variation_prompt(self.schema_desc, batch, n_variations)
            )
        print(
            f"VARIATION_API: {n_variations} variations for "
            f"{len(source_df)} records ({n_batches} batches)"
        )
        all_results = await self._batch_calls(prompts, desc="VARIATION_API")
        all_records = [r for batch in all_results for r in batch]
        df = self._records_to_df(all_records)
        print(f"VARIATION_API: {len(df)} records")
        return df

    # ------------------------------------------------------------------ #
    #  Batch API (sync, 50% cheaper, for full production runs)            #
    # ------------------------------------------------------------------ #

    def _write_batch_jsonl(self, prompts: list[str], path: Path) -> None:
        with open(path, "w") as f:
            for i, prompt in enumerate(prompts):
                body: dict[str, Any] = {
                    "model": self.model,
                    "instructions": INSTRUCTIONS,
                    "input": prompt,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "RecordsBatch",
                            "schema": self._strict_schema,
                            "strict": True,
                        }
                    },
                    "max_output_tokens": 16000,
                }
                if self._is_reasoning:
                    body["reasoning"] = {"effort": "low"}
                else:
                    body["temperature"] = 0.8
                request = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(request) + "\n")

    def _submit_batch(self, jsonl_path: Path, desc: str = "", max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                with open(jsonl_path, "rb") as f:
                    uploaded = self.sync_client.files.create(file=f, purpose="batch")
                job = self.sync_client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/responses",
                    completion_window="24h",
                )
                total = job.request_counts.total if job.request_counts else "?"
                print(f"  {desc}: batch {job.id} submitted ({total} requests)")
                return job.id
            except (APITimeoutError, Exception) as e:
                if attempt >= max_retries - 1:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  {desc}: submit error ({type(e).__name__}), retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def _poll_batch(
        self, batch_id: str, desc: str = "", poll_interval: int = 30,
        max_retries: int = 10,
    ) -> Any:
        consecutive_errors = 0
        while True:
            try:
                status = self.sync_client.batches.retrieve(batch_id)
                consecutive_errors = 0
            except (APITimeoutError, Exception) as e:
                consecutive_errors += 1
                if consecutive_errors >= max_retries:
                    raise
                wait = min(poll_interval * consecutive_errors, 120)
                print(f"  {desc}: poll error ({type(e).__name__}), retry {consecutive_errors}/{max_retries} in {wait}s")
                time.sleep(wait)
                continue
            rc = status.request_counts
            if rc:
                print(
                    f"  {desc}: {rc.completed}/{rc.total} done, "
                    f"{rc.failed} failed [{status.status}]"
                )
            else:
                print(f"  {desc}: [{status.status}]")
            if status.status in ("completed", "failed", "cancelled", "expired"):
                if status.status == "failed" and hasattr(status, "errors") and status.errors:
                    for err in (status.errors.data or []):
                        print(f"  BATCH ERROR: {err.code}: {err.message}")
                return status
            time.sleep(poll_interval)

    def _parse_batch_output(self, output_file_id: str, max_retries: int = 5) -> list[list[dict]]:
        for attempt in range(max_retries):
            try:
                file_content = self.sync_client.files.content(output_file_id)
                break
            except (APITimeoutError, Exception) as e:
                if attempt >= max_retries - 1:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  File download error ({type(e).__name__}), retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
        indexed: list[tuple[int, list[dict]]] = []
        for line in file_content.text.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["custom_id"].split("-")[1])
            records: list[dict] = []
            resp = obj.get("response", {})
            if resp.get("status_code") == 200:
                body = resp.get("body", {})
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        for ci in item.get("content", []):
                            if ci.get("type") == "output_text":
                                text = ci.get("text", "")
                                if text:
                                    try:
                                        batch = RecordsBatch.model_validate_json(
                                            text
                                        )
                                        records = [
                                            r.model_dump() for r in batch.records
                                        ]
                                    except Exception:
                                        pass
            indexed.append((idx, records))
        indexed.sort(key=lambda x: x[0])
        return [recs for _, recs in indexed]

    MAX_REQUESTS_PER_BATCH = 800

    def _save_chunk_results(
        self, records: list[dict], chunk_idx: int, tag: str, work_dir: Path
    ) -> None:
        path = work_dir / f"batch_{tag}_chunk{chunk_idx}.parquet"
        if records:
            self._records_to_df(records).to_parquet(path, index=False)

    def _load_chunk_results(
        self, chunk_idx: int, tag: str, work_dir: Path
    ) -> pd.DataFrame | None:
        path = work_dir / f"batch_{tag}_chunk{chunk_idx}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _save_batch_state(
        self, chunk_idx: int, batch_id: str, tag: str, work_dir: Path
    ) -> None:
        path = work_dir / f"batch_{tag}_active.json"
        path.write_text(json.dumps({"chunk": chunk_idx, "batch_id": batch_id}))

    def _load_batch_state(self, tag: str, work_dir: Path) -> dict | None:
        path = work_dir / f"batch_{tag}_active.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _clear_batch_state(self, tag: str, work_dir: Path) -> None:
        path = work_dir / f"batch_{tag}_active.json"
        if path.exists():
            path.unlink()

    def _run_multi_batch(
        self, prompts: list[str], tag: str, work_dir: Path,
        n_target: int | None = None,
    ) -> list[list[dict]]:
        n_chunks = (len(prompts) + self.MAX_REQUESTS_PER_BATCH - 1) // self.MAX_REQUESTS_PER_BATCH
        print(f"  {n_chunks} sequential chunk(s) of up to {self.MAX_REQUESTS_PER_BATCH} requests")

        active_state = self._load_batch_state(tag, work_dir)
        all_results: list[list[dict]] = []
        total_records = 0

        for ci in range(n_chunks):
            cached = self._load_chunk_results(ci, tag, work_dir)
            if cached is not None:
                print(f"  {tag.upper()} chunk {ci+1}/{n_chunks}: loaded {len(cached)} cached records")
                all_results.append(cached.to_dict(orient="records"))
                total_records += len(cached)
                if n_target and total_records >= n_target:
                    print(f"  Target reached ({total_records:,} >= {n_target:,}), skipping remaining chunks")
                    break
                continue

            if active_state and active_state["chunk"] == ci:
                batch_id = active_state["batch_id"]
                print(f"  {tag.upper()} chunk {ci+1}/{n_chunks}: resuming batch {batch_id}")
            else:
                start = ci * self.MAX_REQUESTS_PER_BATCH
                end = min(start + self.MAX_REQUESTS_PER_BATCH, len(prompts))
                chunk_prompts = prompts[start:end]
                jsonl_path = work_dir / f"batch_{tag}_chunk{ci}.jsonl"
                self._write_batch_jsonl(chunk_prompts, jsonl_path)
                batch_id = self._submit_batch(
                    jsonl_path, desc=f"{tag.upper()} chunk {ci+1}/{n_chunks}"
                )
                self._save_batch_state(ci, batch_id, tag, work_dir)

            status = self._poll_batch(batch_id, desc=f"{tag.upper()} chunk {ci+1}/{n_chunks}")
            chunk_records: list[dict] = []
            if status.status == "completed" and status.output_file_id:
                parsed = self._parse_batch_output(status.output_file_id)
                chunk_records = [r for batch in parsed for r in batch]
            elif status.output_file_id:
                print(f"  Chunk {ci+1}: {status.status}, recovering partial results")
                parsed = self._parse_batch_output(status.output_file_id)
                chunk_records = [r for batch in parsed for r in batch]
            else:
                print(f"  Chunk {ci+1}: {status.status}, no results")

            self._save_chunk_results(chunk_records, ci, tag, work_dir)
            self._clear_batch_state(tag, work_dir)
            all_results.append(chunk_records)
            total_records += len(chunk_records)
            print(f"  Chunk {ci+1}/{n_chunks}: {len(chunk_records)} records saved")

            if n_target and total_records >= n_target:
                print(f"  Target reached ({total_records:,} >= {n_target:,}), skipping remaining chunks")
                break

        return all_results

    def random_api_batch(
        self,
        n_records: int,
        batch_size: int = 10,
        work_dir: Path = Path("."),
    ) -> pd.DataFrame:
        overshoot = int(n_records * 1.25)
        n_calls = (overshoot + batch_size - 1) // batch_size
        n_chunks = (n_calls + self.MAX_REQUESTS_PER_BATCH - 1) // self.MAX_REQUESTS_PER_BATCH
        prompts = [
            _build_random_prompt(self.schema_desc, batch_size)
            for _ in range(n_calls)
        ]
        print(
            f"RANDOM_API_BATCH: {n_records} records "
            f"({n_calls} calls across {n_chunks} batch(es), 25% buffer)"
        )
        all_chunk_results = self._run_multi_batch(prompts, "random", work_dir, n_target=n_records)
        all_records = [r for chunk in all_chunk_results for r in chunk]
        raw = len(all_records)
        df = self._records_to_df(all_records[:n_records])
        print(
            f"RANDOM_API_BATCH: {raw} raw -> {len(df)} returned "
            f"({100 * len(df) / max(raw, 1):.0f}%)"
        )
        return df

    def variation_api_batch(
        self,
        source_df: pd.DataFrame,
        n_variations: int = 2,
        source_batch_size: int = 5,
        work_dir: Path = Path("."),
    ) -> pd.DataFrame:
        source_records = source_df[self.present_cols].to_dict(orient="records")
        n_calls = (len(source_records) + source_batch_size - 1) // source_batch_size
        n_chunks = (n_calls + self.MAX_REQUESTS_PER_BATCH - 1) // self.MAX_REQUESTS_PER_BATCH
        prompts = []
        for i in range(n_calls):
            start = i * source_batch_size
            end = min(start + source_batch_size, len(source_records))
            prompts.append(
                _build_variation_prompt(
                    self.schema_desc, source_records[start:end], n_variations
                )
            )
        print(
            f"VARIATION_API_BATCH: {n_variations} variations for "
            f"{len(source_df)} records ({n_calls} calls across {n_chunks} batch(es))"
        )
        all_chunk_results = self._run_multi_batch(prompts, "variation", work_dir)
        all_records = [r for chunk in all_chunk_results for r in chunk]
        df = self._records_to_df(all_records)
        print(f"VARIATION_API_BATCH: {len(df)} records")
        return df
