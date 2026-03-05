"""Conditional / stratified PE generation with DP-noised aggregates.

Instead of generating records independently, this module:
1. Reads real query results from ``data/results/real/``
2. Computes target aggregate statistics per stratum
3. Adds calibrated Gaussian noise to all aggregates (counts and means)
4. Generates records conditioned on those *noisy* targets

The DP noise ensures that the aggregates embedded in prompts are
differentially private.  The privacy cost (epsilon_agg) composes with
the histogram selection cost (epsilon_hist) for a total budget.

Example: "generate 55 records where cpucode=Tiger Lake, cpu_family=11th Gen i7,
batt group active, batt_duration_mins averaging ~187"
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StratumAllocation:
    """A single stratum in the generation plan."""

    active_group: str  # "batt", "mem", "onoff", "browser_webcat", "hw", "net"
    count: int  # number of records to generate
    cat_constraints: dict[str, str | float]  # fixed categorical values
    numeric_targets: dict[str, float]  # target averages for active group fields
    secondary_hints: str = ""  # additional prompt text (country dist, etc.)


@dataclass
class GenerationPlan:
    """Complete stratified generation plan."""

    allocations: list[StratumAllocation] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return sum(a.count for a in self.allocations)

    def summary(self) -> str:
        by_group: dict[str, list[StratumAllocation]] = {}
        for a in self.allocations:
            by_group.setdefault(a.active_group, []).append(a)
        lines = [
            f"Total: {self.total_records} records across "
            f"{len(self.allocations)} strata"
        ]
        for group in sorted(by_group):
            allocs = by_group[group]
            total = sum(a.count for a in allocs)
            lines.append(f"  {group}: {total} records in {len(allocs)} strata")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Group sparsity from real data (used to allocate records across groups)
# ---------------------------------------------------------------------------

# These match the percentages in the _DISTRIBUTION_PROMPT (api.py).
# disp is excluded because no feasible benchmark queries use it.
_GROUP_SPARSITY_PCT: dict[str, int] = {
    "net": 25,
    "mem": 22,
    "batt": 15,
    "browser_webcat": 20,
    "onoff": 18,
    "hw": 1,
}


def _compute_group_allocation(n_total: int) -> dict[str, int]:
    """Divide *n_total* records across numeric groups proportionally."""
    total_pct = sum(_GROUP_SPARSITY_PCT.values())
    alloc: dict[str, int] = {}
    assigned = 0
    items = sorted(_GROUP_SPARSITY_PCT.items(), key=lambda x: -x[1])
    for i, (group, pct) in enumerate(items):
        if i == len(items) - 1:
            alloc[group] = n_total - assigned
        else:
            n = max(1, round(pct / total_pct * n_total))
            alloc[group] = n
            assigned += n
    return alloc


def _scale_counts(
    counts: pd.Series, target_total: int
) -> pd.Series:
    """Scale a Series of real counts to *target_total*, min 1 per row."""
    real_total = counts.sum()
    if real_total == 0:
        return pd.Series([0] * len(counts), index=counts.index, dtype=int)

    scaled = (counts / real_total * target_total).apply(
        lambda x: max(1, round(x))
    )
    diff = target_total - int(scaled.sum())
    if diff > 0:
        for idx in scaled.nlargest(abs(diff)).index:
            scaled.loc[idx] += 1
    elif diff < 0:
        for idx in scaled.nlargest(len(scaled)).index:
            if diff == 0:
                break
            if scaled.loc[idx] > 1:
                scaled.loc[idx] -= 1
                diff += 1
    return scaled.astype(int)


# ---------------------------------------------------------------------------
# Helpers for reading real query results
# ---------------------------------------------------------------------------

def _read_csv(real_results_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(real_results_dir / name)


def _country_distribution_hint(
    df: pd.DataFrame, col: str = "country"
) -> str:
    """Build a short country distribution string from a query result."""
    total = df["number_of_systems"].sum() if "number_of_systems" in df.columns else len(df)
    if "number_of_systems" in df.columns:
        top = df.nlargest(15, "number_of_systems")
        lines = []
        for _, row in top.iterrows():
            pct = row["number_of_systems"] / total * 100
            lines.append(f"{row[col]} {pct:.0f}%")
        return ", ".join(lines) + ", others spread across remaining countries"
    return ""


def _browser_distribution_hint(df: pd.DataFrame) -> str:
    """Build browser distribution string."""
    parts = []
    for _, row in df.iterrows():
        parts.append(f"{row['browser']} {row['percent_systems']:.1f}%")
    return ", ".join(parts)


def _browser_country_hint(df: pd.DataFrame) -> str:
    """Build most-popular-browser-per-country hint."""
    non_chrome = df[df["browser"] != "chrome"]
    if len(non_chrome) == 0:
        return "chrome is the most popular browser in every country"
    exceptions = []
    for _, row in non_chrome.iterrows():
        exceptions.append(f"{row['country']}: {row['browser']}")
    return (
        "chrome is most popular in most countries. Exceptions: "
        + ", ".join(exceptions)
    )


# ---------------------------------------------------------------------------
# Fields belonging to each numeric group (mirrors _NUMERIC_GROUPS in api.py)
# ---------------------------------------------------------------------------

_GROUP_FIELDS: dict[str, list[str]] = {
    "net": ["net_nrs", "net_received_bytes", "net_sent_bytes"],
    "mem": ["mem_nrs", "mem_avg_pct_used", "mem_sysinfo_ram"],
    "batt": ["batt_num_power_ons", "batt_duration_mins"],
    "browser_webcat": [
        "web_chrome_duration", "web_edge_duration", "web_firefox_duration",
        "web_total_duration", "web_num_instances",
        # plus all webcat_* fields (handled dynamically)
    ],
    "onoff": ["onoff_on_time", "onoff_off_time", "onoff_mods_time", "onoff_sleep_time"],
    "disp": ["disp_num_displays", "disp_total_duration_ac", "disp_total_duration_dc"],
    "hw": [
        "psys_rap_nrs", "psys_rap_avg",
        "pkg_c0_nrs", "pkg_c0_avg",
        "avg_freq_nrs", "avg_freq_avg",
        "temp_nrs", "temp_avg",
        "pkg_power_nrs", "pkg_power_avg",
    ],
}

# All numeric groups that must be zero when a different group is active
_ALL_GROUPS = list(_GROUP_FIELDS.keys())


# ---------------------------------------------------------------------------
# Webcat column mapping (persona_web_cat_usage_analysis CSV → record fields)
# ---------------------------------------------------------------------------

_WEBCAT_CSV_TO_RECORD: dict[str, str] = {
    "content_creation_photo_edit_creation": "webcat_content_creation_photo_edit_creation",
    "content_creation_video_audio_edit_creation": "webcat_content_creation_video_audio_edit_creation",
    "content_creation_web_design_development": "webcat_content_creation_web_design_development",
    "education": "webcat_education",
    "entertainment_music_audio_streaming": "webcat_entertainment_music_audio_streaming",
    "entertainment_other": "webcat_entertainment_other",
    "entertainment_video_streaming": "webcat_entertainment_video_streaming",
    "finance": "webcat_finance",
    "games_other": "webcat_games_other",
    "games_video_games": "webcat_games_video_games",
    "mail": "webcat_mail",
    "news": "webcat_news",
    "unclassified": "webcat_unclassified",
    "private": "webcat_private",
    "productivity_crm": "webcat_productivity_crm",
    "productivity_other": "webcat_productivity_other",
    "productivity_presentations": "webcat_productivity_presentations",
    "productivity_programming": "webcat_productivity_programming",
    "productivity_project_management": "webcat_productivity_project_management",
    "productivity_spreadsheets": "webcat_productivity_spreadsheets",
    "productivity_word_processing": "webcat_productivity_word_processing",
    "recreation_travel": "webcat_recreation_travel",
    "reference": "webcat_reference",
    "search": "webcat_search",
    "shopping": "webcat_shopping",
    "social_social_network": "webcat_social_social_network",
    "social_communication": "webcat_social_communication",
    "social_communication_live": "webcat_social_communication_live",
}


# ---------------------------------------------------------------------------
# Build the generation plan
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DP noise helpers
# ---------------------------------------------------------------------------

def _dp_noise_count(
    count: float, sigma: float, rng: np.random.Generator,
) -> float:
    """Add Gaussian noise to a count query (sensitivity = 1)."""
    return max(0.0, count + rng.normal(0, sigma))


def _dp_noise_mean(
    mean_val: float, n: float, value_range: float,
    sigma: float, rng: np.random.Generator,
) -> float:
    """Add Gaussian noise to a mean query.

    Sensitivity of a mean over *n* records with bounded range is
    ``value_range / n``.  We scale sigma accordingly.
    """
    if n <= 0:
        return mean_val
    sensitivity = value_range / n
    return mean_val + rng.normal(0, sigma * sensitivity)


def _dp_noise_percentage(
    pct: float, n: float, sigma: float, rng: np.random.Generator,
) -> float:
    """Add Gaussian noise to a percentage (0-100 range, sensitivity = 100/n)."""
    if n <= 0:
        return pct
    sensitivity = 100.0 / n
    noisy = pct + rng.normal(0, sigma * sensitivity)
    return max(0.0, min(100.0, noisy))


# Approximate value ranges for numeric fields (used for sensitivity calc)
_FIELD_RANGES: dict[str, float] = {
    "batt_duration_mins": 1000.0,
    "batt_num_power_ons": 50.0,
    "mem_avg_pct_used": 100.0,
    "mem_nrs": 50000.0,
    "mem_sysinfo_ram": 131072.0,  # 128 GB in MB
    "onoff_on_time": 86400.0,  # seconds in a day
    "onoff_off_time": 86400.0,
    "onoff_mods_time": 86400.0,
    "onoff_sleep_time": 86400.0,
    "net_received_bytes": 1e12,
    "net_sent_bytes": 1e12,
    "net_nrs": 50000.0,
    "psys_rap_avg": 200.0,
    "pkg_c0_avg": 100.0,
    "avg_freq_avg": 5000.0,
    "temp_avg": 100.0,
    "pkg_power_avg": 200.0,
    "psys_rap_nrs": 50000.0,
    "pkg_c0_nrs": 50000.0,
    "avg_freq_nrs": 50000.0,
    "temp_nrs": 50000.0,
    "pkg_power_nrs": 50000.0,
    "web_chrome_duration": 1e9,
    "web_edge_duration": 1e9,
    "web_firefox_duration": 1e9,
    "web_total_duration": 1e9,
    "web_num_instances": 10000.0,
}


def _count_aggregate_queries(real_results_dir: Path) -> int:
    """Count total number of aggregate queries we will make.

    This is needed to calibrate per-query sigma via advanced composition.
    Each CSV row contributes: 1 count query + K mean/pct queries.
    We conservatively estimate this from the CSVs.
    """
    total = 0
    # We'll count: for each CSV, rows * (1 count + N numeric targets)
    for name, n_numeric in [
        ("battery_on_duration_cpu_family_gen.csv", 2),  # count + 2 means
        ("ram_utilization_histogram.csv", 3),  # count + 3 means
        ("on_off_mods_sleep_summary_by_cpu_marketcodename_gen.csv", 5),
        ("persona_web_cat_usage_analysis.csv", 27),  # count + ~26 webcat pcts
        ("avg_platform_power_c0_freq_temp_by_chassis.csv", 11),  # count + 10
        ("Xeon_network_consumption.csv", 4),  # count + 3 means
        ("battery_power_on_geographic_summary.csv", 1),  # count only (hints)
        ("popular_browsers_by_count_usage_percentage.csv", 1),  # pct
        ("most_popular_browser_in_each_country_by_system_count.csv", 1),
    ]:
        path = real_results_dir / name
        if path.exists():
            df = pd.read_csv(path)
            total += len(df) * (1 + n_numeric)
    return max(total, 1)


def build_generation_plan(
    real_results_dir: Path,
    n_total: int = 5000,
    epsilon_agg: float = 1.0,
    delta: float = 1e-5,
    seed: int = 42,
) -> tuple[GenerationPlan, dict]:
    """Build a stratified generation plan from real query results.

    All aggregate statistics (counts, means, percentages) are noised
    via the Gaussian mechanism before being embedded in prompts.

    Parameters
    ----------
    real_results_dir : Path
        Directory containing real query result CSVs.
    n_total : int
        Target number of synthetic records.
    epsilon_agg : float
        Privacy budget for aggregate queries.  Noise is calibrated via
        advanced composition over all queries.
    delta : float
        Delta parameter for (epsilon, delta)-DP.
    seed : int
        Random seed for reproducibility of noise.

    Returns
    -------
    tuple[GenerationPlan, dict]
        The generation plan and a privacy accounting dict.
    """
    rng = np.random.default_rng(seed)

    # Count total aggregate queries for composition
    n_queries = _count_aggregate_queries(real_results_dir)
    # Advanced composition: per-query sigma so total is (epsilon_agg, delta)-DP
    # Under advanced composition (Dwork et al.), k queries each at
    # (eps_i, delta_i) compose to (sqrt(2k ln(1/delta')) * eps_i + k*eps_i*(e^eps_i - 1), k*delta_i + delta')
    # We use a simpler RDP-style calibration: sigma_per_query = sqrt(2 * n_queries * ln(1.25/delta)) / epsilon_agg
    sigma_per_query = math.sqrt(2.0 * n_queries * math.log(1.25 / delta)) / epsilon_agg
    logger.info(
        "DP aggregates: epsilon_agg=%.2f, delta=%.1e, n_queries=%d, "
        "sigma_per_query=%.4f",
        epsilon_agg, delta, n_queries, sigma_per_query,
    )
    print(
        f"DP aggregates: epsilon_agg={epsilon_agg}, delta={delta}, "
        f"n_queries={n_queries}, sigma_per_query={sigma_per_query:.4f}"
    )

    privacy_info = {
        "epsilon_agg": epsilon_agg,
        "delta": delta,
        "n_aggregate_queries": n_queries,
        "sigma_per_query": sigma_per_query,
    }

    group_alloc = _compute_group_allocation(n_total)
    allocations: list[StratumAllocation] = []

    # -- Secondary / cross-query data ------------------------------------
    # These hints use DP-noised counts for distributions.
    batt_country_df = _read_csv(
        real_results_dir, "battery_power_on_geographic_summary.csv"
    )
    # Noise country counts before computing distribution hints
    if "number_of_systems" in batt_country_df.columns:
        batt_country_df["number_of_systems"] = batt_country_df[
            "number_of_systems"
        ].apply(lambda x: _dp_noise_count(x, sigma_per_query, rng))
    batt_country_hint = _country_distribution_hint(batt_country_df, "country")

    browser_df = _read_csv(
        real_results_dir, "popular_browsers_by_count_usage_percentage.csv"
    )
    # Noise browser percentages
    n_browser_systems = browser_df["percent_systems"].sum()  # ~100
    browser_df["percent_systems"] = browser_df["percent_systems"].apply(
        lambda x: _dp_noise_percentage(x, n_browser_systems, sigma_per_query, rng)
    )
    browser_hint = _browser_distribution_hint(browser_df)

    browser_country_df = _read_csv(
        real_results_dir,
        "most_popular_browser_in_each_country_by_system_count.csv",
    )
    # Browser-per-country is categorical (which browser is #1) — noise the
    # underlying system_count before deriving the hint.
    if "system_count" in browser_country_df.columns:
        browser_country_df["system_count"] = browser_country_df[
            "system_count"
        ].apply(lambda x: _dp_noise_count(x, sigma_per_query, rng))
    browser_country_hint = _browser_country_hint(browser_country_df)

    pkg_country_df = _read_csv(real_results_dir, "pkg_power_by_country.csv")
    if "number_of_systems" in pkg_country_df.columns:
        pkg_country_df["number_of_systems"] = pkg_country_df[
            "number_of_systems"
        ].apply(lambda x: _dp_noise_count(x, sigma_per_query, rng))
    hw_country_hint = _country_distribution_hint(
        pkg_country_df, "countryname_normalized"
    )

    # ---- BATT group ----------------------------------------------------
    batt_df = _read_csv(
        real_results_dir, "battery_on_duration_cpu_family_gen.csv"
    )
    # Noise counts before scaling
    noisy_batt_counts = batt_df["number_of_systems"].apply(
        lambda x: _dp_noise_count(x, sigma_per_query, rng)
    )
    batt_scaled = _scale_counts(noisy_batt_counts, group_alloc["batt"])
    for i, row in batt_df.iterrows():
        count = int(batt_scaled.iloc[i])  # type: ignore[arg-type]
        if count <= 0:
            continue
        n_real = float(row["number_of_systems"])
        allocations.append(StratumAllocation(
            active_group="batt",
            count=count,
            cat_constraints={
                "cpucode": row["marketcodename"],
                "cpu_family": row["cpugen"],
            },
            numeric_targets={
                "batt_duration_mins": _dp_noise_mean(
                    float(row["avg_duration_mins_on_battery"]),
                    n_real, _FIELD_RANGES["batt_duration_mins"],
                    sigma_per_query, rng,
                ),
                "batt_num_power_ons": _dp_noise_mean(
                    3.0, n_real, _FIELD_RANGES["batt_num_power_ons"],
                    sigma_per_query, rng,
                ),
            },
            secondary_hints=(
                "chassistype MUST be Notebook or '2 in 1' (battery systems only).\n"
                f"Country distribution: {batt_country_hint}"
            ),
        ))

    # ---- MEM group -----------------------------------------------------
    mem_df = _read_csv(real_results_dir, "ram_utilization_histogram.csv")
    noisy_mem_counts = mem_df["count(DISTINCT guid)"].apply(
        lambda x: _dp_noise_count(x, sigma_per_query, rng)
    )
    mem_scaled = _scale_counts(noisy_mem_counts, group_alloc["mem"])
    for i, row in mem_df.iterrows():
        count = int(mem_scaled.iloc[i])  # type: ignore[arg-type]
        if count <= 0:
            continue
        ram_gb = float(row["ram_gb"])
        n_real = float(row["count(DISTINCT guid)"])
        allocations.append(StratumAllocation(
            active_group="mem",
            count=count,
            cat_constraints={"ram": ram_gb},
            numeric_targets={
                "mem_avg_pct_used": _dp_noise_mean(
                    float(row["avg_percentage_used"]),
                    n_real, _FIELD_RANGES["mem_avg_pct_used"],
                    sigma_per_query, rng,
                ),
                "mem_nrs": _dp_noise_mean(
                    5000.0, n_real, _FIELD_RANGES["mem_nrs"],
                    sigma_per_query, rng,
                ),
                "mem_sysinfo_ram": ram_gb * 1024,  # deterministic from ram_gb
            },
        ))

    # ---- ONOFF group ---------------------------------------------------
    onoff_df = _read_csv(
        real_results_dir,
        "on_off_mods_sleep_summary_by_cpu_marketcodename_gen.csv",
    )
    noisy_onoff_counts = onoff_df["number_of_systems"].apply(
        lambda x: _dp_noise_count(x, sigma_per_query, rng)
    )
    onoff_scaled = _scale_counts(noisy_onoff_counts, group_alloc["onoff"])
    for i, row in onoff_df.iterrows():
        count = int(onoff_scaled.iloc[i])  # type: ignore[arg-type]
        if count <= 0:
            continue
        n_real = float(row["number_of_systems"])
        allocations.append(StratumAllocation(
            active_group="onoff",
            count=count,
            cat_constraints={
                "cpucode": row["marketcodename"],
                "cpu_family": row["cpugen"],
            },
            numeric_targets={
                "onoff_on_time": _dp_noise_mean(
                    float(row["avg_on_time"]),
                    n_real, _FIELD_RANGES["onoff_on_time"],
                    sigma_per_query, rng,
                ),
                "onoff_off_time": _dp_noise_mean(
                    float(row["avg_off_time"]),
                    n_real, _FIELD_RANGES["onoff_off_time"],
                    sigma_per_query, rng,
                ),
                "onoff_mods_time": _dp_noise_mean(
                    float(row["avg_modern_sleep_time"]),
                    n_real, _FIELD_RANGES["onoff_mods_time"],
                    sigma_per_query, rng,
                ),
                "onoff_sleep_time": _dp_noise_mean(
                    float(row["avg_sleep_time"]),
                    n_real, _FIELD_RANGES["onoff_sleep_time"],
                    sigma_per_query, rng,
                ),
            },
        ))

    # ---- BROWSER + WEBCAT group ----------------------------------------
    persona_df = _read_csv(
        real_results_dir, "persona_web_cat_usage_analysis.csv"
    )
    # Only personas with actual webcat data (non-NaN webcat columns)
    webcat_sample_col = "content_creation_photo_edit_creation"
    persona_with_webcat = persona_df.dropna(subset=[webcat_sample_col])

    if len(persona_with_webcat) > 0:
        noisy_bw_counts = persona_with_webcat[
            "number_of_systems"
        ].reset_index(drop=True).apply(
            lambda x: _dp_noise_count(x, sigma_per_query, rng)
        )
        bw_scaled = _scale_counts(noisy_bw_counts, group_alloc["browser_webcat"])
        for i, (_, row) in enumerate(persona_with_webcat.iterrows()):
            count = int(bw_scaled.iloc[i])
            if count <= 0:
                continue

            n_real = float(row["number_of_systems"])

            # Build webcat targets from CSV columns — noise each percentage
            webcat_targets: dict[str, float] = {}
            for csv_col, record_col in _WEBCAT_CSV_TO_RECORD.items():
                val = row.get(csv_col)
                if pd.notna(val):
                    webcat_targets[record_col] = _dp_noise_percentage(
                        float(val), n_real, sigma_per_query, rng,
                    )

            allocations.append(StratumAllocation(
                active_group="browser_webcat",
                count=count,
                cat_constraints={"persona": row["persona"]},
                numeric_targets=webcat_targets,
                secondary_hints=(
                    f"Browser distribution across systems: {browser_hint}\n"
                    f"{browser_country_hint}\n"
                    "web_total_duration = sum of all browser durations.\n"
                    "webcat_* fields should sum to roughly 100 (percentages).\n"
                    "Generate web_chrome_duration, web_edge_duration, "
                    "web_firefox_duration as milliseconds (1e3-1e8 range).\n"
                    "web_num_instances: 10-500."
                ),
            ))

    # ---- HW group ------------------------------------------------------
    hw_df = _read_csv(
        real_results_dir,
        "avg_platform_power_c0_freq_temp_by_chassis.csv",
    )
    noisy_hw_counts = hw_df["number_of_systems"].apply(
        lambda x: _dp_noise_count(x, sigma_per_query, rng)
    )
    hw_scaled = _scale_counts(noisy_hw_counts, group_alloc["hw"])
    for i, row in hw_df.iterrows():
        count = int(hw_scaled.iloc[i])  # type: ignore[arg-type]
        if count <= 0:
            continue
        n_real = float(row["number_of_systems"])
        allocations.append(StratumAllocation(
            active_group="hw",
            count=count,
            cat_constraints={"chassistype": row["chassistype"]},
            numeric_targets={
                "psys_rap_avg": _dp_noise_mean(
                    float(row["avg_psys_rap_watts"]),
                    n_real, _FIELD_RANGES["psys_rap_avg"],
                    sigma_per_query, rng,
                ),
                "pkg_c0_avg": _dp_noise_mean(
                    float(row["avg_pkg_c0"]),
                    n_real, _FIELD_RANGES["pkg_c0_avg"],
                    sigma_per_query, rng,
                ),
                "avg_freq_avg": _dp_noise_mean(
                    float(row["avg_freq_mhz"]),
                    n_real, _FIELD_RANGES["avg_freq_avg"],
                    sigma_per_query, rng,
                ),
                "temp_avg": _dp_noise_mean(
                    float(row["avg_temp_centigrade"]),
                    n_real, _FIELD_RANGES["temp_avg"],
                    sigma_per_query, rng,
                ),
                # nrs defaults — noise them too
                "psys_rap_nrs": _dp_noise_mean(
                    500.0, n_real, _FIELD_RANGES["psys_rap_nrs"],
                    sigma_per_query, rng,
                ),
                "pkg_c0_nrs": _dp_noise_mean(
                    500.0, n_real, _FIELD_RANGES["pkg_c0_nrs"],
                    sigma_per_query, rng,
                ),
                "avg_freq_nrs": _dp_noise_mean(
                    500.0, n_real, _FIELD_RANGES["avg_freq_nrs"],
                    sigma_per_query, rng,
                ),
                "temp_nrs": _dp_noise_mean(
                    500.0, n_real, _FIELD_RANGES["temp_nrs"],
                    sigma_per_query, rng,
                ),
                "pkg_power_nrs": _dp_noise_mean(
                    500.0, n_real, _FIELD_RANGES["pkg_power_nrs"],
                    sigma_per_query, rng,
                ),
                "pkg_power_avg": _dp_noise_mean(
                    10.0, n_real, _FIELD_RANGES["pkg_power_avg"],
                    sigma_per_query, rng,
                ),
            },
            secondary_hints=f"Country distribution: {hw_country_hint}",
        ))

    # ---- NET group -----------------------------------------------------
    net_df = _read_csv(real_results_dir, "Xeon_network_consumption.csv")
    # Handle 'n/a' OS values - skip them
    net_df = net_df[net_df["os"] != "n/a"].copy()
    noisy_net_counts = net_df["number_of_systems"].reset_index(
        drop=True
    ).apply(lambda x: _dp_noise_count(x, sigma_per_query, rng))
    net_scaled = _scale_counts(noisy_net_counts, group_alloc["net"])
    for i, (_, row) in enumerate(net_df.iterrows()):
        count = int(net_scaled.iloc[i])
        if count <= 0:
            continue

        n_real = float(row["number_of_systems"])

        # processor_class → cpuname mapping
        if row["processor_class"] == "Server Class":
            cpu_constraint = "Intel Xeon"
        else:
            cpu_constraint = "_non_xeon"  # special marker

        allocations.append(StratumAllocation(
            active_group="net",
            count=count,
            cat_constraints={
                "_processor_class": cpu_constraint,
                "os": row["os"],
            },
            numeric_targets={
                "net_received_bytes": _dp_noise_mean(
                    float(row["avg_bytes_received"]),
                    n_real, _FIELD_RANGES["net_received_bytes"],
                    sigma_per_query, rng,
                ),
                "net_sent_bytes": _dp_noise_mean(
                    float(row["avg_bytes_sent"]),
                    n_real, _FIELD_RANGES["net_sent_bytes"],
                    sigma_per_query, rng,
                ),
                "net_nrs": _dp_noise_mean(
                    5000.0, n_real, _FIELD_RANGES["net_nrs"],
                    sigma_per_query, rng,
                ),
            },
        ))

    return GenerationPlan(allocations=allocations), privacy_info


# ---------------------------------------------------------------------------
# Build conditional prompts per stratum
# ---------------------------------------------------------------------------

_STRATUM_SYSTEM_PROMPT = (
    "You are a tabular data generator. You output ONLY valid JSON conforming "
    "to the provided schema. Do not add commentary, explanations, or any text "
    "outside the JSON structure. Every field value must come from the allowed "
    "set or be a realistic numeric value.\n\n"
    "You are generating Intel DCA telemetry records conditioned on specific "
    "stratum targets. Follow the constraints EXACTLY."
)

# Categorical distribution hints (same as _DISTRIBUTION_PROMPT but condensed)
_CAT_DIST_HINTS = """CATEGORICAL DISTRIBUTIONS (for fields NOT fixed by stratum):
- chassistype: Notebook 72%, Desktop 18%, 2 in 1 6%, Intel NUC/STK 2%, Tablet 1%, Other <1%
- os: Win10 85%, Win11 10%, Win8.1 3%, Win Server 2%
- countryname_normalized: US 22%, India 10%, China 7%, Germany 5%, UK 4%, Brazil 4%, Japan 3%, France 3%, Korea 2%, Italy 2%, Canada 2%, Australia 2%, Mexico 2%, Russia 2%, others spread
- persona: Casual User 27%, Communication 16%, Casual Gamer 16%, Office/Productivity 13%, Web User 8%, Entertainment 6%, Content Creator/IT 5%, Win Store App User 4%, Gamer 2%, File & Network Sharer 2%, Unknown 1%
- modelvendor_normalized: Lenovo 22%, Dell 15%, HP 15%, Asus 8%, Acer 7%, Microsoft 3%, Samsung 3%, HUAWEI 2%, MSI 2%, Toshiba 2%, others spread
- ram: 8 (38%), 16 (28%), 4 (18%), 32 (7%), 12 (4%), 6 (2%), 64 (1%)
- cpuname: Intel Core i5 30%, Intel Core i7 25%, Intel Core i3 15%, Intel Celeron 10%, Intel Pentium 8%, Intel Core i9 5%, Intel Atom 3%, Intel Xeon 2%, Other 2%"""


def build_stratum_prompt(alloc: StratumAllocation, batch_size: int) -> str:
    """Build a conditional generation prompt for one stratum batch."""
    lines: list[str] = []
    lines.append(
        f"Generate exactly {batch_size} synthetic Intel DCA telemetry records "
        f"with these EXACT constraints:\n"
    )

    # Fixed categorical values
    lines.append("FIXED CATEGORICAL VALUES (use these exact values):")
    for key, val in alloc.cat_constraints.items():
        if key == "_processor_class":
            if val == "Intel Xeon":
                lines.append(f"- cpuname: Intel Xeon")
            else:
                lines.append(
                    "- cpuname: any value EXCEPT Intel Xeon "
                    "(use Intel Core i5, i7, i3, Celeron, Pentium, etc.)"
                )
            continue
        if key == "ram":
            lines.append(f"- ram: {val}")
            lines.append(
                f"  (mem_sysinfo_ram should be {float(val) * 1024:.0f})"
            )
            continue
        lines.append(f"- {key}: {val}")

    # Active numeric group + targets
    lines.append(f"\nACTIVE NUMERIC GROUP: {alloc.active_group}")
    lines.append("Target AVERAGES across the batch (individual values should vary around these):")
    for field_name, target in alloc.numeric_targets.items():
        if target == 0:
            lines.append(f"- {field_name}: 0")
        elif abs(target) < 1:
            lines.append(f"- {field_name}: ~{target:.4f}")
        elif abs(target) < 100:
            lines.append(f"- {field_name}: ~{target:.1f}")
        elif abs(target) < 100000:
            lines.append(f"- {field_name}: ~{target:.0f}")
        else:
            lines.append(f"- {field_name}: ~{target:.4e}")

    # Zero groups
    zero_groups = [g for g in _ALL_GROUPS if g != alloc.active_group]
    lines.append(f"\nALL OTHER NUMERIC GROUPS MUST BE EXACTLY 0:")
    for zg in zero_groups:
        fields = _GROUP_FIELDS[zg]
        if zg == "browser_webcat":
            lines.append(f"- {zg}: web_*=0, webcat_*=0")
        else:
            lines.append(f"- {zg}: {', '.join(fields)} = 0")

    # Secondary hints
    if alloc.secondary_hints:
        lines.append(f"\nADDITIONAL CONSTRAINTS:")
        lines.append(alloc.secondary_hints)

    # General categorical distribution hints (for non-fixed categoricals)
    lines.append(f"\n{_CAT_DIST_HINTS}")

    lines.append(
        "\nram is ALWAYS nonzero (pick from standard sizes: 4, 8, 16, 32, 64)."
    )
    lines.append(
        "Vary the non-fixed categorical fields across records. "
        "Do NOT repeat the same combination."
    )

    return "\n".join(lines)
