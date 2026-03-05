"""
PE Model Experiment: Compare OpenAI models with improved prompt engineering.

Key improvements over the original PE pipeline:
1. Enum-constrained Pydantic models (Literal types) to eliminate hallucinated categories
2. Distribution-aware prompts with real marginal percentages and numeric ranges
3. Concrete sparsity examples showing correct active/inactive group patterns
4. Model comparison across gpt-5-mini, gpt-5-nano, gpt-4.1-mini, and others

Usage:
    python scripts/pe_model_experiment.py [--models gpt-5-mini gpt-4.1-mini] [--n-records 100] [--batch-size 5]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel

# Import shared constants from the PE package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pe.constants import (
    ChassisType,
    CountryType,
    CpuCodeType,
    CpuFamilyType,
    CpuNameType,
    OsType,
    PersonaType,
    ProcessorType,
    VALID_CHASSIS,
    VALID_COUNTRIES,
    VALID_OS,
    VALID_PERSONA,
    VendorType,
)
from src.pe.distance import CAT_COLS, NUMERIC_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All categorical value lists and Literal types are now imported from
# src.pe.constants (derived programmatically from real data CSVs).


class ConstrainedTelemetryRecord(BaseModel):
    """Pydantic model with enum-constrained categoricals.

    Uses Literal types imported from src.pe.constants to restrict outputs
    to only valid values from the real dataset, eliminating hallucinated
    categories entirely.
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


class ConstrainedRecordsBatch(BaseModel):
    records: list[ConstrainedTelemetryRecord]


# ---------------------------------------------------------------------------
# Improved prompt with real distribution statistics
# ---------------------------------------------------------------------------

IMPROVED_INSTRUCTIONS = (
    "You are a tabular data generator. You output ONLY valid JSON conforming "
    "to the provided schema. Do not add commentary, explanations, or any text "
    "outside the JSON structure. Every field value must come from the allowed "
    "set or be a realistic numeric value within the specified ranges."
)

# Distribution info derived from real DCA data
DISTRIBUTION_PROMPT = """You are generating synthetic Intel DCA telemetry records. Each record = one client system (Windows PC).

CATEGORICAL DISTRIBUTIONS (match these frequencies):
- chassistype: Notebook 72%, Desktop 18%, 2 in 1 6%, Intel NUC/STK 2%, Tablet 1%, Other <1%, Server/WS <1%
- os: Win10 85%, Win11 10%, Win8.1 3%, Win Server 2%
- countryname_normalized: United States of America 22%, India 10%, China 7%, Germany 5%, United Kingdom of Great Britain and Northern Ireland 4%, Brazil 4%, Japan 3%, France 3%, Korea, Republic of 2%, Italy 2%, Canada 2%, Australia 2%, Mexico 2%, Russian Federation 2%, Poland 1.5%, Netherlands 1.5%, Spain 1.5%, Turkey 1.5%, Indonesia 1%, Thailand 1%, Taiwan, Province of China 1%, Sweden 1%, Switzerland 1%, Colombia 1%, other countries ~15% (spread across remaining countries in the enum)
- persona: Casual User 27%, Communication 16%, Casual Gamer 16%, Office/Productivity 13%, Web User 8%, Entertainment 6%, Content Creator/IT 5%, Win Store App User 4%, Gamer 2%, File & Network Sharer 2%, Unknown 1%
- modelvendor_normalized: Lenovo 22%, Dell 15%, HP 15%, Asus 8%, Acer 7%, Microsoft Corporation 3%, Samsung 3%, HUAWEI 2%, MSI 2%, Toshiba 2%, Fujitsu 1%, Intel 1%, others spread across remaining vendors in the enum
- ram: 8 (38%), 16 (28%), 4 (18%), 32 (7%), 12 (4%), 6 (2%), 64 (1%)

NUMERIC RANGES (for nonzero values only):
- net_received_bytes: 1e8 to 1e14 (median ~1e11), net_sent_bytes: 1e7 to 1e13 (median ~1e10), net_nrs: 100-50000
- mem_avg_pct_used: 20-95 (median 55), mem_nrs: 100-50000, mem_sysinfo_ram: match ram field in MB (e.g. ram=8 -> mem_sysinfo_ram=8192)
- batt_num_power_ons: 1-20 (median 3), batt_duration_mins: 10-600 (median 130)
- web_chrome_duration/web_edge_duration/web_firefox_duration: 1e3 to 1e8 (ms)
- webcat_* fields: 0.0-100.0 (percentage of browsing time; active webcat fields should sum to roughly 100)
- onoff_on_time/off_time/mods_time/sleep_time: 0-1e7 (seconds)
- disp_num_displays: 1-4, disp_total_duration_ac/dc: 1e3 to 1e8

*** CRITICAL SPARSITY RULES (MOST IMPORTANT) ***
There are 7 numeric groups: net, mem, batt, browser+webcat, onoff, disp, hw.
Each system has data in EXACTLY 1 or 2 of these groups. ALL other groups MUST be exactly 0.

Group definitions:
- net: net_nrs, net_received_bytes, net_sent_bytes
- mem: mem_nrs, mem_avg_pct_used, mem_sysinfo_ram
- batt: batt_num_power_ons, batt_duration_mins
- browser+webcat: web_chrome_duration, web_edge_duration, web_firefox_duration, web_total_duration, web_num_instances, AND all webcat_* fields
- onoff: onoff_on_time, onoff_off_time, onoff_mods_time, onoff_sleep_time
- disp: disp_num_displays, disp_total_duration_ac, disp_total_duration_dc
- hw: psys_rap_nrs, psys_rap_avg, pkg_c0_nrs, pkg_c0_avg, avg_freq_nrs, avg_freq_avg, temp_nrs, temp_avg, pkg_power_nrs, pkg_power_avg

Approximate group frequencies (what % of systems have data in each group):
- net: 25%, mem: 22%, batt: 15% (Notebook/2-in-1 only), browser+webcat: 20%, onoff: 18%, disp: 10%, hw: <1%

EXAMPLE of correct sparsity (system with net + mem active):
- net_nrs=5000, net_received_bytes=5e10, net_sent_bytes=3e9 (NONZERO - active)
- mem_nrs=5000, mem_avg_pct_used=62, mem_sysinfo_ram=16384 (NONZERO - active)
- batt_num_power_ons=0, batt_duration_mins=0 (ZERO - inactive)
- web_*=0, webcat_*=0 (all ZERO - inactive)
- onoff_*=0 (all ZERO - inactive)
- disp_*=0 (all ZERO - inactive)
- hw fields=0 (all ZERO - inactive)

EXAMPLE of correct sparsity (system with browser+webcat + onoff active):
- net_*=0 (ZERO), mem_*=0 (ZERO), batt_*=0 (ZERO)
- web_chrome_duration=5e6, web_total_duration=6e6, web_num_instances=150 (NONZERO)
- webcat_entertainment_video_streaming=35, webcat_social_social_network=25, webcat_search=20, webcat_news=10, webcat_mail=10 (sum ~100)
- onoff_on_time=3e6, onoff_off_time=1e6, onoff_mods_time=500000, onoff_sleep_time=2e6 (NONZERO)
- disp_*=0 (ZERO), hw=0 (ZERO)

VIOLATION: Setting net, mem, browser, AND onoff all nonzero for the same system. That is 4 active groups (max allowed is 2).

ram is ALWAYS present and nonzero regardless of group activation.

Do NOT inject your own assumptions. Follow the distributions and sparsity rules exactly."""


def _build_improved_prompt(batch_size: int) -> str:
    return (
        f"{DISTRIBUTION_PROMPT}\n\n"
        f"Generate exactly {batch_size} synthetic telemetry records.\n"
        f"Requirements:\n"
        f"1. Each record MUST have exactly 1 or 2 active numeric groups (all others = 0).\n"
        f"2. Vary countries broadly - use many different countries, not just top 3.\n"
        f"3. Match the categorical frequency distributions above.\n"
        f"4. ram is always nonzero (pick from 4, 8, 16, 32, 64 with given frequencies).\n"
        f"5. Do not repeat the same combination of categoricals across records."
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ModelExperiment:
    def __init__(self, model: str, max_concurrent: int = 20):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=api_key, max_retries=3)
        self.model = model
        self.max_concurrent = max_concurrent
        self._is_reasoning = model.startswith("o") or model.startswith("gpt-5")

    async def generate_batch(self, prompt: str) -> list[dict]:
        """Generate a single batch of records."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": prompt,
                "text_format": ConstrainedRecordsBatch,
                "max_output_tokens": 16000,
            }
            if self._is_reasoning:
                kwargs["instructions"] = IMPROVED_INSTRUCTIONS
                kwargs["reasoning"] = {"effort": "low"}
            else:
                kwargs["instructions"] = IMPROVED_INSTRUCTIONS
                kwargs["temperature"] = 0.8

            response = await self.client.responses.parse(**kwargs)
            if response.output_parsed and response.output_parsed.records:
                return [r.model_dump() for r in response.output_parsed.records]
        except LengthFinishReasonError:
            logger.warning("[%s] Response truncated, dropping batch", self.model)
        except Exception as e:
            logger.warning("[%s] API error: %s", self.model, e)
        return []

    async def run_experiment(
        self, n_records: int, batch_size: int = 5
    ) -> pd.DataFrame:
        """Generate n_records using the improved constrained pipeline."""
        overshoot = int(n_records * 1.3)
        n_batches = (overshoot + batch_size - 1) // batch_size
        prompt = _build_improved_prompt(batch_size)

        print(f"\n{'='*60}")
        print(f"Model: {self.model}")
        print(f"Target: {n_records} records ({n_batches} batches of {batch_size})")
        print(f"{'='*60}")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[list[dict]] = [[] for _ in range(n_batches)]
        completed = 0
        t0 = time.time()

        async def run_one(idx: int) -> None:
            nonlocal completed
            async with semaphore:
                result = await self.generate_batch(prompt)
                results[idx] = result
                completed += 1
                if completed % 10 == 0 or completed == n_batches:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{self.model}] {completed}/{n_batches} ({elapsed:.0f}s, {rate:.1f} batch/s)")

        tasks = [run_one(i) for i in range(n_batches)]
        await asyncio.gather(*tasks)

        all_records = [r for batch in results for r in batch]
        elapsed = time.time() - t0

        # Convert to DataFrame
        all_cols = CAT_COLS + NUMERIC_COLS
        rows = []
        for r in all_records[:n_records]:
            row = {}
            for c in all_cols:
                row[c] = r.get(c, 0 if c in NUMERIC_COLS else "Unknown")
            rows.append(row)

        df = pd.DataFrame(rows)
        for c in [col for col in NUMERIC_COLS if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        for c in [col for col in CAT_COLS if col in df.columns]:
            df[c] = df[c].fillna("Unknown").astype(str)

        print(f"\n  [{self.model}] Generated {len(all_records)} raw -> {len(df)} returned in {elapsed:.1f}s")
        return df


def analyze_batch(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Analyze a batch of generated records for quality metrics."""
    results: dict[str, Any] = {"model": model_name, "n_records": len(df)}

    # Categorical distribution analysis
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        results[f"{col}_unique"] = len(vc)
        results[f"{col}_top3"] = vc.head(3).to_dict()

    # Sparsity analysis
    from src.pe.api import _NUMERIC_GROUPS
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        nonzero_pct = (df[present].abs().sum(axis=1) > 0).mean() * 100
        results[f"sparsity_{gname}_nonzero_pct"] = round(nonzero_pct, 1)

    # Check for hallucinated categoricals (should be 0 with constrained model)
    hallucinated = {}
    valid_sets = {
        "chassistype": set(VALID_CHASSIS),
        "countryname_normalized": set(VALID_COUNTRIES),
        "os": set(VALID_OS),
        "persona": set(VALID_PERSONA),
    }
    for col, valid in valid_sets.items():
        if col in df.columns:
            actual = set(df[col].unique())
            extra = actual - valid
            if extra:
                hallucinated[col] = list(extra)
    results["hallucinated_categories"] = hallucinated

    # Numeric range checks
    if "ram" in df.columns:
        results["ram_distribution"] = df["ram"].value_counts().head(10).to_dict()

    if "mem_avg_pct_used" in df.columns:
        nonzero_mem = df.loc[df["mem_avg_pct_used"] > 0, "mem_avg_pct_used"]
        if len(nonzero_mem) > 0:
            results["mem_pct_stats"] = {
                "mean": round(nonzero_mem.mean(), 1),
                "median": round(nonzero_mem.median(), 1),
                "min": round(nonzero_mem.min(), 1),
                "max": round(nonzero_mem.max(), 1),
            }

    if "batt_duration_mins" in df.columns:
        nonzero_batt = df.loc[df["batt_duration_mins"] > 0, "batt_duration_mins"]
        if len(nonzero_batt) > 0:
            results["batt_duration_stats"] = {
                "mean": round(nonzero_batt.mean(), 1),
                "median": round(nonzero_batt.median(), 1),
                "min": round(nonzero_batt.min(), 1),
                "max": round(nonzero_batt.max(), 1),
            }

    return results


def print_comparison(all_results: list[dict[str, Any]]) -> None:
    """Print a comparison table of all model results."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for res in all_results:
        print(f"\n--- {res['model']} ({res['n_records']} records) ---")

        # Hallucinated categories
        hall = res.get("hallucinated_categories", {})
        if hall:
            print(f"  HALLUCINATED CATEGORIES: {hall}")
        else:
            print("  Hallucinated categories: NONE (all valid)")

        # Key categorical distributions
        for col in ["chassistype", "os", "persona", "countryname_normalized"]:
            key = f"{col}_top3"
            if key in res:
                top3 = res[key]
                formatted = ", ".join(f"{k}: {v:.1%}" for k, v in top3.items())
                print(f"  {col} (top 3): {formatted}")
            uniq_key = f"{col}_unique"
            if uniq_key in res:
                print(f"    unique values: {res[uniq_key]}")

        # Sparsity
        print("  Sparsity (% nonzero):")
        for key, val in sorted(res.items()):
            if key.startswith("sparsity_"):
                gname = key.replace("sparsity_", "").replace("_nonzero_pct", "")
                print(f"    {gname}: {val}%")

        # RAM distribution
        if "ram_distribution" in res:
            print(f"  RAM distribution: {res['ram_distribution']}")

        # Numeric stats
        if "mem_pct_stats" in res:
            print(f"  Memory % used (nonzero): {res['mem_pct_stats']}")
        if "batt_duration_stats" in res:
            print(f"  Battery duration mins (nonzero): {res['batt_duration_stats']}")


async def main():
    parser = argparse.ArgumentParser(description="PE Model Experiment")
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini"],
        help="Models to test",
    )
    parser.add_argument("--n-records", type=int, default=100, help="Records per model")
    parser.add_argument("--batch-size", type=int, default=5, help="Records per API call")
    parser.add_argument("--output-dir", type=str, default="data/pe_experiments", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model in args.models:
        exp = ModelExperiment(model=model)
        df = await exp.run_experiment(n_records=args.n_records, batch_size=args.batch_size)

        # Save raw output
        df.to_parquet(output_dir / f"experiment_{model.replace('/', '_')}.parquet", index=False)

        # Analyze
        analysis = analyze_batch(df, model)
        all_results.append(analysis)

        # Save analysis
        with open(output_dir / f"analysis_{model.replace('/', '_')}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    print_comparison(all_results)

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


from pathlib import Path

if __name__ == "__main__":
    asyncio.run(main())
