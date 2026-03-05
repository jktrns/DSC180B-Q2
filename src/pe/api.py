import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import APITimeoutError, AsyncOpenAI, LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from .distance import CAT_COLS, NUMERIC_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constrained categorical values derived from the real DCA telemetry data.
# Using Literal types forces OpenAI's structured outputs to only produce
# these exact values, eliminating hallucinated categories entirely.
# ---------------------------------------------------------------------------

VALID_CHASSIS = [
    "2 in 1", "Desktop", "Intel NUC/STK", "Notebook",
    "Other/Unknown", "Tablet", "Detachable",
]

VALID_COUNTRIES = [
    "Argentina", "Australia", "Austria", "Bangladesh", "Belgium", "Brazil",
    "Canada", "Chile", "China", "Colombia", "Czech Republic", "Denmark",
    "Ecuador", "Egypt", "France", "Germany", "Greece", "Hong Kong",
    "Hungary", "India", "Indonesia", "Israel", "Italy", "Japan",
    "Korea, Republic of", "Malaysia", "Mexico", "Netherlands", "Norway",
    "Other", "Pakistan", "Peru", "Philippines", "Poland", "Portugal",
    "Romania", "Russian Federation", "Saudi Arabia", "Singapore",
    "South Africa", "Spain", "Sweden", "Switzerland",
    "Taiwan, Province of China", "Thailand", "Turkey", "Ukraine",
    "United Arab Emirates",
    "United Kingdom of Great Britain and Northern Ireland",
    "United States of America", "Viet Nam",
]

VALID_OS = ["Win10", "Win11", "Win8.1", "Win Server", "n/a"]

VALID_PERSONA = [
    "Casual Gamer", "Casual User", "Communication", "Content Creator/IT",
    "Entertainment", "File & Network Sharer", "Gamer",
    "Office/Productivity", "Unknown", "Web User", "Win Store App User",
]

VALID_VENDORS = [
    "Lenovo", "Dell Inc.", "HP", "ASUSTeK COMPUTER INC.", "Acer",
    "Microsoft Corporation", "Samsung", "HUAWEI", "MSI",
    "Micro-Star International", "Toshiba", "TOSHIBA", "Sony",
    "Hewlett-Packard", "Fujitsu", "LG Electronics", "Razer",
    "Panasonic", "Dynabook", "Intel", "Other",
]

VALID_CPUNAMES = [
    "Intel Core i5", "Intel Core i7", "Intel Core i3", "Intel Core i9",
    "Intel Celeron", "Intel Pentium", "Intel Core Ultra 5",
    "Intel Core Ultra 7", "Intel Core Ultra 9", "Intel Xeon",
    "Intel Atom", "Other",
]

VALID_CPUCODES = [
    "Alder Lake", "Tiger Lake", "Comet Lake", "Ice Lake", "Raptor Lake",
    "Whiskey Lake", "Coffee Lake", "Meteor Lake", "Kaby Lake",
    "Skylake", "Cannon Lake", "Rocket Lake", "Jasper Lake",
    "Gemini Lake", "Elkhart Lake", "Arrow Lake", "Lunar Lake", "Other",
]

VALID_CPU_FAMILY = [
    "6th Gen", "7th Gen", "8th Gen", "9th Gen", "10th Gen",
    "11th Gen", "12th Gen", "13th Gen", "14th Gen",
    "Core Ultra Series 1", "Core Ultra Series 2",
    "Pentium/Celeron", "Xeon", "Atom", "Other",
]

VALID_PROCESSORS = [
    "i5-1135G7", "i7-1165G7", "i5-10210U", "i7-10510U", "i5-8250U",
    "i7-8550U", "i5-1235U", "i5-1240P", "i7-1255U", "i7-1260P",
    "i5-8265U", "i7-8565U", "i5-10310U", "i7-10610U", "i5-1145G7",
    "i7-1185G7", "i5-1335U", "i7-1355U", "i5-1345U", "i7-1365U",
    "Other",
]

# Build Literal types from valid value lists
ChassisType = Literal[tuple(VALID_CHASSIS)]  # type: ignore[valid-type]
CountryType = Literal[tuple(VALID_COUNTRIES)]  # type: ignore[valid-type]
OsType = Literal[tuple(VALID_OS)]  # type: ignore[valid-type]
PersonaType = Literal[tuple(VALID_PERSONA)]  # type: ignore[valid-type]
VendorType = Literal[tuple(VALID_VENDORS)]  # type: ignore[valid-type]
CpuNameType = Literal[tuple(VALID_CPUNAMES)]  # type: ignore[valid-type]
CpuCodeType = Literal[tuple(VALID_CPUCODES)]  # type: ignore[valid-type]
CpuFamilyType = Literal[tuple(VALID_CPU_FAMILY)]  # type: ignore[valid-type]
ProcessorType = Literal[tuple(VALID_PROCESSORS)]  # type: ignore[valid-type]


class TelemetryRecord(BaseModel):
    """Pydantic model with enum-constrained categoricals.

    Uses Literal types to restrict outputs to only valid values from the real
    dataset, eliminating hallucinated categories (e.g. "Workstation",
    "Server/WS", "USA", "Japanese").
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


# Distribution-aware prompt with real marginal percentages and sparsity examples
_DISTRIBUTION_PROMPT = """You are generating synthetic Intel DCA telemetry records. Each record = one client system (Windows PC).

CATEGORICAL DISTRIBUTIONS (match these frequencies):
- chassistype: Notebook 72%, Desktop 18%, 2 in 1 6%, Intel NUC/STK 2%, Tablet 1%, Other/Unknown 1%
- os: Win10 85%, Win11 10%, Win8.1 3%, Win Server 1%, n/a 1%
- countryname_normalized: United States of America 22%, India 10%, China 7%, Germany 5%, United Kingdom of Great Britain and Northern Ireland 4%, Brazil 4%, Japan 3%, France 3%, Korea Republic of 2%, Italy 2%, Canada 2%, Australia 2%, Mexico 2%, Russian Federation 2%, Poland 1.5%, Netherlands 1.5%, Spain 1.5%, Turkey 1.5%, Indonesia 1%, Thailand 1%, Taiwan Province of China 1%, Sweden 1%, Switzerland 1%, Colombia 1%, other countries ~15% (spread across remaining countries in the enum)
- persona: Casual User 27%, Communication 16%, Casual Gamer 16%, Office/Productivity 13%, Web User 8%, Entertainment 6%, Content Creator/IT 5%, Win Store App User 4%, Gamer 2%, File & Network Sharer 2%, Unknown 1%
- modelvendor_normalized: Lenovo 22%, Dell Inc. 20%, HP 18%, ASUSTeK COMPUTER INC. 8%, Acer 7%, Microsoft Corporation 3%, Samsung 3%, HUAWEI 2%, MSI 2%, others smaller
- ram: 8 (38%), 16 (28%), 4 (18%), 32 (7%), 12 (4%), 6 (2%), 64 (1%)

NUMERIC RANGES (for nonzero values only):
- net_received_bytes: 1e8 to 1e14 (median ~1e11), net_sent_bytes: 1e7 to 1e13 (median ~1e10), net_nrs: 100-50000
- mem_avg_pct_used: 20-95 (median 55), mem_nrs: 100-50000, mem_sysinfo_ram: match ram in MB (e.g. ram=8 -> 8192)
- batt_num_power_ons: 1-20 (median 3), batt_duration_mins: 10-600 (median 130)
- web_chrome_duration/web_edge_duration/web_firefox_duration: 1e3 to 1e8 (ms)
- webcat_* fields: 0.0-100.0 (percentage of browsing time; active webcat fields should sum to roughly 100)
- onoff_on_time/off_time/mods_time/sleep_time: 0-1e7 (seconds)
- disp_num_displays: 1-4, disp_total_duration_ac/dc: 1e3 to 1e8

*** CRITICAL SPARSITY RULES (MOST IMPORTANT) ***
There are 8 numeric groups: net, mem, batt, browser+webcat, onoff, disp, hw.
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


def _build_schema_description(real_df: pd.DataFrame) -> str:
    """Build schema description with real distribution statistics.

    Uses the static distribution prompt with real-world frequencies rather than
    dynamically computing top-10 values (which lost long-tail coverage).
    The group sparsity is still computed from the real data for accuracy.
    """
    group_sparsity = _compute_group_sparsity(real_df)
    sparsity_info = ", ".join(
        f"{g}: {pct}%" for g, pct in sorted(group_sparsity.items())
    )
    return f"{_DISTRIBUTION_PROMPT}\nReal group sparsity: {sparsity_info}"


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
        model: str = "gpt-4.1-mini",
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
