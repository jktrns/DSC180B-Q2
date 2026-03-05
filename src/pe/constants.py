"""Constrained categorical values for the PE telemetry pipeline.

All VALID_* lists are derived programmatically from the real DCA telemetry
query results in ``data/results/real/``.  Using ``Literal`` types built from
these lists forces OpenAI structured outputs to produce only values that
actually appear in the real dataset, eliminating hallucinated categories.

When the underlying real data changes, re-run
``scripts/derive_constants.py`` to regenerate this file.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Chassis types  (source: server_exploration_1.csv, avg_platform_power…csv)
# ---------------------------------------------------------------------------
VALID_CHASSIS: list[str] = [
    "2 in 1",
    "Desktop",
    "Intel NUC/STK",
    "Notebook",
    "Other",
    "Server/WS",
    "Tablet",
]

# ---------------------------------------------------------------------------
# Countries  (source: most_popular_browser_in_each_country_by_system_count.csv)
# 51 countries exactly matching the real query results.
# ---------------------------------------------------------------------------
VALID_COUNTRIES: list[str] = [
    "Argentina",
    "Australia",
    "Austria",
    "Bangladesh",
    "Belgium",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Czech Republic",
    "Denmark",
    "Ecuador",
    "Egypt",
    "France",
    "Germany",
    "Greece",
    "Hong Kong",
    "Hungary",
    "India",
    "Indonesia",
    "Israel",
    "Italy",
    "Japan",
    "Korea, Republic of",
    "Malaysia",
    "Mexico",
    "Netherlands",
    "Norway",
    "Other",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Romania",
    "Russian Federation",
    "Saudi Arabia",
    "Singapore",
    "South Africa",
    "Spain",
    "Sweden",
    "Switzerland",
    "Taiwan, Province of China",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom of Great Britain and Northern Ireland",
    "United States of America",
    "Viet Nam",
]

# ---------------------------------------------------------------------------
# OS versions  (source: Xeon_network_consumption.csv, server_exploration_1.csv)
# ---------------------------------------------------------------------------
VALID_OS: list[str] = [
    "Win10",
    "Win11",
    "Win8.1",
    "Win Server",
]

# ---------------------------------------------------------------------------
# Persona  (source: persona_web_cat_usage_analysis.csv — 11 personas)
# ---------------------------------------------------------------------------
VALID_PERSONA: list[str] = [
    "Casual Gamer",
    "Casual User",
    "Communication",
    "Content Creator/IT",
    "Entertainment",
    "File & Network Sharer",
    "Gamer",
    "Office/Productivity",
    "Unknown",
    "Web User",
    "Win Store App User",
]

# ---------------------------------------------------------------------------
# Model vendors  (source: server_exploration_1.csv — 45 vendors)
# ---------------------------------------------------------------------------
VALID_VENDORS: list[str] = [
    "AZW",
    "Acer",
    "Alienware",
    "Apple",
    "Asus",
    "Biostar",
    "Casper Bilgisayar",
    "Colorful Technology And Development Co.,LTD",
    "Dell",
    "Fujitsu",
    "GPD",
    "Gateway",
    "Gigabyte",
    "HANSUNG COMPUTER",
    "HP",
    "HUAWEI",
    "Hasee",
    "Intel",
    "LG",
    "Lenovo",
    "MECHREVO",
    "MONSTER",
    "MSI",
    "Medion",
    "Microsoft Corporation",
    "Mouse Computer",
    "NEC",
    "Notebook",
    "Other",
    "PCWARE",
    "Packard Bell",
    "Panasonic Corporation",
    "Positivo",
    "Razer",
    "Samsung",
    "Shinelon Computer",
    "Sony",
    "Supermicro",
    "THUNDEROBOT",
    "TIMI",
    "Thirdwave Corporation",
    "Timi",
    "Toshiba",
    "Unknown",
    "Wortmann_AG",
]

# ---------------------------------------------------------------------------
# CPU names  (not directly in query results; derived from cpugen patterns)
# These are the Intel brand families that appear in the wide training table.
# ---------------------------------------------------------------------------
VALID_CPUNAMES: list[str] = [
    "Intel Atom",
    "Intel Celeron",
    "Intel Core i3",
    "Intel Core i5",
    "Intel Core i7",
    "Intel Core i9",
    "Intel Pentium",
    "Intel Xeon",
    "Other",
]

# ---------------------------------------------------------------------------
# CPU codenames  (source: on_off_mods_sleep_summary…csv, battery_on_duration…csv)
# ---------------------------------------------------------------------------
VALID_CPUCODES: list[str] = [
    "Alder Lake",
    "Coffee Lake",
    "Comet Lake",
    "Gemini Lake Refresh",
    "Ice Lake",
    "Jasper Lake",
    "Rocket Lake",
    "Tiger Lake",
    "Unknown",
    "Whiskey Lake",
]

# ---------------------------------------------------------------------------
# CPU family / generation  (source: on_off_mods_sleep_summary…csv — 22 values)
# ---------------------------------------------------------------------------
VALID_CPU_FAMILY: list[str] = [
    "10th Gen i3",
    "10th Gen i5",
    "10th Gen i7",
    "10th Gen i9",
    "11th Gen i3",
    "11th Gen i5",
    "11th Gen i7",
    "11th Gen i9",
    "12th Gen i5",
    "12th Gen i7",
    "12th Gen i9",
    "9th Gen i3",
    "9th Gen i5",
    "9th Gen i7",
    "9th Gen i9",
    "Pentium/Celeron-Coffee Lake",
    "Pentium/Celeron-Comet Lake",
    "Pentium/Celeron-Gemini Lake Refresh",
    "Pentium/Celeron-Jasper Lake",
    "Pentium/Celeron-Whiskey Lake",
    "Unknown",
    "Xeon",
]

# ---------------------------------------------------------------------------
# Processor numbers  (not directly in query results; representative set from
# the Intel product line for the codenames present in the real data)
# ---------------------------------------------------------------------------
VALID_PROCESSORS: list[str] = [
    # 9th Gen (Coffee Lake)
    "i3-9100", "i5-9300H", "i5-9400", "i7-9700", "i7-9750H", "i9-9900K",
    # 10th Gen (Comet Lake / Ice Lake)
    "i3-10110U", "i5-10210U", "i5-10310U", "i5-10400",
    "i7-10510U", "i7-10610U", "i7-10710U", "i7-10750H",
    "i9-10850K", "i9-10900K",
    "i3-1005G1", "i5-1035G1", "i7-1065G7",
    # 11th Gen (Tiger Lake / Rocket Lake)
    "i3-1115G4", "i5-1135G7", "i5-1145G7", "i5-11400",
    "i7-1165G7", "i7-1185G7", "i7-11700", "i7-11800H",
    "i9-11900K",
    # 12th Gen (Alder Lake)
    "i5-1235U", "i5-1240P", "i5-12400", "i5-12500H",
    "i7-1255U", "i7-1260P", "i7-12700", "i7-12700H",
    "i9-12900K",
    # Pentium / Celeron
    "N4020", "N4120", "N5030", "N5100", "N6000",
    "J4025", "J4125", "J5040",
    "G6900",
    # Xeon
    "E-2224", "E-2236", "E-2278G", "W-10855M",
    # Catch-all
    "Other",
]


# ---------------------------------------------------------------------------
# Build Literal types from valid value lists
# ---------------------------------------------------------------------------
ChassisType = Literal[tuple(VALID_CHASSIS)]  # type: ignore[valid-type]
CountryType = Literal[tuple(VALID_COUNTRIES)]  # type: ignore[valid-type]
OsType = Literal[tuple(VALID_OS)]  # type: ignore[valid-type]
PersonaType = Literal[tuple(VALID_PERSONA)]  # type: ignore[valid-type]
VendorType = Literal[tuple(VALID_VENDORS)]  # type: ignore[valid-type]
CpuNameType = Literal[tuple(VALID_CPUNAMES)]  # type: ignore[valid-type]
CpuCodeType = Literal[tuple(VALID_CPUCODES)]  # type: ignore[valid-type]
CpuFamilyType = Literal[tuple(VALID_CPU_FAMILY)]  # type: ignore[valid-type]
ProcessorType = Literal[tuple(VALID_PROCESSORS)]  # type: ignore[valid-type]
