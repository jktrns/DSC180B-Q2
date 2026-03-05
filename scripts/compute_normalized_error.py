"""
Compute the professor's normalized error metric:

    normalized_error_j = Delta_j(D, D_tilde) / |Delta_j(D1, D2)|

where D1, D2 are random 50/50 splits of the real dataset D,
and D_tilde is a synthetic dataset from one of the methods.

This measures how much distortion the synthetic data introduces
relative to the natural sampling variation from halving the dataset.

Steps:
  1. Collect all GUIDs from the anchor table (system_sysinfo_unique_normalized)
  2. Randomly split GUIDs 50/50 into D1 and D2
  3. For each reporting table, create DuckDB views filtered by GUID split
  4. Run SQL benchmark queries on D1 and D2
  5. Evaluate D1 vs D2 to get baseline variation Delta_j(D1, D2)
  6. Evaluate real vs each synthetic method to get Delta_j(D, D_tilde)
  7. Compute normalized error ratio
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.compare import (
    QUERY_METADATA,
    evaluate_query,
    detailed_results_to_dataframe,
    results_to_dataframe,
)

REPORTING_DIR = PROJECT_ROOT / "data" / "reporting"
QUERIES_DIR = PROJECT_ROOT / "docs" / "queries"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = RESULTS_DIR / "normalized_error"

SEED = 42

# All reporting tables that queries reference
REPORTING_TABLES = [
    "system_sysinfo_unique_normalized",
    "system_cpu_metadata",
    "system_network_consumption",
    "system_memory_utilization",
    "system_batt_dc_events",
    "system_hw_pkg_power",
    "system_display_devices",
    "system_frgnd_apps_types",
    "system_mods_power_consumption",
    "system_mods_top_blocker_hist",
    "system_network_consumption",
    "system_on_off_suspend_time_day",
    "system_os_codename_history",
    "system_pkg_C0",
    "system_pkg_avg_freq_mhz",
    "system_pkg_temp_centigrade",
    "system_psys_rap_watts",
    "system_userwait",
    "system_web_cat_pivot_duration",
    "system_web_cat_usage",
]

SYNTH_METHODS = {
    "widetable": RESULTS_DIR / "synthetic",
    "pertable": RESULTS_DIR / "synth_pertable",
    "mst": RESULTS_DIR / "synth_mst",
    "pe": RESULTS_DIR / "synth_pe",
}


def get_all_guids(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    """Get all unique GUIDs from the anchor table."""
    path = REPORTING_DIR / "system_sysinfo_unique_normalized.parquet"
    result = con.execute(
        f"SELECT DISTINCT guid FROM read_parquet('{path}')"
    ).fetchall()
    guids = np.array([r[0] for r in result])
    print(f"Total unique GUIDs: {len(guids)}")
    return guids


def split_guids(guids: np.ndarray, seed: int = SEED) -> tuple[set[str], set[str]]:
    """Randomly split GUIDs into two equal halves."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(guids)
    mid = len(shuffled) // 2
    d1_guids = set(shuffled[:mid])
    d2_guids = set(shuffled[mid:])
    print(f"D1: {len(d1_guids)} GUIDs, D2: {len(d2_guids)} GUIDs")
    return d1_guids, d2_guids


def create_split_views(
    con: duckdb.DuckDBPyConnection,
    guid_table_name: str,
    split_prefix: str,
) -> None:
    """
    Create DuckDB views for each reporting table, filtered by GUIDs
    in the specified guid table.
    """
    for table_name in set(REPORTING_TABLES):
        parquet_path = REPORTING_DIR / f"{table_name}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping")
            continue

        view_name = f"{split_prefix}_{table_name}"
        # All reporting tables have a 'guid' column
        con.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT t.*
            FROM read_parquet('{parquet_path}') t
            WHERE t.guid IN (SELECT guid FROM {guid_table_name})
        """)


def adapt_sql_for_split(sql: str, split_prefix: str) -> str:
    """Adapt SQL to use split views instead of parquet files."""
    def replacer(match: re.Match) -> str:
        table = match.group(1)
        return f"{split_prefix}_{table}"
    return re.sub(r"reporting\.(\w+)", replacer, sql)


def run_benchmark_on_split(
    con: duckdb.DuckDBPyConnection,
    split_prefix: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run all benchmark queries using the split views."""
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for qname in sorted(QUERY_METADATA.keys()):
        qfile = QUERIES_DIR / f"{qname}.json"
        if not qfile.exists():
            print(f"  Query {qname}: file not found")
            continue

        with open(qfile) as f:
            data = json.load(f)
            if isinstance(data, list):
                data = data[0]

        sql = data["sql"]
        adapted = adapt_sql_for_split(sql, split_prefix)

        try:
            df = con.execute(adapted).df()
            results[qname] = df
            df.to_csv(output_dir / f"{qname}.csv", index=False)
        except Exception as e:
            print(f"  Query {qname} failed: {e}")

    return results


def compute_metric_values(detail_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """
    Extract per-query, per-column, per-metric_type raw values from detail DataFrame.
    Returns dict of (query, column, metric_type) -> value.
    """
    values = {}
    for _, row in detail_df.iterrows():
        key = (row["query"], row["column"], row["metric_type"])
        values[key] = row["value"]
    return values


def compute_normalized_errors(
    baseline_values: dict[tuple[str, str, str], float],
    method_values: dict[tuple[str, str, str], float],
) -> list[dict]:
    """
    Compute normalized error for each metric:
    normalized = method_value / |baseline_value|

    For metrics where lower is better (RE, TV): both values are distances
    For metrics where higher is better (jaccard, spearman, accuracy):
      we compute the "error" as (1 - value) so that it becomes a distance metric.
    """
    HIGHER_IS_BETTER = {
        "jaccard", "mean_jaccard", "spearman_rho", "mean_spearman_rho",
        "categorical_accuracy",
    }

    rows = []
    for key in sorted(set(baseline_values.keys()) | set(method_values.keys())):
        query, column, metric_type = key

        baseline_val = baseline_values.get(key)
        method_val = method_values.get(key)

        if baseline_val is None or method_val is None:
            continue
        if not np.isfinite(baseline_val) or not np.isfinite(method_val):
            continue

        # Convert to "error" (distance) for higher-is-better metrics
        if metric_type in HIGHER_IS_BETTER:
            baseline_err = 1.0 - baseline_val
            method_err = 1.0 - method_val
        else:
            baseline_err = abs(baseline_val)
            method_err = abs(method_val)

        # Compute normalized error
        if abs(baseline_err) < 1e-10:
            # Baseline variation is ~0, can't normalize meaningfully
            # If method error is also ~0, that's perfect (normalized = 0)
            # If method error is large, that's bad (normalized = inf)
            if method_err < 1e-10:
                normalized = 0.0
            else:
                normalized = float("inf")
        else:
            normalized = method_err / abs(baseline_err)

        rows.append({
            "query": query,
            "column": column,
            "metric_type": metric_type,
            "method_value": method_val,
            "baseline_value": baseline_val,
            "method_error": method_err,
            "baseline_error": baseline_err,
            "normalized_error": normalized,
        })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # Increase memory limit for large parquet operations
    con.execute("SET memory_limit='8GB'")

    # Step 1: Get all GUIDs
    print("=" * 60)
    print("Step 1: Collecting GUIDs from anchor table...")
    guids = get_all_guids(con)

    # Step 2: Split GUIDs
    print("\nStep 2: Splitting GUIDs 50/50...")
    d1_guids, d2_guids = split_guids(guids)

    # Store GUID sets in DuckDB temp tables for efficient filtering
    print("\nStep 3: Creating GUID temp tables...")
    d1_df = pd.DataFrame({"guid": list(d1_guids)})
    d2_df = pd.DataFrame({"guid": list(d2_guids)})
    con.execute("CREATE TABLE d1_guids AS SELECT * FROM d1_df")
    con.execute("CREATE TABLE d2_guids AS SELECT * FROM d2_df")

    # Step 3: Create filtered views
    print("\nStep 4: Creating filtered views for D1...")
    create_split_views(con, "d1_guids", "d1")
    print("Creating filtered views for D2...")
    create_split_views(con, "d2_guids", "d2")

    # Step 4: Run benchmark on each split
    d1_output = OUTPUT_DIR / "d1"
    d2_output = OUTPUT_DIR / "d2"

    print("\nStep 5: Running benchmark queries on D1...")
    d1_results = run_benchmark_on_split(con, "d1", d1_output)
    print(f"  D1: {len(d1_results)} queries completed")

    print("\nStep 6: Running benchmark queries on D2...")
    d2_results = run_benchmark_on_split(con, "d2", d2_output)
    print(f"  D2: {len(d2_results)} queries completed")

    # Step 5: Evaluate D1 vs D2 (baseline variation)
    print("\nStep 7: Evaluating D1 vs D2 (baseline variation)...")
    baseline_results = []
    for qname in sorted(QUERY_METADATA.keys()):
        d1_path = d1_output / f"{qname}.csv"
        d2_path = d2_output / f"{qname}.csv"
        if not d1_path.exists() or not d2_path.exists():
            print(f"  Skipping {qname}: missing split results")
            continue
        d1_df = pd.read_csv(d1_path)
        d2_df = pd.read_csv(d2_path)
        result = evaluate_query(qname, d1_df, d2_df)
        baseline_results.append(result)

    baseline_summary = results_to_dataframe(baseline_results)
    baseline_detail = detailed_results_to_dataframe(baseline_results)
    baseline_summary.to_csv(OUTPUT_DIR / "baseline_d1_vs_d2_summary.csv", index=False)
    baseline_detail.to_csv(OUTPUT_DIR / "baseline_d1_vs_d2_detail.csv", index=False)
    print(f"  Baseline: {len(baseline_results)} queries evaluated")

    # Step 6: Evaluate real vs each synthetic method
    print("\nStep 8: Evaluating real vs synthetic for each method...")
    method_details = {}
    for method_name, synth_dir in SYNTH_METHODS.items():
        print(f"\n  --- {method_name} ---")
        method_results = []
        for qname in sorted(QUERY_METADATA.keys()):
            real_path = RESULTS_DIR / "real" / f"{qname}.csv"
            synth_path = synth_dir / f"{qname}.csv"
            if not real_path.exists():
                continue
            if not synth_path.exists():
                print(f"    Skipping {qname}: no synth CSV")
                continue
            real_df = pd.read_csv(real_path)
            synth_df = pd.read_csv(synth_path)
            result = evaluate_query(qname, real_df, synth_df)
            method_results.append(result)

        method_summary = results_to_dataframe(method_results)
        method_detail = detailed_results_to_dataframe(method_results)
        method_summary.to_csv(OUTPUT_DIR / f"method_{method_name}_summary.csv", index=False)
        method_detail.to_csv(OUTPUT_DIR / f"method_{method_name}_detail.csv", index=False)
        method_details[method_name] = method_detail
        print(f"    {method_name}: {len(method_results)} queries evaluated")

    # Step 7: Compute normalized errors
    print("\nStep 9: Computing normalized errors...")
    baseline_vals = compute_metric_values(baseline_detail)

    all_normalized = []
    for method_name, detail_df in method_details.items():
        method_vals = compute_metric_values(detail_df)
        norm_rows = compute_normalized_errors(baseline_vals, method_vals)
        for row in norm_rows:
            row["method"] = method_name
        all_normalized.extend(norm_rows)

    norm_df = pd.DataFrame(all_normalized)
    norm_df.to_csv(OUTPUT_DIR / "normalized_errors.csv", index=False)

    # Step 8: Create summary table
    print("\n" + "=" * 60)
    print("NORMALIZED ERROR SUMMARY")
    print("=" * 60)

    if len(norm_df) > 0:
        # Filter out infinities for summary stats
        finite_df = norm_df[np.isfinite(norm_df["normalized_error"])]

        # Per-method summary
        print("\n--- Per-method mean normalized error ---")
        method_summary = finite_df.groupby("method")["normalized_error"].agg(
            ["mean", "median", "std", "count"]
        )
        print(method_summary.to_string())
        method_summary.to_csv(OUTPUT_DIR / "normalized_error_per_method.csv")

        # Per-method, per-query-type summary
        print("\n--- Per-method, per-query-type mean normalized error ---")
        # Map queries to types
        query_types = {q: m["type"] for q, m in QUERY_METADATA.items()}
        finite_df = finite_df.copy()
        finite_df["query_type"] = finite_df["query"].map(query_types)
        pivot = finite_df.groupby(["method", "query_type"])["normalized_error"].agg(
            ["mean", "median", "count"]
        )
        print(pivot.to_string())
        pivot.to_csv(OUTPUT_DIR / "normalized_error_by_method_querytype.csv")

        # Per-method, per-query summary (for the LaTeX table)
        print("\n--- Per-method, per-query mean normalized error ---")
        query_method = finite_df.groupby(["query", "method"])["normalized_error"].mean().unstack()
        print(query_method.to_string())
        query_method.to_csv(OUTPUT_DIR / "normalized_error_per_query_method.csv")

        # Overall comparison
        print("\n--- Overall method ranking (lower is better) ---")
        ranking = finite_df.groupby("method")["normalized_error"].median().sort_values()
        for method, val in ranking.items():
            print(f"  {method}: median normalized error = {val:.4f}")

    print(f"\nAll results saved to {OUTPUT_DIR}")
    con.close()


if __name__ == "__main__":
    main()
