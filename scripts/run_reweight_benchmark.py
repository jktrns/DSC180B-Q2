"""Post-hoc reweighting benchmark for PE synthetic data.

Loads PE synthetic records, solves for optimal per-record weights so that
weighted aggregates match real query results, creates a reweighted dataset,
and re-runs the full benchmark pipeline to compare scores.

Usage:
    python scripts/run_reweight_benchmark.py
    python scripts/run_reweight_benchmark.py --n-records 5000 --lambda-reg 0.05
    python scripts/run_reweight_benchmark.py --skip-optimize  # re-run eval only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.eval.benchmark import run_benchmark
from src.eval.compare import (
    QUERY_METADATA,
    detailed_results_to_dataframe,
    evaluate_all,
    results_to_dataframe,
)
from src.eval.decompose import decompose_wide_table
from src.pe.api import _NUMERIC_GROUPS
from src.pe.reweight import (
    compute_all_weighted_queries,
    create_reweighted_table,
    prepare_wide_table,
    solve_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_synth_wide(source_path: Path, n_records: int) -> pd.DataFrame:
    """Load synthetic wide table, sampling n_records if needed."""
    df = pd.read_parquet(source_path)
    logger.info("Loaded %d records from %s", len(df), source_path)

    if len(df) > n_records:
        df = df.iloc[:n_records].reset_index(drop=True)
        logger.info("Using first %d records", n_records)

    # Ensure guid column exists
    if "guid" not in df.columns:
        df.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(df))])

    return df


def run_baseline_benchmark(
    synth_wide: pd.DataFrame,
    output_dir: Path,
    queries_dir: Path,
    real_results_dir: Path,
) -> pd.DataFrame:
    """Run the standard (unweighted) benchmark and return evaluation df."""
    reporting_dir = output_dir / "baseline" / "reporting"
    results_dir = output_dir / "baseline" / "query_results"

    print("\n=== BASELINE: Decomposing ===")
    counts = decompose_wide_table(synth_wide, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    print("\n=== BASELINE: Running queries ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(query_names, queries_dir, reporting_dir, results_dir)
    print(f"  {len(results)}/{len(query_names)} queries succeeded")

    print("\n=== BASELINE: Evaluating ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "baseline_evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(output_dir / "baseline_evaluation_detail.csv", index=False)

    return eval_df


def run_reweighted_benchmark(
    reweighted_wide: pd.DataFrame,
    output_dir: Path,
    queries_dir: Path,
    real_results_dir: Path,
) -> pd.DataFrame:
    """Run benchmark on the reweighted dataset and return evaluation df."""
    reporting_dir = output_dir / "reweighted" / "reporting"
    results_dir = output_dir / "reweighted" / "query_results"

    print("\n=== REWEIGHTED: Decomposing ===")
    counts = decompose_wide_table(reweighted_wide, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    print("\n=== REWEIGHTED: Running queries ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(query_names, queries_dir, reporting_dir, results_dir)
    print(f"  {len(results)}/{len(query_names)} queries succeeded")

    print("\n=== REWEIGHTED: Evaluating ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "reweighted_evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(
        output_dir / "reweighted_evaluation_detail.csv", index=False)

    return eval_df


def print_comparison(baseline_df: pd.DataFrame, reweighted_df: pd.DataFrame):
    """Print a side-by-side comparison of baseline vs reweighted scores."""
    print("\n" + "=" * 78)
    print("COMPARISON: BASELINE vs REWEIGHTED")
    print("=" * 78)
    print(f"{'Query':<55} {'Base':>6} {'Rewt':>6} {'Δ':>6}")
    print("-" * 78)

    for _, base_row in baseline_df.iterrows():
        query = base_row["query"]
        rw_row = reweighted_df[reweighted_df["query"] == query]

        if base_row["error"]:
            print(f"  {query:<53} {'SKIP':>6} {'SKIP':>6}")
            continue

        base_score = base_row["score"]
        if len(rw_row) > 0 and not rw_row.iloc[0]["error"]:
            rw_score = rw_row.iloc[0]["score"]
            delta = rw_score - base_score
            delta_str = f"{delta:+.3f}"
            base_str = f"{base_score:.3f}"
            rw_str = f"{rw_score:.3f}"
        else:
            rw_str = "SKIP"
            delta_str = ""
            base_str = f"{base_score:.3f}"

        print(f"  {query:<53} {base_str:>6} {rw_str:>6} {delta_str:>6}")

    # Summary
    base_valid = baseline_df[baseline_df["error"] == ""]
    rw_valid = reweighted_df[reweighted_df["error"] == ""]

    base_passed = base_valid[base_valid["passed"]]
    rw_passed = rw_valid[rw_valid["passed"]]

    print("-" * 78)
    print(f"  {'Queries passed (≥0.5):':<53} "
          f"{len(base_passed):>3}/{len(base_valid):<2} "
          f"{len(rw_passed):>3}/{len(rw_valid):<2}")

    if len(base_valid) > 0 and len(rw_valid) > 0:
        base_avg = base_valid["score"].mean()
        rw_avg = rw_valid["score"].mean()
        print(f"  {'Average score:':<53} "
              f"{base_avg:.3f}  {rw_avg:.3f}  {rw_avg - base_avg:+.3f}")
        base_med = base_valid["score"].median()
        rw_med = rw_valid["score"].median()
        print(f"  {'Median score:':<53} "
              f"{base_med:.3f}  {rw_med:.3f}  {rw_med - base_med:+.3f}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc reweighting benchmark for PE synthetic data")
    parser.add_argument(
        "--source", type=str,
        default="data/reporting/pe_wide_table.parquet",
        help="Path to synthetic wide table parquet")
    parser.add_argument("--n-records", type=int, default=50000)
    parser.add_argument("--lambda-reg", type=float, default=0.001,
                        help="Regularization strength")
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--replicate-scale", type=int, default=10,
                        help="Scale factor for row replication")
    parser.add_argument("--output-dir", type=str,
                        default="data/results/pe_reweighted")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip optimization, use saved weights")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = Path("docs/queries")
    real_results_dir = Path("data/results/real")

    # ---- Step 1: Load synthetic data ----
    print("=== LOADING SYNTHETIC DATA ===")
    synth_wide = load_synth_wide(Path(args.source), args.n_records)
    print(f"  Shape: {synth_wide.shape}")

    # Sparsity check
    print("\n=== SPARSITY CHECK ===")
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in synth_wide.columns]
        if present:
            nonzero = (synth_wide[present].abs().sum(axis=1) > 0).sum()
            print(f"  {gname}: {nonzero}/{len(synth_wide)} nonzero "
                  f"({100 * nonzero / len(synth_wide):.1f}%)")

    # ---- Step 2: Run baseline benchmark ----
    baseline_df = run_baseline_benchmark(
        synth_wide, output_dir, queries_dir, real_results_dir)

    # ---- Step 3: Solve for optimal weights ----
    weights_path = output_dir / "weights.npy"

    if args.skip_optimize and weights_path.exists():
        print("\n=== LOADING SAVED WEIGHTS ===")
        weights = np.load(weights_path)
        print(f"  Loaded weights: shape={weights.shape}")
    else:
        print("\n=== SOLVING FOR OPTIMAL WEIGHTS ===")
        prepared = prepare_wide_table(synth_wide)
        t0 = time.time()
        weights = solve_weights(
            prepared,
            real_results_dir,
            lambda_reg=args.lambda_reg,
            max_weight=args.max_weight,
        )
        elapsed = time.time() - t0
        print(f"  Optimization completed in {elapsed:.1f}s")
        np.save(weights_path, weights)
        print(f"  Saved weights to {weights_path}")

    # Weight statistics
    print("\n=== WEIGHT STATISTICS ===")
    print(f"  Min:      {weights.min():.4f}")
    print(f"  Median:   {np.median(weights):.4f}")
    print(f"  Mean:     {weights.mean():.4f}")
    print(f"  Max:      {weights.max():.4f}")
    print(f"  Std:      {weights.std():.4f}")
    print(f"  Nonzero:  {(weights > 1e-6).sum()}/{len(weights)}")
    print(f"  Sum:      {weights.sum():.1f}")

    # Weight distribution
    bins = [0, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
    hist, _ = np.histogram(weights, bins=bins)
    print("  Distribution:")
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        hi_str = f"{hi:.1f}" if hi < float("inf") else "∞"
        print(f"    [{lo:.2f}, {hi_str}): {hist[i]}")

    # ---- Step 4: Compute direct weighted query results (exact) ----
    print("\n=== DIRECT WEIGHTED QUERY RESULTS (EXACT) ===")
    direct_dir = output_dir / "direct_weighted"
    direct_results = compute_all_weighted_queries(
        prepared, weights, direct_dir)
    print(f"  Produced {len(direct_results)} query result files")

    # Evaluate direct weighted results
    print("\n=== EVALUATING DIRECT WEIGHTED RESULTS ===")
    direct_eval = evaluate_all(real_results_dir, direct_dir)
    direct_df = results_to_dataframe(direct_eval)
    direct_df.to_csv(output_dir / "direct_evaluation.csv", index=False)
    direct_detail = detailed_results_to_dataframe(direct_eval)
    direct_detail.to_csv(
        output_dir / "direct_evaluation_detail.csv", index=False)

    # ---- Step 5: Create reweighted dataset (row replication) ----
    print("\n=== CREATING REWEIGHTED DATASET (ROW REPLICATION) ===")
    reweighted_wide = create_reweighted_table(
        synth_wide, weights, scale=args.replicate_scale)
    print(f"  Reweighted shape: {reweighted_wide.shape}")

    # Save reweighted wide table
    reweighted_wide.to_parquet(
        output_dir / "reweighted_wide.parquet", index=False)

    # ---- Step 6: Run reweighted benchmark (standard pipeline) ----
    reweighted_df = run_reweighted_benchmark(
        reweighted_wide, output_dir, queries_dir, real_results_dir)

    # ---- Step 7: Compare all three approaches ----
    print("\n" + "=" * 90)
    print("COMPARISON: BASELINE vs DIRECT-WEIGHTED vs ROW-REPLICATION")
    print("=" * 90)
    print(f"{'Query':<50} {'Base':>6} {'Direct':>7} {'Replic':>7}")
    print("-" * 90)

    for _, base_row in baseline_df.iterrows():
        query = base_row["query"]
        if base_row["error"]:
            print(f"  {query:<48} {'ERR':>6} {'--':>7} {'--':>7}")
            continue

        bs = base_row["score"]
        dr = direct_df[direct_df["query"] == query]
        rr = reweighted_df[reweighted_df["query"] == query]

        ds = dr.iloc[0]["score"] if len(dr) > 0 and not dr.iloc[0]["error"] else None
        rs = rr.iloc[0]["score"] if len(rr) > 0 and not rr.iloc[0]["error"] else None

        bs_s = f"{bs:.3f}"
        ds_s = f"{ds:.3f}" if ds is not None else "--"
        rs_s = f"{rs:.3f}" if rs is not None else "--"
        print(f"  {query:<48} {bs_s:>6} {ds_s:>7} {rs_s:>7}")

    # Summary
    base_valid = baseline_df[baseline_df["error"] == ""]
    direct_valid = direct_df[direct_df["error"] == ""]
    rw_valid = reweighted_df[reweighted_df["error"] == ""]

    base_passed = base_valid[base_valid["passed"]]
    direct_passed = direct_valid[direct_valid["passed"]]
    rw_passed = rw_valid[rw_valid["passed"]]

    print("-" * 90)
    print(f"  {'Passed (>=0.5):':<48} "
          f"{len(base_passed):>3}/{len(base_valid):<3} "
          f"{len(direct_passed):>4}/{len(direct_valid):<3} "
          f"{len(rw_passed):>4}/{len(rw_valid):<3}")
    if len(base_valid) > 0:
        ba = base_valid["score"].mean()
        da = direct_valid["score"].mean() if len(direct_valid) > 0 else 0
        ra = rw_valid["score"].mean() if len(rw_valid) > 0 else 0
        print(f"  {'Average score:':<48} {ba:>6.3f} {da:>7.3f} {ra:>7.3f}")
    print("=" * 90)

    # Also print the old-style comparison
    print_comparison(baseline_df, reweighted_df)

    # Save comparison
    comparison = baseline_df[["query", "type", "score", "passed", "error"]].copy()
    comparison = comparison.rename(columns={
        "score": "baseline_score", "passed": "baseline_passed"})
    rw_scores = reweighted_df[["query", "score", "passed"]].copy()
    rw_scores = rw_scores.rename(columns={
        "score": "reweighted_score", "passed": "reweighted_passed"})
    comparison = comparison.merge(rw_scores, on="query", how="left")
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
