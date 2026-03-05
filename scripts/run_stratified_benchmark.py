"""Stratified PE benchmark: conditional generation from real query targets.

Instead of generating records independently, this script:
1. Reads real query results to compute per-stratum targets (with DP noise)
2. Generates records conditioned on those noisy targets
3. Runs DP histogram selection on the generated pool
4. Decomposes into reporting tables, runs queries, evaluates

Privacy budget:  epsilon_agg (aggregate noise) + epsilon_hist (histogram)
                 composed via basic composition.

Usage:
    python scripts/run_stratified_benchmark.py --n-records 5000
    python scripts/run_stratified_benchmark.py --skip-generation  # re-run eval only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval.benchmark import run_benchmark
from src.eval.compare import (
    QUERY_METADATA,
    detailed_results_to_dataframe,
    evaluate_all,
    results_to_dataframe,
)
from src.eval.decompose import decompose_wide_table
from src.pe.api import PEApi, _NUMERIC_GROUPS
from src.pe.distance import WorkloadDistance
from src.pe.histogram import dp_nn_histogram, select_candidates
from src.pe.privacy import calibrate_sigma, compute_epsilon
from src.pe.stratified import build_generation_plan


def main():
    parser = argparse.ArgumentParser(
        description="Stratified PE benchmark (conditional generation + DP)"
    )
    parser.add_argument("--n-records", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument(
        "--output-dir", type=str, default="data/results/pe_stratified"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, re-run decompose+benchmark+eval only",
    )
    # Privacy parameters
    parser.add_argument(
        "--epsilon-agg", type=float, default=1.0,
        help="Privacy budget for DP-noised aggregates (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-hist", type=float, default=3.0,
        help="Privacy budget for DP histogram selection (default: 3.0)",
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5,
        help="Delta parameter for (epsilon, delta)-DP (default: 1e-5)",
    )
    parser.add_argument(
        "--pool-multiplier", type=int, default=3,
        help="Generate pool_multiplier * n_records candidates, then select "
             "n_records via DP histogram (default: 3)",
    )
    parser.add_argument(
        "--real-subsample", type=int, default=None,
        help="Subsample real records for DP histogram voting (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporting_dir = output_dir / "reporting"
    results_dir = output_dir / "query_results"
    queries_dir = Path("docs/queries")
    real_results_dir = Path("data/results/real")

    total_epsilon = args.epsilon_agg + args.epsilon_hist
    print(f"Privacy budget: epsilon_agg={args.epsilon_agg} + "
          f"epsilon_hist={args.epsilon_hist} = {total_epsilon} total, "
          f"delta={args.delta}")

    # ---- Step 1: Build generation plan (with DP-noised aggregates) ------
    print("\n=== BUILDING STRATIFIED GENERATION PLAN (DP-noised) ===")
    plan, privacy_info = build_generation_plan(
        real_results_dir,
        n_total=args.n_records,
        epsilon_agg=args.epsilon_agg,
        delta=args.delta,
    )
    print(plan.summary())

    # ---- Step 2: Generate synthetic data ---------------------------------
    if args.skip_generation:
        print("\n=== LOADING EXISTING WIDE TABLE ===")
        synth_df = pd.read_parquet(output_dir / "synth_wide.parquet")
        print(f"Loaded: {synth_df.shape}")
    else:
        print("\n=== GENERATING SYNTHETIC DATA (STRATIFIED) ===")
        real_df_path = Path("data/reporting/pe_wide_table.parquet")
        if not real_df_path.exists():
            # Fallback to legacy path
            real_df_path = Path("data/pe_experiments/experiment_gpt-5-mini.parquet")
        if real_df_path.exists():
            real_df = pd.read_parquet(real_df_path)
        else:
            from src.pe.distance import CAT_COLS, NUMERIC_COLS
            real_df = pd.DataFrame(
                {c: ["Unknown"] for c in CAT_COLS}
                | {c: [0.0] for c in NUMERIC_COLS}
            )

        api = PEApi(
            real_df=real_df,
            model=args.model,
            max_concurrent=args.max_concurrent,
        )

        # Generate a larger pool for DP histogram selection
        pool_target = args.n_records * args.pool_multiplier
        pool_plan, _ = build_generation_plan(
            real_results_dir,
            n_total=pool_target,
            epsilon_agg=args.epsilon_agg,
            delta=args.delta,
        )

        t0 = time.time()
        pool_df = asyncio.run(
            api.stratified_api(pool_plan, batch_size=args.batch_size)
        )
        gen_time = time.time() - t0
        print(f"Generated pool: {len(pool_df)} records in {gen_time:.0f}s")

        # Add guid column to pool
        pool_df.insert(
            0, "guid", [f"pe_strat_{i:07d}" for i in range(len(pool_df))]
        )
        pool_df.to_parquet(output_dir / "pool_wide.parquet", index=False)

        # ---- Step 3: DP histogram selection --------------------------------
        print("\n=== DP HISTOGRAM SELECTION ===")
        if real_df_path.exists():
            voting_df = pd.read_parquet(real_df_path)
        else:
            print("WARNING: No real_df available for voting. "
                  "Skipping DP histogram (results have no DP guarantee).")
            synth_df = pool_df.head(args.n_records).reset_index(drop=True)
            synth_df.to_parquet(output_dir / "synth_wide.parquet", index=False)
            print(f"Saved wide table (no DP selection): {synth_df.shape}")
            _run_evaluation(
                synth_df, output_dir, reporting_dir, results_dir,
                queries_dir, real_results_dir, args, privacy_info,
                dp_histogram_applied=False,
            )
            return

        if args.real_subsample and args.real_subsample < len(voting_df):
            voting_df = voting_df.sample(
                args.real_subsample, random_state=42
            ).reset_index(drop=True)
            print(f"Subsampled {args.real_subsample:,} real records for voting")

        # Calibrate histogram noise
        T = 1  # single iteration
        sigma_hist = calibrate_sigma(
            args.epsilon_hist, args.delta, T, sensitivity=1.0
        )
        actual_epsilon_hist = compute_epsilon(
            sigma_hist, args.delta, T, sensitivity=1.0
        )
        print(f"Histogram: sigma={sigma_hist:.4f}, "
              f"actual_epsilon_hist={actual_epsilon_hist:.4f}")

        dist = WorkloadDistance(voting_df)
        pool_for_hist = pool_df.drop(columns=["guid"], errors="ignore")

        t0 = time.time()
        print(f"Computing DP nearest-neighbor histogram "
              f"({len(voting_df)} real x {len(pool_for_hist)} synth)...")
        histogram = dp_nn_histogram(
            voting_df, pool_for_hist, dist, sigma_hist,
        )
        hist_time = time.time() - t0
        print(f"Histogram computed in {hist_time:.1f}s")

        nonzero_bins = (histogram > 0).sum()
        print(f"Nonzero bins: {nonzero_bins}/{len(histogram)}")

        synth_df = select_candidates(
            pool_df, histogram, args.n_records, method="rank"
        )
        synth_df = synth_df.reset_index(drop=True)
        synth_df["guid"] = [f"pe_strat_{i:07d}" for i in range(len(synth_df))]
        synth_df.to_parquet(output_dir / "synth_wide.parquet", index=False)
        print(f"Selected {len(synth_df)} records via DP histogram")
        print(f"Saved wide table: {synth_df.shape}")

        privacy_info["epsilon_hist"] = actual_epsilon_hist
        privacy_info["sigma_hist"] = sigma_hist
        privacy_info["total_epsilon"] = args.epsilon_agg + actual_epsilon_hist
        privacy_info["histogram_time"] = hist_time
        privacy_info["pool_size"] = len(pool_df)
        privacy_info["selected_size"] = len(synth_df)

    # ---- Evaluation -------------------------------------------------------
    dp_applied = not args.skip_generation
    _run_evaluation(
        synth_df, output_dir, reporting_dir, results_dir,
        queries_dir, real_results_dir, args, privacy_info,
        dp_histogram_applied=dp_applied,
    )


def _run_evaluation(
    synth_df: pd.DataFrame,
    output_dir: Path,
    reporting_dir: Path,
    results_dir: Path,
    queries_dir: Path,
    real_results_dir: Path,
    args: argparse.Namespace,
    privacy_info: dict,
    dp_histogram_applied: bool,
):
    """Run sparsity check, decompose, benchmark, evaluate, and compare."""

    # ---- Sparsity check ---------------------------------------------------
    print("\n=== SPARSITY CHECK ===")
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in synth_df.columns]
        if present:
            nonzero = (synth_df[present].abs().sum(axis=1) > 0).sum()
            print(
                f"  {gname}: {nonzero}/{len(synth_df)} nonzero "
                f"({100 * nonzero / len(synth_df):.1f}%)"
            )

    # ---- Decompose --------------------------------------------------------
    print("\n=== DECOMPOSING INTO REPORTING TABLES ===")
    counts = decompose_wide_table(synth_df, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    # ---- Run benchmark queries --------------------------------------------
    print("\n=== RUNNING BENCHMARK QUERIES ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(
        query_names, queries_dir, reporting_dir, results_dir
    )
    print(f"{len(results)}/{len(query_names)} queries succeeded")

    # ---- Evaluate ----------------------------------------------------------
    print("\n=== EVALUATING AGAINST REAL DATA ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(output_dir / "evaluation_detail.csv", index=False)

    # Save privacy accounting
    privacy_info["dp_histogram_applied"] = dp_histogram_applied
    with open(output_dir / "privacy_accounting.json", "w") as f:
        json.dump(privacy_info, f, indent=2)

    # ---- Print summary -----------------------------------------------------
    print("\n" + "=" * 70)
    dp_status = "with DP" if dp_histogram_applied else "NO DP histogram"
    total_eps = privacy_info.get(
        "total_epsilon", privacy_info.get("epsilon_agg", "?")
    )
    print(
        f"STRATIFIED BENCHMARK RESULTS "
        f"({len(synth_df)} records, {args.model}, {dp_status}, "
        f"epsilon={total_eps})"
    )
    print("=" * 70 + "\n")
    for _, row in eval_df.iterrows():
        if row["error"]:
            status = "SKIP"
            score_str = f'({row["error"]})'
        elif row["passed"]:
            status = "PASS"
            score_str = f'{row["score"]:.3f}'
        else:
            status = "FAIL"
            score_str = f'{row["score"]:.3f}'
        print(f"  [{status}] {row['query']}: {score_str}")

    valid = eval_df[eval_df["error"] == ""]
    skipped = eval_df[eval_df["error"] != ""]
    passed = valid[valid["passed"]]
    print(f"\nSummary:")
    print(f"  Queries run:     {len(valid)}/{len(eval_df)}")
    print(f"  Queries skipped: {len(skipped)} (missing reporting tables)")
    print(f"  Passed (>=0.5):  {len(passed)}/{len(valid)}")
    if len(valid) > 0:
        print(f"  Average score:   {valid['score'].mean():.3f}")
        print(f"  Median score:    {valid['score'].median():.3f}")

    print(f"\nPrivacy accounting:")
    print(f"  epsilon_agg (aggregates):  "
          f"{privacy_info.get('epsilon_agg', 'N/A')}")
    print(f"  epsilon_hist (histogram):  "
          f"{privacy_info.get('epsilon_hist', 'N/A')}")
    print(f"  total epsilon:             {total_eps}")
    print(f"  delta:                     {privacy_info.get('delta', 'N/A')}")
    print(f"  DP histogram applied:      {dp_histogram_applied}")

    # ---- Compare with previous midscale results ----------------------------
    prev_eval = Path("data/results/pe_midscale/evaluation.csv")
    if prev_eval.exists():
        print("\n=== COMPARISON WITH PREVIOUS (INDEPENDENT) PE ===")
        prev_df = pd.read_csv(prev_eval)
        prev_df["error"] = prev_df["error"].fillna("")
        prev_valid = prev_df[prev_df["error"] == ""]
        prev_passed = prev_valid[prev_valid["passed"]]

        print(f"  Previous: {len(prev_passed)}/{len(prev_valid)} passed, "
              f"avg={prev_valid['score'].mean():.3f}")
        print(f"  Stratified: {len(passed)}/{len(valid)} passed, "
              f"avg={valid['score'].mean():.3f}")

        merged = valid.merge(
            prev_valid[["query", "score"]],
            on="query",
            how="outer",
            suffixes=("_strat", "_prev"),
        )
        for _, row in merged.iterrows():
            s_strat = row.get("score_strat", float("nan"))
            s_prev = row.get("score_prev", float("nan"))
            d = ""
            if pd.notna(s_strat) and pd.notna(s_prev):
                diff = s_strat - s_prev
                d = f" (delta={diff:+.3f})"
            print(
                f"  {row['query']}: "
                f"strat={s_strat:.3f} vs prev={s_prev:.3f}{d}"
            )


if __name__ == "__main__":
    main()
