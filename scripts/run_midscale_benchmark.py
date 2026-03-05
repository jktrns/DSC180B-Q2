"""Mid-scale PE benchmark: generate synthetic data, decompose, run queries, evaluate.

Usage:
    python scripts/run_midscale_benchmark.py --n-records 5000 --batch-size 10
    python scripts/run_midscale_benchmark.py --skip-generation  # re-run eval only
"""

from __future__ import annotations

import argparse
import asyncio
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


def main():
    parser = argparse.ArgumentParser(description="Mid-scale PE benchmark")
    parser.add_argument("--n-records", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data/results/pe_midscale")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, re-run decompose+benchmark+eval only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporting_dir = output_dir / "reporting"
    results_dir = output_dir / "query_results"
    queries_dir = Path("docs/queries")
    real_results_dir = Path("data/results/real")

    # ---- Step 1: Generate synthetic data ----
    if args.skip_generation:
        print("=== LOADING EXISTING WIDE TABLE ===")
        synth_df = pd.read_parquet(output_dir / "synth_wide.parquet")
        print(f"Loaded: {synth_df.shape}")
    else:
        print("=== GENERATING SYNTHETIC DATA ===")
        # Use real wide training table for schema description (sparsity stats).
        # Falls back to prior experiment parquet if available.
        real_df_path = Path("data/reporting/wide_training_table.parquet")
        if not real_df_path.exists():
            real_df_path = Path("data/pe_experiments/experiment_gpt-5-mini.parquet")
        if not real_df_path.exists():
            print(f"ERROR: No real data found. Need wide_training_table.parquet or experiment parquet.")
            sys.exit(1)
        print(f"Loading real data from: {real_df_path}")
        real_df = pd.read_parquet(real_df_path)
        api = PEApi(
            real_df=real_df,
            model=args.model,
            max_concurrent=args.max_concurrent,
        )

        t0 = time.time()
        synth_df = asyncio.run(
            api.random_api(n_records=args.n_records, batch_size=args.batch_size)
        )
        elapsed = time.time() - t0
        print(f"Generated {len(synth_df)} records in {elapsed:.0f}s")

        # Add guid column
        synth_df.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(synth_df))])
        synth_df.to_parquet(output_dir / "synth_wide.parquet", index=False)
        print(f"Saved wide table: {synth_df.shape}")

    # ---- Step 2: Sparsity check ----
    print("\n=== SPARSITY CHECK ===")
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in synth_df.columns]
        if present:
            nonzero = (synth_df[present].abs().sum(axis=1) > 0).sum()
            print(f"  {gname}: {nonzero}/{len(synth_df)} nonzero "
                  f"({100 * nonzero / len(synth_df):.1f}%)")

    # ---- Step 3: Decompose ----
    print("\n=== DECOMPOSING INTO REPORTING TABLES ===")
    counts = decompose_wide_table(synth_df, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    # ---- Step 4: Run benchmark queries ----
    print("\n=== RUNNING BENCHMARK QUERIES ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(query_names, queries_dir, reporting_dir, results_dir)
    print(f"{len(results)}/{len(query_names)} queries succeeded")

    # ---- Step 5: Evaluate ----
    print("\n=== EVALUATING AGAINST REAL DATA ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(output_dir / "evaluation_detail.csv", index=False)

    # ---- Step 6: Print summary ----
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS ({len(synth_df)} records, {args.model})")
    print("=" * 60 + "\n")
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
    passed = valid[valid["passed"] == True]
    print(f"\nSummary:")
    print(f"  Queries run:     {len(valid)}/{len(eval_df)}")
    print(f"  Queries skipped: {len(skipped)} (missing reporting tables)")
    print(f"  Passed (>=0.5):  {len(passed)}/{len(valid)}")
    if len(valid) > 0:
        print(f"  Average score:   {valid['score'].mean():.3f}")
        print(f"  Median score:    {valid['score'].median():.3f}")


if __name__ == "__main__":
    main()
