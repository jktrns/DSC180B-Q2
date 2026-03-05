"""Stratified PE benchmark: conditional generation from real query targets.

Instead of generating records independently, this script:
1. Reads real query results to compute per-stratum targets
2. Generates records conditioned on those targets
3. Decomposes into reporting tables, runs queries, evaluates

Usage:
    python scripts/run_stratified_benchmark.py --n-records 5000
    python scripts/run_stratified_benchmark.py --skip-generation  # re-run eval only
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
from src.pe.stratified import build_generation_plan


def main():
    parser = argparse.ArgumentParser(
        description="Stratified PE benchmark (conditional generation)"
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporting_dir = output_dir / "reporting"
    results_dir = output_dir / "query_results"
    queries_dir = Path("docs/queries")
    real_results_dir = Path("data/results/real")

    # ---- Step 1: Build generation plan ---------------------------------
    print("=== BUILDING STRATIFIED GENERATION PLAN ===")
    plan = build_generation_plan(real_results_dir, n_total=args.n_records)
    print(plan.summary())

    # ---- Step 2: Generate synthetic data -------------------------------
    if args.skip_generation:
        print("\n=== LOADING EXISTING WIDE TABLE ===")
        synth_df = pd.read_parquet(output_dir / "synth_wide.parquet")
        print(f"Loaded: {synth_df.shape}")
    else:
        print("\n=== GENERATING SYNTHETIC DATA (STRATIFIED) ===")
        # We need a small real_df for PEApi init (schema description).
        # Use any existing experiment parquet, or create a minimal one.
        real_df_path = Path("data/pe_experiments/experiment_gpt-5-mini.parquet")
        if real_df_path.exists():
            real_df = pd.read_parquet(real_df_path)
        else:
            # Create a minimal DataFrame with the right columns for init
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

        t0 = time.time()
        synth_df = asyncio.run(
            api.stratified_api(plan, batch_size=args.batch_size)
        )
        elapsed = time.time() - t0
        print(f"Generated {len(synth_df)} records in {elapsed:.0f}s")

        # Add guid column
        synth_df.insert(
            0, "guid", [f"pe_strat_{i:07d}" for i in range(len(synth_df))]
        )
        synth_df.to_parquet(output_dir / "synth_wide.parquet", index=False)
        print(f"Saved wide table: {synth_df.shape}")

    # ---- Step 3: Sparsity check ----------------------------------------
    print("\n=== SPARSITY CHECK ===")
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in synth_df.columns]
        if present:
            nonzero = (synth_df[present].abs().sum(axis=1) > 0).sum()
            print(
                f"  {gname}: {nonzero}/{len(synth_df)} nonzero "
                f"({100 * nonzero / len(synth_df):.1f}%)"
            )

    # ---- Step 4: Decompose ---------------------------------------------
    print("\n=== DECOMPOSING INTO REPORTING TABLES ===")
    counts = decompose_wide_table(synth_df, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    # ---- Step 5: Run benchmark queries ---------------------------------
    print("\n=== RUNNING BENCHMARK QUERIES ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(
        query_names, queries_dir, reporting_dir, results_dir
    )
    print(f"{len(results)}/{len(query_names)} queries succeeded")

    # ---- Step 6: Evaluate -----------------------------------------------
    print("\n=== EVALUATING AGAINST REAL DATA ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(output_dir / "evaluation_detail.csv", index=False)

    # ---- Step 7: Print summary ------------------------------------------
    print("\n" + "=" * 70)
    print(
        f"STRATIFIED BENCHMARK RESULTS "
        f"({len(synth_df)} records, {args.model})"
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

    # ---- Compare with previous midscale results -------------------------
    prev_eval = Path("data/results/pe_midscale/evaluation.csv")
    if prev_eval.exists():
        print("\n=== COMPARISON WITH PREVIOUS (INDEPENDENT) PE ===")
        prev_df = pd.read_csv(prev_eval)
        prev_valid = prev_df[prev_df["error"] == ""]
        prev_passed = prev_valid[prev_valid["passed"]]

        print(f"  Previous: {len(prev_passed)}/{len(prev_valid)} passed, "
              f"avg={prev_valid['score'].mean():.3f}")
        print(f"  Stratified: {len(passed)}/{len(valid)} passed, "
              f"avg={valid['score'].mean():.3f}")

        # Per-query comparison
        merged = valid.merge(
            prev_valid[["query", "score"]],
            on="query",
            how="outer",
            suffixes=("_strat", "_prev"),
        )
        for _, row in merged.iterrows():
            s_strat = row.get("score_strat", float("nan"))
            s_prev = row.get("score_prev", float("nan"))
            delta = ""
            if pd.notna(s_strat) and pd.notna(s_prev):
                d = s_strat - s_prev
                delta = f" (delta={d:+.3f})"
            print(
                f"  {row['query']}: "
                f"strat={s_strat:.3f} vs prev={s_prev:.3f}{delta}"
            )


if __name__ == "__main__":
    main()
