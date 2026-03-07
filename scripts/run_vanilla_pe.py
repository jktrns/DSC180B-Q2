"""Vanilla PE experiment: independent generation + DP histogram selection.

Uses async OpenAI API calls with chunked gathering to avoid deadlocks.

Usage:
    python scripts/run_vanilla_pe.py --n-synth 5000 --L 3
    python scripts/run_vanilla_pe.py --skip-generation  # re-run eval only
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
from openai import AsyncOpenAI

from src.eval.benchmark import run_benchmark
from src.eval.compare import (
    QUERY_METADATA,
    detailed_results_to_dataframe,
    evaluate_all,
    results_to_dataframe,
)
from src.eval.decompose import decompose_wide_table
from src.pe.api import (
    INSTRUCTIONS,
    NUMERIC_COLS,
    CAT_COLS,
    RecordsBatch,
    _build_random_prompt,
    _build_schema_description,
    _NUMERIC_GROUPS,
)
from src.pe.distance import WorkloadDistance
from src.pe.histogram import dp_nn_histogram, select_candidates
from src.pe.privacy import calibrate_sigma, compute_epsilon


async def call_api_once(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    is_reasoning: bool,
) -> list[dict]:
    try:
        kwargs: dict = {
            "model": model,
            "instructions": INSTRUCTIONS,
            "input": prompt,
            "text_format": RecordsBatch,
            "max_output_tokens": 16000,
        }
        if is_reasoning:
            kwargs["reasoning"] = {"effort": "low"}
        else:
            kwargs["temperature"] = 0.8
        response = await client.responses.parse(**kwargs)
        if response.output_parsed and response.output_parsed.records:
            return [r.model_dump() for r in response.output_parsed.records]
    except Exception:
        pass
    return []


async def generate_records_async(
    client: AsyncOpenAI,
    model: str,
    schema_desc: str,
    present_cols: list[str],
    n_records: int,
    batch_size: int = 5,
    max_concurrent: int = 50,
) -> pd.DataFrame:
    overshoot = int(n_records * 1.25)
    n_batches = (overshoot + batch_size - 1) // batch_size
    is_reasoning = model.startswith("gpt-5") or model.startswith("o")
    print(f"RANDOM_API: {n_records} records ({n_batches} batches of {batch_size}, 25% buffer, concurrent={max_concurrent})")
    prompts = [_build_random_prompt(schema_desc, batch_size) for _ in range(n_batches)]
    all_records: list[dict] = []
    t0 = time.time()
    chunk_size = max_concurrent
    for chunk_start in range(0, n_batches, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_batches)
        chunk_prompts = prompts[chunk_start:chunk_end]
        tasks = [call_api_once(client, model, p, is_reasoning) for p in chunk_prompts]
        results = await asyncio.gather(*tasks)
        for batch_records in results:
            all_records.extend(batch_records)
        elapsed = time.time() - t0
        rate = chunk_end / elapsed if elapsed > 0 else 0
        print(f"  RANDOM_API: {chunk_end}/{n_batches} ({elapsed:.0f}s, {rate:.1f}/s, {len(all_records)} records)")
        if len(all_records) >= n_records:
            print(f"  Got enough records ({len(all_records)} >= {n_records}), stopping early")
            break
    rows = []
    for r in all_records[:n_records]:
        row = {}
        for c in present_cols:
            val = r.get(c, 0 if c in NUMERIC_COLS else "Unknown")
            row[c] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in [col for col in NUMERIC_COLS if col in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    for c in [col for col in CAT_COLS if col in df.columns]:
        df[c] = df[c].fillna("Unknown").astype(str)
    print(f"RANDOM_API: {len(all_records)} raw -> {len(df)} returned")
    return df


async def run_generation(args, real_df, output_dir):
    if args.real_subsample < len(real_df):
        voting_df = real_df.sample(args.real_subsample, random_state=42).reset_index(drop=True)
        print(f"Subsampled {args.real_subsample:,} from {len(real_df):,} real records for voting")
    else:
        voting_df = real_df
    sigma = calibrate_sigma(args.epsilon, args.delta, args.T)
    print(f"PE config: N_synth={args.n_synth}, T={args.T}, L={args.L}, epsilon={args.epsilon}, delta={args.delta}, sigma={sigma:.4f}")
    all_cols = CAT_COLS + NUMERIC_COLS
    present_cols = [c for c in all_cols if c in real_df.columns]
    schema_desc = _build_schema_description(real_df)
    client = AsyncOpenAI()
    n_candidates = args.n_synth * args.L
    print(f"\n--- Generating initial population (N={n_candidates}) ---")
    t0 = time.time()
    population = await generate_records_async(client, args.model, schema_desc, present_cols, n_candidates, batch_size=args.batch_size, max_concurrent=args.max_concurrent)
    gen_time = time.time() - t0
    print(f"Population generated: {len(population)} records in {gen_time:.0f}s")
    if len(population) == 0:
        raise RuntimeError("No records generated. Check API key and model.")
    population.to_parquet(output_dir / "population.parquet", index=False)
    print("\n--- DP histogram selection ---")
    dist = WorkloadDistance(real_df)
    t0 = time.time()
    print(f"Computing DP nearest-neighbor histogram ({len(voting_df)} real x {len(population)} synth)...")
    histogram = dp_nn_histogram(voting_df, population, dist, sigma, real_chunk=5000, synth_chunk=10000)
    hist_time = time.time() - t0
    print(f"Histogram computed in {hist_time:.1f}s")
    nonzero_bins = (histogram > 0).sum()
    print(f"Nonzero bins: {nonzero_bins}/{len(histogram)}")
    synth_df = select_candidates(population, histogram, args.n_synth, method="rank")
    print(f"Selected top {args.n_synth} candidates")
    synth_df.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(synth_df))])
    total_time = gen_time + hist_time
    actual_eps = compute_epsilon(sigma, args.delta, args.T)
    print(f"\nVanilla PE complete: {len(synth_df)} records in {total_time:.0f}s")
    print(f"Actual epsilon: {actual_eps:.4f}")
    synth_df.to_parquet(output_dir / "synth_wide.parquet", index=False)
    with open(output_dir / "pe_history.json", "w") as f:
        json.dump({"sigma": sigma, "epsilon": args.epsilon, "actual_epsilon": actual_eps, "delta": args.delta, "T": args.T, "L": args.L, "n_synth": args.n_synth, "n_candidates": len(population), "generation_time": gen_time, "histogram_time": hist_time, "total_time": total_time, "nonzero_bins": int(nonzero_bins)}, f, indent=2, default=str)
    return synth_df


def main():
    parser = argparse.ArgumentParser(description="Vanilla PE benchmark")
    parser.add_argument("--n-synth", type=int, default=5000)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--real-subsample", type=int, default=50000, help="Subsample real records for voting")
    parser.add_argument("--output-dir", type=str, default="data/results/pe_vanilla")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporting_dir = output_dir / "reporting"
    results_dir = output_dir / "query_results"
    queries_dir = Path("docs/queries")
    real_results_dir = Path("data/results/real")
    if args.skip_generation:
        synth_df = pd.read_parquet(output_dir / "synth_wide.parquet")
        print(f"Loaded existing: {synth_df.shape}")
    else:
        real_df = pd.read_parquet("data/reporting/wide_training_table.parquet")
        print(f"Real data: {real_df.shape}")
        synth_df = asyncio.run(run_generation(args, real_df, output_dir))
    # Sparsity check
    print("\n=== SPARSITY CHECK ===")
    cols_no_guid = [c for c in synth_df.columns if c != "guid"]
    for gname, cols in _NUMERIC_GROUPS.items():
        present = [c for c in cols if c in cols_no_guid]
        if present:
            nonzero = (synth_df[present].abs().sum(axis=1) > 0).sum()
            print(f"  {gname}: {nonzero}/{len(synth_df)} nonzero ({100 * nonzero / len(synth_df):.1f}%)")
    # Decompose
    print("\n=== DECOMPOSING INTO REPORTING TABLES ===")
    counts = decompose_wide_table(synth_df, reporting_dir)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")
    # Benchmark
    print("\n=== RUNNING BENCHMARK QUERIES ===")
    query_names = list(QUERY_METADATA.keys())
    results = run_benchmark(query_names, queries_dir, reporting_dir, results_dir)
    print(f"{len(results)}/{len(query_names)} queries succeeded")
    # Evaluate
    print("\n=== EVALUATING AGAINST REAL DATA ===")
    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_df.to_csv(output_dir / "evaluation.csv", index=False)
    detail_df = detailed_results_to_dataframe(eval_results)
    detail_df.to_csv(output_dir / "evaluation_detail.csv", index=False)
    # Print summary
    print("\n" + "=" * 70)
    print(f"VANILLA PE RESULTS ({len(synth_df)} records, {args.model}, epsilon={args.epsilon})")
    print("=" * 70 + "\n")
    for _, row in eval_df.iterrows():
        if row["error"]:
            status = "SKIP"
            score_str = f"({row['error']})"
        elif row["passed"]:
            status = "PASS"
            score_str = f"{row['score']:.3f}"
        else:
            status = "FAIL"
            score_str = f"{row['score']:.3f}"
        print(f"  [{status}] {row['query']}: {score_str}")
    valid = eval_df[eval_df["error"] == ""]
    skipped = eval_df[eval_df["error"] != ""]
    passed = valid[valid["passed"]]
    print("\nSummary:")
    print(f"  Queries run:     {len(valid)}/{len(eval_df)}")
    print(f"  Queries skipped: {len(skipped)}")
    print(f"  Passed (>=0.5):  {len(passed)}/{len(valid)}")
    if len(valid) > 0:
        print(f"  Average score:   {valid['score'].mean():.3f}")
        print(f"  Median score:    {valid['score'].median():.3f}")


if __name__ == "__main__":
    main()
