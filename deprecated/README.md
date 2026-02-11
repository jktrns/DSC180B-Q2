# Differentially Private Synthetic Telemetry Data Benchmark
By: Jason Tran, Phuc Tran, Mehak Kapur, Hana Tjendrawasi

## Overview

This repository contains the code, queries, and documentation for a project comparing two approaches to generating **differentially private synthetic data** from large scale system telemetry:

1. **DP-SGD based generative models**
2. **Training free private evolution**

The goal of the project is to evaluate how well each approach preserves the utility of sensitive telemetry data while providing formal privacy guarantees, with a particular focus on **summary statistics and SQL based analytical queries** that reflect real world industry use cases.

The project is motivated by the need to safely analyze and share highly sensitive device telemetry data, such as crashes, performance metrics, and power usage, without exposing information about individual devices.

---

## Project Objectives

- Generate synthetic telemetry data under differential privacy
- Compare DP-SGD and private evolution at equal privacy budgets
- Measure utility using summary statistics and fixed SQL queries
- Evaluate tradeoffs in accuracy, cost, and deployment complexity
- Ground evaluation in realistic industry style telemetry analysis

---

## Dataset Description

The project uses Intel laptop telemetry data stored as **Parquet files**, exported from two database schemas:

- `university_analysis_pad` (data dictionary and metadata)
- `university_prod` (raw telemetry data)

Due to the size of the full dataset (over 9 TB), analysis is performed on carefully selected subsets using sampling and time window filtering.

### Selected Tables

The project focuses on the following eight tables:

1. `system_sysinfo_unique_normalized`  
   Anchor table defining the device level privacy unit.

2. `frgnd_system_usage_by_app`  
   Records foreground application usage and associated system resource metrics.

3. `userwait_v2`  
   Captures user wait and application responsiveness events.

4. `os_network_consumption_v2`  
   Records system-level network usage statistics.

5. `ucsd_apps_execclass_final`  
   Provides application execution class labels for grouping and analysis.

   
---

## Data Access and Tooling

### DuckDB

All analysis is performed using **DuckDB**, which allows direct querying of Parquet files without loading the full dataset into memory.

Key features used:
- `read_parquet` for on disk querying
- SQL based aggregation and filtering
- Efficient handling of large tabular data on local machines

### Data Reduction Strategy

To make the data manageable:
- Time ranges are restricted
- Device identifiers are sampled when needed
- Raw event tables are aggregated into device level summaries

All differential privacy methods operate on **aggregated device level tables**, not raw event logs.

---

## Project Pipeline

1. Inspect schemas and manifest metadata
2. Select a subset of relevant telemetry tables
3. Load Parquet data into DuckDB
4. Aggregate raw events into device level features
5. Compute baseline summary statistics on real data
6. Generate synthetic data using:
   - DP-SGD based generative models
   - Training free private evolution
7. Run identical SQL queries on real and synthetic data
8. Compare utility, privacy behavior, and computational cost

---

## Evaluation Metrics

### Utility Metrics
- Summary statistics (count, mean, median, variance, percentiles)
- Grouped SQL query accuracy
- Relative error between real and synthetic results
- Downstream model performance (accuracy, F1 score, AUC where applicable)

### Privacy Considerations
- Formal differential privacy guarantees (epsilon, delta)
- Bounded contribution per device
- Analysis of rare event behavior

### Cost Metrics
- Runtime
- Memory usage
- Engineering complexity
- Ease of deployment

---

## Repository Structure (Example)

