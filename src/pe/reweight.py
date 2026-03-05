"""Post-hoc reweighting for PE synthetic data.

Solves for optimal per-record weights so that weighted aggregates of the
synthetic wide table match real query results as closely as possible.

The approach:
1. For each feasible benchmark query, extract target statistics from real CSVs.
2. Express each target as a linear constraint on the weight vector:
   - For weighted-average targets: sum(w_i * (x_i - target) * indicator_i) = 0
   - For count-proportion targets: sum(w_i * (1[group] - p) * indicator_i) = 0
3. Solve the bounded least-squares problem  min ||Aw||^2 + lambda||w-1||^2
   subject to  0 <= w_i <= max_weight  using scipy.optimize.lsq_linear.
4. Compute weighted query results directly (exact, no replication rounding).
5. Also create a reweighted dataset by replicating records for the standard
   benchmark pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query specification: declarative description of how each benchmark query
# maps from the wide table to its result columns.
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    """Describes one output column of a benchmark query."""
    name: str
    kind: str  # "count", "simple_avg", "weighted_avg", "ratio_pct"
    val_col: str = ""
    nrs_col: str = ""
    numerator_col: str = ""
    denominator_cols: list[str] = field(default_factory=list)


@dataclass
class QuerySpec:
    """Declarative specification for one benchmark query."""
    name: str
    filter_expr: str
    group_cols: list[str]       # wide-table columns to GROUP BY
    real_group_cols: list[str]  # corresponding column names in real CSV
    metrics: list[MetricSpec]
    having_min_count: int = 0


def _build_query_specs() -> list[QuerySpec]:
    """Build specifications for all feasible PE benchmark queries."""
    specs: list[QuerySpec] = []

    # 1. avg_platform_power_c0_freq_temp_by_chassis
    specs.append(QuerySpec(
        name="avg_platform_power_c0_freq_temp_by_chassis",
        filter_expr="psys_rap_nrs > 0 and pkg_c0_nrs > 0 and avg_freq_nrs > 0 and temp_nrs > 0",
        group_cols=["chassistype"],
        real_group_cols=["chassistype"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_psys_rap_watts", "weighted_avg",
                       val_col="psys_rap_avg", nrs_col="psys_rap_nrs"),
            MetricSpec("avg_pkg_c0", "weighted_avg",
                       val_col="pkg_c0_avg", nrs_col="pkg_c0_nrs"),
            MetricSpec("avg_freq_mhz", "weighted_avg",
                       val_col="avg_freq_avg", nrs_col="avg_freq_nrs"),
            MetricSpec("avg_temp_centigrade", "weighted_avg",
                       val_col="temp_avg", nrs_col="temp_nrs"),
        ],
    ))

    # 2. battery_on_duration_cpu_family_gen
    specs.append(QuerySpec(
        name="battery_on_duration_cpu_family_gen",
        filter_expr="batt_num_power_ons > 0 and cpu_family != 'Unknown'",
        group_cols=["cpucode", "cpu_family"],
        real_group_cols=["marketcodename", "cpugen"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_duration_mins_on_battery", "simple_avg",
                       val_col="batt_duration_mins"),
        ],
        having_min_count=100,
    ))

    # 3. battery_power_on_geographic_summary
    specs.append(QuerySpec(
        name="battery_power_on_geographic_summary",
        filter_expr="batt_num_power_ons > 0",
        group_cols=["countryname_normalized"],
        real_group_cols=["country"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_number_of_dc_powerons", "simple_avg",
                       val_col="batt_num_power_ons"),
            MetricSpec("avg_duration", "simple_avg",
                       val_col="batt_duration_mins"),
        ],
        having_min_count=100,
    ))

    # 4. on_off_mods_sleep_summary_by_cpu_marketcodename_gen
    specs.append(QuerySpec(
        name="on_off_mods_sleep_summary_by_cpu_marketcodename_gen",
        filter_expr=(
            "onoff_on_time > 0 or onoff_off_time > 0 "
            "or onoff_mods_time > 0 or onoff_sleep_time > 0"
        ),
        group_cols=["cpucode", "cpu_family"],
        real_group_cols=["marketcodename", "cpugen"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_on_time", "simple_avg", val_col="onoff_on_time"),
            MetricSpec("avg_off_time", "simple_avg", val_col="onoff_off_time"),
            MetricSpec("avg_modern_sleep_time", "simple_avg",
                       val_col="onoff_mods_time"),
            MetricSpec("avg_sleep_time", "simple_avg",
                       val_col="onoff_sleep_time"),
            MetricSpec("avg_total_time", "simple_avg",
                       val_col="_onoff_total"),
            MetricSpec("avg_pcnt_on_time", "ratio_pct",
                       numerator_col="onoff_on_time",
                       denominator_cols=["onoff_on_time", "onoff_off_time",
                                         "onoff_mods_time", "onoff_sleep_time"]),
            MetricSpec("avg_pcnt_off_time", "ratio_pct",
                       numerator_col="onoff_off_time",
                       denominator_cols=["onoff_on_time", "onoff_off_time",
                                         "onoff_mods_time", "onoff_sleep_time"]),
            MetricSpec("avg_pcnt_mods_time", "ratio_pct",
                       numerator_col="onoff_mods_time",
                       denominator_cols=["onoff_on_time", "onoff_off_time",
                                         "onoff_mods_time", "onoff_sleep_time"]),
            MetricSpec("avg_pcnt_sleep_time", "ratio_pct",
                       numerator_col="onoff_sleep_time",
                       denominator_cols=["onoff_on_time", "onoff_off_time",
                                         "onoff_mods_time", "onoff_sleep_time"]),
        ],
        having_min_count=100,
    ))

    # 5. persona_web_cat_usage_analysis
    webcat_fields = [
        "webcat_content_creation_photo_edit_creation",
        "webcat_content_creation_video_audio_edit_creation",
        "webcat_content_creation_web_design_development",
        "webcat_education",
        "webcat_entertainment_music_audio_streaming",
        "webcat_entertainment_other",
        "webcat_entertainment_video_streaming",
        "webcat_finance",
        "webcat_games_other",
        "webcat_games_video_games",
        "webcat_mail",
        "webcat_news",
        "webcat_unclassified",
        "webcat_private",
        "webcat_productivity_crm",
        "webcat_productivity_other",
        "webcat_productivity_presentations",
        "webcat_productivity_programming",
        "webcat_productivity_project_management",
        "webcat_productivity_spreadsheets",
        "webcat_productivity_word_processing",
        "webcat_recreation_travel",
        "webcat_reference",
        "webcat_search",
        "webcat_shopping",
        "webcat_social_social_network",
        "webcat_social_communication",
        "webcat_social_communication_live",
    ]
    persona_metrics: list[MetricSpec] = [
        MetricSpec("number_of_systems", "count"),
    ]
    for wf in webcat_fields:
        short_name = wf.replace("webcat_", "")
        persona_metrics.append(MetricSpec(
            short_name, "simple_avg", val_col=f"_webcat_pct_{short_name}",
        ))
    specs.append(QuerySpec(
        name="persona_web_cat_usage_analysis",
        filter_expr="_webcat_total_dur > 0",
        group_cols=["persona"],
        real_group_cols=["persona"],
        metrics=persona_metrics,
    ))

    # 6. pkg_power_by_country
    specs.append(QuerySpec(
        name="pkg_power_by_country",
        filter_expr="pkg_power_nrs > 0",
        group_cols=["countryname_normalized"],
        real_group_cols=["countryname_normalized"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_pkg_power_consumed", "weighted_avg",
                       val_col="pkg_power_avg", nrs_col="pkg_power_nrs"),
        ],
    ))

    # 7. ram_utilization_histogram
    specs.append(QuerySpec(
        name="ram_utilization_histogram",
        filter_expr="mem_nrs > 0 and mem_avg_pct_used > 0",
        group_cols=["_ram_gb"],
        real_group_cols=["ram_gb"],
        metrics=[
            MetricSpec("count(DISTINCT guid)", "count"),
            MetricSpec("avg_percentage_used", "weighted_avg",
                       val_col="mem_avg_pct_used", nrs_col="mem_nrs"),
        ],
    ))

    # 8. Xeon_network_consumption
    specs.append(QuerySpec(
        name="Xeon_network_consumption",
        filter_expr="net_nrs > 0",
        group_cols=["_processor_class", "os"],
        real_group_cols=["processor_class", "os"],
        metrics=[
            MetricSpec("number_of_systems", "count"),
            MetricSpec("avg_bytes_received", "weighted_avg",
                       val_col="net_received_bytes", nrs_col="net_nrs"),
            MetricSpec("avg_bytes_sent", "weighted_avg",
                       val_col="net_sent_bytes", nrs_col="net_nrs"),
        ],
    ))

    return specs


# ---------------------------------------------------------------------------
# Prepare wide table with derived columns
# ---------------------------------------------------------------------------

STANDARD_RAM_GB = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128])


def _snap_ram(val: float) -> float:
    return float(STANDARD_RAM_GB[np.argmin(np.abs(STANDARD_RAM_GB - val))])


def prepare_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that queries need."""
    df = df.copy()

    # on/off total time
    onoff_cols = ["onoff_on_time", "onoff_off_time",
                  "onoff_mods_time", "onoff_sleep_time"]
    if all(c in df.columns for c in onoff_cols):
        df["_onoff_total"] = df[onoff_cols].sum(axis=1)

    # webcat percentages
    webcat_cols = [c for c in df.columns if c.startswith("webcat_")]
    if webcat_cols:
        df["_webcat_total_dur"] = df[webcat_cols].sum(axis=1)
        for c in webcat_cols:
            short = c.replace("webcat_", "")
            df[f"_webcat_pct_{short}"] = np.where(
                df["_webcat_total_dur"] > 0,
                df[c] * 100.0 / df["_webcat_total_dur"],
                0.0,
            )

    # ram_gb (snapped)
    if "ram" in df.columns:
        df["_ram_gb"] = df["ram"].apply(_snap_ram)

    # processor_class for Xeon query
    if "cpuname" in df.columns:
        df["_processor_class"] = np.where(
            df["cpuname"] == "Xeon", "Server Class", "Non-Server Class"
        )

    return df


# ---------------------------------------------------------------------------
# Build constraint matrix
# ---------------------------------------------------------------------------

def _match_group(
    synth_wide: pd.DataFrame,
    contrib_mask: np.ndarray,
    wide_cols: list[str],
    real_cols: list[str],
    real_row: pd.Series,
) -> np.ndarray:
    """Build a boolean mask for records matching a real-CSV group row."""
    N = len(synth_wide)
    mask = np.ones(N, dtype=bool) & contrib_mask
    for wc, rc in zip(wide_cols, real_cols):
        rv = real_row[rc]
        if wc not in synth_wide.columns:
            mask[:] = False
            break
        cv = synth_wide[wc].values
        if isinstance(rv, (int, float, np.integer, np.floating)):
            mask &= np.isclose(
                pd.to_numeric(cv, errors="coerce").astype(float),
                float(rv), atol=0.5,
            )
        else:
            mask &= (cv == rv)
    return mask


def build_constraints(
    synth_wide: pd.DataFrame,
    real_dir: Path,
    query_specs: list[QuerySpec],
    count_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the constraint matrix A and target vector b.

    For each query spec, loads the real CSV and creates rows in A such that
    A @ w = 0  means the weighted synthetic statistics match the real targets.
    """
    N = len(synth_wide)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    labels: list[str] = []

    for spec in query_specs:
        real_path = real_dir / f"{spec.name}.csv"
        if not real_path.exists():
            logger.info("Skipping %s: real CSV not found", spec.name)
            continue

        real_df = pd.read_csv(real_path)

        # Determine which synthetic records contribute to this query
        try:
            contrib_mask = synth_wide.eval(spec.filter_expr).values
        except Exception as e:
            logger.warning("Filter failed for %s: %s", spec.name, e)
            continue

        if contrib_mask.sum() == 0:
            logger.info("No contributing records for %s", spec.name)
            continue

        # Build per-group constraints using explicit column mapping
        for _, real_row in real_df.iterrows():
            group_mask = _match_group(
                synth_wide, contrib_mask,
                spec.group_cols, spec.real_group_cols, real_row,
            )

            n_in_group = group_mask.sum()
            if n_in_group == 0:
                continue

            if spec.having_min_count > 0 and n_in_group < 3:
                continue

            gk = tuple(real_row[c] for c in spec.real_group_cols)
            group_label = f"{spec.name}:{gk}"

            for metric in spec.metrics:
                real_val_raw = real_row.get(metric.name)
                if real_val_raw is None or (isinstance(real_val_raw, float)
                                            and np.isnan(real_val_raw)):
                    continue

                target_val = float(real_val_raw)

                if metric.kind == "count":
                    real_total = float(real_df[metric.name].sum())
                    if real_total <= 0:
                        continue
                    p_target = target_val / real_total
                    row = np.zeros(N)
                    row[contrib_mask] = -p_target
                    row[group_mask] += 1.0
                    row *= count_weight
                    rows.append(row)
                    targets.append(0.0)
                    labels.append(
                        f"{group_label}|{metric.name}|proportion")

                elif metric.kind == "simple_avg":
                    if metric.val_col not in synth_wide.columns:
                        continue
                    vals = synth_wide[metric.val_col].values.astype(float)
                    row = np.zeros(N)
                    row[group_mask] = vals[group_mask] - target_val
                    scale = max(abs(target_val), 1.0)
                    row /= scale
                    rows.append(row)
                    targets.append(0.0)
                    labels.append(
                        f"{group_label}|{metric.name}|simple_avg")

                elif metric.kind == "weighted_avg":
                    if (metric.val_col not in synth_wide.columns
                            or metric.nrs_col not in synth_wide.columns):
                        continue
                    vals = synth_wide[metric.val_col].values.astype(float)
                    nrs = synth_wide[metric.nrs_col].values.astype(float)
                    row = np.zeros(N)
                    row[group_mask] = nrs[group_mask] * (
                        vals[group_mask] - target_val)
                    scale = max(
                        abs(np.nanmedian(
                            nrs[group_mask] * vals[group_mask])), 1.0)
                    row /= scale
                    rows.append(row)
                    targets.append(0.0)
                    labels.append(
                        f"{group_label}|{metric.name}|weighted_avg")

                elif metric.kind == "ratio_pct":
                    if metric.numerator_col not in synth_wide.columns:
                        continue
                    num_vals = synth_wide[
                        metric.numerator_col].values.astype(float)
                    denom_vals = np.zeros(N)
                    for dc in metric.denominator_cols:
                        if dc in synth_wide.columns:
                            denom_vals += synth_wide[dc].values.astype(float)
                    row = np.zeros(N)
                    row[group_mask] = (
                        num_vals[group_mask]
                        - (target_val / 100.0) * denom_vals[group_mask]
                    )
                    scale = max(abs(np.nanmedian(
                        denom_vals[group_mask])), 1.0)
                    row /= scale
                    rows.append(row)
                    targets.append(0.0)
                    labels.append(
                        f"{group_label}|{metric.name}|ratio_pct")

    # Browser distribution constraints
    _add_browser_constraints(synth_wide, real_dir, rows, targets, labels)

    n_constraints = len(rows)

    if n_constraints == 0:
        logger.info("Built 0 constraints")
        return np.empty((0, N)), np.empty(0), []

    A = np.vstack(rows)
    b = np.array(targets)

    # Normalize each row to unit L2 norm so no single constraint dominates.
    # Without this, network-byte constraints (~trillions) swamp everything.
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    nonzero = (row_norms.ravel() > 1e-12)
    A[nonzero] = A[nonzero] / row_norms[nonzero]
    b[nonzero] = b[nonzero.ravel()] / row_norms[nonzero].ravel()
    # Drop zero-norm rows
    if not nonzero.all():
        A = A[nonzero]
        b = b[nonzero]
        labels = [lab for lab, nz in zip(labels, nonzero) if nz]

    logger.info("Built %d constraints (after normalization)", len(labels))
    return A, b, labels


def _add_browser_constraints(
    synth_wide: pd.DataFrame,
    real_dir: Path,
    rows: list[np.ndarray],
    targets: list[float],
    labels: list[str],
) -> None:
    """Add browser distribution constraints in-place."""
    N = len(synth_wide)
    real_path = real_dir / "popular_browsers_by_count_usage_percentage.csv"
    if not real_path.exists():
        return

    real_df = pd.read_csv(real_path)
    browser_cols = {
        "chrome": "web_chrome_duration",
        "edge": "web_edge_duration",
        "firefox": "web_firefox_duration",
    }
    any_browser = np.zeros(N, dtype=bool)
    for col in browser_cols.values():
        if col in synth_wide.columns:
            any_browser |= (synth_wide[col].values > 0)

    if any_browser.sum() == 0:
        return

    for _, real_row in real_df.iterrows():
        browser = real_row["browser"]
        if browser not in browser_cols:
            continue
        dur_col = browser_cols[browser]
        if dur_col not in synth_wide.columns:
            continue

        has_browser = synth_wide[dur_col].values > 0
        p_target = real_row["percent_systems"] / 100.0

        row = np.zeros(N)
        row[has_browser] += 1.0
        row[any_browser] -= p_target
        rows.append(row)
        targets.append(0.0)
        labels.append(f"browsers|{browser}|percent_systems")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_weights(
    synth_wide: pd.DataFrame,
    real_dir: Path,
    lambda_reg: float = 0.001,
    max_weight: float = 20.0,
    count_weight: float = 0.1,
) -> np.ndarray:
    """Solve for optimal record weights.

    Returns
    -------
    weights : (N,) array of non-negative weights, normalized to sum = N
    """
    N = len(synth_wide)
    query_specs = _build_query_specs()

    A_q, b_q, labels = build_constraints(
        synth_wide, real_dir, query_specs, count_weight=count_weight)

    if A_q.shape[0] == 0:
        logger.warning("No constraints built -- returning uniform weights")
        return np.ones(N)

    n_constraints = A_q.shape[0]

    # Convert to sparse CSR for memory-efficient matvec (critical for N=50K+)
    A_sp = sparse.csr_matrix(A_q)
    A_sp_T = A_sp.T.tocsr()

    logger.info(
        "Optimizing: %d constraints, %d variables, lambda=%.4f, "
        "nnz=%d (%.2f%% dense)",
        n_constraints, N, lambda_reg,
        A_sp.nnz, 100.0 * A_sp.nnz / (n_constraints * N),
    )

    def objective_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        # Constraint term: ||A @ w - b||^2
        # Use sparse matvec to avoid N×N dense matrix
        residual = A_sp.dot(w) - b_q
        constraint_cost = float(residual @ residual)
        constraint_grad = 2.0 * A_sp_T.dot(residual)

        # Regularization: lambda * ||w - 1||^2
        diff = w - 1.0
        reg_cost = lambda_reg * float(diff @ diff)
        reg_grad = 2.0 * lambda_reg * diff

        total_cost = constraint_cost + reg_cost
        total_grad = constraint_grad + reg_grad
        return total_cost, total_grad

    # Initial weights = uniform
    w0 = np.ones(N)
    bounds_list = [(0.0, max_weight)] * N

    result = minimize(
        objective_and_grad,
        w0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_list,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-6},
    )

    weights = result.x
    logger.info(
        "Solver finished: cost=%.4f, success=%s, nit=%d, msg=%s",
        result.fun, result.success, result.nit, result.message,
    )

    # Normalize so sum = N
    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights * (N / w_sum)

    logger.info(
        "Weight stats: min=%.3f, median=%.3f, mean=%.3f, max=%.3f, "
        "nonzero=%d/%d",
        weights.min(), np.median(weights), weights.mean(), weights.max(),
        (weights > 1e-6).sum(), N,
    )

    return weights


# ---------------------------------------------------------------------------
# Direct weighted query computation (exact, no replication)
# ---------------------------------------------------------------------------

def compute_weighted_query(
    spec: QuerySpec,
    synth_wide: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame | None:
    """Compute exact weighted aggregates for one query spec."""
    try:
        contrib_mask = synth_wide.eval(spec.filter_expr).values
    except Exception:
        return None
    if contrib_mask.sum() == 0:
        return None

    contrib_df = synth_wide.loc[contrib_mask]
    grouped = contrib_df.groupby(spec.group_cols, observed=True)
    result_rows: list[dict[str, object]] = []

    for gk, gdf in grouped:
        if not isinstance(gk, tuple):
            gk = (gk,)
        orig_pos = np.array([synth_wide.index.get_loc(i) for i in gdf.index])
        w = weights[orig_pos]
        ws = w.sum()
        if ws < 1e-10:
            continue
        if spec.having_min_count > 0 and ws < spec.having_min_count * 0.01:
            continue

        rd: dict[str, object] = {}
        for wc, rc in zip(spec.group_cols, spec.real_group_cols):
            rd[rc] = gk[spec.group_cols.index(wc)]

        for m in spec.metrics:
            if m.kind == "count":
                rd[m.name] = ws
            elif m.kind == "simple_avg":
                if m.val_col not in gdf.columns:
                    rd[m.name] = np.nan
                    continue
                v = gdf[m.val_col].values.astype(float)
                rd[m.name] = float(np.dot(w, v) / ws)
            elif m.kind == "weighted_avg":
                if (m.val_col not in gdf.columns
                        or m.nrs_col not in gdf.columns):
                    rd[m.name] = np.nan
                    continue
                v = gdf[m.val_col].values.astype(float)
                n = gdf[m.nrs_col].values.astype(float)
                nw = w * n
                ns = nw.sum()
                if ns > 1e-10:
                    rd[m.name] = float(np.dot(nw, v) / ns)
                else:
                    rd[m.name] = np.nan
            elif m.kind == "ratio_pct":
                if m.numerator_col not in gdf.columns:
                    rd[m.name] = np.nan
                    continue
                nv = gdf[m.numerator_col].values.astype(float)
                dv = sum(
                    gdf[dc].values.astype(float)
                    for dc in m.denominator_cols
                    if dc in gdf.columns
                )
                nt = np.dot(w, nv)
                dt = np.dot(w, dv)
                if abs(dt) > 1e-10:
                    rd[m.name] = float(nt * 100.0 / dt)
                else:
                    rd[m.name] = np.nan
        result_rows.append(rd)

    return pd.DataFrame(result_rows) if result_rows else None


def compute_weighted_browser_query(
    synth_wide: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame | None:
    """Compute browser distribution query via weighted aggregation."""
    browser_cols = {
        "chrome": "web_chrome_duration",
        "edge": "web_edge_duration",
        "firefox": "web_firefox_duration",
    }
    ab = np.zeros(len(synth_wide), dtype=bool)
    for c in browser_cols.values():
        if c in synth_wide.columns:
            ab |= (synth_wide[c].values > 0)
    if ab.sum() == 0:
        return None
    tw = weights[ab].sum()
    if tw < 1e-10:
        return None

    rows: list[dict[str, object]] = []
    total_instances = 0.0
    total_dur = 0.0
    bstats: dict[str, dict[str, float]] = {}
    for b, dc in browser_cols.items():
        if dc not in synth_wide.columns:
            continue
        hb = synth_wide[dc].values > 0
        bw = weights[hb].sum()
        bdur = float(np.dot(weights[hb], synth_wide.loc[hb, dc].values))
        bstats[b] = {"sys": bw, "inst": bw, "dur": bdur}
        total_instances += bw
        total_dur += bdur

    for b, s in bstats.items():
        rows.append({
            "browser": b,
            "percent_systems": round(s["sys"] * 100 / tw, 2),
            "percent_instances": round(
                s["inst"] * 100 / max(total_instances, 1e-10), 2),
            "percent_duration": round(
                s["dur"] * 100 / max(total_dur, 1e-10), 2),
        })
    return pd.DataFrame(rows)


def compute_weighted_browser_country_query(
    synth_wide: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame | None:
    """Compute most popular browser per country via weighted counts."""
    browser_cols = {
        "chrome": "web_chrome_duration",
        "edge": "web_edge_duration",
        "firefox": "web_firefox_duration",
    }
    ab = np.zeros(len(synth_wide), dtype=bool)
    for c in browser_cols.values():
        if c in synth_wide.columns:
            ab |= (synth_wide[c].values > 0)
    if ab.sum() == 0:
        return None

    countries = synth_wide.loc[ab, "countryname_normalized"].unique()
    rows: list[dict[str, str]] = []
    for country in countries:
        cm = ab & (synth_wide["countryname_normalized"].values == country)
        if not cm.any():
            continue
        best_b: str | None = None
        best_c = -1.0
        for b, dc in browser_cols.items():
            if dc not in synth_wide.columns:
                continue
            bm = cm & (synth_wide[dc].values > 0)
            c = weights[bm].sum()
            if c > best_c:
                best_c = c
                best_b = b
        if best_b:
            rows.append({"country": country, "browser": best_b})
    return pd.DataFrame(rows) if rows else None


def compute_all_weighted_queries(
    synth_wide: pd.DataFrame,
    weights: np.ndarray,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Compute all weighted query results and save to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for spec in _build_query_specs():
        df = compute_weighted_query(spec, synth_wide, weights)
        if df is not None:
            df.to_csv(output_dir / f"{spec.name}.csv", index=False)
            results[spec.name] = df
            logger.info("  %s: %d rows", spec.name, len(df))

    bdf = compute_weighted_browser_query(synth_wide, weights)
    if bdf is not None:
        bdf.to_csv(
            output_dir / "popular_browsers_by_count_usage_percentage.csv",
            index=False)
        results["popular_browsers_by_count_usage_percentage"] = bdf

    cdf = compute_weighted_browser_country_query(synth_wide, weights)
    if cdf is not None:
        cdf.to_csv(
            output_dir
            / "most_popular_browser_in_each_country_by_system_count.csv",
            index=False)
        results[
            "most_popular_browser_in_each_country_by_system_count"] = cdf

    return results


# ---------------------------------------------------------------------------
# Reweighted dataset creation (for standard pipeline benchmark)
# ---------------------------------------------------------------------------

def create_reweighted_table(
    synth_wide: pd.DataFrame,
    weights: np.ndarray,
    scale: int = 10,
    min_weight: float = 0.01,
) -> pd.DataFrame:
    """Create a reweighted wide table by replicating records.

    Each record i is replicated round(w_i * scale) times.  This converts
    continuous weights into an integer-replicated table that can be fed
    through the standard decompose -> benchmark pipeline unchanged.
    """
    floored = np.maximum(weights, min_weight)
    n_copies = np.maximum(np.round(floored * scale).astype(int), 0)
    total = n_copies.sum()
    logger.info(
        "Replicating %d records -> %d rows (scale=%d)",
        len(synth_wide), total, scale,
    )

    idx = np.repeat(np.arange(len(synth_wide)), n_copies)
    expanded = synth_wide.iloc[idx].reset_index(drop=True)
    expanded["guid"] = [f"pe_rw_{i:07d}" for i in range(len(expanded))]

    return expanded
