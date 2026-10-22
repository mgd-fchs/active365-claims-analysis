#!/usr/bin/env python3
"""
Python scaffolding around an R did::att_gt pipeline.

- Pull + filter data in DuckDB (age baseline, gender baseline, optional engagement strata)
- Aggregate to person-period panel and construct t_index/g_index
- Optional calendar window restriction
- Either:
  (A) enforce a fully balanced panel 
  (B) keep an unbalanced panel but require minimum pre/post support per individual
- Export did_input.parquet for the R script
- Write stratum_summary.csv and (optional) support_event_time.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_YEAR_DEFAULT = 2000
OUTCOMES_DEFAULT = []
COST_COLS = [
    "total_claim_cost", "inpatient_cost", "outpatient_cost",
    "inpatient_cost_planned", "inpatient_cost_emergency",
    "inpatient_cost_ill", "inpatient_cost_acc",
    "outpatient_cost_ill", "outpatient_cost_acc",
]

def winsorize_costs(
    df: pd.DataFrame,
    *,
    pct: float = 0.99,
    cols: Optional[List[str]] = None,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Upper-tail winsorization of cost outcomes at the person-period level.
    - Cutoff = `pct` quantile among POSITIVE values (costs are zero-inflated).
    - One pooled cutoff per column across treated/control and all periods.
    - Writes <col>_w; originals are left untouched.
    """
    x = df.copy()
    cols = [c for c in (cols or COST_COLS) if c in x.columns]
    rows = []
    for c in cols:
        v = pd.to_numeric(x[c], errors="coerce")
        pos = v[v > 0]
        if pos.empty:
            x[f"{c}_w"] = v
            rows.append({"col": c, "cutoff": np.nan, "n_pos": 0, "n_capped": 0})
            continue
        cut = float(pos.quantile(pct))
        x[f"{c}_w"] = v.clip(upper=cut)
        rows.append({"col": c, "cutoff": round(cut, 2), "n_pos": len(pos), "n_capped": int((v > cut).sum())})

    summary = pd.DataFrame(rows)
    print(f"\n===== WINSORIZATION (p{pct * 100:g} among positives) =====")
    print(summary.to_string(index=False))
    print("===== END WINSORIZATION =====\n")
    if out_path is not None:
        summary.to_csv(out_path, index=False)
    return x

@dataclass
class RunConfig:
    db_path: str
    out_dir: str

    source_table: str
    work_table: str

    age_min: int
    age_max: int
    gender: str  # "both" | "male" | "female"

    granularity: str  # "quarterly" | "half-yearly" | "yearly"
    base_year: int

    min_e: int
    max_e: int

    cal_min: Optional[str]  # YYYY-MM-DD
    cal_max: Optional[str]  # YYYY-MM-DD

    control_group: str  # "notyettreated" | "nevertreated"
    base_period: str  # "universal" | "varying"

    # Panel handling
    unbalanced: bool
    min_pre_periods: int
    min_post_periods: int

    # Optional engagement strat
    engagement_strat: bool
    engagement_definition: Optional[str]
    engagement_granularity: Optional[str]
    engagement_group_max: Optional[int]
    engagement_group_min: Optional[int]

    # BMI
    weight_group: str
    
    # Income
    income_strat: bool
    income_granularity: Optional[str]
    income_group_min: Optional[int]
    income_group_max: Optional[int]

    # Prior healthcare cost
    prior_cost_strat: bool
    prior_cost_definition: Optional[str]   # "total" | "inpatient"
    prior_cost_granularity: Optional[str]  # "tercile" | "quartile" | "decile"
    prior_cost_group_min: Optional[int]
    prior_cost_group_max: Optional[int]

    # Preexisting PCG (binary)
    pcg_strat: bool
    pcg_value: Optional[int]  # 0 or 1

    # Preexisting T2D (binary)
    t2d_strat: bool
    t2d_value: Optional[int]  # 0 or 1

    # Steps-diff category (avg_daily_steps_180 - avg_daily_steps_90)
    steps_diff_strat: bool
    steps_diff_category: Optional[str]  # "increase" | "stable" | "decrease"

    # Roustness
    placebo_time: bool
    winsor_pct: float

    outcomes: List[str]


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="DuckDB prep + panel trimming + run R did::att_gt script.")
    p.add_argument("--db", dest="db_path", required=True, help="Path to DuckDB file.")
    p.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory for CSVs/parquet.")

    p.add_argument(
        "--source-table",
        default="claims_person_month_summary_final_w_start_engagementq",
        help="Immutable source table to copy from each run.",
    )
    p.add_argument(
        "--work-table",
        default="claims_person_month_summary_final_w_start_engagementq",
        help="Working table name (will be replaced).",
    )

    p.add_argument("--age-min", type=int, required=True, help="AGE_STRAT_MIN (inclusive).")
    p.add_argument("--age-max", type=int, required=True, help="AGE_STRAT_MAX (inclusive).")

    p.add_argument("--cal-min", default=None, help="Min calendar date (YYYY-MM-DD) for t_period_start restriction.")
    p.add_argument("--cal-max", default=None, help="Max calendar date (YYYY-MM-DD) for t_period_start restriction.")

    p.add_argument(
        "--gender",
        choices=["both", "male", "female"],
        default="both",
        help="Gender filter. male = M only, female = F only, both = no gender restriction.",
    )

    p.add_argument("--placebo-time", action="store_true", help="Randomly permute treatment timing across treated units.")
    
    p.add_argument(
        "--granularity",
        choices=["quarterly", "half-yearly", "yearly", "monthly"],
        default="quarterly",
        help="Time aggregation for DiD.",
    )
    p.add_argument("--base-year", type=int, default=BASE_YEAR_DEFAULT, help="Base year for t_index/g_index coding.")

    p.add_argument("--min-e", type=int, required=True, help="MIN_E for dynamic event study in R (inclusive).")
    p.add_argument("--max-e", type=int, required=True, help="MAX_E for dynamic event study in R (inclusive).")

    p.add_argument("--control-group", default="notyettreated", choices=["notyettreated", "nevertreated"])
    p.add_argument("--base-period", dest="base_period", choices=["universal", "varying"], default="universal")

    # panel handling flag
    p.add_argument(
        "--unbalanced",
        action="store_true",
        help="If set: do NOT enforce a balanced panel. Instead keep unbalanced but require min pre/post periods per id.",
    )
    p.add_argument("--min-pre-periods", type=int, default=4, help="Min pre-treatment periods required per id (default 4).")
    p.add_argument("--min-post-periods", type=int, default=4, help="Min post-treatment periods required per id (default 4).")

    # Optional engagement stratification
    p.add_argument("--engagement-strat", action="store_true", help="Enable engagement-based stratification.")
    p.add_argument(
        "--engagement-definition",
        type=str,
        choices=["alltime_frequency", "30d_frequency", "alltime_points", "frequency_points", "minimal_engager", "session_user"],
        help="Engagement definition to use (e.g., 30d or alltime).",
    )
    p.add_argument("--engagement-granularity", type=str, choices=["quartile", "decile", "none"], help="Engagement grouping granularity.")
    p.add_argument("--engagement-group-min", type=int, help="Minimum engagement group (inclusive).")
    p.add_argument("--engagement-group-max", type=int, help="Maximum engagement group (inclusive).")

    p.add_argument(
        "--outcomes",
        default=",".join(OUTCOMES_DEFAULT),
        help=f"Comma-separated outcomes. Default: {','.join(OUTCOMES_DEFAULT)}",
    )

    # Income stratum (empirical quantiles within the analyzed subgroup)
    p.add_argument("--income-strat", action="store_true",
                help="Enable income-based stratification using empirical quantiles within the analyzed subgroup.")
    p.add_argument("--income-granularity", choices=["tercile", "quartile", "decile"],
                default="tercile")
    p.add_argument("--income-group-min", type=int, help="Min income quantile (inclusive, 1-based).")
    p.add_argument("--income-group-max", type=int, help="Max income quantile (inclusive, 1-based).")

    # Prior healthcare cost stratum (empirical quantiles within the analyzed subgroup)
    p.add_argument("--prior-cost-strat", action="store_true",
                help="Enable prior-healthcare-cost-based stratification using empirical quantiles within the analyzed subgroup.")
    p.add_argument(
        "--prior-cost-definition",
        choices=["total", "inpatient"],
        default="total",
        help="Which prior-cost column to bin on: 'total' uses prior_total_cost, 'inpatient' uses prior_inpatient_cost.",
    )
    p.add_argument("--prior-cost-granularity", choices=["tercile", "quartile", "decile"],
                default="quartile")
    p.add_argument("--prior-cost-group-min", type=int, help="Min prior-cost quantile (inclusive, 1-based).")
    p.add_argument("--prior-cost-group-max", type=int, help="Max prior-cost quantile (inclusive, 1-based).")

    # Preexisting PCG stratification (binary)
    p.add_argument("--pcg-strat", action="store_true",
                help="Enable stratification by preexisting PCG status (binary).")
    p.add_argument("--pcg-value", type=int, choices=[0, 1],
                help="Which preexisting_pcg value to keep: 0 (no preexisting condition) or 1 (has one).")
    
    # Preexisting T2D stratification (binary)
    p.add_argument("--t2d-strat", action="store_true",
                help="Enable stratification by preexisting T2D status (binary).")
    p.add_argument("--t2d-value", type=int, choices=[0, 1],
                help="Which preexisting_t2d value to keep: 0 (no preexisting condition) or 1 (has one).")

    # Steps-diff stratification (categorical: increase / stable / decrease)
    p.add_argument("--steps-diff-strat", action="store_true",
                help="Enable stratification by steps_diff category (avg_daily_steps_180 - avg_daily_steps_90).")
    p.add_argument(
        "--steps-diff-category",
        type=str,
        choices=["increase", "stable", "decrease"],
        help="Which steps_diff category to keep: increase (>=500), stable (0-499), decrease (<0).",
    )

    # BMI stratification
    p.add_argument(
    "--weight-group",
    choices=["underweight", "normal", "overweight", "obese", "all"],
    default="all",
    help="BMI category filter using WHO cutoffs. Applied at baseline (claim_month==start_month).",
    )

    p.add_argument("--winsor-pct", type=float, default=0.99,
        help="Upper-tail winsorization cutoff for cost outcomes (quantile among positive values, 0-1). "
        "Winsorized columns are written as <col>_w; select them via --outcomes.")

    a = p.parse_args()

    if not 0 < a.winsor_pct < 1:
       raise ValueError("--winsor-pct must be between 0 and 1.")

    outcomes = [x.strip() for x in a.outcomes.split(",") if x.strip()]
    if not outcomes:
        raise ValueError("No outcomes provided.")

    os.makedirs(a.out_dir, exist_ok=True)

    return RunConfig(
        db_path=a.db_path,
        out_dir=a.out_dir,
        source_table=a.source_table,
        work_table=a.work_table,
        age_min=a.age_min,
        age_max=a.age_max,
        gender=a.gender,
        cal_min=a.cal_min,
        cal_max=a.cal_max,
        granularity=a.granularity,
        base_year=a.base_year,
        min_e=a.min_e,
        max_e=a.max_e,
        control_group=a.control_group,
        base_period=a.base_period,
        unbalanced=a.unbalanced,
        min_pre_periods=a.min_pre_periods,
        min_post_periods=a.min_post_periods,
        placebo_time=a.placebo_time,
        engagement_strat=a.engagement_strat,
        engagement_definition=a.engagement_definition,
        engagement_granularity=a.engagement_granularity,
        engagement_group_min=a.engagement_group_min,
        engagement_group_max=a.engagement_group_max,
        weight_group=a.weight_group,
        income_strat=a.income_strat,
        income_granularity=a.income_granularity,
        income_group_min=a.income_group_min,
        income_group_max=a.income_group_max,
        prior_cost_strat=a.prior_cost_strat,
        prior_cost_definition=a.prior_cost_definition,
        prior_cost_granularity=a.prior_cost_granularity,
        prior_cost_group_min=a.prior_cost_group_min,
        prior_cost_group_max=a.prior_cost_group_max,
        pcg_strat=a.pcg_strat,
        pcg_value=a.pcg_value,
        t2d_strat=a.t2d_strat,
        t2d_value=a.t2d_value,
        steps_diff_strat=a.steps_diff_strat,
        steps_diff_category=a.steps_diff_category,
        winsor_pct=a.winsor_pct,
        outcomes=outcomes,
    )


def check_start_month_granularity(con, source_table: str) -> None:
    """
    Diagnostic: are adoption dates (start_month) actually recorded at
    quarterly granularity (always falling on Jan/Apr/Jul/Oct 1), or do
    they vary by calendar month? Prints the sorted unique start_month
    values plus a distinct-month breakdown so this is easy to eyeball
    before deciding whether --granularity monthly would add real
    resolution or just extra empty periods.
    """
    print("\n===== START_MONTH GRANULARITY CHECK =====")

    unique_start_months = con.execute(f"""
        SELECT DISTINCT TRY_CAST(start_month AS DATE) AS start_month
        FROM {source_table}
        WHERE start_month IS NOT NULL
        ORDER BY start_month
    """).fetchdf()

    print(f"Unique start_month values ({len(unique_start_months)} total):")
    print(unique_start_months.to_string(index=False))

    month_breakdown = con.execute(f"""
        SELECT
            EXTRACT(month FROM TRY_CAST(start_month AS DATE))::INTEGER AS start_month_num,
            COUNT(DISTINCT eth_id) AS n_people
        FROM {source_table}
        WHERE start_month IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    print("\nDistinct calendar months represented in start_month, with people counts:")
    print(month_breakdown.to_string(index=False))

    quarter_start_months = {1, 4, 7, 10}
    observed_months = set(month_breakdown["start_month_num"].dropna().astype(int).tolist())

    if observed_months.issubset(quarter_start_months):
        print(
            "\n[RESULT] start_month only ever falls on quarter-start months "
            f"({sorted(quarter_start_months)}). Adoption dates are effectively "
            "quarterly-granular in this data -- switching --granularity to "
            "'monthly' would NOT add real resolution, only more (empty-between) "
            "periods."
        )
    else:
        extra = sorted(observed_months - quarter_start_months)
        print(
            f"\n[RESULT] start_month varies by month (non-quarter-start months present: {extra}). "
            "Adoption dates carry genuine monthly resolution -- 'monthly' granularity "
            "would meaningfully reduce adoption-period contamination."
        )

    print("===== END START_MONTH GRANULARITY CHECK =====\n")


def balance_panel_for_did(
    df: pd.DataFrame,
    id_col: str = "id_num",
    t_col: str = "t_index",
    g_col: str = "g_index",
    required_cols: Optional[List[str]] = None,
    drop_missing_covars: bool = True,
    drop_early_treated: bool = True,
    enforce_balanced_panel: bool = True,
    out_dir: Optional[str] = None,
    tag: str = "",
) -> pd.DataFrame:
    """
    Balanced-panel enforcement.
    Keeps only ids observed at *all* t in the post-filtered sample.
    Also drops "early treated" ids with g <= min(t) if requested.
    """
    x = df.copy()
    if required_cols is None:
        required_cols = []

    if drop_missing_covars and required_cols:
        before = len(x)
        x = x.dropna(subset=required_cols).copy()
        if out_dir:
            x.to_parquet(os.path.join(out_dir, f"df_after_dropna{tag}.parquet"), index=False)
        print(f"[BALANCE] dropna({required_cols}): {before} -> {len(x)} rows")

    before = len(x)
    x = x.drop_duplicates(subset=[id_col, t_col]).copy()
    print(f"[BALANCE] drop_duplicates({id_col},{t_col}): {before} -> {len(x)} rows")

    g_nunique = x.groupby(id_col)[g_col].nunique(dropna=True)
    bad_ids = g_nunique[g_nunique > 1].index.tolist()
    if bad_ids:
        if out_dir:
            pd.DataFrame({id_col: bad_ids}).to_csv(os.path.join(out_dir, f"bad_ids_multiple_g{tag}.csv"), index=False)
        raise ValueError(f"[BALANCE] Found {len(bad_ids)} ids with multiple non-NA {g_col} values. Fix g_index construction.")

    if drop_early_treated:
        min_t = int(x[t_col].min())
        first_g = (
            x.loc[x[g_col].notna(), [id_col, g_col]]
            .groupby(id_col)[g_col]
            .min()
        )
        drop_ids = first_g[first_g <= min_t].index
        print(f"[BALANCE] early-treated drop (g <= min_t={min_t}): {len(drop_ids)} ids")
        if len(drop_ids) > 0:
            if out_dir:
                pd.DataFrame({id_col: drop_ids}).to_csv(os.path.join(out_dir, f"dropped_early_treated_ids{tag}.csv"), index=False)
            x = x[~x[id_col].isin(drop_ids)].copy()

    if enforce_balanced_panel:
        all_t = np.sort(x[t_col].unique())
        T = len(all_t)
        counts = x.groupby(id_col)[t_col].nunique()
        keep_ids = counts[counts == T].index
        drop_ids = counts[counts != T].index

        print(f"[BALANCE] balanced panel keep: {len(keep_ids)} ids; drop: {len(drop_ids)} ids; T={T} periods")
        if len(drop_ids) > 0 and out_dir:
            pd.DataFrame({id_col: drop_ids, "n_t": counts.loc[drop_ids].values}).to_csv(
                os.path.join(out_dir, f"dropped_unbalanced_ids{tag}.csv"), index=False
            )

        x = x[x[id_col].isin(keep_ids)].copy()

    x = x.sort_values([id_col, t_col]).reset_index(drop=True)

    treated_ids = (
        x.loc[x[g_col].notna(), [id_col, g_col]]
        .drop_duplicates(subset=[id_col])[id_col]
        .nunique()
    )
    print(f"[BALANCE] FINAL: rows={len(x)} ids={x[id_col].nunique()} periods={x[t_col].nunique()} treated_ids={treated_ids}")

    if out_dir:
        x.to_parquet(os.path.join(out_dir, f"df_balanced{tag}.parquet"), index=False)

    return x

def summarize_final_did_sample(
    df: pd.DataFrame,
    *,
    id_col: str = "id_num",
    t_col: str = "t_index",
    g_col: str = "g_index",
    gender_col: str = "gender_num",
    age_col: str = "age",
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize the final DiD sample (df AFTER panel trimming).

    Outputs:
      - n_unique_person_periods: number of unique (id, t) rows (person-quarters / person-periods)
      - n_unique_people: number of unique ids
      - gender counts + percentages among unique people (gender_num: 0=F, 1=M, NA=unknown/other)
      - age stats (mean/median/sd) among treated people at event_time == 0 (t_index == g_index)

    Notes:
      - "person-quarters" here means unique person-period observations, i.e. unique (id_col, t_col).
      - Age at eventtime 0 is computed on treated ids with an observed row at t==g.
        (If some treated people have no event_time==0 row after trimming/windowing, they are excluded.)
    """
    x = df.copy()

    # Ensure unique person-periods in the summary
    x = x.drop_duplicates(subset=[id_col, t_col]).copy()

    # 1) Unique person-periods and people
    n_unique_person_periods = int(x[[id_col, t_col]].drop_duplicates().shape[0])
    n_unique_people = int(x[id_col].nunique())

    # 2) Gender distribution among unique people (use one record per id)
    # Prefer baseline gender if available; here we take the first non-null by time.
    per_id = (
        x.sort_values([id_col, t_col])
         .groupby(id_col, as_index=False)
         .agg(
             gender_num=(gender_col, lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
         )
    )

    # Map to labels for readability
    def _gender_label(v):
        if pd.isna(v):
            return "unknown"
        if int(v) == 0:
            return "female"
        if int(v) == 1:
            return "male"
        return "unknown"

    per_id["gender_label"] = per_id["gender_num"].apply(_gender_label)

    gender_counts = per_id["gender_label"].value_counts(dropna=False)
    gender_perc = (gender_counts / n_unique_people * 100.0).round(2)

    # 3) Age stats at event time 0 (treated only, row where t==g)
    # event_time 0 means t_index == g_index
    et0 = x[(x[g_col].notna()) & (x[t_col] == x[g_col])].copy()

    # One row per treated id at event time 0 (in case of duplicates)
    et0 = et0.sort_values([id_col, t_col]).drop_duplicates(subset=[id_col]).copy()

    ages = pd.to_numeric(et0[age_col], errors="coerce").dropna()
    age_mean = float(ages.mean()) if len(ages) else np.nan
    age_median = float(ages.median()) if len(ages) else np.nan
    age_sd = float(ages.std(ddof=1)) if len(ages) else np.nan
    n_treated_with_et0 = int(et0[id_col].nunique())

    # Build a single-row summary table
    row: Dict[str, object] = {
        "n_unique_person_periods": n_unique_person_periods,
        "n_unique_people": n_unique_people,
        "n_treated_with_event_time0_row": n_treated_with_et0,
        "age_et0_mean": round(age_mean, 4) if pd.notna(age_mean) else np.nan,
        "age_et0_median": round(age_median, 4) if pd.notna(age_median) else np.nan,
        "age_et0_sd": round(age_sd, 4) if pd.notna(age_sd) else np.nan,
    }

    # Add gender counts + percentages
    for lbl in ["female", "male", "unknown"]:
        c = int(gender_counts.get(lbl, 0))
        p = float(gender_perc.get(lbl, 0.0))
        row[f"gender_{lbl}_n"] = c
        row[f"gender_{lbl}_pct"] = p

    out = pd.DataFrame([row])

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)

    return out

def summarize_preadoption_costs(
    df: pd.DataFrame,
    *,
    id_col: str = "id_num",
    t_col: str = "t_index",
    g_col: str = "g_index",
    pre_window: Tuple[int, int] = (-4, -1),
    outcome_cols: Tuple[str, ...] = ("total_claim_cost", "inpatient_cost"),
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each treated id, compute the mean per-period value over event_time in
    [pre_window[0], pre_window[1]] for each outcome in outcome_cols.
    Then report the cross-person mean, SD, and n for each outcome — one row
    per outcome (long format).

    Returns a DataFrame with columns:
        outcome, preadoption_mean, preadoption_sd, n_treated
    """
    x = df.copy()
    x["event_time"] = pd.to_numeric(x[t_col], errors="coerce") - pd.to_numeric(x[g_col], errors="coerce")

    pre = x[
        x[g_col].notna()
        & (x["event_time"] >= pre_window[0])
        & (x["event_time"] <= pre_window[1])
    ].copy()

    if pre.empty:
        raise ValueError(
            f"No treated pre-adoption observations found in event_time window {pre_window}."
        )

    for col in outcome_cols:
        pre[col] = pd.to_numeric(pre[col], errors="coerce")

    # Per-person mean per-period value over the pre-window, for each outcome
    person_means = pre.groupby(id_col)[list(outcome_cols)].mean()

    rows = []
    for col in outcome_cols:
        vals = person_means[col].dropna()
        rows.append({
            "outcome": col,
            "preadoption_mean": round(float(vals.mean()), 4) if len(vals) else np.nan,
            "preadoption_sd": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else np.nan,
            "n_treated": int(vals.shape[0]),
        })

    out = pd.DataFrame(rows)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)

    return out


def trim_unbalanced_min_pre_post(
    df: pd.DataFrame,
    *,
    id_col: str = "id_num",
    t_col: str = "t_index",
    g_col: str = "g_index",
    required_cols: Optional[List[str]] = None,
    drop_missing_covars: bool = True,
    drop_early_treated: bool = False,
    min_pre: int = 4,
    min_post: int = 4,
    out_dir: Optional[str] = None,
    tag: str = "",
) -> pd.DataFrame:
    """
    Unbalanced-panel trimming:
    - Does NOT enforce full balance.
    - Requires per-id support:
        * If treated (g non-NA): at least min_pre periods with t < g AND at least min_post periods with t > g.
        * If never-treated (g NA): require at least (min_pre + min_post) observed periods total.
    Also handles:
    - optional dropna on required_cols
    - optional early-treated drop (usually redundant if min_pre >= 1)
    Writes debug drops if out_dir is provided.
    """
    x = df.copy()
    if required_cols is None:
        required_cols = []

    if drop_missing_covars and required_cols:
        before = len(x)
        x = x.dropna(subset=required_cols).copy()
        if out_dir:
            x.to_parquet(os.path.join(out_dir, f"df_after_dropna{tag}.parquet"), index=False)
        print(f"[UNBAL] dropna({required_cols}): {before} -> {len(x)} rows")

    before = len(x)
    x = x.drop_duplicates(subset=[id_col, t_col]).copy()
    print(f"[UNBAL] drop_duplicates({id_col},{t_col}): {before} -> {len(x)} rows")

    n_periods_per_id = x.groupby(id_col)[t_col].nunique()
    print(
        f"[UNBAL] periods/person -> "
        f"mean={n_periods_per_id.mean():.2f}, "
        f"median={n_periods_per_id.median():.1f}, "
        f"min={n_periods_per_id.min()}, "
        f"max={n_periods_per_id.max()}"
    )

    # sanity: stable g within id among treated
    g_nunique = x.groupby(id_col)[g_col].nunique(dropna=True)
    bad_ids = g_nunique[g_nunique > 1].index.tolist()
    if bad_ids:
        if out_dir:
            pd.DataFrame({id_col: bad_ids}).to_csv(os.path.join(out_dir, f"bad_ids_multiple_g{tag}.csv"), index=False)
        raise ValueError(f"[UNBAL] Found {len(bad_ids)} ids with multiple non-NA {g_col} values. Fix g_index construction.")

    if drop_early_treated:
        min_t = int(x[t_col].min())
        first_g = (
            x.loc[x[g_col].notna(), [id_col, g_col]]
            .groupby(id_col)[g_col]
            .min()
        )
        drop_ids = first_g[first_g <= min_t].index
        print(f"[UNBAL] early-treated drop (g <= min_t={min_t}): {len(drop_ids)} ids")
        if len(drop_ids) > 0:
            if out_dir:
                pd.DataFrame({id_col: drop_ids}).to_csv(os.path.join(out_dir, f"dropped_early_treated_ids{tag}.csv"), index=False)
            x = x[~x[id_col].isin(drop_ids)].copy()

    # compute per-id support
    # (use the id-level g as min non-NA g, which should be unique by sanity check)
    id_g = (
        x.loc[x[g_col].notna(), [id_col, g_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[g_col]
    )

    # total observed periods per id
    n_obs = x.groupby(id_col)[t_col].nunique()

    # treated: pre/post counts relative to g
    treated = x[x[g_col].notna()].copy()
    treated["g_id"] = treated[id_col].map(id_g)
    pre_counts = treated[treated[t_col] < treated["g_id"]].groupby(id_col)[t_col].nunique()
    post_counts = treated[treated[t_col] > treated["g_id"]].groupby(id_col)[t_col].nunique()

    # align with ids
    treated_ids = id_g.index
    pre_counts = pre_counts.reindex(treated_ids).fillna(0).astype(int)
    post_counts = post_counts.reindex(treated_ids).fillna(0).astype(int)

    keep_treated_ids = treated_ids[(pre_counts >= min_pre) & (post_counts >= min_post)]
    drop_treated_ids = treated_ids.difference(keep_treated_ids)

    pre_stats = pre_counts.loc[keep_treated_ids]
    post_stats = post_counts.loc[keep_treated_ids]
    print(
        f"[UNBAL] pre-adoption periods (kept treated) -> "
        f"mean={pre_stats.mean():.2f}, median={pre_stats.median():.1f}, "
        f"min={pre_stats.min()}, max={pre_stats.max()}"
    )
    print(
        f"[UNBAL] post-adoption periods (kept treated) -> "
        f"mean={post_stats.mean():.2f}, median={post_stats.median():.1f}, "
        f"min={post_stats.min()}, max={post_stats.max()}"
    )

    # never-treated ids: require enough total periods
    never_ids = n_obs.index.difference(treated_ids)
    keep_never_ids = never_ids[n_obs.reindex(never_ids).fillna(0).astype(int) >= (min_pre + min_post)]
    drop_never_ids = never_ids.difference(keep_never_ids)
    keep_ids = keep_treated_ids.union(keep_never_ids)
    print(f"[UNBAL] min_pre={min_pre} min_post={min_post}")
    print(f"[UNBAL] treated keep={len(keep_treated_ids)} drop={len(drop_treated_ids)}")
    print(f"[UNBAL] never  keep={len(keep_never_ids)} drop={len(drop_never_ids)}")
    
    if out_dir:
        # detailed drop reasons
        if len(drop_treated_ids) > 0:
            dd = pd.DataFrame(
                {
                    id_col: drop_treated_ids,
                    "pre_periods": pre_counts.reindex(drop_treated_ids).astype(int).values,
                    "post_periods": post_counts.reindex(drop_treated_ids).astype(int).values,
                    "n_periods_total": n_obs.reindex(drop_treated_ids).astype(int).values,
                }
            )
            dd.to_csv(os.path.join(out_dir, f"dropped_treated_minprepost{tag}.csv"), index=False)
        if len(drop_never_ids) > 0:
            dn = pd.DataFrame(
                {
                    id_col: drop_never_ids,
                    "n_periods_total": n_obs.reindex(drop_never_ids).astype(int).values,
                }
            )
            dn.to_csv(os.path.join(out_dir, f"dropped_never_minperiods{tag}.csv"), index=False)

    x = x[x[id_col].isin(keep_ids)].copy()
    x = x.sort_values([id_col, t_col]).reset_index(drop=True)

    treated_ids_final = (
        x.loc[x[g_col].notna(), [id_col, g_col]]
        .drop_duplicates(subset=[id_col])[id_col]
        .nunique()
    )
    print(f"[UNBAL] FINAL: rows={len(x)} ids={x[id_col].nunique()} periods={x[t_col].nunique()} treated_ids={treated_ids_final}")

    if out_dir:
        x.to_parquet(os.path.join(out_dir, f"df_unbalanced_trimmed{tag}.parquet"), index=False)

    return x


def build_person_period_sql(
    work_table: str,
    granularity: str,
    base_year: int,
    dest_table: str = "claims_person_period_summary_w_start_idx",
    treatment_start_shift_periods: int = 0,
) -> str:
    g = granularity.lower().strip()
    if g not in {"monthly", "quarterly", "half-yearly", "yearly"}:
        raise ValueError("granularity must be one of: monthly, quarterly, half-yearly, yearly")

    if treatment_start_shift_periods < 0:
        raise ValueError("treatment_start_shift_periods must be >= 0")

    k = {
        "yearly": 1,
        "half-yearly": 2,
        "quarterly": 4,
        "monthly": 12,
    }[g]

    if g == "yearly":
        t_year_expr = "EXTRACT(year FROM d_claim)::INTEGER"
        t_sub_expr = "1"
        t_start_expr = "DATE_TRUNC('year', d_claim)"

        # effective treatment start shifted forward by N years
        g_date_shifted_expr = (
            f"DATE_TRUNC('year', d_start) + INTERVAL '{treatment_start_shift_periods} year'"
        )
        g_year_expr = f"EXTRACT(year FROM ({g_date_shifted_expr}))::INTEGER"
        g_sub_expr = "1"
        g_start_expr = f"DATE_TRUNC('year', ({g_date_shifted_expr}))"

    elif g == "half-yearly":
        t_year_expr = "EXTRACT(year FROM d_claim)::INTEGER"
        t_sub_expr = "CASE WHEN EXTRACT(month FROM d_claim)::INTEGER <= 6 THEN 1 ELSE 2 END"
        t_start_expr = (
            "MAKE_DATE(EXTRACT(year FROM d_claim)::INTEGER, "
            "CASE WHEN EXTRACT(month FROM d_claim)::INTEGER <= 6 THEN 1 ELSE 7 END, 1)"
        )

        # shift by 6-month blocks
        g_date_shifted_expr = (
            f"DATE_TRUNC('month', d_start) + INTERVAL '{6 * treatment_start_shift_periods} month'"
        )
        g_year_expr = f"EXTRACT(year FROM ({g_date_shifted_expr}))::INTEGER"
        g_sub_expr = (
            f"CASE WHEN EXTRACT(month FROM ({g_date_shifted_expr}))::INTEGER <= 6 THEN 1 ELSE 2 END"
        )
        g_start_expr = (
            f"MAKE_DATE(EXTRACT(year FROM ({g_date_shifted_expr}))::INTEGER, "
            f"CASE WHEN EXTRACT(month FROM ({g_date_shifted_expr}))::INTEGER <= 6 THEN 1 ELSE 7 END, 1)"
        )

    elif g == "quarterly":
        t_year_expr = "EXTRACT(year FROM d_claim)::INTEGER"
        t_sub_expr = "EXTRACT(quarter FROM d_claim)::INTEGER"
        t_start_expr = "DATE_TRUNC('quarter', d_claim)"

        # shift by 3-month blocks
        g_date_shifted_expr = (
            f"DATE_TRUNC('month', d_start) + INTERVAL '{3 * treatment_start_shift_periods} month'"
        )
        g_year_expr = f"EXTRACT(year FROM ({g_date_shifted_expr}))::INTEGER"
        g_sub_expr = f"EXTRACT(quarter FROM ({g_date_shifted_expr}))::INTEGER"
        g_start_expr = f"DATE_TRUNC('quarter', ({g_date_shifted_expr}))"

    else:  # monthly
        t_year_expr = "EXTRACT(year FROM d_claim)::INTEGER"
        t_sub_expr = "EXTRACT(month FROM d_claim)::INTEGER"
        t_start_expr = "DATE_TRUNC('month', d_claim)"

        # shift by calendar months
        g_date_shifted_expr = (
            f"DATE_TRUNC('month', d_start) + INTERVAL '{treatment_start_shift_periods} month'"
        )
        g_year_expr = f"EXTRACT(year FROM ({g_date_shifted_expr}))::INTEGER"
        g_sub_expr = f"EXTRACT(month FROM ({g_date_shifted_expr}))::INTEGER"
        g_start_expr = f"DATE_TRUNC('month', ({g_date_shifted_expr}))"

    return f"""
    DROP TABLE IF EXISTS {dest_table};

    CREATE TABLE {dest_table} AS
    WITH base AS (
      SELECT
        eth_id,
        start_month,
        TRY_CAST(claim_month AS DATE) AS d_claim,
        TRY_CAST(start_month AS DATE) AS d_start,

        COALESCE(n_inpatient_claims, 0) AS n_inpatient_claims,
        COALESCE(n_outpatient_claims, 0) AS n_outpatient_claims,
        COALESCE(total_claim_cost, 0) AS total_claim_cost,
        COALESCE(inpatient_cost, 0) AS inpatient_cost,
        COALESCE(outpatient_cost, 0) AS outpatient_cost,
        COALESCE(n_inpatient_emergency, 0) AS n_inpatient_emergency,
        COALESCE(n_inpatient_planned,   0) AS n_inpatient_planned,
        COALESCE(inpatient_cost_planned,   0) AS inpatient_cost_planned,
        COALESCE(inpatient_cost_emergency,   0) AS inpatient_cost_emergency,
        COALESCE(n_inpatient_ill,                  0) AS n_inpatient_ill,
        COALESCE(n_inpatient_acc,                  0) AS n_inpatient_acc,
        COALESCE(inpatient_cost_ill,               0) AS inpatient_cost_ill,
        COALESCE(inpatient_cost_acc,               0) AS inpatient_cost_acc,
        COALESCE(n_outpatient_ill,                 0) AS n_outpatient_ill,
        COALESCE(n_outpatient_acc,                 0) AS n_outpatient_acc,
        COALESCE(outpatient_cost_ill,              0) AS outpatient_cost_ill,
        COALESCE(outpatient_cost_acc,              0) AS outpatient_cost_acc,
        franchise,

        age,
        gender
      FROM {work_table}
      WHERE TRY_CAST(claim_month AS DATE) IS NOT NULL
    ),
    labeled AS (
      SELECT
        eth_id,

        {t_start_expr} AS t_period_start,
        {t_year_expr}  AS t_year,
        {t_sub_expr}   AS t_sub,

        CASE
          WHEN start_month IS NULL OR d_start IS NULL THEN NULL
          ELSE {g_start_expr}
        END AS g_period_start,

        CASE
          WHEN start_month IS NULL OR d_start IS NULL THEN NULL
          ELSE {g_year_expr}
        END AS g_year,

        CASE
          WHEN start_month IS NULL OR d_start IS NULL THEN NULL
          ELSE {g_sub_expr}
        END AS g_sub,

        n_inpatient_claims,
        n_outpatient_claims,
        total_claim_cost,
        inpatient_cost,
        outpatient_cost,
        n_inpatient_emergency,
        n_inpatient_planned,
        inpatient_cost_planned,
        inpatient_cost_emergency,
        n_inpatient_ill,
        n_inpatient_acc,
        inpatient_cost_ill,
        inpatient_cost_acc,
        n_outpatient_ill,
        n_outpatient_acc,
        outpatient_cost_ill,
        outpatient_cost_acc,
        franchise,
        age,
        gender
      FROM base
    ),
    agg AS (
      SELECT
        eth_id,
        t_period_start,
        t_year,
        t_sub,

        MIN(g_year) AS g_year,
        MIN(g_sub)  AS g_sub,

        SUM(n_inpatient_claims) AS n_inpatient_claims,
        SUM(n_outpatient_claims) AS n_outpatient_claims,
        SUM(total_claim_cost) AS total_claim_cost,
        SUM(inpatient_cost) AS inpatient_cost,
        SUM(outpatient_cost) AS outpatient_cost,
        SUM(n_inpatient_emergency) AS n_inpatient_emergency,
        SUM(n_inpatient_planned)   AS n_inpatient_planned,
        SUM(inpatient_cost_planned)   AS inpatient_cost_planned,
        SUM(inpatient_cost_emergency)   AS inpatient_cost_emergency,
        SUM(n_inpatient_ill)                   AS n_inpatient_ill,
        SUM(n_inpatient_acc)                   AS n_inpatient_acc,
        SUM(inpatient_cost_ill)                AS inpatient_cost_ill,
        SUM(inpatient_cost_acc)                AS inpatient_cost_acc,
        SUM(n_outpatient_ill)                  AS n_outpatient_ill,
        SUM(n_outpatient_acc)                  AS n_outpatient_acc,
        SUM(outpatient_cost_ill)               AS outpatient_cost_ill,
        SUM(outpatient_cost_acc)               AS outpatient_cost_acc,
        MAX(franchise) AS franchise,

        MAX(age) AS age,
        MAX(gender) AS gender
      FROM labeled
      GROUP BY eth_id, t_period_start, t_year, t_sub
    )
    SELECT
      *,
      (t_year * 100 + t_sub) AS t_period_code,
      CASE
        WHEN g_year IS NULL OR g_sub IS NULL THEN NULL
        ELSE (g_year * 100 + g_sub)
      END AS g_period_code,

      ((t_year - {base_year}) * {k} + (t_sub - 1)) AS t_index,
      CASE
        WHEN g_year IS NULL OR g_sub IS NULL THEN NULL
        ELSE ((g_year - {base_year}) * {k} + (g_sub - 1))
      END AS g_index
    FROM agg;
    """

def assign_placebo_g_index(
    df: pd.DataFrame,
    *,
    id_col: str = "id_num",
    g_col: str = "g_index",
    seed: int = 123,
) -> pd.DataFrame:
    x = df.copy()

    treated_map = (
        x.loc[x[g_col].notna(), [id_col, g_col]]
        .drop_duplicates(subset=[id_col])
        .copy()
    )

    chk = treated_map.groupby(id_col)[g_col].nunique()
    if (chk > 1).any():
        raise ValueError("g_index is not unique within id.")

    rng = np.random.default_rng(seed)
    placebo_g = rng.permutation(treated_map[g_col].to_numpy())

    placebo_map = treated_map[[id_col]].copy()
    placebo_map[g_col] = placebo_g

    never_ids = x.loc[x[g_col].isna(), [id_col]].drop_duplicates()

    x = x.drop(columns=[g_col])
    x = x.merge(placebo_map, on=id_col, how="left")

    return x

def build_window_sql(
    source_table: str,
    dest_table: str = "claims_person_period_summary_w_start_idx_windowed",
) -> str:
    # Keep all rows; this is just a placeholder if you later want to add true windowing.
    return f"""
    DROP TABLE IF EXISTS {dest_table};
    CREATE TABLE {dest_table} AS
    SELECT *, (t_index - g_index) AS event_time
    FROM {source_table};
    """

def export_support_by_event_time(
    df_input: pd.DataFrame,
    *,
    min_e: int,
    max_e: int,
    out_path: str,
    control_group: str,
    fig_out_path: str | None = None,
    id_col: str = "id_num",
    t_col: str = "t_index",
    g_col: str = "g_index",
    age_col: str = "age",
    female_col: str = "gender_num",
    calendar_time_label_col: str | None = None,  # kept for API compatibility, ignored for plotting
    periods_per_year: int = 4,
    base_year: int = 2000,
) -> None:
    """
    Export support summary by event time, and optionally export a 3-panel figure
    showing composition by calendar time for the observations that are eligible
    in the treated and control groups.

    CSV output:
    - treated_n_ids / treated_n_obs by event_time in [min_e, max_e]
    - control availability at calendar times contributing to those event_times:
        * notyettreated: g is NA OR g > t
        * nevertreated:  g is NA

    Optional figure output (if fig_out_path is not None):
    - unique individuals by calendar time
    - % female by calendar time
    - median age by calendar time

    Notes
    -----
    - Assumes df_input is already the trimmed / balanced analysis sample.
    - Group composition plots are based on the actual observations eligible at each
      calendar time under the requested event window and control-group definition.
    - Plot formatting is aligned to the previously used calendar-time plotting style.
    """
    
    color_map = {
        "treated": "#5b2a86",
        "not_yet_treated": "#8b1e3f",
    }
    if control_group not in {"notyettreated", "nevertreated"}:
        raise ValueError("control_group must be either 'notyettreated' or 'nevertreated'")

    required_cols = {id_col, t_col, g_col, age_col, female_col}
    missing = required_cols - set(df_input.columns)
    if missing:
        raise ValueError(f"df_input is missing required columns: {sorted(missing)}")

    x = df_input.copy()

    # Numeric coercion
    for c in [t_col, age_col, female_col]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x[g_col] = pd.to_numeric(x[g_col], errors="coerce")

    # Event time
    x["event_time"] = x[t_col] - x[g_col]

    # Treated observations inside requested event window
    treated = x.loc[
        x[g_col].notna(),
        [id_col, t_col, g_col, "event_time", age_col, female_col],
    ].copy()

    treated["event_time"] = pd.to_numeric(treated["event_time"], errors="coerce")
    treated = treated[
        treated["event_time"].notna()
        & (treated["event_time"] >= min_e)
        & (treated["event_time"] <= max_e)
    ].copy()
    treated["event_time"] = treated["event_time"].astype(int)

    # Eligible controls
    ctrl = x.loc[:, [id_col, t_col, g_col, age_col, female_col]].copy()
    if control_group == "notyettreated":
        ctrl = ctrl[ctrl[g_col].isna() | (ctrl[g_col] > ctrl[t_col])].copy()
    else:
        ctrl = ctrl[ctrl[g_col].isna()].copy()

    # ------------------------------------------------------------------
    # 1) CSV export: support by event time
    # ------------------------------------------------------------------
    treated_support = (
        treated.groupby("event_time")
        .agg(
            treated_n_ids=(id_col, "nunique"),
            treated_n_obs=(id_col, "size"),
        )
        .reset_index()
    )

    t_sets = treated.groupby("event_time")[t_col].unique().to_dict()
    rows = []
    for e, t_list in t_sets.items():
        csub = ctrl[ctrl[t_col].isin(t_list)]
        rows.append(
            {
                "event_time": int(e),
                "control_n_ids": int(csub[id_col].nunique()),
                "control_n_obs": int(len(csub)),
                "n_calendar_periods": int(len(np.unique(t_list))),
                "total_unique_people": int(x[id_col].nunique()),
            }
        )

    control_support = pd.DataFrame(rows)

    support = (
        treated_support.merge(control_support, on="event_time", how="left")
        .sort_values("event_time")
        .reset_index(drop=True)
    )

    support.to_csv(out_path, index=False)

    # Stop here if no figure requested
    if fig_out_path is None:
        return

    # Keep only calendar periods that actually contribute to the requested event window
    included_t = np.sort(treated[t_col].dropna().unique())
    if len(included_t) == 0:
        raise ValueError("No treated observations fall inside the requested event-time window.")

    treated_plot = treated[treated[t_col].isin(included_t)].copy()
    ctrl_plot = ctrl[ctrl[t_col].isin(included_t)].copy()

    # One row per person per calendar time
    treated_plot = treated_plot[[id_col, t_col, age_col, female_col]].drop_duplicates([id_col, t_col])
    ctrl_plot = ctrl_plot[[id_col, t_col, age_col, female_col]].drop_duplicates([id_col, t_col])

    def _summarize_by_t(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[t_col, "n_people", "pct_female", "median_age", "group"])

        out = (
            df.groupby(t_col)
            .agg(
                n_people=(id_col, "nunique"),
                pct_female=(female_col, lambda s: 100 * pd.to_numeric(s, errors="coerce").mean()),
                median_age=(age_col, lambda s: pd.to_numeric(s, errors="coerce").median()),
            )
            .reset_index()
        )
        out["group"] = group_name
        return out

    treated_summary = _summarize_by_t(treated_plot, "treated")
    ctrl_group_name = "not_yet_treated" if control_group == "notyettreated" else "never_treated"
    ctrl_summary = _summarize_by_t(ctrl_plot, ctrl_group_name)

    plot_df = pd.concat([treated_summary, ctrl_summary], ignore_index=True)

    # ------------------------------------------------------------------
    # 2) Figure export: calendar-time composition
    # ------------------------------------------------------------------
    if fig_out_path is None:
        return

    # Calendar periods touched by the requested event-time window
    included_t = np.sort(treated[t_col].dropna().unique())
    if len(included_t) == 0:
        raise ValueError("No treated observations fall inside the requested event-time window.")

    # IMPORTANT:
    # For the figure, "treated" should mean actually treated at calendar time t,
    # not merely ever-treated with event_time inside [min_e, max_e].
    treated_plot = x.loc[
        x[g_col].notna()
        & (x[t_col] >= x[g_col])
        & (x[t_col].isin(included_t)),
        [id_col, t_col, age_col, female_col],
    ].copy()

    ctrl_plot = ctrl.loc[
        ctrl[t_col].isin(included_t),
        [id_col, t_col, age_col, female_col],
    ].copy()

    # One row per person per calendar time
    treated_plot = treated_plot.drop_duplicates([id_col, t_col])
    ctrl_plot = ctrl_plot.drop_duplicates([id_col, t_col])

    def _summarize_by_t(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[t_col, "n_people", "pct_female", "median_age", "group"])

        out = (
            df.groupby(t_col)
            .agg(
                n_people=(id_col, "nunique"),
                pct_female=(female_col, lambda s: 100 * pd.to_numeric(s, errors="coerce").mean()),
                median_age=(age_col, lambda s: pd.to_numeric(s, errors="coerce").median()),
            )
            .reset_index()
        )
        out["group"] = group_name
        return out

    treated_summary = _summarize_by_t(treated_plot, "treated")
    ctrl_group_name = "not_yet_treated" if control_group == "notyettreated" else "never_treated"
    ctrl_summary = _summarize_by_t(ctrl_plot, ctrl_group_name)

    plot_df = pd.concat([treated_summary, ctrl_summary], ignore_index=True)

    # Granularity-aware tick placement/labeling.
    # periods_per_year=4 -> quarterly (t_index steps of 4 per year),
    # periods_per_year=2 -> half-yearly, =1 -> yearly, =12 -> monthly.
    def period_to_year(t):
        return base_year + (int(t) // periods_per_year)

    q1_ticks = sorted(plot_df.loc[plot_df[t_col] % periods_per_year == 0, t_col].dropna().unique())
    q1_labels = [period_to_year(t) for t in q1_ticks]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

    for g, sub in plot_df.groupby("group"):
        sub = sub.sort_values(t_col)
        axes[0].plot(sub[t_col], sub["n_people"], marker="o", label=g, color=color_map.get(g, "black"))
    axes[0].set_title("Support by calendar time")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Unique individuals")
    axes[0].legend()
    axes[0].set_ylim(0, 110000)

    for g, sub in plot_df.groupby("group"):
        sub = sub.sort_values(t_col)
        axes[1].plot(sub[t_col], sub["pct_female"], marker="o", label=g, color=color_map.get(g, "black"))
    axes[1].set_title("% female by calendar time")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("% female")
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    for g, sub in plot_df.groupby("group"):
        sub = sub.sort_values(t_col)
        axes[2].plot(sub[t_col], sub["median_age"], marker="o", label=g, color=color_map.get(g, "black"))
    axes[2].set_title("Median age by calendar time")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Median age")
    axes[2].set_ylim(30, 50)
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(q1_ticks)
        ax.set_xticklabels(q1_labels, rotation=0)

    plt.tight_layout()
    fig.savefig(fig_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def subgroup_check(con, run_table: str, cfg: RunConfig, label: str) -> None:
    check = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT eth_id) AS n_ids,
            MIN(age) AS min_age,
            MAX(age) AS max_age,
            STRING_AGG(DISTINCT CAST(gender AS VARCHAR), ', ') AS genders
        FROM {run_table}
    """).fetchdf()

    print(f"\n===== {label} =====")
    print(check.to_string(index=False))
    print("=" * (len(label) + 12))

    if int(check["n_rows"].iloc[0]) == 0:
        raise ValueError(f"{label}: filter produced zero rows.")


def assert_subgroup_filter(con, run_table: str, cfg: RunConfig) -> None:
    # Check baseline age, not all period ages
    age_check = con.execute(f"""
        SELECT
            MIN(age) AS min_baseline_age,
            MAX(age) AS max_baseline_age
        FROM {run_table}
        WHERE TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
    """).fetchdf()

    min_age = age_check["min_baseline_age"].iloc[0]
    max_age = age_check["max_baseline_age"].iloc[0]

    if pd.isna(min_age) or pd.isna(max_age):
        raise ValueError("No baseline rows found after filtering.")

    if min_age < cfg.age_min or max_age > cfg.age_max:
        raise ValueError(
            f"Age filter failed. Expected baseline age {cfg.age_min}-{cfg.age_max}, "
            f"got {min_age}-{max_age}."
        )

    if cfg.gender != "both":
        expected = "M" if cfg.gender == "male" else "F"

        observed = set(
            con.execute(f"""
                SELECT DISTINCT UPPER(TRIM(CAST(gender AS VARCHAR))) AS gender
                FROM {run_table}
                WHERE gender IS NOT NULL
            """).fetchdf()["gender"].dropna()
        )

        if observed != {expected}:
            raise ValueError(
                f"Gender filter failed. Expected only {expected}, got {observed}."
            )

    print("\n===== SUBGROUP FILTER ASSERTION PASSED =====")
    print(f"gender={cfg.gender}, age={cfg.age_min}-{cfg.age_max}")
    print("===========================================\n")


def main() -> None:
    cfg = parse_args()
    con = duckdb.connect(cfg.db_path)

    # NEW: check whether adoption dates (start_month) are truly quarterly
    # or actually vary by month, before doing anything else.
    check_start_month_granularity(con, cfg.source_table)

    run_table = f"{cfg.work_table}__run_tmp"

    # Important: avoid TEMP/persistent table ambiguity
    con.execute(f"DROP TABLE IF EXISTS {run_table}")
    con.execute(f"DROP TABLE IF EXISTS temp.{run_table}")

    con.execute(f"""
        CREATE TABLE {run_table} AS
        SELECT *
        FROM {cfg.source_table}
    """)

    subgroup_check(con, run_table, cfg, "INITIAL FULL SOURCE COPY")
    
    # 1) Baseline age filter
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {run_table} AS
        WITH eligible AS (
            SELECT DISTINCT eth_id
            FROM {run_table}
            WHERE eth_id IS NOT NULL
              AND TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
              AND age BETWEEN ? AND ?
        )
        SELECT t.*
        FROM {run_table} t
        INNER JOIN eligible e
            ON t.eth_id = e.eth_id
        """,
        [cfg.age_min, cfg.age_max],
    )

    subgroup_check(con, run_table, cfg, "AFTER BASELINE AGE FILTER")

    # 2) Gender filter
    if cfg.gender != "both":
        gval = "M" if cfg.gender == "male" else "F"

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND UPPER(TRIM(CAST(gender AS VARCHAR))) = ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [gval],
        )

    subgroup_check(con, run_table, cfg, "AFTER GENDER FILTER")

    # hard fail if age/gender did not actually apply
    assert_subgroup_filter(con, run_table, cfg)

    # 2a) BMI / weight group filter
    if cfg.weight_group != "all":
        bmi_ranges = {
            "underweight": (None, 18.5),
            "normal": (18.5, 25.0),
            "overweight": (25.0, 30.0),
            "obese": (30.0, None),
        }

        lo, hi = bmi_ranges[cfg.weight_group]
        conds = [
            "bmi IS NOT NULL",
            "TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)",
        ]
        params = []

        if lo is not None:
            conds.append("bmi >= ?")
            params.append(lo)
        if hi is not None:
            conds.append("bmi < ?")
            params.append(hi)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND {' AND '.join(conds)}
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            params,
        )

        subgroup_check(con, run_table, cfg, "AFTER BMI FILTER")

    # 2b) Engagement stratification
    if cfg.engagement_strat:
        col_map = {
            ("30d_frequency", "decile"): "engagement_decile_30d",
            ("30d_frequency", "quartile"): "engagement_q_30d",
            ("alltime_frequency", "decile"): "engagement_decile_alltime",
            ("alltime_frequency", "quartile"): "engagement_q_alltime",
            ("alltime_points", "decile"): "engagement_decile_points_alltime",
            ("alltime_points", "quartile"): "engagement_q_points_alltime",
            ("frequency_points", "decile"): "engagement_decile_frequency_points",
            ("frequency_points", "quartile"): "engagement_q_frequency_points",
            ("minimal_engager", "none"): "minimal_engager",
            ("session_user", "none"): "session_user"
        }

        key = (cfg.engagement_definition, cfg.engagement_granularity)

        if key not in col_map:
            raise ValueError(f"Unsupported engagement setting: {key}")

        if cfg.engagement_group_min is None or cfg.engagement_group_max is None:
            raise ValueError("Engagement strat enabled but min/max not provided.")

        eng_col = col_map[key]

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND {eng_col} BETWEEN ? AND ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [cfg.engagement_group_min, cfg.engagement_group_max],
        )

        subgroup_check(con, run_table, cfg, "AFTER ENGAGEMENT FILTER")

    # 2c) Income stratification
    if cfg.income_strat:
        if cfg.income_group_min is None or cfg.income_group_max is None:
            raise ValueError("--income-strat requires --income-group-min and --income-group-max.")

        n_bins = {"tercile": 3, "quartile": 4, "decile": 10}[cfg.income_granularity]

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH baseline AS (
                SELECT eth_id, income_index
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
                  AND income_index IS NOT NULL
            ),
            ranked AS (
                SELECT
                    eth_id,
                    NTILE({n_bins}) OVER (ORDER BY income_index) AS income_bin
                FROM baseline
            ),
            eligible AS (
                SELECT DISTINCT eth_id
                FROM ranked
                WHERE income_bin BETWEEN ? AND ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [cfg.income_group_min, cfg.income_group_max],
        )

        subgroup_check(con, run_table, cfg, "AFTER INCOME FILTER")

    # 2d) Prior healthcare cost stratification
    if cfg.prior_cost_strat:
        if cfg.prior_cost_group_min is None or cfg.prior_cost_group_max is None:
            raise ValueError("--prior-cost-strat requires --prior-cost-group-min and --prior-cost-group-max.")

        cost_col_map = {
            "total": "prior_total_cost",
            "inpatient": "prior_inpatient_cost",
        }
        if cfg.prior_cost_definition not in cost_col_map:
            raise ValueError(f"Unsupported prior-cost definition: {cfg.prior_cost_definition}")
        cost_col = cost_col_map[cfg.prior_cost_definition]

        n_bins = {"tercile": 3, "quartile": 4, "decile": 10}[cfg.prior_cost_granularity]

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH baseline AS (
                SELECT eth_id, {cost_col} AS prior_cost_val
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
                  AND {cost_col} IS NOT NULL
            ),
            ranked AS (
                SELECT
                    eth_id,
                    NTILE({n_bins}) OVER (ORDER BY prior_cost_val) AS prior_cost_bin
                FROM baseline
            ),
            eligible AS (
                SELECT DISTINCT eth_id
                FROM ranked
                WHERE prior_cost_bin BETWEEN ? AND ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [cfg.prior_cost_group_min, cfg.prior_cost_group_max],
        )

        subgroup_check(con, run_table, cfg, "AFTER PRIOR COST FILTER")

    # 2e) Preexisting PCG stratification (binary)
    if cfg.pcg_strat:
        if cfg.pcg_value is None:
            raise ValueError("--pcg-strat requires --pcg-value (0 or 1).")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
                  AND preexisting_pcg = ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [cfg.pcg_value],
        )

        subgroup_check(con, run_table, cfg, "AFTER PCG FILTER")

    # 2e) Preexisting T2D stratification (binary)
    if cfg.t2d_strat:
        if cfg.t2d_value is None:
            raise ValueError("--t2d-strat requires --t2d-value (0 or 1).")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND TRY_CAST(claim_month AS DATE) = TRY_CAST(start_month AS DATE)
                  AND preexisting_dm2 = ?
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """,
            [cfg.t2d_value],
        )

        subgroup_check(con, run_table, cfg, "AFTER T2D FILTER")

    # 2f) Steps-diff stratification (categorical: increase / stable / decrease)
    if cfg.steps_diff_strat:
        if cfg.steps_diff_category is None:
            raise ValueError("--steps-diff-strat requires --steps-diff-category (increase/stable/decrease).")

        category_cond_map = {
            "increase": "steps_diff >= 1000",
            "stable": "steps_diff > -1000 AND steps_diff < 1000",
            "decrease": "steps_diff <= -1000",
        }
        cond = category_cond_map[cfg.steps_diff_category]

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {run_table} AS
            WITH eligible AS (
                SELECT DISTINCT eth_id
                FROM {run_table}
                WHERE eth_id IS NOT NULL
                  AND steps_diff IS NOT NULL
                  AND {cond}
            )
            SELECT t.*
            FROM {run_table} t
            INNER JOIN eligible e
                ON t.eth_id = e.eth_id
            """
        )

        subgroup_check(con, run_table, cfg, "AFTER STEPS-DIFF FILTER")

    # Final hard check before building DiD input
    assert_subgroup_filter(con, run_table, cfg)

    # 3) Build person-period panel
    con.execute(build_person_period_sql(run_table, cfg.granularity, cfg.base_year))

    # 4) Add event_time
    con.execute(build_window_sql("claims_person_period_summary_w_start_idx"))

    # 5) Add numeric id and gender_num
    con.execute("DROP TABLE IF EXISTS claims_person_period_summary_w_start_idx_numid")
    con.execute("""
        CREATE TABLE claims_person_period_summary_w_start_idx_numid AS
        SELECT
            *,
            DENSE_RANK() OVER (ORDER BY eth_id) AS id_num
        FROM claims_person_period_summary_w_start_idx_windowed
    """)

    con.execute("DROP TABLE IF EXISTS mdid_input")
    con.execute("""
        CREATE TABLE mdid_input AS
        SELECT
            t.*,
            CASE
                WHEN gender IS NULL THEN NULL
                WHEN UPPER(TRIM(CAST(gender AS VARCHAR))) = 'F' THEN 0
                WHEN UPPER(TRIM(CAST(gender AS VARCHAR))) = 'M' THEN 1
                ELSE NULL
            END AS gender_num
        FROM claims_person_period_summary_w_start_idx_numid t
    """)

    df = con.execute("SELECT * FROM mdid_input").fetchdf()
    
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        print(f"[WARN] Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    if cfg.placebo_time:
        df = assign_placebo_g_index(df, id_col="id_num", g_col="g_index", seed=999)

    df["t_period_start"] = pd.to_datetime(df["t_period_start"])

    if cfg.cal_min is not None:
        df = df[df["t_period_start"] >= pd.to_datetime(cfg.cal_min)].copy()

    if cfg.cal_max is not None:
        df = df[df["t_period_start"] <= pd.to_datetime(cfg.cal_max)].copy()

    df = df.drop_duplicates(subset=["id_num", "t_index"]).copy()
    df["id_num"] = df["id_num"].astype(int)
    df["t_index"] = df["t_index"].astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender_num"] = pd.to_numeric(df["gender_num"], errors="coerce")

    print("\n===== DATA DIAGNOSTICS BEFORE PANEL TRIMMING =====")
    print(f"Rows: {len(df)}")
    print(f"Unique ids: {df['id_num'].nunique()}")
    print(f"Unique periods: {df['t_index'].nunique()}")
    print("Gender_num counts:")
    print(df["gender_num"].value_counts(dropna=False))
    print("Age range in person-period data:")
    print(df["age"].min(), df["age"].max())
    print("===== END DIAGNOSTICS =====\n")

    # Extra post-build gender assertion
    if cfg.gender != "both":
        expected_num = 1 if cfg.gender == "male" else 0
        observed_gender_nums = set(df["gender_num"].dropna().astype(int).unique())

        if observed_gender_nums != {expected_num}:
            raise ValueError(
                f"Post-build gender filter failed. Expected gender_num={expected_num}, "
                f"got {observed_gender_nums}."
            )

    tag = f"_{cfg.base_period}_{'unbalanced' if cfg.unbalanced else 'balanced'}"
    required_cols = ["age", "gender_num"]

    if cfg.unbalanced:
        df = trim_unbalanced_min_pre_post(
            df,
            id_col="id_num",
            t_col="t_index",
            g_col="g_index",
            required_cols=required_cols,
            drop_missing_covars=True,
            drop_early_treated=False,
            min_pre=cfg.min_pre_periods,
            min_post=cfg.min_post_periods,
            out_dir=cfg.out_dir,
            tag=tag,
        )
    else:
        df = balance_panel_for_did(
            df,
            id_col="id_num",
            t_col="t_index",
            g_col="g_index",
            required_cols=required_cols,
            drop_missing_covars=True,
            drop_early_treated=True,
            enforce_balanced_panel=True,
            out_dir=cfg.out_dir,
            tag=tag,
        )

    df = winsorize_costs(
       df,
       pct=cfg.winsor_pct,
       out_path=os.path.join(cfg.out_dir, "winsorization_cutoffs.csv"),
    )

    # Save final IDs
    df["eth_id"].drop_duplicates().to_csv(
        os.path.join(cfg.out_dir, "unique_ids.csv"),
        index=False,
        header=["eth_id"],
    )

    did_summary = summarize_final_did_sample(
        df,
        id_col="id_num",
        t_col="t_index",
        g_col="g_index",
        gender_col="gender_num",
        age_col="age",
        out_path=os.path.join(cfg.out_dir, "did_sample_summary.csv"),
    )

    print("\n===== FINAL DID SAMPLE SUMMARY =====")
    print(did_summary.to_string(index=False))
    print("===== END SUMMARY =====\n")

    preadoption_cols = [c for c in cfg.outcomes if c in df.columns]
    missing_cols = [c for c in cfg.outcomes if c not in df.columns]
    if missing_cols:
        print(f"[WARN] outcomes not found in df, skipping for pre-adoption summary: {missing_cols}")
    
    if preadoption_cols:
        preadoption_summary = summarize_preadoption_costs(
            df,
            id_col="id_num",
            t_col="t_index",
            g_col="g_index",
            pre_window=(-4, -1),
            outcome_cols=tuple(preadoption_cols),
            out_path=os.path.join(cfg.out_dir, "preadoption_cost_summary.csv"),
        )
    
        print("\n===== PRE-ADOPTION SUMMARY (event_time -4 to -1) =====")
        print(preadoption_summary.to_string(index=False))
        print("===== END PRE-ADOPTION SUMMARY =====\n")
    else:
        print("[WARN] No valid outcome columns found — skipping pre-adoption summary.")

    # 7) Export parquet for R
    in_path = os.path.join(cfg.out_dir, "did_input.parquet")
    df.to_parquet(in_path, index=False)

    # 8) Stratum summary
    n_unique_people = int(df["id_num"].nunique())

    stratum_summary = pd.DataFrame(
        [{
            "age_min": cfg.age_min,
            "age_max": cfg.age_max,
            "gender": cfg.gender,
            "granularity": cfg.granularity,
            "base_year": cfg.base_year,
            "min_e": cfg.min_e,
            "max_e": cfg.max_e,
            "n_unique_people": n_unique_people,
            "cal_min": cfg.cal_min,
            "cal_max": cfg.cal_max,
            "min_t_period_start": str(df["t_period_start"].min().date()) if len(df) else None,
            "max_t_period_start": str(df["t_period_start"].max().date()) if len(df) else None,
            "control_group": cfg.control_group,
            "base_period": cfg.base_period,
            "unbalanced": cfg.unbalanced,
            "min_pre_periods": cfg.min_pre_periods,
            "min_post_periods": cfg.min_post_periods,
            "engagement_strat": cfg.engagement_strat,
            "placebo_time": cfg.placebo_time,
            "engagement_definition": cfg.engagement_definition,
            "engagement_granularity": cfg.engagement_granularity,
            "engagement_group_min": cfg.engagement_group_min,
            "engagement_group_max": cfg.engagement_group_max,
            "weight_group": cfg.weight_group,
            "income_strat": cfg.income_strat,
            "income_granularity": cfg.income_granularity,
            "income_group_min": cfg.income_group_min,
            "income_group_max": cfg.income_group_max,
            "prior_cost_strat": cfg.prior_cost_strat,
            "prior_cost_definition": cfg.prior_cost_definition,
            "prior_cost_granularity": cfg.prior_cost_granularity,
            "prior_cost_group_min": cfg.prior_cost_group_min,
            "prior_cost_group_max": cfg.prior_cost_group_max,
            "pcg_strat": cfg.pcg_strat,
            "pcg_value": cfg.pcg_value,
            "t2d_strat": cfg.t2d_strat,
            "t2d_value": cfg.t2d_value,
            "steps_diff_strat": cfg.steps_diff_strat,
            "steps_diff_category": cfg.steps_diff_category,
            "outcomes": ",".join(cfg.outcomes),
        }]
    )

    stratum_summary.to_csv(os.path.join(cfg.out_dir, "stratum_summary.csv"), index=False)

    # 9) Support by event time
    periods_per_year_map = {"yearly": 1, "half-yearly": 2, "quarterly": 4, "monthly": 12}
    export_support_by_event_time(
        df_input=df,
        min_e=cfg.min_e,
        max_e=cfg.max_e,
        out_path=os.path.join(cfg.out_dir, "support_event_time.csv"),
        fig_out_path=os.path.join(cfg.out_dir, "support_event_time.png"),
        control_group=cfg.control_group,
        periods_per_year=periods_per_year_map[cfg.granularity],
        base_year=cfg.base_year,
    )

    # 10) Run R
    print("Running in R...")

    r_script = "diff-in-diff/mdid_standard_pckg.R"

    cmd = [
        "Rscript",
        r_script,
        in_path,
        cfg.out_dir,
        str(cfg.min_e),
        str(cfg.max_e),
        cfg.control_group,
        cfg.base_period,
        ",".join(cfg.outcomes)
    ]

    proc = subprocess.run(cmd, text=True, capture_output=False)

    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed with return code {proc.returncode}: {cmd}")

    # Cleanup run table
    con.execute(f"DROP TABLE IF EXISTS {run_table}")


if __name__ == "__main__":
    main()