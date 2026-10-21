#!/usr/bin/env python3
"""
Staggered-adoption DiD data preparation, step 3: step-count change after adoption.

Builds on the table written by did_data_preprocessing_types.py and attaches, per person:
  - steps_period_1 : mean daily steps on days 1-90 after the first app transaction
  - steps_period_2 : mean daily steps on days 91-180
  - steps_diff     : steps_period_2 - steps_period_1
A period's mean is only computed if steps data exist on >= 75% of its days (68/90),
otherwise NULL; steps_diff is NULL unless both periods qualify.

Start date = MIN(TRANSACTION_DATETIME) per person in a365_transaction_check;
day 1 = the day after that. Steps come from a365_health_activities (DATA_TYPE = 'steps').

Output goes to stdout and to a log file in OUT_DIR.

Usage:
    python did_data_preprocessing_steps.py
    python did_data_preprocessing_steps.py --db ./insurance.duckdb
    nohup python -u did_data_preprocessing_steps.py > steps_prep.out 2>&1 &
"""
import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import psutil

# Add repository root to Python's import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data_preparation.config import connect_db

# -----------------------------
# Config
# -----------------------------
OUT_DIR = "./results/steps_diff"
DB_NAME = "./insurance.duckdb"

TARGET = "claims_person_month_summary_final_w_inpatient_type"   # output of did_data_preprocessing_types.py

TRANSACTION_TABLE = "a365_transaction_check"
ACTIVITY_TABLE = "a365_health_activities"
DATA_TYPE_FILTER = "steps"

PERIOD_1_START, PERIOD_1_END = 1, 90      # day offsets after first transaction, inclusive
PERIOD_2_START, PERIOD_2_END = 91, 180
MIN_COVERAGE_FRAC = 0.75                   # min share of days in a period with steps data
STEPS_DIFF_INCREASE_THRESHOLD = 1000       # reporting only

OUTPUT_COL_1 = "steps_period_1"
OUTPUT_COL_2 = "steps_period_2"
OUTPUT_COL_DIFF = "steps_diff"
# Column names used by earlier versions of the notebook; dropped if present
LEGACY_COLS = ["avg_daily_steps_90", "avg_daily_steps_180"]

# DB settings (same as main script)
DUCKDB_TMP = "./duckdb_tmp"
MEMORY_LIMIT = "48GB"
THREADS = 6
MAX_TEMP_DIR = "12GiB"

P1_LEN = PERIOD_1_END - PERIOD_1_START + 1
P2_LEN = PERIOD_2_END - PERIOD_2_START + 1
MIN_DAYS_P1 = int(-(-MIN_COVERAGE_FRAC * P1_LEN // 1))   # ceil
MIN_DAYS_P2 = int(-(-MIN_COVERAGE_FRAC * P2_LEN // 1))
OFFSET_MIN = min(PERIOD_1_START, PERIOD_2_START)
OFFSET_MAX = max(PERIOD_1_END, PERIOD_2_END)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

_T0 = time.time()


# -----------------------------
# Helpers
# -----------------------------
class Tee:
    """Write everything printed to stdout and to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title):
    print(f"\n=== {title}  [t+{time.time() - _T0:,.0f}s, "
          f"free RAM {psutil.virtual_memory().available / 1e9:.1f} GB] ===")


def checkpoint(con):
    con.execute("CHECKPOINT")
    gc.collect()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def show(con, sql, transpose=False):
    df = con.execute(sql).fetchdf()
    print(df.T.to_string(header=False) if transpose else df.to_string(index=False))


def columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


# -----------------------------
# Main
# -----------------------------
def main(db_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DUCKDB_TMP, exist_ok=True)

    log_path = Path(OUT_DIR) / f"steps_prep_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"Log file: {log_path.resolve()}")
    print(f"DB: {os.path.abspath(db_name)}")

    con = connect_db(str(db_name))
    try:
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET max_temp_directory_size = '{MAX_TEMP_DIR}'")
        con.execute(f"SET temp_directory = '{DUCKDB_TMP}'")
        con.execute("SET preserve_insertion_order = false")
        print(con.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads'), "
            "current_setting('temp_directory')"
        ).fetchall())

        # ------------------------------------------------------------------
        section(f"Baseline snapshot of {TARGET}")
        # ------------------------------------------------------------------
        n_rows_before = scalar(con, f"SELECT COUNT(*) FROM {TARGET}")
        n_ids_before = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {TARGET}")
        print(f"BEFORE merge: n_rows={n_rows_before:,}, n_ids={n_ids_before:,}")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eth_ids_before AS
            SELECT DISTINCT eth_id FROM {TARGET}
        """)

        existing = columns(con, TARGET)
        outputs = (OUTPUT_COL_1, OUTPUT_COL_2, OUTPUT_COL_DIFF)
        # exact names, legacy names, and DuckDB-suffixed duplicates from earlier re-runs (e.g. steps_period_1_1)
        to_drop = [
            c for c in existing
            if c in outputs or c in LEGACY_COLS
            or any(c.startswith(f"{o}_") and c[len(o) + 1:].isdigit() for o in outputs)
        ]
        if to_drop:
            print(f"Existing steps columns found, will be replaced: {to_drop}")
        else:
            print("No existing steps columns found on target table.")

        # ------------------------------------------------------------------
        section("Build steps feature table")
        # ------------------------------------------------------------------
        print(f"Periods: [{PERIOD_1_START},{PERIOD_1_END}] & [{PERIOD_2_START},{PERIOD_2_END}], "
              f"min coverage {MIN_COVERAGE_FRAC:.0%} -> {MIN_DAYS_P1}/{MIN_DAYS_P2} days")

        # Start date for panel persons only (keeps the activity scan small)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE person_start_date AS
            SELECT
                t.CSS_MARKETING_ID AS eth_id,
                MIN(CAST(t.TRANSACTION_DATETIME AS DATE)) AS first_transaction_date
            FROM {TRANSACTION_TABLE} t
            JOIN eth_ids_before i ON i.eth_id = t.CSS_MARKETING_ID
            WHERE t.TRANSACTION_DATETIME IS NOT NULL
            GROUP BY t.CSS_MARKETING_ID
        """)
        print(f"person_start_date: {scalar(con, 'SELECT COUNT(*) FROM person_start_date'):,} persons")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE steps_windowed AS
            SELECT
                d.CSS_MARKETING_ID AS eth_id,
                (CAST(d.ACTIVITY_DATE AS DATE) - p.first_transaction_date) AS day_offset,
                d.VALUE AS steps
            FROM {ACTIVITY_TABLE} d
            JOIN person_start_date p ON d.CSS_MARKETING_ID = p.eth_id
            WHERE d.DATA_TYPE = '{DATA_TYPE_FILTER}'
              AND d.VALUE IS NOT NULL
              AND (CAST(d.ACTIVITY_DATE AS DATE) - p.first_transaction_date)
                  BETWEEN {OFFSET_MIN} AND {OFFSET_MAX}
        """)
        checkpoint(con)

        # Coverage counts rows, so it assumes one steps record per person-day. Check that.
        n_rows_w, n_person_days = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT (eth_id, day_offset)) FROM steps_windowed
        """).fetchone()
        print(f"steps_windowed: {n_rows_w:,} rows, {n_person_days:,} distinct person-days")
        if n_rows_w != n_person_days:
            print(f"WARNING: {n_rows_w - n_person_days:,} extra rows on person-days with >1 steps record. "
                  "Coverage counts and averages are row-based, so such days are over-weighted.")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE steps_windowed_raw AS
            SELECT
                eth_id,
                COUNT(*)   FILTER (WHERE day_offset BETWEEN {PERIOD_1_START} AND {PERIOD_1_END}) AS n_days_p1,
                AVG(steps) FILTER (WHERE day_offset BETWEEN {PERIOD_1_START} AND {PERIOD_1_END}) AS avg_p1_raw,
                COUNT(*)   FILTER (WHERE day_offset BETWEEN {PERIOD_2_START} AND {PERIOD_2_END}) AS n_days_p2,
                AVG(steps) FILTER (WHERE day_offset BETWEEN {PERIOD_2_START} AND {PERIOD_2_END}) AS avg_p2_raw
            FROM steps_windowed
            GROUP BY eth_id
        """)
        con.execute("DROP TABLE IF EXISTS steps_windowed")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE steps_diff_table AS
            SELECT
                eth_id,
                CASE WHEN n_days_p1 >= {MIN_DAYS_P1} THEN avg_p1_raw END AS "{OUTPUT_COL_1}",
                CASE WHEN n_days_p2 >= {MIN_DAYS_P2} THEN avg_p2_raw END AS "{OUTPUT_COL_2}",
                CASE WHEN n_days_p1 >= {MIN_DAYS_P1} AND n_days_p2 >= {MIN_DAYS_P2}
                     THEN avg_p2_raw - avg_p1_raw END AS "{OUTPUT_COL_DIFF}"
            FROM steps_windowed_raw
        """)
        checkpoint(con)

        n_dupes = scalar(con, """
            SELECT COUNT(*) FROM (SELECT eth_id FROM steps_diff_table GROUP BY eth_id HAVING COUNT(*) > 1)
        """)
        if n_dupes:
            raise RuntimeError(
                f"steps_diff_table has {n_dupes} eth_id(s) with duplicate rows; the merge would fan out. "
                "Aborting before touching the target table."
            )
        print(f"steps_diff_table built: {scalar(con, 'SELECT COUNT(*) FROM steps_diff_table'):,} unique eth_id rows")
        print("Duplicate check passed: 0 eth_ids with >1 row in steps_diff_table")

        print("\nDays with steps data per period (persons with any steps data in days 1-180):")
        show(con, f"""
            SELECT
                MEDIAN(n_days_p1) AS median_days_p1,
                MEDIAN(n_days_p2) AS median_days_p2,
                COUNT(*) FILTER (WHERE n_days_p1 >= {MIN_DAYS_P1}) AS n_meet_p1,
                COUNT(*) FILTER (WHERE n_days_p2 >= {MIN_DAYS_P2}) AS n_meet_p2
            FROM steps_windowed_raw
        """, transpose=True)
        con.execute("DROP TABLE IF EXISTS steps_windowed_raw")

        # ------------------------------------------------------------------
        section(f"Merge onto {TARGET}")
        # ------------------------------------------------------------------
        # LEFT JOIN driven by the target table: cannot add eth_ids; with <=1 feature row
        # per eth_id it cannot add rows either.
        exclude = f"EXCLUDE ({', '.join(to_drop)})" if to_drop else ""
        con.execute(f"""
            CREATE OR REPLACE TABLE {TARGET} AS
            SELECT
                t.* {exclude},
                sd."{OUTPUT_COL_1}",
                sd."{OUTPUT_COL_2}",
                sd."{OUTPUT_COL_DIFF}"
            FROM {TARGET} t
            LEFT JOIN steps_diff_table sd ON t.eth_id = sd.eth_id
        """)
        checkpoint(con)
        print(f"Merge complete: {OUTPUT_COL_1}, {OUTPUT_COL_2}, {OUTPUT_COL_DIFF} attached.")

        # ------------------------------------------------------------------
        section("Hard safety checks")
        # ------------------------------------------------------------------
        n_rows_after = scalar(con, f"SELECT COUNT(*) FROM {TARGET}")
        n_ids_after = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {TARGET}")
        print(f"AFTER merge:  n_rows={n_rows_after:,}, n_ids={n_ids_after:,}")
        print(f"BEFORE merge: n_rows={n_rows_before:,}, n_ids={n_ids_before:,}")

        if n_rows_after != n_rows_before:
            raise RuntimeError(f"ROW COUNT CHANGED: {n_rows_before} -> {n_rows_after}.")
        if n_ids_after != n_ids_before:
            raise RuntimeError(f"UNIQUE eth_id COUNT CHANGED: {n_ids_before} -> {n_ids_after}.")

        lost_ids, gained_ids = con.execute(f"""
            WITH after AS (SELECT DISTINCT eth_id FROM {TARGET})
            SELECT
                (SELECT COUNT(*) FROM eth_ids_before b ANTI JOIN after a USING (eth_id)),
                (SELECT COUNT(*) FROM after a ANTI JOIN eth_ids_before b USING (eth_id))
        """).fetchone()
        print(f"eth_ids lost in merge:   {lost_ids}")
        print(f"eth_ids gained in merge: {gained_ids}")
        if lost_ids or gained_ids:
            raise RuntimeError(f"eth_id SET MISMATCH: {lost_ids} lost, {gained_ids} gained.")

        found = {c for c in columns(con, TARGET) if c in (OUTPUT_COL_1, OUTPUT_COL_2, OUTPUT_COL_DIFF)}
        missing = {OUTPUT_COL_1, OUTPUT_COL_2, OUTPUT_COL_DIFF} - found
        if missing:
            raise RuntimeError(f"Expected columns missing after build: {missing}")
        dup_names = [c for c in columns(con, TARGET)
                     if any(c.startswith(f"{o}_") and c[len(o) + 1:].isdigit() for o in outputs)]
        if dup_names:
            raise RuntimeError(f"Suffixed duplicate columns present: {dup_names}")
        show(con, f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{TARGET}'
              AND column_name IN ('{OUTPUT_COL_1}', '{OUTPUT_COL_2}', '{OUTPUT_COL_DIFF}')
            ORDER BY ordinal_position
        """)
        print("\n[OK] Row count, id count, exact id set and columns all as expected.")

        # ------------------------------------------------------------------
        section("Summary statistics (person level)")
        # ------------------------------------------------------------------
        person_steps = con.execute(f"""
            SELECT eth_id,
                   ANY_VALUE("{OUTPUT_COL_1}") AS "{OUTPUT_COL_1}",
                   ANY_VALUE("{OUTPUT_COL_2}") AS "{OUTPUT_COL_2}",
                   ANY_VALUE("{OUTPUT_COL_DIFF}") AS "{OUTPUT_COL_DIFF}"
            FROM {TARGET}
            GROUP BY eth_id
        """).fetchdf()

        n_total = len(person_steps)
        rows = []
        for col in (OUTPUT_COL_1, OUTPUT_COL_2, OUTPUT_COL_DIFF):
            s = person_steps[col]
            n_valid = int(s.notna().sum())
            rows.append({
                "variable": col,
                "n_valid": n_valid,
                "n_missing": n_total - n_valid,
                "pct_missing": round(100 * (n_total - n_valid) / n_total, 1),
                "mean": round(s.mean(), 1) if n_valid else None,
                "median": round(s.median(), 1) if n_valid else None,
                "sd": round(s.std(), 1) if n_valid > 1 else None,
                "min": round(s.min(), 1) if n_valid else None,
                "max": round(s.max(), 1) if n_valid else None,
            })
        print(pd.DataFrame(rows).to_string(index=False))

        valid_diff = person_steps[OUTPUT_COL_DIFF].dropna()
        n_valid = len(valid_diff)
        if n_valid:
            n_inc = int((valid_diff >= STEPS_DIFF_INCREASE_THRESHOLD).sum())
            n_dec = int((valid_diff < 0).sum())
            n_stable = n_valid - n_inc - n_dec
            print(f"\nAmong {n_valid:,} persons with valid {OUTPUT_COL_DIFF}:")
            print(f"  >= {STEPS_DIFF_INCREASE_THRESHOLD} step increase : {n_inc:,} ({100 * n_inc / n_valid:.1f}%)")
            print(f"  0-{STEPS_DIFF_INCREASE_THRESHOLD - 1} step change    : {n_stable:,} ({100 * n_stable / n_valid:.1f}%)")
            print(f"  decrease (<0)        : {n_dec:,} ({100 * n_dec / n_valid:.1f}%)")

        # ------------------------------------------------------------------
        section("Coverage diagnostic")
        # ------------------------------------------------------------------
        # Separates "no steps data at all" from "some data, but not 75% in both periods"
        n_any_data = scalar(con, "SELECT COUNT(*) FROM steps_diff_table")
        n_any_period = int((person_steps[OUTPUT_COL_1].notna() | person_steps[OUTPUT_COL_2].notna()).sum())
        print(f"total persons:                                  {n_total:,}")
        print(f"  any steps data in days {OFFSET_MIN}-{OFFSET_MAX}:                  {n_any_data:,}")
        print(f"  at least one period meets coverage:           {n_any_period:,}")
        print(f"  both periods meet coverage (valid steps_diff): {n_valid:,}")
        del person_steps
        gc.collect()

        # ------------------------------------------------------------------
        section("Cleanup")
        # ------------------------------------------------------------------
        for t in ("person_start_date", "steps_diff_table", "eth_ids_before"):
            con.execute(f"DROP TABLE IF EXISTS {t}")
        checkpoint(con)
        print("Temp tables dropped. Target table changes are persisted.")

        section("Done")

    finally:
        con.close()
        gc.collect()
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_NAME, help="Path to insurance.duckdb")
    args = ap.parse_args()
    main(args.db)