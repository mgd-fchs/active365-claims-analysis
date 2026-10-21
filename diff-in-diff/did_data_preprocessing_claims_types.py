#!/usr/bin/env python3
"""
Staggered-adoption DiD data preparation, step 2: claim types, franchise, session users.

Builds on the person-month table written by did_data_preprocessing_main.py and adds:
  - inpatient admissions by type (emergency / non-emergency), length of stay, costs by type
  - in- and outpatient claims/costs by treatment reason (illness / accident / excl. maternity)
  - franchise (deductible) of the contract active in each person-month
  - session_user flag (>=1 app session in the first 30 days after adoption)

Sanity checks and summary stats are printed to stdout AND written to a log file in OUT_DIR.

Usage:
    python did_data_preprocessing_types.py
    python did_data_preprocessing_types.py --db ./insurance.duckdb --diagnostics
    python did_data_preprocessing_types.py --examples 3      # print 3 random person timelines (individual-level!)
    nohup python -u did_data_preprocessing_types.py > types_prep.out 2>&1 &
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data_preparation.config import connect_db

# -----------------------------
# Config
# -----------------------------
OUT_DIR = "./results/types_of_cost_reduction"
DB_NAME = "./insurance.duckdb"

BASE = "claims_person_month_summary"                           # output of did_data_preprocessing_main.py
OUT = "claims_person_month_summary_final_w_inpatient_type"     # output of this script

# DB settings (same as main script)
DUCKDB_TMP = "./duckdb_tmp"
MEMORY_LIMIT = "48GB"
THREADS = 6
MAX_TEMP_DIR = "12GiB"

# Claims used for the type/reason splits: KVG only, excluding these provider types.
TYPE_PARTNER_EXCLUSION = [
    # "Alters- und Pflegeheime",
    "Blutspendezentren",
    "Fitness",
    # "Geburtshäuser",
    # "Hebammen",
    # "Kur- und Erholungszentren",
    # "Kurhäuser",
    # "Organisationen der Hebammen",
]

SESSION_WINDOW_DAYS = 30

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


def scalar(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


def show(con, sql, transpose=False, params=None):
    df = con.execute(sql, params or []).fetchdf()
    print((df.T.to_string(header=False) if transpose else df.to_string(index=False)))


def sql_list(values):
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


# -----------------------------
# Main
# -----------------------------
def main(db_name, diagnostics, n_examples):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DUCKDB_TMP, exist_ok=True)

    log_path = Path(OUT_DIR) / f"types_prep_{datetime.now():%Y%m%d_%H%M%S}.log"
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

        partner_excl = sql_list(TYPE_PARTNER_EXCLUSION)

        # ------------------------------------------------------------------
        section(f"Base table checks ({BASE})")
        # ------------------------------------------------------------------
        n_rows_base = scalar(con, f"SELECT COUNT(*) FROM {BASE}")
        n_ids_base = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {BASE}")
        n_keys_base = scalar(con, f"SELECT COUNT(*) FROM (SELECT DISTINCT eth_id, claim_month FROM {BASE})")
        print(f"rows: {n_rows_base:,} | persons: {n_ids_base:,} | distinct (eth_id, claim_month): {n_keys_base:,} "
              f"| duplicate person-month rows: {n_rows_base - n_keys_base:,}")

        print("\nAge distribution at start_month for age < 18 (age 0 excluded):")
        df_u18 = con.execute(f"""
            SELECT age, COUNT(DISTINCT eth_id) AS n_eth_ids
            FROM {BASE}
            WHERE claim_month = start_month AND age IS NOT NULL AND age < 18 AND age != 0
            GROUP BY age ORDER BY age
        """).fetchdf()
        print(df_u18.to_string(index=False))
        print(f"Total: {df_u18['n_eth_ids'].sum():,}")

        df_nobase = con.execute(f"""
            WITH has_baseline AS (
                SELECT DISTINCT eth_id FROM {BASE} WHERE claim_month = start_month
            )
            SELECT b.eth_id, ANY_VALUE(b.start_month) AS start_month,
                   MIN(b.claim_month) AS first_observed_month,
                   MAX(b.claim_month) AS last_observed_month
            FROM {BASE} b
            LEFT JOIN has_baseline h USING (eth_id)
            WHERE h.eth_id IS NULL AND b.start_month IS NOT NULL
            GROUP BY b.eth_id
        """).fetchdf()
        n_nobase = len(df_nobase)
        print(f"\nAdopters missing a row at claim_month = start_month: {n_nobase:,}")
        if n_nobase:
            after = (df_nobase["start_month"] > df_nobase["last_observed_month"]).sum()
            before = (df_nobase["start_month"] < df_nobase["first_observed_month"]).sum()
            print(f"  start_month AFTER last observed month:  {after:,}")
            print(f"  start_month BEFORE first observed month: {before:,}")
            print(f"  start_month in a GAP within observed range: {n_nobase - after - before:,}")
        del df_nobase
        gc.collect()

        # ------------------------------------------------------------------
        section("Raw source distributions")
        # ------------------------------------------------------------------
        print("Inpatient spital_aufnahmeart:")
        show(con, """
            SELECT TRIM(spital_aufnahmeart) AS spital_aufnahmeart, COUNT(*) AS n
            FROM inpatient GROUP BY 1 ORDER BY n DESC
        """)
        print("\nInpatient behandlungsgrund:")
        show(con, """
            SELECT TRIM(behandlungsgrund) AS behandlungsgrund, COUNT(*) AS n,
                   SUM(TRY_CAST(betrag AS DOUBLE)) AS total_cost
            FROM inpatient GROUP BY 1 ORDER BY n DESC
        """)
        print("\nInpatient aufenthaltsdauer (length of stay):")
        show(con, """
            SELECT COUNT(*) AS total_inpatient_rows,
                   COUNT(*) FILTER (WHERE aufenthaltsdauer IS NULL) AS n_null_aufenthaltsdauer,
                   COUNT(*) FILTER (WHERE aufenthaltsdauer = 0)     AS n_zero_aufenthaltsdauer
            FROM inpatient
        """)
        if diagnostics:
            # full scan of the outpatient table (~1bn rows) -> only on request
            print("\nOutpatient behandlungsgrund:")
            show(con, """
                SELECT TRIM(behandlungsgrund) AS behandlungsgrund, COUNT(*) AS n,
                       SUM(TRY_CAST(betrag AS DOUBLE)) AS total_cost
                FROM outpatient GROUP BY 1 ORDER BY n DESC
            """)
            print("\nNULL behandlungsgrund in source tables:")
            show(con, """
                SELECT 'inpatient' AS tbl, COUNT(*) AS n_null FROM inpatient WHERE behandlungsgrund IS NULL
                UNION ALL
                SELECT 'outpatient', COUNT(*) FROM outpatient WHERE behandlungsgrund IS NULL
            """)

        # ------------------------------------------------------------------
        section("Monthly claim-type aggregates (panel persons only)")
        # ------------------------------------------------------------------
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE tmp_panel_ids AS
            SELECT DISTINCT eth_id FROM {BASE} WHERE eth_id IS NOT NULL
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE tmp_inpatient_monthly AS
            WITH src AS (
                SELECT
                    i.eth_id,
                    DATE_TRUNC('month', TRY_CAST(i.behandlungsbeginn AS DATE)) AS claim_month,
                    TRIM(i.spital_aufnahmeart) AS aufnahmeart,
                    TRIM(i.behandlungsgrund)   AS grund,
                    i.aufenthaltsdauer,
                    TRY_CAST(i.betrag AS DOUBLE) AS betrag
                FROM inpatient i
                JOIN tmp_panel_ids p ON p.eth_id = i.eth_id
                WHERE TRY_CAST(i.behandlungsbeginn AS DATE) IS NOT NULL
                  AND UPPER(TRIM(i.gesetz)) = 'KVG'
                  AND TRIM(i.partnerart) NOT IN ({partner_excl})
            )
            SELECT
                eth_id, claim_month,
                COUNT(*) FILTER (WHERE aufnahmeart = 'notfall') AS n_inpatient_emergency,
                COUNT(*) FILTER (WHERE aufnahmeart = 'geplant') AS n_inpatient_planned,
                SUM(aufenthaltsdauer) AS los_total,
                -- normalised LOS = total days / number of stays (NULL if no stay of that type)
                SUM(aufenthaltsdauer) FILTER (WHERE aufnahmeart = 'notfall') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE aufnahmeart = 'notfall'), 0) AS los_emergency,
                SUM(aufenthaltsdauer) FILTER (WHERE aufnahmeart = 'geplant') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE aufnahmeart = 'geplant'), 0) AS los_planned,
                SUM(betrag) FILTER (WHERE aufnahmeart = 'notfall') AS inpatient_cost_emergency,
                SUM(betrag) FILTER (WHERE aufnahmeart = 'geplant') AS inpatient_cost_planned,
                SUM(betrag) AS inpatient_cost_typebase,       -- reconciliation only, not kept
                COUNT(*)    AS n_inpatient_typebase,          -- reconciliation only, not kept
                COUNT(*) FILTER (WHERE grund = 'Krankheit') AS n_inpatient_ill,
                COUNT(*) FILTER (WHERE grund = 'Unfall')    AS n_inpatient_acc,
                SUM(betrag) FILTER (WHERE grund = 'Krankheit') AS inpatient_cost_ill,
                SUM(betrag) FILTER (WHERE grund = 'Unfall')    AS inpatient_cost_acc,
                COUNT(*)    FILTER (WHERE grund != 'Mutterschaft') AS n_inpatient_excl_mutterschaft,
                SUM(betrag) FILTER (WHERE grund != 'Mutterschaft') AS inpatient_cost_excl_mutterschaft
            FROM src
            GROUP BY eth_id, claim_month
        """)
        checkpoint(con)
        print(f"tmp_inpatient_monthly: {scalar(con, 'SELECT COUNT(*) FROM tmp_inpatient_monthly'):,} person-months")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE tmp_outpatient_monthly AS
            WITH src AS (
                SELECT
                    o.eth_id,
                    DATE_TRUNC('month', TRY_CAST(o.behandlungsbeginn AS DATE)) AS claim_month,
                    TRIM(o.behandlungsgrund) AS grund,
                    TRY_CAST(o.betrag AS DOUBLE) AS betrag
                FROM outpatient o
                JOIN tmp_panel_ids p ON p.eth_id = o.eth_id
                WHERE TRY_CAST(o.behandlungsbeginn AS DATE) IS NOT NULL
                  AND UPPER(TRIM(o.gesetz)) = 'KVG'
                  AND TRIM(o.partnerart) NOT IN ({partner_excl})
            )
            SELECT
                eth_id, claim_month,
                SUM(betrag) AS outpatient_cost_typebase,      -- reconciliation only, not kept
                COUNT(*)    AS n_outpatient_typebase,         -- reconciliation only, not kept
                COUNT(*) FILTER (WHERE grund = 'Krankheit') AS n_outpatient_ill,
                COUNT(*) FILTER (WHERE grund = 'Unfall')    AS n_outpatient_acc,
                SUM(betrag) FILTER (WHERE grund = 'Krankheit') AS outpatient_cost_ill,
                SUM(betrag) FILTER (WHERE grund = 'Unfall')    AS outpatient_cost_acc,
                COUNT(*)    FILTER (WHERE grund != 'Mutterschaft') AS n_outpatient_excl_mutterschaft,
                SUM(betrag) FILTER (WHERE grund != 'Mutterschaft') AS outpatient_cost_excl_mutterschaft
            FROM src
            GROUP BY eth_id, claim_month
        """)
        checkpoint(con)
        print(f"tmp_outpatient_monthly: {scalar(con, 'SELECT COUNT(*) FROM tmp_outpatient_monthly'):,} person-months")

        # ------------------------------------------------------------------
        section(f"Join onto panel -> {OUT}")
        # ------------------------------------------------------------------
        con.execute(f"DROP TABLE IF EXISTS {OUT}")
        con.execute(f"""
            CREATE TABLE {OUT} AS
            SELECT
                b.*,
                COALESCE(i.n_inpatient_emergency, 0)            AS n_inpatient_emergency,
                COALESCE(i.n_inpatient_planned, 0)              AS n_inpatient_planned,
                i.los_emergency                                 AS los_emergency,   -- NULL = no stay
                i.los_planned                                   AS los_planned,     -- NULL = no stay
                COALESCE(i.los_total, 0)                        AS los_total,
                COALESCE(i.inpatient_cost_emergency, 0)         AS inpatient_cost_emergency,
                COALESCE(i.inpatient_cost_planned, 0)           AS inpatient_cost_planned,
                COALESCE(i.n_inpatient_ill, 0)                  AS n_inpatient_ill,
                COALESCE(i.n_inpatient_acc, 0)                  AS n_inpatient_acc,
                COALESCE(i.inpatient_cost_ill, 0)               AS inpatient_cost_ill,
                COALESCE(i.inpatient_cost_acc, 0)               AS inpatient_cost_acc,
                COALESCE(i.n_inpatient_excl_mutterschaft, 0)    AS n_inpatient_excl_mutterschaft,
                COALESCE(i.inpatient_cost_excl_mutterschaft, 0) AS inpatient_cost_excl_mutterschaft,
                COALESCE(o.n_outpatient_ill, 0)                 AS n_outpatient_ill,
                COALESCE(o.n_outpatient_acc, 0)                 AS n_outpatient_acc,
                COALESCE(o.outpatient_cost_ill, 0)              AS outpatient_cost_ill,
                COALESCE(o.outpatient_cost_acc, 0)              AS outpatient_cost_acc,
                COALESCE(o.n_outpatient_excl_mutterschaft, 0)   AS n_outpatient_excl_mutterschaft,
                COALESCE(o.outpatient_cost_excl_mutterschaft, 0) AS outpatient_cost_excl_mutterschaft
            FROM {BASE} b
            LEFT JOIN tmp_inpatient_monthly i
              ON b.eth_id = i.eth_id AND CAST(b.claim_month AS DATE) = CAST(i.claim_month AS DATE)
            LEFT JOIN tmp_outpatient_monthly o
              ON b.eth_id = o.eth_id AND CAST(b.claim_month AS DATE) = CAST(o.claim_month AS DATE)
        """)
        checkpoint(con)

        n_rows_out = scalar(con, f"SELECT COUNT(*) FROM {OUT}")
        print(f"rows: base {n_rows_base:,} -> {OUT} {n_rows_out:,}")
        assert n_rows_out == n_rows_base, "Row count changed during claim-type join"

        # ------------------------------------------------------------------
        section("Checks: length of stay")
        # ------------------------------------------------------------------
        show(con, f"""
            SELECT
                SUM(n_inpatient_emergency) AS total_emergency_stays,
                SUM(n_inpatient_planned)   AS total_planned_stays,
                SUM(los_emergency * n_inpatient_emergency) / NULLIF(SUM(n_inpatient_emergency), 0) AS avg_los_emergency,
                SUM(los_planned * n_inpatient_planned)     / NULLIF(SUM(n_inpatient_planned), 0)   AS avg_los_planned,
                COUNT(*) FILTER (WHERE los_emergency IS NULL) AS n_null_los_emergency,
                COUNT(*) FILTER (WHERE los_planned   IS NULL) AS n_null_los_planned,
                COUNT(*) FILTER (WHERE los_emergency = 0)     AS n_zero_los_emergency,
                COUNT(*) FILTER (WHERE los_planned   = 0)     AS n_zero_los_planned,
                MIN(los_emergency) AS min_los_emergency, MAX(los_emergency) AS max_los_emergency,
                MIN(los_planned)   AS min_los_planned,   MAX(los_planned)   AS max_los_planned
            FROM {OUT}
        """, transpose=True)

        # ------------------------------------------------------------------
        section("Checks: decomposition by treatment reason")
        # ------------------------------------------------------------------
        # Reconciliation: base totals (main-script filters) vs. claims eligible for the
        # type/reason split (KVG + extended partner exclusion). Difference is NOT maternity.
        print("Base totals vs. type/reason-eligible claims (panel person-months):")
        show(con, f"""
            WITH base AS (
                SELECT SUM(n_inpatient_claims) AS n_inp, SUM(inpatient_cost) AS c_inp,
                       SUM(n_outpatient_claims) AS n_out, SUM(outpatient_cost) AS c_out
                FROM {BASE}
            ),
            typ AS (
                SELECT
                    (SELECT SUM(n_inpatient_typebase)    FROM tmp_inpatient_monthly i  JOIN (SELECT DISTINCT eth_id, CAST(claim_month AS DATE) m FROM {BASE}) k ON k.eth_id = i.eth_id AND k.m = CAST(i.claim_month AS DATE)) AS n_inp,
                    (SELECT SUM(inpatient_cost_typebase) FROM tmp_inpatient_monthly i  JOIN (SELECT DISTINCT eth_id, CAST(claim_month AS DATE) m FROM {BASE}) k ON k.eth_id = i.eth_id AND k.m = CAST(i.claim_month AS DATE)) AS c_inp,
                    (SELECT SUM(n_outpatient_typebase)   FROM tmp_outpatient_monthly o JOIN (SELECT DISTINCT eth_id, CAST(claim_month AS DATE) m FROM {BASE}) k ON k.eth_id = o.eth_id AND k.m = CAST(o.claim_month AS DATE)) AS n_out,
                    (SELECT SUM(outpatient_cost_typebase) FROM tmp_outpatient_monthly o JOIN (SELECT DISTINCT eth_id, CAST(claim_month AS DATE) m FROM {BASE}) k ON k.eth_id = o.eth_id AND k.m = CAST(o.claim_month AS DATE)) AS c_out
            )
            SELECT 'inpatient' AS source, base.n_inp AS n_base, typ.n_inp AS n_type_eligible,
                   base.c_inp AS cost_base, typ.c_inp AS cost_type_eligible
            FROM base, typ
            UNION ALL
            SELECT 'outpatient', base.n_out, typ.n_out, base.c_out, typ.c_out
            FROM base, typ
        """)

        for src, n_col, c_col in [("INPATIENT", "n_inpatient_claims", "inpatient_cost"),
                                  ("OUTPATIENT", "n_outpatient_claims", "outpatient_cost")]:
            p = src.lower()
            print(f"\n{src} cost decomposition:")
            show(con, f"""
                SELECT
                    SUM({c_col})                       AS total_{p}_cost_base,
                    SUM({p}_cost_ill)                  AS cost_ill,
                    SUM({p}_cost_acc)                  AS cost_acc,
                    SUM({p}_cost_excl_mutterschaft)    AS cost_excl_mutterschaft,
                    SUM({p}_cost_ill) + SUM({p}_cost_acc) AS ill_plus_acc,
                    SUM({c_col}) - SUM({p}_cost_ill) - SUM({p}_cost_acc) AS cost_other
                FROM {OUT}
            """)
            print(f"\n{src} claim count decomposition:")
            show(con, f"""
                SELECT
                    SUM({n_col})                       AS total_n_base,
                    SUM(n_{p}_ill)                     AS n_ill,
                    SUM(n_{p}_acc)                     AS n_acc,
                    SUM(n_{p}_excl_mutterschaft)       AS n_excl_mutterschaft,
                    SUM(n_{p}_ill) + SUM(n_{p}_acc)    AS ill_plus_acc,
                    SUM({n_col}) - SUM(n_{p}_ill) - SUM(n_{p}_acc) AS n_other
                FROM {OUT}
            """)

        con.execute("DROP TABLE IF EXISTS tmp_inpatient_monthly")
        con.execute("DROP TABLE IF EXISTS tmp_outpatient_monthly")
        con.execute("DROP TABLE IF EXISTS tmp_panel_ids")
        checkpoint(con)

        # ------------------------------------------------------------------
        section("Checks: totals and non-zero rows, base vs. output")
        # ------------------------------------------------------------------
        for tbl in (BASE, OUT):
            print(f"\n{tbl}:")
            show(con, f"""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(*) FILTER (WHERE total_claim_cost > 0)    AS rows_with_cost,
                    COUNT(*) FILTER (WHERE n_outpatient_claims > 0) AS rows_with_outpatient,
                    COUNT(*) FILTER (WHERE n_inpatient_claims > 0)  AS rows_with_inpatient,
                    SUM(n_inpatient_claims)  AS total_inpatient_claims,
                    SUM(n_outpatient_claims) AS total_outpatient_claims,
                    SUM(inpatient_cost)      AS total_inpatient_cost,
                    SUM(outpatient_cost)     AS total_outpatient_cost,
                    SUM(total_claim_cost)    AS total_claim_cost
                FROM {tbl}
            """, transpose=True)

        # ------------------------------------------------------------------
        section("Franchise (deductible) per person-month")
        # ------------------------------------------------------------------
        # Franchise can be 0 for minors -> coded NA. If several contracts overlap a month,
        # the most recently started one is used. No overlapping contract -> NA.
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE tmp_month_keys AS
            SELECT DISTINCT eth_id, CAST(claim_month AS DATE) AS m FROM {OUT}
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE tmp_month_contracts AS
            WITH contracts AS (
                SELECT
                    eth_id,
                    TRY_CAST(beginndatum AS DATE)                            AS c_begin,
                    COALESCE(TRY_CAST(endedatum AS DATE), DATE '9999-12-31') AS c_end,
                    franchise_betrag
                FROM contracts_obligatory
            )
            SELECT k.eth_id, k.m, c.c_begin, c.c_end, c.franchise_betrag
            FROM tmp_month_keys k
            JOIN contracts c
              ON  c.eth_id = k.eth_id
              AND c.c_begin <= LAST_DAY(k.m)
              AND c.c_end   >= k.m
        """)
        checkpoint(con)

        show(con, """
            WITH per_month AS (
                SELECT k.eth_id, k.m,
                       COUNT(c.eth_id) AS n_contracts,
                       COUNT(*) FILTER (WHERE c.franchise_betrag = 0) AS n_zero
                FROM tmp_month_keys k
                LEFT JOIN tmp_month_contracts c ON c.eth_id = k.eth_id AND c.m = k.m
                GROUP BY 1, 2
            )
            SELECT
                COUNT(*)                                                         AS n_person_months,
                COUNT(*) FILTER (WHERE n_contracts = 0)                          AS n_no_contract,
                ROUND(100.0 * COUNT(*) FILTER (WHERE n_contracts = 0) / COUNT(*), 2) AS pct_no_contract,
                COUNT(*) FILTER (WHERE n_contracts > 1)                          AS n_multiple_contracts,
                COUNT(*) FILTER (WHERE n_contracts > 0 AND n_zero > 0)           AS n_franchise_zero
            FROM per_month
        """, transpose=True)

        con.execute("""
            CREATE OR REPLACE TEMP TABLE tmp_franchise AS
            SELECT eth_id, m, NULLIF(franchise_betrag, 0) AS franchise
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY eth_id, m
                    ORDER BY c_begin DESC, c_end DESC, franchise_betrag DESC  -- deterministic tie-break
                ) AS rn
                FROM tmp_month_contracts
            )
            WHERE rn = 1
        """)
        con.execute("DROP TABLE IF EXISTS tmp_month_contracts")
        con.execute("DROP TABLE IF EXISTS tmp_month_keys")

        existing = [r[1] for r in con.execute(f"PRAGMA table_info('{OUT}')").fetchall()]
        exclude = "EXCLUDE (franchise)" if "franchise" in existing else ""
        con.execute(f"""
            CREATE OR REPLACE TABLE {OUT} AS
            SELECT b.* {exclude}, f.franchise
            FROM {OUT} b
            LEFT JOIN tmp_franchise f
              ON f.eth_id = b.eth_id AND f.m = CAST(b.claim_month AS DATE)
        """)
        con.execute("DROP TABLE IF EXISTS tmp_franchise")
        checkpoint(con)

        assert scalar(con, f"SELECT COUNT(*) FROM {OUT}") == n_rows_base, "Row count changed during franchise join"
        show(con, f"SELECT COUNT(*) FILTER (WHERE franchise IS NULL) AS n_franchise_na FROM {OUT}")
        print("\nFranchise distribution (person-months):")
        show(con, f"""
            SELECT franchise, COUNT(*) AS n_person_months
            FROM {OUT} GROUP BY 1 ORDER BY 1 NULLS LAST
        """)

        # ------------------------------------------------------------------
        section("Session users (>=1 session in first 30 days)")
        # ------------------------------------------------------------------
        # Adoption date = first app activity day (tmp_engagement_person.first_day,
        # from the main script); fallback to start_month if missing.
        con.execute(f"""
            CREATE OR REPLACE TABLE session_user_flag AS
            WITH ids AS (
                SELECT DISTINCT eth_id, CAST(start_month AS DATE) AS start_month
                FROM {OUT}
            ),
            adopt AS (
                SELECT
                    i.eth_id,
                    COALESCE(CAST(e.first_day AS DATE), i.start_month) AS adoption_date,
                    e.first_day IS NULL                                AS adoption_from_start_month
                FROM ids i
                LEFT JOIN tmp_engagement_person e ON i.eth_id = e.CSS_MARKETING_ID
            ),
            sess AS (
                SELECT a.eth_id, COUNT(s.CSS_MARKETING_ID) AS n_sessions_30d
                FROM adopt a
                LEFT JOIN a365_health_sessions s
                  ON  s.CSS_MARKETING_ID = a.eth_id
                  AND s.DATE_TIME >= a.adoption_date
                  AND s.DATE_TIME <  a.adoption_date + INTERVAL {SESSION_WINDOW_DAYS} DAY
                GROUP BY a.eth_id
            )
            SELECT
                a.eth_id, a.adoption_date, a.adoption_from_start_month, s.n_sessions_30d,
                CASE WHEN s.n_sessions_30d > 0 THEN 1 ELSE 0 END AS session_user
            FROM adopt a
            JOIN sess s USING (eth_id)
        """)

        existing = [r[1] for r in con.execute(f"PRAGMA table_info('{OUT}')").fetchall()]
        exclude = "EXCLUDE (session_user)" if "session_user" in existing else ""
        con.execute(f"""
            CREATE OR REPLACE TABLE {OUT} AS
            SELECT b.* {exclude}, COALESCE(f.session_user, 0) AS session_user
            FROM {OUT} b
            LEFT JOIN session_user_flag f USING (eth_id)
        """)
        checkpoint(con)
        assert scalar(con, f"SELECT COUNT(*) FROM {OUT}") == n_rows_base, "Row count changed during session_user join"

        show(con, """
            SELECT
                COUNT(*)                                           AS n_people,
                SUM(session_user)                                  AS n_session_users,
                ROUND(100.0 * AVG(session_user), 1)                AS pct_session_users,
                COUNT(*) FILTER (WHERE adoption_from_start_month)  AS n_adoption_fallback,
                MEDIAN(n_sessions_30d) FILTER (WHERE session_user = 1) AS median_sessions_among_users
            FROM session_user_flag
        """, transpose=True)
        # Sessions before the adoption date would suggest first_day isn't the true first activity
        show(con, """
            SELECT COUNT(DISTINCT f.eth_id) AS n_with_sessions_before_adoption
            FROM session_user_flag f
            JOIN a365_health_sessions s
              ON s.CSS_MARKETING_ID = f.eth_id AND s.DATE_TIME < f.adoption_date
        """)

        # ------------------------------------------------------------------
        if n_examples > 0:
            section(f"Example timelines ({n_examples} random people; individual-level, do not share)")
            con.execute("SELECT setseed(0.42)")
            sample_ids = con.execute(f"""
                SELECT eth_id FROM (SELECT DISTINCT eth_id FROM {OUT}) ORDER BY random() LIMIT {int(n_examples)}
            """).fetchdf()["eth_id"].tolist()
            for pid in sample_ids:
                print("\n" + "=" * 90)
                print(f"eth_id: {pid[:16]}...")
                print("--- Contract periods ---")
                show(con, """
                    SELECT beginndatum, endedatum, grundversicherungstyp, franchise_betrag
                    FROM contracts_obligatory WHERE eth_id = ?
                    ORDER BY TRY_CAST(beginndatum AS DATE)
                """, params=[pid])
                print("--- Claim months ---")
                show(con, f"""
                    SELECT CAST(claim_month AS DATE) AS claim_month, age, total_claim_cost, franchise
                    FROM {OUT} WHERE eth_id = ? ORDER BY claim_month
                """, params=[pid])

        # ------------------------------------------------------------------
        section("Done")
        # ------------------------------------------------------------------
        print(f"{OUT}: {scalar(con, f'SELECT COUNT(*) FROM {OUT}'):,} rows, "
              f"{scalar(con, f'SELECT COUNT(DISTINCT eth_id) FROM {OUT}'):,} persons")
        print("Columns:", ", ".join(r[1] for r in con.execute(f"PRAGMA table_info('{OUT}')").fetchall()))

    finally:
        con.close()
        gc.collect()
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_NAME, help="Path to insurance.duckdb")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Also run full-table scans of the raw outpatient table (slow)")
    ap.add_argument("--examples", type=int, default=0,
                    help="Print N random person timelines (individual-level data; default 0)")
    args = ap.parse_args()
    main(args.db, args.diagnostics, args.examples)