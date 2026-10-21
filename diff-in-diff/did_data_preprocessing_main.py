#!/usr/bin/env python3
"""
Staggered-adoption DiD data preparation.
CHECKPOINT + gc after every large table rewrite to keep the memory bounded.

Usage:
    python did_data_preprocessing_main.py
    python did_data_preprocessing_main.py --db ../insurance.duckdb --diagnostics
    nohup python -u did_data_preprocessing_main.py > dataprep.log 2>&1 &
"""
import argparse
import gc
import os
import sys
import time
import duckdb
import pandas as pd
import psutil
import sys
from pathlib import Path

# Add repository root to Python's import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data_preparation.config import connect_db

# -----------------------------
# Config
# -----------------------------
COLORS = {
    "non_users": "#004E89",
    "users": "#A0001C",
}

OUT_DIR = "./results/staggered_did_data"
DB_NAME = "./insurance.duckdb"
KVG_ONLY = True  # restrict to legal basis (mandatory insurance = 'KVG')
EXCL_COVID_TESTS = False

# Granularity of analysis
GRANULARITY = "quarterly"  

# DB settings
DUCKDB_TMP = "./duckdb_tmp"
MEMORY_LIMIT = "48GB"
THREADS = 6
MAX_TEMP_DIR = "12GiB"

MAIN = "claims_person_month_summary"

# Providers to exclude:
partner_exclusion_list = [
    "Blutspendezentren",  # 'blood donation centres'
    "Fitness",            # 'fitness' providers
]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# unbuffered output
sys.stdout.reconfigure(line_buffering=True)

_T0 = time.time()


def section(title):
    print(f"\n=== {title}  [t+{time.time() - _T0:,.0f}s, free RAM {psutil.virtual_memory().available / 1e9:.1f} GB] ===")


def checkpoint(con):
    con.execute("CHECKPOINT")
    gc.collect()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def drop_cols_if_exist(con, table, cols):
    existing = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    for col in cols:
        if col in existing:
            con.execute(f"ALTER TABLE {table} DROP COLUMN {col}")


def main(db_name, diagnostics):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DUCKDB_TMP, exist_ok=True)
    con = duckdb.connect(db_name)
    print(os.path.abspath(db_name))

    try:
        # --- memory hygiene -------------------------------------------------
        con = connect_db(str(DB_NAME))
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET max_temp_directory_size = '{MAX_TEMP_DIR}'")
        con.execute(f"SET temp_directory = '{DUCKDB_TMP}'")
        con.execute("SET preserve_insertion_order = false")

        print(con.execute(
            "SELECT current_setting('memory_limit'), "
            "current_setting('threads'), "
            "current_setting('temp_directory')"
        ).fetchall())

        if diagnostics:
            print(con.execute("SHOW ALL TABLES;").fetchdf()[["name"]].to_string())

        con.execute("""
        DROP TABLE IF EXISTS claims_person_month_summary
        """)
        checkpoint(con)

        # ------------------------------------------------------------------
        section("Initial population")
        # ------------------------------------------------------------------
        _prisma_initial = scalar(con, """
        SELECT COUNT(DISTINCT eth_id) AS n
        FROM (
            SELECT eth_id FROM inpatient WHERE eth_id IS NOT NULL
            UNION
            SELECT eth_id FROM outpatient WHERE eth_id IS NOT NULL
        )
        """)
        print(f"[PRISMA] Initial overall population (any person with >=1 inpatient or outpatient claim, 2018–2024): {_prisma_initial:,}")

        _prisma_initial = scalar(con, """
        SELECT COUNT(DISTINCT eth_id) AS n
        FROM (
            SELECT eth_id FROM inpatient WHERE eth_id IS NOT NULL
            UNION
            SELECT eth_id FROM outpatient WHERE eth_id IS NOT NULL
        ) claimants
        WHERE eth_id IN (
            SELECT DISTINCT eth_id FROM contracts_obligatory WHERE eth_id IS NOT NULL
        )
        """)
        print(f"[PRISMA] Initial overall population (any person with >=1 inpatient or outpatient claim, 2018–2024, AND held mandatory insurance at some point): {_prisma_initial:,}")

        # ------------------------------------------------------------------
        section("Filter people (OKP ∩ A365 ∩ has transaction)")
        # ------------------------------------------------------------------
        df_okp_in_a365 = con.execute("""
        SELECT DISTINCT
            o.eth_id AS CSS_MARKETING_ID
        FROM contracts_obligatory o
        JOIN a365_user_data a
          ON o.eth_id = a.CSS_MARKETING_ID
        WHERE o.eth_id IS NOT NULL
        """).fetchdf()

        df_a365_start_month = con.execute("""
        WITH okp_in_a365 AS (
          SELECT DISTINCT o.eth_id AS CSS_MARKETING_ID
          FROM contracts_obligatory o
          JOIN a365_user_data a ON o.eth_id = a.CSS_MARKETING_ID
          WHERE o.eth_id IS NOT NULL
        ),
        first_tx AS (
          SELECT
              t.CSS_MARKETING_ID,
              MIN(t.TRANSACTION_DATETIME) AS first_transaction_datetime
          FROM a365_transaction_check t
          JOIN okp_in_a365 k ON k.CSS_MARKETING_ID = t.CSS_MARKETING_ID
          WHERE t.TRANSACTION_DATETIME IS NOT NULL
          GROUP BY 1
        )
        SELECT
            CSS_MARKETING_ID,
            first_transaction_datetime,
            DATE_TRUNC('month', first_transaction_datetime) AS start_month,
            EXTRACT(year  FROM first_transaction_datetime)::INT AS start_year,
            EXTRACT(month FROM first_transaction_datetime)::INT AS start_month_num
        FROM first_tx
        """).fetchdf()

        # Keep only IDs that actually have a transaction record
        df_okp_in_a365_with_start = df_okp_in_a365.merge(
            df_a365_start_month[["CSS_MARKETING_ID", "start_month", "first_transaction_datetime"]],
            on="CSS_MARKETING_ID",
            how="inner",
        )
        del df_okp_in_a365, df_a365_start_month
        gc.collect()

        _prisma_okp_a365 = len(df_okp_in_a365_with_start)
        print(f"[PRISMA] OKP-insured AND in A365 AND has transaction record: {_prisma_okp_a365:,} unique eth_ids proceed")

        # Strip timestamps (work with date only)
        df_okp_in_a365_with_start["first_transaction_datetime"] = (
            df_okp_in_a365_with_start["first_transaction_datetime"].dt.date
        )
        df_okp_in_a365_with_start["start_month"] = (
            df_okp_in_a365_with_start["start_month"].dt.to_period("M").dt.to_timestamp().dt.date
        )
        print(f"unique CSS_MARKETING_ID: {df_okp_in_a365_with_start.CSS_MARKETING_ID.nunique():,}")

        con.register("partner_exclusion", pd.DataFrame({"partnerart": partner_exclusion_list}))

        # ------------------------------------------------------------------
        section("Inpatient claims")
        # ------------------------------------------------------------------
        con.register("ids_okp_a365", df_okp_in_a365_with_start[["CSS_MARKETING_ID"]])
        con.execute("""
        CREATE TEMP TABLE claims_inpatient_a365 AS
        SELECT i.*
        FROM inpatient i
        JOIN ids_okp_a365 ids ON i.eth_id = ids.CSS_MARKETING_ID
        """)

        if diagnostics:
            print(con.execute("""
            SELECT gesetz, COUNT(*) AS n FROM inpatient GROUP BY gesetz ORDER BY gesetz
            """).fetchdf())
            print(con.execute("""
            SELECT COUNT(*) AS n_claims_total, COUNT(DISTINCT eth_id) AS n_unique_people
            FROM claims_inpatient_a365
            """).fetchdf())

        _n_inp_before_excl = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_inpatient_a365")

        # NOTE: identical to notebook — no KVG filter is applied to inpatient here.
        con.execute("""
        CREATE TEMP TABLE claims_inpatient_a365_reduced AS
        SELECT
            eth_id,
            DATE_TRUNC('month', behandlungsbeginn) AS behandlungs_month,
            spital_aufnahmeart,
            betrag
        FROM claims_inpatient_a365 i
        WHERE behandlungsbeginn IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM partner_exclusion p WHERE p.partnerart = i.partnerart
          )
        """)

        _n_inp_after_excl = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_inpatient_a365_reduced")
        print(f"[PRISMA] Inpatient before partner excl: {_n_inp_before_excl:,}")
        print(f"[PRISMA] Inpatient after  partner excl: {_n_inp_after_excl:,} ({_n_inp_before_excl - _n_inp_after_excl:,} dropped)")

        con.execute("DROP TABLE IF EXISTS claims_inpatient_a365")
        con.unregister("ids_okp_a365")

        _prisma_inp_before = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_inpatient_a365_reduced")
        print(f"[PRISMA] Inpatient after KVG filter + non-null behandlungsbeginn: {_prisma_inp_before:,} unique eth_ids with >=1 inpatient claim")
        checkpoint(con)

        # ------------------------------------------------------------------
        section("Outpatient claims")
        # ------------------------------------------------------------------
        con.register("ids_okp_a365", df_okp_in_a365_with_start[["CSS_MARKETING_ID"]])
        con.execute("DROP TABLE IF EXISTS claims_outpatient_a365")
        con.execute("""
        CREATE TABLE claims_outpatient_a365 AS
        SELECT i.*
        FROM outpatient i
        JOIN ids_okp_a365 ids ON i.eth_id = ids.CSS_MARKETING_ID
        """)
        checkpoint(con)

        print(con.execute("""
            SELECT gesetz, COUNT(*) AS instance_count
            FROM claims_outpatient_a365
            GROUP BY gesetz
            ORDER BY instance_count DESC
        """).fetchdf())

        if diagnostics:
            print(con.execute("""
            SELECT COUNT(*) AS n_claims_total, COUNT(DISTINCT eth_id) AS n_unique_people
            FROM claims_outpatient_a365
            """).fetchdf())

        if KVG_ONLY:
            con.execute("""
            CREATE OR REPLACE TABLE claims_outpatient_a365 AS
            SELECT * FROM claims_outpatient_a365
            WHERE UPPER(TRIM(gesetz)) = 'KVG'
            """)
            checkpoint(con)

        if EXCL_COVID_TESTS:
            df_covid_before = con.execute("""
                SELECT
                    CASE WHEN TRIM(tarifposition) = '01.03.1000' THEN 'covid_selftest' ELSE 'other' END AS tarif_group,
                    COUNT(*) AS n_claims,
                    COUNT(DISTINCT eth_id) AS n_unique_people,
                    SUM(betrag) AS total_cost
                FROM claims_outpatient_a365
                GROUP BY 1
            """).fetchdf()
            print("===== BEFORE COVID exclusion =====")
            print(df_covid_before)
            total_before = con.execute("SELECT COUNT(*), COUNT(DISTINCT eth_id) FROM claims_outpatient_a365").fetchone()
            print(f"Total rows before: {total_before[0]:,}, unique people before: {total_before[1]:,}")

            con.execute("""
            CREATE OR REPLACE TABLE claims_outpatient_a365 AS
            SELECT * FROM claims_outpatient_a365
            WHERE TRIM(tarifposition) != '01.03.1000'
            """)
            checkpoint(con)

            df_covid_after = con.execute("""
                SELECT
                    CASE WHEN TRIM(tarifposition) = '01.03.1000' THEN 'covid_selftest' ELSE 'other' END AS tarif_group,
                    COUNT(*) AS n_claims,
                    COUNT(DISTINCT eth_id) AS n_unique_people,
                    SUM(betrag) AS total_cost
                FROM claims_outpatient_a365
                GROUP BY 1
            """).fetchdf()
            print("\n===== AFTER COVID exclusion =====")
            print(df_covid_after)
            total_after = con.execute("SELECT COUNT(*), COUNT(DISTINCT eth_id) FROM claims_outpatient_a365").fetchone()
            print(f"Total rows after: {total_after[0]:,}, unique people after: {total_after[1]:,}")
            print(f"\nRows removed: {total_before[0] - total_after[0]:,}")
            sel = df_covid_before[df_covid_before["tarif_group"] == "covid_selftest"]["n_unique_people"]
            if len(sel):
                print(f"People potentially affected (had ≥1 self-test claim): {sel.values[0]:,}")

        _prisma_outp_kvg = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_outpatient_a365")
        print(f"[PRISMA] Outpatient after KVG filter: {_prisma_outp_kvg:,} unique eth_ids with >=1 outpatient claim")

        con.execute("""
        CREATE TEMP TABLE claims_outpatient_a365_filtered AS
        SELECT o.*
        FROM claims_outpatient_a365 o
        LEFT JOIN partner_exclusion p ON o.partnerart = p.partnerart
        WHERE p.partnerart IS NULL
        """)
        con.execute("DROP TABLE IF EXISTS claims_outpatient_a365")
        con.unregister("ids_okp_a365")
        checkpoint(con)

        _prisma_outp_before_excl = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_outpatient_a365_filtered")
        _prisma_outp_excl_dropped = _prisma_outp_kvg - _prisma_outp_before_excl
        print(f"[PRISMA] Outpatient after partner exclusion (Blutspendezentren, Fitness): {_prisma_outp_before_excl:,} proceed, {_prisma_outp_excl_dropped:,} dropped")

        if diagnostics:
            print(con.execute("""
            SELECT COUNT(*) AS n_claims_total, COUNT(DISTINCT eth_id) AS n_unique_people
            FROM claims_outpatient_a365_filtered
            """).fetchdf())

        con.execute("""
        CREATE TEMP TABLE claims_outpatient_a365_reduced AS
        SELECT
            eth_id,
            DATE_TRUNC('month', behandlungsbeginn) AS behandlungs_month,
            spital_aufnahmeart,
            betrag
        FROM claims_outpatient_a365_filtered
        WHERE behandlungsbeginn IS NOT NULL
        """)
        con.execute("DROP TABLE IF EXISTS claims_outpatient_a365_filtered")
        checkpoint(con)

        _prisma_outp_reduced = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_outpatient_a365_reduced")
        _prisma_outp_nonnull_dropped = _prisma_outp_before_excl - _prisma_outp_reduced
        print(f"[PRISMA] Outpatient after non-null behandlungsbeginn filter: {_prisma_outp_reduced:,} proceed, {_prisma_outp_nonnull_dropped:,} dropped")

        # ------------------------------------------------------------------
        section("Merge in- and outpatient, rename to English")
        # ------------------------------------------------------------------
        con.execute("""
        CREATE TEMP TABLE claims_outpatient_a365_reduced_renamed AS
        SELECT eth_id, behandlungs_month AS claim_month,
               spital_aufnahmeart AS admission_type, betrag AS cost
        FROM claims_outpatient_a365_reduced
        """)
        con.execute("""
        CREATE TEMP TABLE claims_inpatient_a365_reduced_renamed AS
        SELECT eth_id, behandlungs_month AS claim_month,
               spital_aufnahmeart AS admission_type, betrag AS cost
        FROM claims_inpatient_a365_reduced
        """)
        con.execute("DROP TABLE IF EXISTS claims_inpatient_a365_reduced")
        con.execute("DROP TABLE IF EXISTS claims_outpatient_a365_reduced")

        con.execute("DROP TABLE IF EXISTS claims_person_month_summary")
        con.execute("""
        CREATE TABLE claims_person_month_summary AS
        WITH
        inpatient_pm AS (
          SELECT eth_id, claim_month,
                 COUNT(*) AS n_inpatient_claims, SUM(cost) AS inpatient_cost
          FROM claims_inpatient_a365_reduced_renamed
          GROUP BY 1,2
        ),
        outpatient_pm AS (
          SELECT eth_id, claim_month,
                 COUNT(*) AS n_outpatient_claims, SUM(cost) AS outpatient_cost
          FROM claims_outpatient_a365_reduced_renamed
          GROUP BY 1,2
        ),
        ids_months AS (
          SELECT eth_id, claim_month FROM inpatient_pm
          UNION
          SELECT eth_id, claim_month FROM outpatient_pm
        )
        SELECT
          m.eth_id,
          m.claim_month,
          COALESCE(i.n_inpatient_claims, 0) AS n_inpatient_claims,
          COALESCE(o.n_outpatient_claims, 0) AS n_outpatient_claims,
          COALESCE(i.inpatient_cost, 0) AS inpatient_cost,
          COALESCE(o.outpatient_cost, 0) AS outpatient_cost,
          COALESCE(i.inpatient_cost, 0) + COALESCE(o.outpatient_cost, 0) AS total_claim_cost
        FROM ids_months m
        LEFT JOIN inpatient_pm i  ON m.eth_id = i.eth_id AND m.claim_month = i.claim_month
        LEFT JOIN outpatient_pm o ON m.eth_id = o.eth_id AND m.claim_month = o.claim_month
        """)
        con.execute("DROP TABLE IF EXISTS claims_inpatient_a365_reduced_renamed")
        con.execute("DROP TABLE IF EXISTS claims_outpatient_a365_reduced_renamed")
        checkpoint(con)

        _prisma_cpm = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary")
        print(f"[PRISMA] claims_person_month_summary (union of inpatient + outpatient claimants): {_prisma_cpm:,} unique eth_ids")

        print(con.execute("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT eth_id) AS persons, COUNT(DISTINCT claim_month) AS months
        FROM claims_person_month_summary
        """).fetchdf())

        # ------------------------------------------------------------------
        section("Descriptives")
        # ------------------------------------------------------------------
        print(con.execute("""
        SELECT COUNT(DISTINCT eth_id) AS n_unique_individuals FROM claims_person_month_summary
        """).fetchdf())
        print(con.execute("""
        SELECT
            AVG(n_inpatient_claims)  AS mean_n_inpatient,
            MEDIAN(n_inpatient_claims) AS median_n_inpatient,
            STDDEV_POP(n_inpatient_claims) AS sd_n_inpatient,
            AVG(n_outpatient_claims) AS mean_n_outpatient,
            MEDIAN(n_outpatient_claims) AS median_n_outpatient,
            STDDEV_POP(n_outpatient_claims) AS sd_n_outpatient,
            AVG(outpatient_cost) AS mean_outpatient_cost,
            MEDIAN(outpatient_cost) AS median_outpatient_cost,
            STDDEV_POP(outpatient_cost) AS sd_outpatient_cost,
            AVG(inpatient_cost) AS mean_inpatient_cost,
            MEDIAN(inpatient_cost) AS median_inpatient_cost,
            STDDEV_POP(inpatient_cost) AS sd_inpatient_cost,
            AVG(total_claim_cost) AS mean_total_cost,
            MEDIAN(total_claim_cost) AS median_total_cost,
            STDDEV_POP(total_claim_cost) AS sd_total_cost
        FROM claims_person_month_summary
        """).fetchdf().T)
        print(con.execute("""
        WITH months_per_person AS (
          SELECT eth_id, COUNT(DISTINCT claim_month) AS n_months
          FROM claims_person_month_summary
          GROUP BY eth_id
        )
        SELECT AVG(n_months) AS mean_months, MEDIAN(n_months) AS median_months,
               STDDEV_POP(n_months) AS sd_months
        FROM months_per_person
        """).fetchdf())

        # ------------------------------------------------------------------
        section("Merge with A365 start time data")
        # ------------------------------------------------------------------
        con.register(
            "okp_a365_start",
            df_okp_in_a365_with_start.drop(columns=["first_transaction_datetime"]),
        )
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_with_start")
        con.execute("""
        CREATE TEMP TABLE claims_person_month_summary_with_start AS
        SELECT c.*, s.start_month
        FROM claims_person_month_summary c
        LEFT JOIN okp_a365_start s ON c.eth_id = s.CSS_MARKETING_ID
        """)
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary")
        checkpoint(con)

        _prisma_with_start = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary_with_start")
        print(f"[PRISMA] After merging start_month (left join, no exclusion): {_prisma_with_start:,} unique eth_ids proceed")

        # ------------------------------------------------------------------
        section("Persons spell match (demographics)")
        # ------------------------------------------------------------------
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_w_persons_mcov")
        con.execute("""
        CREATE TEMP TABLE claims_person_month_summary_w_persons_mcov AS
        WITH base AS (
          SELECT s.*, DATE_TRUNC('month', TRY_CAST(s.claim_month AS DATE)) AS claim_m
          FROM claims_person_month_summary_with_start s
        ),
        spells AS (
          SELECT
            eth_id,
            DATE_TRUNC('month', TRY_CAST(beginndatum AS DATE)) AS begin_m,
            DATE_TRUNC('month', TRY_CAST(endedatum  AS DATE)) AS end_m,
            alter_kunde,
            geschlecht
          FROM persons
        )
        SELECT
          b.*,
          CASE WHEN p.eth_id IS NULL THEN 0 ELSE 1 END AS in_coverage,
          p.alter_kunde AS age,
          p.geschlecht  AS gender,
          p.begin_m     AS date_begin,
          p.end_m       AS date_end
        FROM base b
        LEFT JOIN LATERAL (
          SELECT s.eth_id, s.alter_kunde, s.geschlecht, s.begin_m, s.end_m
          FROM spells s
          WHERE s.eth_id = b.eth_id
            AND s.begin_m <= b.claim_m
            AND s.end_m  >= b.claim_m
          ORDER BY s.begin_m DESC
          LIMIT 1
        ) p ON TRUE
        """)
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_with_start")
        checkpoint(con)

        _prisma_cov_in = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary_w_persons_mcov WHERE in_coverage = 1")
        _prisma_cov_out = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary_w_persons_mcov WHERE in_coverage = 0")
        print(f"[PRISMA] Persons spell match: {_prisma_cov_in:,} proceed (in_coverage=1), {_prisma_cov_out:,} dropped (no matching persons spell)")

        # ------------------------------------------------------------------
        section("Zero buffer")
        # ------------------------------------------------------------------
        con.execute("ALTER TABLE claims_person_month_summary_w_persons_mcov DROP COLUMN claim_m")
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_w_persons_mcov_zerobuf")
        con.execute("""
        CREATE TABLE claims_person_month_summary_w_persons_mcov_zerobuf AS
        WITH spells AS (
          SELECT DISTINCT eth_id, date_begin, date_end, age, gender
          FROM claims_person_month_summary_w_persons_mcov
          WHERE in_coverage = 1 AND date_begin IS NOT NULL AND date_end IS NOT NULL
        ),
        spell_months AS (
          SELECT s.eth_id, gs AS claim_month, s.date_begin, s.date_end, s.age, s.gender
          FROM spells s
          CROSS JOIN generate_series(s.date_begin, s.date_end, INTERVAL '1 month') AS t(gs)
        ),
        agg AS (
          SELECT
            eth_id, claim_month, date_begin, date_end,
            MAX(age) AS age,
            MAX(gender) AS gender,
            SUM(n_inpatient_claims) AS n_inpatient_claims,
            SUM(n_outpatient_claims) AS n_outpatient_claims,
            SUM(outpatient_cost) AS outpatient_cost,
            SUM(inpatient_cost) AS inpatient_cost,
            SUM(total_claim_cost) AS total_claim_cost
          FROM claims_person_month_summary_w_persons_mcov
          WHERE in_coverage = 1
          GROUP BY 1,2,3,4
        )
        SELECT
          m.eth_id,
          m.claim_month,
          COALESCE(a.n_inpatient_claims, 0) AS n_inpatient_claims,
          COALESCE(a.n_outpatient_claims, 0) AS n_outpatient_claims,
          COALESCE(a.outpatient_cost, 0) AS outpatient_cost,
          COALESCE(a.inpatient_cost, 0) AS inpatient_cost,
          COALESCE(a.total_claim_cost, 0) AS total_claim_cost,
          m.age,
          m.gender,
          m.date_begin,
          m.date_end
        FROM spell_months m
        LEFT JOIN agg a
          ON m.eth_id = a.eth_id
         AND m.claim_month = a.claim_month
         AND m.date_begin = a.date_begin
         AND m.date_end = a.date_end
        """)
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_w_persons_mcov")
        checkpoint(con)

        _prisma_zerobuf = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary_w_persons_mcov_zerobuf")
        _prisma_zerobuf_dropped = _prisma_cov_in - _prisma_zerobuf
        print(f"[PRISMA] After zero-buffer construction (in_coverage=1 spells only): {_prisma_zerobuf:,} proceed, {_prisma_zerobuf_dropped:,} dropped")

        print(con.execute("""
        WITH unique_persons AS (
            SELECT eth_id, ANY_VALUE(AGE) AS age, ANY_VALUE(GENDER) AS gender
            FROM claims_person_month_summary_w_persons_mcov_zerobuf
            WHERE eth_id IS NOT NULL AND AGE IS NOT NULL AND GENDER IN ('M', 'F')
            GROUP BY eth_id
        )
        SELECT gender, COUNT(*) AS n_persons, AVG(age) AS mean_age,
               MEDIAN(age) AS median_age, STDDEV_SAMP(age) AS sd_age
        FROM unique_persons
        GROUP BY gender
        ORDER BY gender;
        """).fetchdf())

        # ------------------------------------------------------------------
        section("Final panel + start_month")
        # ------------------------------------------------------------------
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_final")
        con.execute("""
        CREATE TABLE claims_person_month_summary_final AS
        SELECT eth_id, claim_month, n_inpatient_claims, n_outpatient_claims,
               outpatient_cost, inpatient_cost, total_claim_cost,
               age, gender, date_begin, date_end
        FROM claims_person_month_summary_w_persons_mcov_zerobuf
        """)
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_w_persons_mcov_zerobuf")
        checkpoint(con)

        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_final_w_start")
        con.execute("""
        CREATE TABLE claims_person_month_summary_final_w_start AS
        SELECT c.*, s.start_month
        FROM claims_person_month_summary_final c
        LEFT JOIN (
            SELECT DISTINCT CSS_MARKETING_ID AS eth_id, start_month
            FROM okp_a365_start
        ) s ON c.eth_id = s.eth_id
        """)
        con.execute("DROP TABLE IF EXISTS claims_person_month_summary_final")
        checkpoint(con)

        _prisma_final_w_start = scalar(con, "SELECT COUNT(DISTINCT eth_id) FROM claims_person_month_summary_final_w_start")
        print(f"[PRISMA] claims_person_month_summary_final_w_start: {_prisma_final_w_start:,} unique eth_ids")

        # pandas copy no longer needed after this point
        con.unregister("okp_a365_start")
        del df_okp_in_a365_with_start
        gc.collect()

        # ------------------------------------------------------------------
        section("Engagement")
        # ------------------------------------------------------------------
        print("duckdb:", duckdb.__version__)
        print("file GB:", os.path.getsize(db_name) / 1e9)
        print("free RAM GB:", psutil.virtual_memory().available / 1e9)

        con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
        con.execute(f"PRAGMA threads={THREADS}")

        con.execute("""
        CREATE OR REPLACE TEMP TABLE tx_sample AS
        SELECT t.CSS_MARKETING_ID, DATE(t.TRANSACTION_DATETIME) AS tx_date
        FROM a365_transaction_check t
        WHERE t.TRANSACTION_DATETIME IS NOT NULL
          AND t.CSS_MARKETING_ID IN (
              SELECT DISTINCT eth_id
              FROM claims_person_month_summary_final_w_start
              WHERE eth_id IS NOT NULL
          )
        """)

        con.execute("""
        CREATE OR REPLACE TEMP TABLE eng_person AS
        WITH first_tx AS (
            SELECT CSS_MARKETING_ID, MIN(tx_date) AS first_day
            FROM tx_sample
            GROUP BY 1
            HAVING MIN(tx_date) >= DATE '2021-01-01'
        )
        SELECT
            s.CSS_MARKETING_ID,
            f.first_day,
            COUNT(DISTINCT s.tx_date) FILTER (
                WHERE s.tx_date >= f.first_day
                  AND s.tx_date <  f.first_day + INTERVAL 30 DAY
            ) AS n_active_days_30d,
            30 AS window_days_30d,
            COUNT(DISTINCT s.tx_date) FILTER (
                WHERE s.tx_date >= f.first_day
                  AND s.tx_date <  f.first_day + INTERVAL 30 DAY
            )::DOUBLE / 30.0 AS engagement_density_30d,
            MIN(s.tx_date) AS first_day_all,
            MAX(s.tx_date) AS last_day_all,
            COUNT(DISTINCT s.tx_date) AS n_active_days_all,
            DATE_DIFF('day', MIN(s.tx_date), MAX(s.tx_date)) + 1 AS window_days_all,
            COUNT(DISTINCT s.tx_date)::DOUBLE
                / (DATE_DIFF('day', MIN(s.tx_date), MAX(s.tx_date)) + 1) AS engagement_density_all
        FROM tx_sample s
        JOIN first_tx f USING (CSS_MARKETING_ID)
        GROUP BY s.CSS_MARKETING_ID, f.first_day
        """)

        con.execute("""
        CREATE OR REPLACE TABLE tmp_engagement_person AS
        SELECT
            *,
            NTILE(4)  OVER (ORDER BY engagement_density_30d) AS engagement_q_30d,
            NTILE(10) OVER (ORDER BY engagement_density_30d) AS engagement_decile_30d,
            NTILE(4)  OVER (ORDER BY engagement_density_all) AS engagement_q_alltime,
            NTILE(10) OVER (ORDER BY engagement_density_all) AS engagement_decile_alltime
        FROM eng_person
        """)

        print(con.execute("""
        SELECT
            engagement_q_30d AS quartile,
            COUNT(*) AS n,
            MIN(n_active_days_30d) AS min_active_days,
            MAX(n_active_days_30d) AS max_active_days,
            ROUND(MIN(engagement_density_30d), 3) AS min_density,
            ROUND(MAX(engagement_density_30d), 3) AS max_density
        FROM tmp_engagement_person
        GROUP BY engagement_q_30d
        ORDER BY engagement_q_30d
        """).df())

        con.execute("DROP TABLE IF EXISTS tx_sample")
        con.execute("DROP TABLE IF EXISTS eng_person")
        checkpoint(con)

        # minimal_engager: points earned on exactly one distinct day within first 30 days
        con.execute("""
        CREATE OR REPLACE TEMP TABLE point_days AS
        SELECT
            CAST(t.CSS_MARKETING_ID AS VARCHAR) AS CSS_MARKETING_ID,
            COUNT(DISTINCT DATE(t.TRANSACTION_DATETIME)) AS n_point_days_30d
        FROM a365_transaction_check t
        JOIN tmp_engagement_person e
            ON CAST(t.CSS_MARKETING_ID AS VARCHAR) = CAST(e.CSS_MARKETING_ID AS VARCHAR)
        WHERE t.TRANSACTION_DATETIME IS NOT NULL
          AND UPPER(TRIM(t.TRANSACTION_TYPE)) = 'CREDIT'
          AND t.TRANSACTION_VALUE > 0
          AND DATE(t.TRANSACTION_DATETIME) >= e.first_day
          AND DATE(t.TRANSACTION_DATETIME) <  e.first_day + INTERVAL 30 DAY
        GROUP BY 1
        """)
        con.execute("""
        CREATE OR REPLACE TABLE tmp_engagement_person AS
        SELECT
            e.*,
            COALESCE(p.n_point_days_30d, 0) AS n_point_days_30d,
            CASE
                WHEN p.n_point_days_30d = 1  THEN 1
                WHEN p.n_point_days_30d >= 2 THEN 0
                ELSE NULL
            END AS minimal_engager
        FROM tmp_engagement_person e
        LEFT JOIN point_days p
            ON CAST(e.CSS_MARKETING_ID AS VARCHAR) = p.CSS_MARKETING_ID
        """)
        con.execute("DROP TABLE IF EXISTS point_days")
        checkpoint(con)

        print(con.execute("""
            SELECT minimal_engager, COUNT(*) AS n,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
            FROM tmp_engagement_person
            GROUP BY 1 ORDER BY 1
        """).df())

        _prisma_eng_total = scalar(con, "SELECT COUNT(DISTINCT CSS_MARKETING_ID) FROM tmp_engagement_person")
        _prisma_eng_null = scalar(con, """
            SELECT COUNT(DISTINCT eth_id)
            FROM claims_person_month_summary_final_w_start
            WHERE eth_id NOT IN (SELECT CSS_MARKETING_ID FROM tmp_engagement_person)
        """)
        print(f"[PRISMA] Engagement assignment (first tx >= 2021-01-01): {_prisma_eng_total:,} get engagement metrics, {_prisma_eng_null:,} will have NULL engagement cols (stay in panel)")
        print("free RAM GB:", psutil.virtual_memory().available / 1e9)

        con.execute(f"""
        CREATE OR REPLACE TABLE {MAIN} AS
        SELECT
            t.*,
            e.engagement_density_30d,
            e.engagement_q_30d,
            e.engagement_decile_30d,
            e.engagement_density_all,
            e.engagement_q_alltime,
            e.engagement_decile_alltime,
            e.minimal_engager
        FROM claims_person_month_summary_final_w_start t
        LEFT JOIN tmp_engagement_person e ON t.eth_id = e.CSS_MARKETING_ID
        """)
        checkpoint(con)

        _prisma_engq = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {MAIN}")
        print(f"[PRISMA] claims_person_month_summary_final_w_start_engagementq: {_prisma_engq:,} unique eth_ids (final analysis table before covariate joins)")

        print(con.execute("""
        WITH tx AS (
          SELECT CSS_MARKETING_ID, DATE(TRANSACTION_DATETIME) AS tx_date
          FROM a365_transaction_check
          WHERE TRANSACTION_DATETIME IS NOT NULL
        ),
        span AS (
          SELECT CSS_MARKETING_ID,
                 DATE_DIFF('day', MIN(tx_date), MAX(tx_date)) + 1 AS span_days
          FROM tx GROUP BY 1
        )
        SELECT AVG(CASE WHEN span_days <= 90 THEN 1 ELSE 0 END) AS share_span_le_90,
               quantile_cont(span_days, 0.5) AS median_span_days
        FROM span;
        """).fetch_df())

        # ------------------------------------------------------------------
        section("Prior-year HC utilization")
        # ------------------------------------------------------------------
        drop_cols_if_exist(con, MAIN, ("prior_total_cost", "prior_inpatient_cost"))
        con.execute("DROP TABLE IF EXISTS tmp_prior_year_cost")
        con.execute(f"""
        CREATE TEMP TABLE tmp_prior_year_cost AS
        SELECT
            eth_id,
            SUM(CASE WHEN claim_month >= (start_month - INTERVAL '12 months')
                      AND claim_month <  start_month
                     THEN total_claim_cost ELSE 0 END) AS prior_total_cost,
            SUM(CASE WHEN claim_month >= (start_month - INTERVAL '12 months')
                      AND claim_month <  start_month
                     THEN inpatient_cost ELSE 0 END) AS prior_inpatient_cost
        FROM {MAIN}
        WHERE start_month IS NOT NULL
        GROUP BY eth_id
        """)
        print(con.execute("""
            SELECT COUNT(*) AS n_people,
                   AVG(prior_total_cost)     AS mean_prior_total,
                   AVG(prior_inpatient_cost) AS mean_prior_inpatient
            FROM tmp_prior_year_cost
        """).fetchdf())

        con.execute(f"""
        CREATE OR REPLACE TABLE {MAIN} AS
        SELECT t.*, p.prior_total_cost, p.prior_inpatient_cost
        FROM {MAIN} t
        LEFT JOIN tmp_prior_year_cost p USING (eth_id)
        """)
        con.execute("DROP TABLE IF EXISTS tmp_prior_year_cost")
        checkpoint(con)

        print(con.execute(f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT eth_id) AS n_people,
                SUM(CASE WHEN prior_total_cost     IS NULL THEN 1 ELSE 0 END) AS n_null_total,
                SUM(CASE WHEN prior_inpatient_cost IS NULL THEN 1 ELSE 0 END) AS n_null_inpatient,
                AVG(prior_total_cost)     AS mean_prior_total,
                AVG(prior_inpatient_cost) AS mean_prior_inpatient
            FROM {MAIN}
        """).fetchdf())

        # ------------------------------------------------------------------
        section("PCG flag")
        # ------------------------------------------------------------------
        pcg_cols = con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'pcg_wide'
              AND column_name LIKE 'pcg_%'
              AND column_name NOT IN ('pcg_jahr', 'pcg_versionsnummer')
            ORDER BY column_name
        """).fetchdf()["column_name"].tolist()
        pcg_any_condition = " OR ".join([f"p.{col} = 1" for col in pcg_cols])

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE preexisting_pcg_flag AS
            SELECT DISTINCT
                s.eth_id,
                CASE WHEN p.eth_id IS NOT NULL THEN 1 ELSE 0 END AS preexisting_pcg
            FROM (
                SELECT DISTINCT eth_id, EXTRACT(year FROM TRY_CAST(start_month AS DATE)) AS adoption_year
                FROM {MAIN}
                WHERE start_month IS NOT NULL
            ) s
            LEFT JOIN pcg_wide p
                ON p.eth_id = s.eth_id
               AND p.pcg_jahr <= s.adoption_year
               AND ({pcg_any_condition})
        """)
        print(con.execute("""
            SELECT preexisting_pcg, COUNT(*) AS n FROM preexisting_pcg_flag GROUP BY preexisting_pcg
        """).fetchdf())

        drop_cols_if_exist(con, MAIN, ("preexisting_pcg",))
        con.execute(f"""
            CREATE OR REPLACE TABLE {MAIN} AS
            SELECT t.*, COALESCE(f.preexisting_pcg, 0) AS preexisting_pcg
            FROM {MAIN} t
            LEFT JOIN preexisting_pcg_flag f USING (eth_id)
        """)
        con.execute("DROP TABLE IF EXISTS preexisting_pcg_flag")
        checkpoint(con)

        print(con.execute(f"""
            SELECT preexisting_pcg, COUNT(*) AS n FROM {MAIN} GROUP BY preexisting_pcg
        """).fetchdf())

        # ------------------------------------------------------------------
        section("Income index + BMI")
        # ------------------------------------------------------------------
        before = scalar(con, f"SELECT COUNT(*) FROM {MAIN}")

        drop_cols_if_exist(con, MAIN, ("income_index",))
        con.execute(f"""
            CREATE OR REPLACE TABLE {MAIN} AS
            WITH income_ranked AS (
                SELECT
                    eth_id,
                    einkommensindex AS income_index,
                    ROW_NUMBER() OVER (
                        PARTITION BY eth_id
                        ORDER BY TRY_CAST(beginndatum AS DATE) DESC NULLS LAST
                    ) AS rn
                FROM income_index
                WHERE eth_id IS NOT NULL
            ),
            income_latest AS (
                SELECT eth_id, income_index FROM income_ranked WHERE rn = 1
            )
            SELECT t.*, il.income_index
            FROM {MAIN} t
            LEFT JOIN income_latest il ON t.eth_id = il.eth_id
        """)
        checkpoint(con)

        drop_cols_if_exist(con, MAIN, ("bmi",))
        con.execute(f"""
            CREATE OR REPLACE TABLE {MAIN} AS
            WITH user_clean AS (
                SELECT
                    CSS_MARKETING_ID AS eth_id,
                    (WEIGHT / POWER(HEIGHT / 100.0, 2)) AS bmi_raw
                FROM a365_user_data
                WHERE CSS_MARKETING_ID IS NOT NULL
                  AND HEIGHT IS NOT NULL
                  AND WEIGHT IS NOT NULL
                  AND HEIGHT BETWEEN 100 AND 250
                  AND WEIGHT BETWEEN 25 AND 300
            ),
            user_bmi AS (
                SELECT eth_id, AVG(bmi_raw) AS bmi
                FROM user_clean
                WHERE bmi_raw BETWEEN 10 AND 80
                GROUP BY eth_id
            )
            SELECT t.*, ub.bmi
            FROM {MAIN} t
            LEFT JOIN user_bmi ub ON t.eth_id = ub.eth_id
        """)
        checkpoint(con)

        _prisma_post_covjoin = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {MAIN}")
        _prisma_income_null = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {MAIN} WHERE income_index IS NULL")
        _prisma_bmi_null = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {MAIN} WHERE bmi IS NULL")
        print(f"[PRISMA] After income + BMI join (left joins, no exclusion): {_prisma_post_covjoin:,} unique eth_ids remain")
        print(f"[PRISMA]   income_index NULL: {_prisma_income_null:,} eth_ids")
        print(f"[PRISMA]   bmi NULL: {_prisma_bmi_null:,} eth_ids")

        after = scalar(con, f"SELECT COUNT(*) FROM {MAIN}")
        print("before:", before)
        print("after:", after)
        assert before == after, "Row count changed during covariate joins (duplicate keys?)"

        # ------------------------------------------------------------------
        section("start_month coverage filter")
        # ------------------------------------------------------------------
        n_total, n_adopters, n_no_baseline = con.execute(f"""
            SELECT
                COUNT(DISTINCT eth_id) AS n_total,
                COUNT(DISTINCT CASE WHEN start_month IS NOT NULL THEN eth_id END) AS n_adopters,
                COUNT(DISTINCT CASE
                    WHEN start_month IS NOT NULL
                     AND eth_id NOT IN (
                         SELECT eth_id FROM {MAIN} WHERE claim_month = start_month
                     )
                    THEN eth_id
                END) AS n_no_baseline_row
            FROM {MAIN}
        """).fetchone()
        print(f"[PRISMA] Adopters before start_month coverage check: {n_adopters:,}")
        print(f"[PRISMA] Adopters with no baseline row (start_month not covered by contract): {n_no_baseline:,}")
        print(f"[PRISMA] Adopters proceeding after start_month coverage check: {n_adopters - n_no_baseline:,}")

        con.execute(f"""
            CREATE OR REPLACE TABLE {MAIN} AS
            SELECT t.*
            FROM {MAIN} t
            WHERE t.start_month IS NULL
               OR t.eth_id IN (
                   SELECT eth_id FROM {MAIN} WHERE claim_month = start_month
               )
        """)
        checkpoint(con)

        _n_after = scalar(con, f"SELECT COUNT(DISTINCT eth_id) FROM {MAIN}")
        print(f"[PRISMA] Unique eth_ids in {MAIN} after start_month coverage filter: {_n_after:,}")

        section("Done")

    finally:
        con.close()
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_NAME, help="Path to insurance.duckdb")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Also print non-PRISMA inspection tables")
    args = ap.parse_args()
    main(args.db, args.diagnostics)