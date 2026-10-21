#!/usr/bin/env python3
"""
Set up DuckDB views over S3 parquet files.

CLI:
  python3 prepare_db.py --db_name insurance.duckdb

Creates views over source data (insurance claims).
No data is copied locally — queries go directly to S3.
"""

import argparse
import duckdb
import os
from pathlib import Path
from dotenv import load_dotenv
from config import connect_db, s3

# Load .env from the repository root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BUCKET = os.environ["A365_S3_BUCKET"]
PREFIX = os.environ["A365_S3_PREFIX"]

def s3(path: str) -> str:
    return f"s3://{BUCKET}/{PREFIX}{path}"


def main():
    parser = argparse.ArgumentParser(
        description="Set up DuckDB views over S3 Parquet files."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        required=True,
        help="Path to DuckDB database file"
    )
    args = parser.parse_args()

    con = connect_db(args.db_name)

    # -----------------------------------------------------------------------
    # Core views
    # -----------------------------------------------------------------------
    print("Creating view: income_index")
    con.execute(f"CREATE OR REPLACE VIEW income_index AS SELECT * FROM read_parquet('{s3('einkommensindex.parquet')}')")

    print("Creating view: applications")
    con.execute(f"CREATE OR REPLACE VIEW applications AS SELECT * FROM read_parquet('{s3('antraege.parquet')}')")

    print("Creating view: persons_raw")
    con.execute(f"CREATE OR REPLACE VIEW persons_raw AS SELECT * FROM read_parquet('{s3('person.parquet')}')")

    # persons is deduplicated — keep as TABLE so dedup runs once on setup
    print("Creating table: persons (deduplicated)")
    con.execute("CREATE OR REPLACE TABLE persons AS SELECT DISTINCT * FROM persons_raw")

    # -----------------------------------------------------------------------
    # Official PCG definitions
    # -----------------------------------------------------------------------
    print("Creating view: pcg_wide")
    con.execute(f"CREATE OR REPLACE VIEW pcg_wide AS SELECT * FROM read_parquet('{s3('pcg_ra.parquet')}')")

    print("Creating view: pcg_ra_dict")
    con.execute(f"CREATE OR REPLACE VIEW pcg_ra_dict AS SELECT * FROM read_csv_auto('{s3('pcg_ra_dict.csv')}')")

    print("Creating table: pcg_long")
    con.execute("""
        CREATE OR REPLACE TABLE pcg_long AS
        SELECT
            eth_id,
            ra_jahr,
            pcg_jahr,
            pcg_versionsnummer,
            pcg_name,
            pcg_value
        FROM pcg_wide
        UNPIVOT (
            pcg_value FOR pcg_name IN (
                pcg_abh,
                pcg_adh,
                pcg_aik,
                pcg_alz,
                pcg_ast,
                pcg_bsr,
                pcg_car,
                pcg_cop,
                pcg_dep,
                pcg_dm1,
                pcg_dm2,
                pcg_epi,
                pcg_gla,
                pcg_hch,
                pcg_hiv,
                pcg_kho,
                pcg_kre,
                pcg_krk,
                pcg_mcr,
                pcg_msk,
                pcg_nie,
                pcg_pah,
                pcg_par,
                pcg_pso,
                pcg_psy,
                pcg_rhe,
                pcg_smc,
                pcg_smn,
                pcg_thy,
                pcg_tra,
                pcg_was,
                pcg_zfp,
                pcg_zns,
                pcg_dm2hyp
            )
        )
        WHERE pcg_value = 1
    """)

    # -----------------------------------------------------------------------
    # Contracts
    # -----------------------------------------------------------------------
    print("Creating view: contracts_obligatory")
    con.execute(f"CREATE OR REPLACE VIEW contracts_obligatory AS SELECT * FROM read_parquet('{s3('vertrag_okp.parquet')}')")

    print("Creating view: contracts_supplementary")
    con.execute(f"CREATE OR REPLACE VIEW contracts_supplementary AS SELECT * FROM read_parquet('{s3('vertrag_vvg.parquet')}')")

    # -----------------------------------------------------------------------
    # Claims: inpatient + outpatient
    # Views over a glob — no data copied, filter pushed down to S3.
    # The old script loaded these into local tables (that's where your 52 GB came from).
    # -----------------------------------------------------------------------
    print("Creating view: inpatient")
    con.execute(f"CREATE OR REPLACE VIEW inpatient AS SELECT * FROM read_parquet('{s3('leistungen_*.parquet')}') WHERE TRIM(behandlungsart) = 'stationaer'")

    print("Creating view: outpatient")
    con.execute(f"CREATE OR REPLACE VIEW outpatient AS SELECT * FROM read_parquet('{s3('leistungen_*.parquet')}') WHERE TRIM(behandlungsart) = 'ambulant'")

    con.close()
    print("\n[done]")


if __name__ == "__main__":
    main()