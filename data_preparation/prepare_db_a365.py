#!/usr/bin/env python3
"""
Load A365 parquet files from S3 into DuckDB as views.

CLI:
  python3 prepare_db_a365.py --db_name insurance.duckdb

Reads parquet files from source and creates views:
  a365_daily_challenges, a365_health_activities, etc.

"""

from __future__ import annotations

import argparse
import sys

import duckdb



#!/usr/bin/env python3
"""
Create DuckDB views over Active365 usage data stored in S3.

Usage:
    python data_preparation/prepare_db_a365.py --db_name insurance.duckdb

Requires authorised access to the underlying data.
"""

import argparse
import sys

from config import connect_db, s3


PARQUET_FILES: dict[str, str] = {
    "daily_challenges":     "daily_challenges.parquet",
    "health_activities":    "health_activities.parquet",
    "health_sessions":      "health_sessions.parquet",
    "onboarding_questions": "onboarding_questions.parquet",
    "program_recipe":       "program_recipe.parquet",
    "transaction_check":    "transaction_check.parquet",
    "user_data":            "user_data.parquet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create DuckDB views over Active365 S3 Parquet files."
    )
    p.add_argument(
        "--db_name",
        required=True,
        help="Path to DuckDB database file"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    con = connect_db(args.db_name)

    try:
        for key, filename in PARQUET_FILES.items():
            view_name = f"a365_{key}"
            path = s3(filename, source="usage")

            print(f"[view] {view_name}")

            con.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{path}')
            """)

            print(f"[ok] {view_name}")

    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    finally:
        con.close()

    print("\n[done]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())