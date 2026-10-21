import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BUCKET = os.environ["A365_S3_BUCKET"]
CLAIMS_PREFIX = os.environ["A365_S3_PREFIX"]
USAGE_PREFIX = os.environ["A365_USAGE_S3_PREFIX"]


def s3(path: str, source: str = "claims") -> str:
    if source == "claims":
        prefix = CLAIMS_PREFIX
    elif source == "usage":
        prefix = USAGE_PREFIX
    else:
        raise ValueError(f"Unknown data source: {source}")

    return f"s3://{BUCKET}/{prefix.strip('/')}/{path}"


def connect_db(db_name: str):
    con = duckdb.connect(db_name)

    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL aws; LOAD aws;")

    con.execute("""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE S3,
            PROVIDER CREDENTIAL_CHAIN,
            REGION '{region}'
        );
    """.format(region=os.environ["AWS_REGION"]))

    return con