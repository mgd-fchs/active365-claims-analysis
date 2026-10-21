
# Active365 Claims Analysis

Analysis code accompanying the study evaluating the effects of adopting the active365 digital health intervention on healthcare utilisation and costs.

## 1. Environment setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data preparation

The underlying health insurance claims and digital health intervention (DHI) usage data cannot be shared publicly due to privacy and contractual restrictions. However, we provide the scripts used to construct the analytical database from the original files stored on AWS S3. Running these scripts requires authorised access to the respective data sources.

`prepare_db.py` creates DuckDB views of the insurance claims data.

```bash
python prepare_db.py --db_name insurance.duckdb
```

`prepare_db_a365.py` creates additional views of the Active365 usage data within the same DuckDB database.

```bash
python prepare_db_a365.py --db_name insurance.duckdb
```

Both scripts must be run sequentially using the same database path.

## 3. Data documentation

See [data_documentation.md](data_documentation.md) for descriptions of the data sources, variable definitions, coding conventions, and linkage between datasets.
