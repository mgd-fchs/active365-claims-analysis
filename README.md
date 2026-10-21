
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
python data_preparation/prepare_db.py --db_name insurance.duckdb
```

`prepare_db_a365.py` creates additional views of the Active365 usage data within the same DuckDB database.

```bash
python data_preparation/prepare_db_a365.py --db_name insurance.duckdb
```

Both scripts must be run sequentially using the same database path.

## 3. Data documentation

See [data_documentation.md](data_documentation.md) for descriptions of the data sources, variable definitions, coding conventions, and linkage between datasets.

## 4. DiD data preparation

Data preparation for diff-in-diff analyses is split into three scripts in `diff-in-diff/`, which run in sequence on `insurance.duckdb`. Each step builds on the table written by the previous one. We split it up for runtime and memory reasons, so that only the steps an analysis needs have to be run:

- **Main DiD analyses:** step 1 only.
- **Analyses by claim type** (e.g. emergency vs. planned admissions, illness vs. accident): steps 1 and 2.
- **Exploratory behavioural-pathway analyses** (change in daily steps): all three steps.

This builds a per-person-per month dataframe, which the DiD analysis script then aggregates to a quarterly granularity.

To run:
```bash
python diff-in-diff/did_data_preprocessing_main.py          # step 1
python diff-in-diff/did_data_preprocessing_claims_types.py  # step 2 (requires step 1)
python diff-in-diff/did_data_preprocessing_steps_diff.py    # step 3 (requires step 2)
```

### Step 1: `did_data_preprocessing_main.py`

Runs the joins and data preparation for the main DiD analyses and writes the person-month panel `claims_person_month_summary`. The script:

1. restricts claims to people with mandatory health insurance (KVG) who have used the app and have at least one app transaction; adoption month = month of the first transaction;
2. keeps KVG outpatient claims and excludes blood-donation and fitness providers;
3. aggregates in- and outpatient claims and costs per person-month;
4. matches each month to a valid contract period (age, sex) and fills months without claims with zeros;
5. adds engagement metrics (activity density in the first 30 days and overall), prior-year healthcare costs, a pre-existing chronic condition flag (pharmaceutical cost groups, PCG), income index and BMI;
6. drops adopters whose adoption month is not covered by a contract.

The script outputs the sample sizes corresponding to the PRISMA chart in the publication up to and including the step matching unique IDs to a valid contract period (N = 143,719). The last three exclusion steps are handled and output in the DiD script itself.

### Step 2: `did_data_preprocessing_claims_types.py`

Adds claim-type variables to the panel and writes `claims_person_month_summary_final_w_inpatient_type`:

- inpatient admissions, costs and mean length of stay, separately for emergency and planned admissions;
- in- and outpatient claims and costs by reason for treatment: illness, accident, and all claims excluding maternity;
- the deductible (*franchise*) of the contract active in each month (0 coded as missing);
- `session_user`: at least one app session in the first 30 days after adoption.

These variables use KVG claims only.

### Step 3: `did_data_preprocessing_steps_diff.py`

Adds person-level step-count variables to `claims_person_month_summary_final_w_inpatient_type`:

- `steps_period_1`, `steps_period_2`: mean daily steps on days 1–90 and 91–180 after the first app transaction. Each is only computed if step data are available on at least 75% of the days in that period (68 of 90).
- `steps_diff`: `steps_period_2 − steps_period_1`, available only if both periods meet the coverage threshold.

## 5. Sample descriptives

`descriptive_analyses` contains code to reproduce the descriptive figures and tables, e.g. the sample characteristics table, the PCG prevalence plots and the outpatient pre-trend checks. Both notebooks run against `insurance.duckdb`. They use the outputs of the DiD steps: `unique_ids_final.csv` (the final analysed sample, N = 100,281).

### `sample_descriptives.ipynb`

Compares the DiD analysis sample with a comparison population of all insured adults. That population is everyone with ≥1 inpatient or outpatient claim in 2018–2024 and a mandatory (KVG) contract overlapping that period, who was aged ≥18 in at least one person record (N = 1,623,841). The comparison population includes the DHI adopters.

| Section | Output |
|---|---|
| Comparison-population waterfall | Printed counts per inclusion step |
| Income index distribution | `results/descriptives/income_distribution_did_vs_pop.png` |
| Prevalence of pharmaceutical cost groups (PCGs), top 15 | `results/descriptives/pcg_prevalence_did_vs_pop.png`, `pcg_prevalence_top15_did_vs_pop.png` |
| Characteristics table (age, sex, canton, income, yearly costs and claims, deductible, number of PCGs) | `results/descriptives/characteristics_table_did_vs_pop.csv` |
| App-only characteristics for adopters (self-reported BMI with WHO categories, daily steps, sessions per active month) | `adopters_bmi_steps_sessions_summary.csv` |
| Adopter characteristics by adoption year (2020–2024), including pre-/post-adoption observation time | `did_sample_characteristics_by_adoption_year.csv` |
| Claims count and volume by legal basis (*Gesetz*), source and year, comparison population | `results/descriptives/gesetz_wide_filtered_pop.tex` |
| Missingness by variable, analysis sample vs. comparison population | `results/descriptives/missingness_by_adoption_status.csv` |

### `outpatient_pretrend_investigation.ipynb`

Investigates the rise in outpatient claims in the quarters just before adoption (pre-trends violation). It compares months −1 to −3 with months −4 to −6 before each person's adoption, using KVG outpatient claims only:
